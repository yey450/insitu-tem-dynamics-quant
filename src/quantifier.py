"""原位序列纳米颗粒定量分析模块。

逐帧处理流程：
    1. 背景展平：下采样灰度闭运算估计照明背景，I / I_bg 得到相对透射率；
    2. 去噪：OpenCV 快速非局部均值（NLM）+ 轻度高斯平滑；
    3. 自适应分割：Otsu 全局阈值与局部自适应阈值联合判决，形态学清理；
    4. 分水岭分离：距离变换 + h-maxima 标记，分离粘连颗粒；
    5. 统计：剔除触碰有效区域边界的颗粒，提取数量、等效直径（nm）与圆度。

动力学拟合：对平均半径 R(t) 拟合广义幂律粗化模型

    R(t)^n - R_0^n = k t   →   R(t) = (R_0^n + k t)^(1/n)

其中 n = 3 对应扩散控制的 LSW 熟化，n = 2 对应界面反应控制。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.optimize import curve_fit
from skimage import filters, measure, morphology, segmentation


def _remove_small(arr: np.ndarray, min_area: int) -> np.ndarray:
    """移除面积小于 ``min_area`` 的连通域，兼容 scikit-image 0.26 前后的参数变化。"""
    try:  # scikit-image >= 0.26：max_size 移除 “≤ max_size” 的对象
        return morphology.remove_small_objects(arr, max_size=min_area - 1)
    except TypeError:  # 旧版本：min_size 移除 “< min_size” 的对象
        return morphology.remove_small_objects(arr, min_size=min_area)


@dataclass
class SegmentationParams:
    """分割参数（物理尺度以 nm 给出，内部按像素尺寸换算）。"""

    background_scale_nm: float = 15.0   # 背景估计闭运算尺度（需大于最大颗粒直径）
    nlm_strength: float = 0.9           # NLM 强度系数（乘以估计噪声标准差）
    smooth_sigma_nm: float = 0.25
    local_block_nm: float = 30.0        # 局部自适应阈值窗口
    min_radius_nm: float = 0.6          # 最小可检测等效半径
    watershed_h_nm: float = 0.35        # h-maxima 动态阈值
    border_exclusion: bool = True


@dataclass
class QuantificationResult:
    labels: np.ndarray            # (T, H, W) int32 逐帧标签图
    particles: pd.DataFrame       # 逐颗粒测量
    summary: pd.DataFrame         # 逐帧统计
    contrast: np.ndarray          # (T, H, W) float32 预处理后的衬度图（颗粒为正）


# ----------------------------------------------------------------------------
# 预处理
# ----------------------------------------------------------------------------
def flatten_background(frame: np.ndarray, pixel_size_nm: float, scale_nm: float) -> np.ndarray:
    """估计缓变照明背景（暗颗粒被闭运算抹除）并返回相对透射率 I / I_bg。"""
    img = frame.astype(np.float32)
    factor = 4
    small = cv2.resize(img, None, fx=1 / factor, fy=1 / factor, interpolation=cv2.INTER_AREA)
    size = max(int(round(scale_nm / pixel_size_nm / factor)), 3)
    bg_small = ndimage.grey_closing(small, size=(size, size))
    bg_small = ndimage.gaussian_filter(bg_small, size / 2)
    bg = cv2.resize(bg_small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    return img / np.maximum(bg, 1e-6)


def denoise(contrast: np.ndarray, pixel_size_nm: float, params: SegmentationParams) -> np.ndarray:
    """快速非局部均值去噪（8 位量化后进行）+ 轻度高斯平滑。"""
    lo, hi = np.percentile(contrast, [0.1, 99.9])
    scaled = np.clip((contrast - lo) / (hi - lo + 1e-12), 0, 1)
    u8 = (scaled * 255).astype(np.uint8)
    # 以相邻像素差分的 MAD 估计噪声标准差（对结构不敏感）
    sigma = 1.4826 * np.median(np.abs(np.diff(u8.astype(np.float32), axis=1))) / np.sqrt(2)
    h = float(np.clip(params.nlm_strength * sigma, 3, 60))
    den = cv2.fastNlMeansDenoising(u8, None, h=h, templateWindowSize=7, searchWindowSize=21)
    den = den.astype(np.float32) / 255 * (hi - lo) + lo
    return ndimage.gaussian_filter(den, params.smooth_sigma_nm / pixel_size_nm)


def preprocess_frame(frame: np.ndarray, pixel_size_nm: float, params: SegmentationParams) -> np.ndarray:
    """返回颗粒为正值的去噪衬度图 C = 1 - I/I_bg。"""
    flat = flatten_background(frame, pixel_size_nm, params.background_scale_nm)
    return 1.0 - denoise(flat, pixel_size_nm, params)


# ----------------------------------------------------------------------------
# 分割
# ----------------------------------------------------------------------------
def segment_frame(
    contrast: np.ndarray,
    pixel_size_nm: float,
    params: SegmentationParams,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """自适应阈值 + 分水岭分离，返回标签图。"""
    vals = contrast[valid_mask] if valid_mask is not None else contrast.ravel()
    t_global = filters.threshold_otsu(vals)
    block = int(round(params.local_block_nm / pixel_size_nm)) | 1
    t_local = filters.threshold_local(contrast, block_size=block, method="gaussian")
    # 局部阈值用于适应残余背景起伏；全局阈值的一半作为下限以抑制背景误检
    mask = (contrast > np.maximum(t_local, 0.5 * t_global)) & (contrast > 0.75 * t_global)
    if valid_mask is not None:
        mask &= valid_mask

    min_r_px = params.min_radius_nm / pixel_size_nm
    min_area = max(int(np.pi * min_r_px ** 2), 4)
    mask = ndimage.binary_opening(mask, structure=morphology.disk(1))
    mask = ndimage.binary_fill_holes(mask)
    mask = _remove_small(mask, min_area)

    # 分水岭：平滑距离图的 h-maxima 作为种子，分离颈缩连接的粘连颗粒
    dist = ndimage.distance_transform_edt(mask)
    dist = ndimage.gaussian_filter(dist, 1.0)
    h = params.watershed_h_nm / pixel_size_nm
    markers = measure.label(morphology.h_maxima(dist, h) & mask)
    labels = segmentation.watershed(-dist, markers, mask=mask)

    # 剔除过小碎片与触碰有效区域边界的颗粒（避免截断偏差）
    labels = _remove_small(labels, min_area)
    if params.border_exclusion:
        invalid = ~valid_mask if valid_mask is not None else np.zeros_like(mask)
        invalid = invalid.copy()
        invalid[[0, -1], :] = True
        invalid[:, [0, -1]] = True
        touching = np.unique(labels[ndimage.binary_dilation(invalid, iterations=2)])
        labels[np.isin(labels, touching[touching > 0])] = 0
    labels, _, _ = segmentation.relabel_sequential(labels)
    return labels.astype(np.int32)


def measure_particles(labels: np.ndarray, pixel_size_nm: float) -> pd.DataFrame:
    """提取等效直径（nm）、面积、圆度等形貌参数。"""
    props = measure.regionprops_table(
        labels,
        properties=("label", "area", "perimeter", "equivalent_diameter_area", "centroid", "eccentricity"),
    )
    df = pd.DataFrame(props).rename(
        columns={"centroid-0": "centroid_y_px", "centroid-1": "centroid_x_px"}
    )
    df["area_nm2"] = df["area"] * pixel_size_nm ** 2
    df["diameter_nm"] = df["equivalent_diameter_area"] * pixel_size_nm
    df["radius_nm"] = df["diameter_nm"] / 2
    perim = np.maximum(df["perimeter"].to_numpy(), 1e-6)
    df["circularity"] = np.clip(4 * np.pi * df["area"] / perim ** 2, 0, 1)
    return df.drop(columns=["equivalent_diameter_area"])


def quantify_stack(
    aligned: np.ndarray,
    times_s: np.ndarray,
    pixel_size_nm: float,
    valid_mask: Optional[np.ndarray] = None,
    params: Optional[SegmentationParams] = None,
) -> QuantificationResult:
    """对配准后的序列逐帧分割与统计。"""
    params = params or SegmentationParams()
    labels_all = np.zeros(aligned.shape, dtype=np.int32)
    contrast_all = np.zeros(aligned.shape, dtype=np.float32)
    per_particle: List[pd.DataFrame] = []
    rows: List[Dict] = []
    for k in range(aligned.shape[0]):
        contrast = preprocess_frame(aligned[k], pixel_size_nm, params)
        labels = segment_frame(contrast, pixel_size_nm, params, valid_mask)
        df = measure_particles(labels, pixel_size_nm)
        df.insert(0, "frame", k)
        df.insert(1, "time_s", times_s[k])
        per_particle.append(df)
        labels_all[k], contrast_all[k] = labels, contrast
        n = len(df)
        d = df["diameter_nm"].to_numpy()
        rows.append(
            dict(
                frame=k,
                time_s=times_s[k],
                n_particles=n,
                mean_diameter_nm=d.mean() if n else np.nan,
                std_diameter_nm=d.std(ddof=1) if n > 1 else np.nan,
                median_diameter_nm=np.median(d) if n else np.nan,
                mean_radius_nm=d.mean() / 2 if n else np.nan,
                sem_radius_nm=d.std(ddof=1) / 2 / np.sqrt(n) if n > 1 else np.nan,
                mean_circularity=df["circularity"].mean() if n else np.nan,
            )
        )
    return QuantificationResult(
        labels=labels_all,
        particles=pd.concat(per_particle, ignore_index=True),
        summary=pd.DataFrame(rows),
        contrast=contrast_all,
    )


# ----------------------------------------------------------------------------
# 动力学拟合
# ----------------------------------------------------------------------------
def coarsening_model(t: np.ndarray, r0: float, k: float, n: float) -> np.ndarray:
    """R(t) = (R0^n + k t)^(1/n)。"""
    return np.power(np.clip(r0 ** n + k * t, 1e-12, None), 1.0 / n)


def _r_squared(y: np.ndarray, y_fit: np.ndarray) -> float:
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def fit_coarsening_kinetics(
    times_s: np.ndarray,
    radius_nm: np.ndarray,
    sigma: Optional[np.ndarray] = None,
    fixed_n: Optional[float] = None,
) -> Dict[str, float]:
    """拟合 R^n - R0^n = k t。``fixed_n`` 为 None 时同时拟合指数 n。

    返回包含 n、k、R0 及其标准误差、R² 的字典（k 的单位为 nm^n / s）。
    """
    t = np.asarray(times_s, float)
    r = np.asarray(radius_nm, float)
    ok = np.isfinite(r)
    if sigma is not None:
        sigma = np.asarray(sigma, float)
        ok &= np.isfinite(sigma) & (sigma > 0)
    t, r = t[ok], r[ok]
    sig = sigma[ok] if sigma is not None else None
    if t.size < 3:
        raise ValueError("动力学拟合至少需要 3 个有效时间点。")

    n0 = fixed_n if fixed_n is not None else 3.0
    k0 = max((r[-1] ** n0 - r[0] ** n0) / max(t[-1] - t[0], 1e-9), 1e-6)
    if fixed_n is None:
        f = coarsening_model
        p0, lb, ub = [r[0], k0, n0], [1e-3, 0.0, 1.0], [np.inf, np.inf, 6.0]
    else:
        def f(tt, r0, k):
            return coarsening_model(tt, r0, k, fixed_n)
        p0, lb, ub = [r[0], k0], [1e-3, 0.0], [np.inf, np.inf]

    popt, pcov = curve_fit(f, t, r, p0=p0, sigma=sig, absolute_sigma=False,
                           bounds=(lb, ub), maxfev=20000)
    perr = np.sqrt(np.clip(np.diag(pcov), 0, None))
    r_fit = f(t, *popt)
    out = dict(
        R0_nm=float(popt[0]), R0_err=float(perr[0]),
        k=float(popt[1]), k_err=float(perr[1]),
        n=float(popt[2]) if fixed_n is None else float(fixed_n),
        n_err=float(perr[2]) if fixed_n is None else 0.0,
        r_squared=_r_squared(r, r_fit),
        rmse_nm=float(np.sqrt(np.mean((r - r_fit) ** 2))),
        n_fixed=fixed_n is not None,
    )
    return out
