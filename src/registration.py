"""原位序列亚像素漂移校正模块。

基于相位相关（``skimage.registration.phase_cross_correlation``，Guizar-Sicairos
矩阵乘法 DFT 上采样）计算每一帧相对参考帧（默认第 0 帧）的亚像素位移，
并通过三次样条插值完成对齐。

约定：
    * ``shift_px`` = (row, col)：将第 k 帧对齐到参考帧所需施加的平移量；
    * ``drift_px`` = (dx, dy) = -(col, row)：第 k 帧中样品相对参考帧的漂移，
      与 ``simulator`` 中的真实值定义一致，可直接比较。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.filters import window
from skimage.registration import phase_cross_correlation


@dataclass
class RegistrationResult:
    aligned: np.ndarray        # (T, H, W) float32 对齐后的序列
    shift_px: np.ndarray       # (T, 2) 校正平移 [row, col]
    drift_px: np.ndarray       # (T, 2) 估计漂移 [dx, dy]
    errors: np.ndarray         # (T,) 相位相关归一化误差
    valid_mask: np.ndarray     # (H, W) bool，所有帧均有真实数据覆盖的公共区域
    reference_index: int = 0


def preprocess_for_registration(
    frame: np.ndarray,
    pixel_size_nm: float,
    lowpass_nm: float = 0.3,
    highpass_nm: float = 6.0,
) -> np.ndarray:
    """带通滤波 + 标准化 + Hann 窗：抑制散粒噪声、照明梯度与边界周期性伪影。"""
    img = frame.astype(np.float64)
    low = ndimage.gaussian_filter(img, lowpass_nm / pixel_size_nm)
    background = ndimage.gaussian_filter(img, highpass_nm / pixel_size_nm)
    band = low - background
    band = (band - band.mean()) / (band.std() + 1e-12)
    return band * window("hann", band.shape)


def estimate_drift(
    stack: np.ndarray,
    pixel_size_nm: float,
    reference_index: int = 0,
    upsample_factor: int = 50,
    normalization: Optional[str] = None,
):
    """逐帧计算相对参考帧的亚像素平移，返回 (shift_px, drift_px, errors)。"""
    ref = preprocess_for_registration(stack[reference_index], pixel_size_nm)
    n = stack.shape[0]
    shifts = np.zeros((n, 2))
    errors = np.zeros(n)
    for k in range(n):
        if k == reference_index:
            continue
        mov = preprocess_for_registration(stack[k], pixel_size_nm)
        result = phase_cross_correlation(
            ref, mov, upsample_factor=upsample_factor, normalization=normalization
        )
        shifts[k] = result[0]
        errors[k] = result[1]
    drift = -shifts[:, ::-1]  # (row, col) → (dx, dy)，并取反得到样品漂移
    return shifts, drift, errors


def apply_shifts(stack: np.ndarray, shifts: np.ndarray, order: int = 3):
    """按给定平移对齐序列，并计算所有帧均被真实数据覆盖的公共有效区域。"""
    aligned = np.empty(stack.shape, dtype=np.float32)
    valid = np.ones(stack.shape[1:], dtype=bool)
    ones = np.ones(stack.shape[1:], dtype=np.float32)
    for k, s in enumerate(shifts):
        aligned[k] = ndimage.shift(stack[k].astype(np.float32), s, order=order, mode="nearest")
        cover = ndimage.shift(ones, s, order=1, mode="constant", cval=0.0)
        valid &= cover > 0.999
    return aligned, valid


def register_stack(
    stack: np.ndarray,
    pixel_size_nm: float,
    reference_index: int = 0,
    upsample_factor: int = 50,
) -> RegistrationResult:
    """完整的漂移估计 + 对齐流程。"""
    shifts, drift, errors = estimate_drift(
        stack, pixel_size_nm, reference_index=reference_index, upsample_factor=upsample_factor
    )
    aligned, valid = apply_shifts(stack, shifts)
    return RegistrationResult(aligned, shifts, drift, errors, valid, reference_index)


def drift_comparison_table(
    reg: RegistrationResult,
    pixel_size_nm: float,
    times_s: np.ndarray,
    ground_truth: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """输出估计漂移轨迹（及与真实值的对比、误差）。"""
    df = pd.DataFrame(
        {
            "frame": np.arange(len(reg.drift_px)),
            "time_s": times_s,
            "est_drift_x_px": reg.drift_px[:, 0],
            "est_drift_y_px": reg.drift_px[:, 1],
            "phase_corr_error": reg.errors,
        }
    )
    if ground_truth is not None:
        df["true_drift_x_px"] = ground_truth["drift_x_px"].to_numpy()
        df["true_drift_y_px"] = ground_truth["drift_y_px"].to_numpy()
        df["error_x_px"] = df["est_drift_x_px"] - df["true_drift_x_px"]
        df["error_y_px"] = df["est_drift_y_px"] - df["true_drift_y_px"]
        df["error_norm_nm"] = np.hypot(df["error_x_px"], df["error_y_px"]) * pixel_size_nm
    return df


def drift_rmse(table: pd.DataFrame) -> float:
    """估计漂移相对真实值的二维 RMSE（像素）。"""
    if "error_x_px" not in table:
        return float("nan")
    return float(np.sqrt(np.mean(table["error_x_px"] ** 2 + table["error_y_px"] ** 2)))
