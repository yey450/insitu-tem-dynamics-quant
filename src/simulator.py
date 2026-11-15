"""原位 TEM 纳米颗粒演化序列仿真模块。

物理模型概述
------------
1. **粗化动力学**：采用平均场 Lifshitz–Slyozov–Wagner (LSW) Ostwald 熟化模型

       dr_i/dt = K / r_i * (1 / r_c - 1 / r_i),   r_c = <r>

   当 r_c 取算术平均半径时，体系总体积（Σ r_i^3）严格守恒；大颗粒长大、小颗粒
   溶解，平均半径渐近满足 <R>^3 - R_0^3 ∝ t。半径低于溶解阈值的颗粒被移除。
2. **成像模型**：明场质厚衬度，I = I_bg · exp(-μ · t(x, y))，其中 t 为球形颗粒的
   投影厚度 2·sqrt(r^2 - d^2)；背景为静态非晶碳支持膜纹理，并以高斯 PSF 模糊。
3. **热漂移**：样品整体（颗粒 + 碳膜）做带方向持续性的随机游走，累计偏移
   10–30 像素，以亚像素精度（三次样条插值）施加。
4. **探测器噪声**：低剂量泊松散粒噪声 + 高斯读出噪声，叠加逐帧剂量（亮度）波动
   与缓慢移动的非均匀照明。

坐标约定
--------
``drift_x_px`` / ``drift_y_px`` 表示第 k 帧中样品相对第 0 帧在图像坐标系下的位移
（x 向右、y 向下为正）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage


@dataclass
class SimulationConfig:
    """仿真参数（物理量统一使用 nm / s 单位，内部再换算为像素）。"""

    n_frames: int = 20
    image_size: int = 512
    pixel_size_nm: float = 0.25
    frame_interval_s: float = 5.0
    seed: Optional[int] = 42

    # --- 颗粒群与粗化动力学 ---
    n_particles: int = 95
    mean_radius_nm: float = 3.0
    radius_sigma: float = 0.32          # 对数正态分布形状参数
    coarsening_rate: float = 0.9        # LSW 速率常数 K（nm^3/s）
    dissolution_radius_nm: float = 0.5  # 低于该半径视为完全溶解
    substeps_per_frame: int = 400       # 帧间 ODE 积分子步数
    min_gap_nm: float = 0.3             # 颗粒间最小表面间距

    # --- 成像衬度 ---
    absorption_per_nm: float = 0.20     # 质厚衬度吸收系数 μ（1/nm）
    psf_sigma_nm: float = 0.22          # 点扩散函数高斯宽度
    carbon_texture_amp: float = 0.045   # 非晶碳膜纹理相对幅值

    # --- 热漂移 ---
    drift_range_px: Tuple[float, float] = (10.0, 30.0)
    drift_jitter_px: float = 0.35

    # --- 低剂量噪声与亮度波动 ---
    dose_e_per_px: float = 40.0
    read_noise_e: float = 2.0
    brightness_fluctuation: float = 0.06
    illumination_falloff: float = 0.10
    detector_gain: float = 8.0
    detector_offset: float = 200.0


@dataclass
class SimulationResult:
    """仿真输出。"""

    stack: np.ndarray                 # (T, H, W) uint16 原始序列
    times_s: np.ndarray               # (T,) 采集时间
    drift_px: np.ndarray              # (T, 2) 真实漂移 [dx, dy]
    radii_nm: np.ndarray              # (T, N) 每个颗粒的真实半径（0 表示已溶解）
    centers_px: np.ndarray            # (N, 2) 颗粒在第 0 帧坐标系下的中心 [x, y]
    ground_truth: pd.DataFrame        # 逐帧真实值
    config: SimulationConfig = field(repr=False)


# ----------------------------------------------------------------------------
# 1. 粗化动力学
# ----------------------------------------------------------------------------
def simulate_coarsening(
    r0_nm: np.ndarray,
    times_s: np.ndarray,
    rate_k: float,
    r_dissolve_nm: float,
    substeps: int,
) -> np.ndarray:
    """对平均场 LSW 方程做显式积分，返回各帧时刻的颗粒半径矩阵 (T, N)。"""
    r = r0_nm.astype(np.float64).copy()
    history = np.zeros((len(times_s), r.size))
    history[0] = r
    for i in range(1, len(times_s)):
        dt = (times_s[i] - times_s[i - 1]) / substeps
        for _ in range(substeps):
            alive = r > 0
            if alive.sum() < 2:
                break
            ra = r[alive]
            r_crit = ra.mean()
            ra = ra + rate_k / ra * (1.0 / r_crit - 1.0 / ra) * dt
            ra[ra < r_dissolve_nm] = 0.0
            r[alive] = ra
        history[i] = r
    return history


# ----------------------------------------------------------------------------
# 2. 空间布局
# ----------------------------------------------------------------------------
def place_particles(
    max_radii_px: np.ndarray,
    canvas_size: int,
    gap_px: float,
    rng: np.random.Generator,
    max_attempts: int = 4000,
) -> Tuple[np.ndarray, np.ndarray]:
    """按最大半径从大到小随机序贯放置（硬球不重叠），返回中心坐标与成功放置的索引。"""
    order = np.argsort(max_radii_px)[::-1]
    centers, radii, kept = [], [], []
    for idx in order:
        r = max_radii_px[idx]
        for _ in range(max_attempts):
            c = rng.uniform(r, canvas_size - r, size=2)
            if not centers:
                ok = True
            else:
                d = np.hypot(*(np.asarray(centers) - c).T)
                ok = np.all(d >= np.asarray(radii) + r + gap_px)
            if ok:
                centers.append(c)
                radii.append(r)
                kept.append(idx)
                break
    kept = np.asarray(kept, dtype=int)
    centers = np.asarray(centers, dtype=float).reshape(-1, 2)
    sort = np.argsort(kept)
    return centers[sort], kept[sort]


def generate_drift(
    n_frames: int,
    drift_range_px: Tuple[float, float],
    jitter_px: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """生成带方向持续性的热漂移轨迹 (T, 2)，第 0 帧为原点，末帧累计位移落在给定区间内。"""
    drift = np.zeros((n_frames, 2))
    if n_frames < 2:
        return drift
    total = rng.uniform(*drift_range_px)
    theta = rng.uniform(0, 2 * np.pi) + np.cumsum(rng.normal(0, 0.18, n_frames - 1))
    speed = np.clip(1.0 + rng.normal(0, 0.3, n_frames - 1), 0.2, None)
    steps = np.stack([np.cos(theta), np.sin(theta)], axis=1) * speed[:, None]
    steps += rng.normal(0, jitter_px / max(total / (n_frames - 1), 1e-6), steps.shape)
    drift[1:] = np.cumsum(steps, axis=0)
    drift *= total / np.linalg.norm(drift[-1])
    return drift


# ----------------------------------------------------------------------------
# 3. 图像形成
# ----------------------------------------------------------------------------
def make_carbon_texture(size: int, pixel_size_nm: float, amp: float, rng) -> np.ndarray:
    """多尺度相关高斯噪声，模拟非晶碳支持膜的静态纹理。"""
    fine = ndimage.gaussian_filter(rng.normal(size=(size, size)), 0.4 / pixel_size_nm)
    coarse = ndimage.gaussian_filter(rng.normal(size=(size, size)), 2.5 / pixel_size_nm)
    fine /= fine.std() + 1e-12
    coarse /= coarse.std() + 1e-12
    return 1.0 + amp * (0.7 * fine + 0.3 * coarse)


def render_thickness(
    canvas_size: int, centers_px: np.ndarray, radii_px: np.ndarray, pixel_size_nm: float
) -> np.ndarray:
    """计算球形颗粒在画布上的投影厚度图（单位 nm）。"""
    thick = np.zeros((canvas_size, canvas_size), dtype=np.float64)
    for (cx, cy), r in zip(centers_px, radii_px):
        if r <= 0:
            continue
        x0, x1 = max(int(cx - r) - 1, 0), min(int(cx + r) + 2, canvas_size)
        y0, y1 = max(int(cy - r) - 1, 0), min(int(cy + r) + 2, canvas_size)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        thick[y0:y1, x0:x1] += 2.0 * np.sqrt(np.clip(r * r - d2, 0, None))
    return thick * pixel_size_nm


def illumination_profile(size: int, center: np.ndarray, falloff: float) -> np.ndarray:
    """探测器坐标系下的高斯型非均匀照明（相对强度）。"""
    ax = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(ax, ax)
    return 1.0 - falloff * ((xx - center[0]) ** 2 + (yy - center[1]) ** 2) / 2.0


# ----------------------------------------------------------------------------
# 4. 主流程
# ----------------------------------------------------------------------------
def simulate_insitu_series(config: SimulationConfig = SimulationConfig()) -> SimulationResult:
    """生成完整的原位 TEM 仿真序列及真实值。"""
    cfg = config
    rng = np.random.default_rng(cfg.seed)
    px = cfg.pixel_size_nm
    size = cfg.image_size
    times = np.arange(cfg.n_frames) * cfg.frame_interval_s

    # 漂移轨迹与画布边距（画布需容纳漂移造成的视场外移）
    drift = generate_drift(cfg.n_frames, cfg.drift_range_px, cfg.drift_jitter_px, rng)
    margin = int(np.ceil(np.abs(drift).max())) + 24
    canvas = size + 2 * margin

    # 粗化动力学：先在时间上积分，得到每个颗粒的最大半径以安全布局
    r0 = rng.lognormal(np.log(cfg.mean_radius_nm), cfg.radius_sigma, cfg.n_particles)
    r0 = np.clip(r0, 1.0, 3.0 * cfg.mean_radius_nm)
    radii_hist = simulate_coarsening(
        r0, times, cfg.coarsening_rate, cfg.dissolution_radius_nm, cfg.substeps_per_frame
    )
    max_r_px = radii_hist.max(axis=0) / px
    centers_canvas, kept = place_particles(max_r_px, canvas, cfg.min_gap_nm / px, rng)
    radii_hist = radii_hist[:, kept]

    carbon = make_carbon_texture(canvas, px, cfg.carbon_texture_amp, rng)
    beam_center = np.cumsum(rng.normal(0, 0.03, size=(cfg.n_frames, 2)), axis=0)
    stack = np.empty((cfg.n_frames, size, size), dtype=np.uint16)
    centers_frame0 = centers_canvas - margin

    gt_rows = []
    for k in range(cfg.n_frames):
        r_px = radii_hist[k] / px
        thickness = render_thickness(canvas, centers_canvas, r_px, px)
        ideal = carbon * np.exp(-cfg.absorption_per_nm * thickness)
        ideal = ndimage.gaussian_filter(ideal, cfg.psf_sigma_nm / px)

        # 亚像素热漂移：样品整体平移 (dy, dx)，再裁剪到探测器视场
        dx, dy = drift[k]
        shifted = ndimage.shift(ideal, (dy, dx), order=3, mode="reflect")
        frame = shifted[margin:margin + size, margin:margin + size]

        # 低剂量成像：剂量波动 × 照明不均 → 泊松散粒噪声 + 高斯读出噪声
        brightness = np.clip(1.0 + rng.normal(0, cfg.brightness_fluctuation), 0.8, 1.2)
        illum = illumination_profile(size, beam_center[k], cfg.illumination_falloff)
        expected = cfg.dose_e_per_px * brightness * illum * np.clip(frame, 0, None)
        electrons = rng.poisson(expected).astype(np.float64)
        electrons += rng.normal(0, cfg.read_noise_e, electrons.shape)
        adu = electrons * cfg.detector_gain + cfg.detector_offset
        stack[k] = np.clip(np.round(adu), 0, 65535).astype(np.uint16)

        # 真实值：完全位于该帧视场内的存活颗粒
        cx = centers_frame0[:, 0] + dx
        cy = centers_frame0[:, 1] + dy
        inside = (
            (r_px > 0)
            & (cx - r_px >= 0) & (cx + r_px <= size - 1)
            & (cy - r_px >= 0) & (cy + r_px <= size - 1)
        )
        r_vis = radii_hist[k][inside]
        alive_all = radii_hist[k][radii_hist[k] > 0]
        gt_rows.append(
            dict(
                frame=k,
                time_s=times[k],
                drift_x_px=dx,
                drift_y_px=dy,
                drift_x_nm=dx * px,
                drift_y_nm=dy * px,
                true_n_particles_fov=int(inside.sum()),
                true_mean_radius_nm=float(r_vis.mean()) if r_vis.size else np.nan,
                true_mean_diameter_nm=float(2 * r_vis.mean()) if r_vis.size else np.nan,
                true_std_diameter_nm=float(2 * r_vis.std()) if r_vis.size else np.nan,
                true_mean_radius_ensemble_nm=float(alive_all.mean()),
                true_n_particles_ensemble=int(alive_all.size),
                brightness_factor=brightness,
            )
        )

    return SimulationResult(
        stack=stack,
        times_s=times,
        drift_px=drift,
        radii_nm=radii_hist,
        centers_px=centers_frame0,
        ground_truth=pd.DataFrame(gt_rows),
        config=cfg,
    )


def save_simulation(result: SimulationResult, data_dir: Path | str = "data") -> Tuple[Path, Path]:
    """保存多帧 TIFF（ImageJ 兼容，含比例尺与帧间隔元数据）与真实值 CSV。"""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    tif_path = data_dir / "synthetic_insitu_stack.tif"
    csv_path = data_dir / "ground_truth.csv"
    px = result.config.pixel_size_nm
    tifffile.imwrite(
        tif_path,
        result.stack,
        imagej=True,
        resolution=(1.0 / px, 1.0 / px),
        metadata={"axes": "TYX", "unit": "nm", "finterval": result.config.frame_interval_s},
    )
    result.ground_truth.to_csv(csv_path, index=False, float_format="%.6g")
    return tif_path, csv_path


def load_stack(tif_path: Path | str) -> np.ndarray:
    """读取多帧 TIFF 为 (T, H, W) 数组。"""
    stack = tifffile.imread(tif_path)
    if stack.ndim == 2:
        stack = stack[None]
    return stack
