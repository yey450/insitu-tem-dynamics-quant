"""期刊风格综合定量分析图。

2×2 布局：
    A. 初始帧 t0 与末帧 t_end 原始图像（含比例尺）；
    B. 亚像素漂移校正轨迹（X / Y Drift Trajectory，估计值 vs 真实值）；
    C. 末帧分割轮廓叠加图（按等效直径着色）；
    D. 平均粒径演化曲线与幂律动力学拟合。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import cm, colors  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from skimage import measure  # noqa: E402

from .quantifier import coarsening_model  # noqa: E402

# Okabe–Ito 色盲友好配色
C_X, C_Y, C_FIT, C_GT, C_N = "#0072B2", "#D55E00", "#009E73", "#4D4D4D", "#B0B0B0"

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8.5,
    "axes.titlesize": 8.5,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "mathtext.default": "regular",
    "savefig.dpi": 300,
}


def _robust_limits(img: np.ndarray, lo: float = 0.5, hi: float = 99.5):
    return np.percentile(img, [lo, hi])


def _nice_bar_length(fov_nm: float) -> float:
    """选取约为视场 1/5 的“整”比例尺长度。"""
    target = fov_nm / 5
    for v in (1, 2, 5, 10, 20, 25, 50, 100, 200, 500):
        if v >= target * 0.7:
            return float(v)
    return float(round(target, -2))


def add_scale_bar(ax, pixel_size_nm: float, image_width_px: int, length_nm: Optional[float] = None,
                  color: str = "white", loc_frac=(0.06, 0.07)):
    """在图像坐标系中绘制比例尺（不依赖额外第三方包）。"""
    length_nm = length_nm or _nice_bar_length(image_width_px * pixel_size_nm)
    length_px = length_nm / pixel_size_nm
    h = image_width_px * 0.012
    x0 = image_width_px * loc_frac[0]
    y0 = image_width_px * (1 - loc_frac[1])
    ax.add_patch(Rectangle((x0, y0 - h), length_px, h, color=color, lw=0))
    ax.text(x0 + length_px / 2, y0 - h * 1.8, f"{length_nm:g} nm", color=color,
            ha="center", va="bottom", fontsize=7, fontweight="bold")


def _panel_label(ax, label: str, x: float = -0.16, y: float = 1.06):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=12, fontweight="bold",
            va="bottom", ha="left")


def _show_image(ax, img, title=None):
    vmin, vmax = _robust_limits(img)
    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.6)
    if title:
        ax.set_title(title, pad=3)


def plot_summary(
    raw_stack: np.ndarray,
    times_s: np.ndarray,
    pixel_size_nm: float,
    drift_table: pd.DataFrame,
    aligned_last: np.ndarray,
    labels_last: np.ndarray,
    summary: pd.DataFrame,
    fit: Optional[Dict[str, float]],
    out_path: Path | str,
    ground_truth: Optional[pd.DataFrame] = None,
    valid_mask: Optional[np.ndarray] = None,
    fit_lsw: Optional[Dict[str, float]] = None,
) -> Path:
    """绘制并保存 2×2 综合定量分析图。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    width_px = raw_stack.shape[-1]

    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(7.2, 6.6))
        gs = GridSpec(2, 2, figure=fig, left=0.075, right=0.90, top=0.94, bottom=0.075,
                      wspace=0.42, hspace=0.34)

        # ---------------- A：初末帧原始图像对比 ----------------
        gs_a = gs[0, 0].subgridspec(1, 2, wspace=0.05)
        ax_a0 = fig.add_subplot(gs_a[0])
        ax_a1 = fig.add_subplot(gs_a[1])
        _show_image(ax_a0, raw_stack[0], f"$t_0$ = {times_s[0]:.0f} s")
        _show_image(ax_a1, raw_stack[-1], f"$t_{{end}}$ = {times_s[-1]:.0f} s")
        for ax in (ax_a0, ax_a1):
            add_scale_bar(ax, pixel_size_nm, width_px)
            ax.set_box_aspect(1)
        ax_a0.text(0.04, 0.96, "raw", transform=ax_a0.transAxes, color="white",
                   fontsize=6.5, va="top")
        ax_a1.text(0.04, 0.96, "raw (drifted)", transform=ax_a1.transAxes, color="white",
                   fontsize=6.5, va="top")
        _panel_label(ax_a0, "A", x=-0.30, y=1.12)

        # ---------------- B：漂移轨迹 ----------------
        ax_b = fig.add_subplot(gs[0, 1])
        t = drift_table["time_s"].to_numpy()
        if "true_drift_x_px" in drift_table:
            ax_b.plot(t, drift_table["true_drift_x_px"], "-", color=C_X, lw=1.0, alpha=0.45)
            ax_b.plot(t, drift_table["true_drift_y_px"], "-", color=C_Y, lw=1.0, alpha=0.45)
        ax_b.plot(t, drift_table["est_drift_x_px"], "o", ms=3.2, mfc="white", mec=C_X, mew=1.0,
                  label="X drift")
        ax_b.plot(t, drift_table["est_drift_y_px"], "s", ms=3.0, mfc="white", mec=C_Y, mew=1.0,
                  label="Y drift")
        rmse_title = None
        if "true_drift_x_px" in drift_table:
            ax_b.plot([], [], "-", color=C_GT, lw=1.0, alpha=0.6, label="ground truth")
            rmse = np.sqrt(np.mean(drift_table["error_x_px"] ** 2 + drift_table["error_y_px"] ** 2))
            rmse_title = f"RMSE = {rmse:.3f} px ({rmse * pixel_size_nm * 1e3:.0f} pm)"
        ax_b.axhline(0, color="k", lw=0.5, ls=":")
        ax_b.set_xlabel("Time (s)")
        ax_b.set_ylabel("Drift (px)")
        ax_b.set_title("Sub-pixel drift trajectory (phase correlation)", pad=3)
        # RMSE 作为图例标题，loc="best" 自动选择不遮挡数据点的空白区域
        ax_b.legend(loc="best", handlelength=1.6, title=rmse_title,
                    title_fontsize=7, alignment="left")
        sec = ax_b.secondary_yaxis("right", functions=(lambda p: p * pixel_size_nm,
                                                       lambda n: n / pixel_size_nm))
        sec.set_ylabel("Drift (nm)")
        sec.tick_params(direction="in", labelsize=7.5)
        _panel_label(ax_b, "B")

        # ---------------- C：末帧分割轮廓 ----------------
        ax_c = fig.add_subplot(gs[1, 0])
        _show_image(ax_c, aligned_last)
        if valid_mask is not None and not valid_mask.all():
            shade = np.zeros((*valid_mask.shape, 4))
            shade[~valid_mask] = (0, 0, 0, 0.55)
            ax_c.imshow(shade, interpolation="nearest")
        props = measure.regionprops(labels_last)
        diam = np.array([p.equivalent_diameter_area * pixel_size_nm for p in props])
        if diam.size:
            norm = colors.Normalize(vmin=np.percentile(diam, 2), vmax=np.percentile(diam, 98))
            cmap = matplotlib.colormaps["plasma"]
            for p, d in zip(props, diam):
                minr, minc, _, _ = p.bbox
                sub = np.pad(p.image.astype(float), 1)
                for cnt in measure.find_contours(sub, 0.5):
                    ax_c.plot(cnt[:, 1] + minc - 1, cnt[:, 0] + minr - 1,
                              color=cmap(norm(d)), lw=0.8)
            sm = cm.ScalarMappable(norm=norm, cmap=cmap)
            cax = ax_c.inset_axes([1.03, 0.0, 0.04, 1.0])
            cb = fig.colorbar(sm, cax=cax)
            cb.set_label("Equivalent diameter (nm)", fontsize=7.5)
            cb.ax.tick_params(labelsize=7, direction="in")
            cb.outline.set_linewidth(0.6)
        add_scale_bar(ax_c, pixel_size_nm, width_px)
        ax_c.set_title(f"Segmentation at $t_{{end}}$ (drift-corrected), N = {len(props)}", pad=3)
        _panel_label(ax_c, "C")

        # ---------------- D：粒径演化与动力学拟合 ----------------
        ax_d = fig.add_subplot(gs[1, 1])
        ax_n = ax_d.twinx()
        ts = summary["time_s"].to_numpy()
        width = (ts[1] - ts[0]) * 0.6 if ts.size > 1 else 1.0
        ax_n.bar(ts, summary["n_particles"], width=width, color=C_N, alpha=0.35, lw=0, zorder=0)
        ax_n.set_ylabel("Particle count N", color="#7F7F7F")
        ax_n.tick_params(axis="y", colors="#7F7F7F", direction="in", labelsize=7.5)
        ax_n.set_ylim(0, summary["n_particles"].max() * 2.2)
        ax_d.set_zorder(ax_n.get_zorder() + 1)
        ax_d.patch.set_visible(False)

        r = summary["mean_radius_nm"].to_numpy()
        ax_d.errorbar(ts, r, yerr=summary["sem_radius_nm"], fmt="o", ms=3.5, mfc="white",
                      mec="k", mew=0.9, ecolor="k", elinewidth=0.7, capsize=1.8,
                      label=r"measured $\langle R \rangle$ ± SEM")
        if ground_truth is not None and "true_mean_radius_nm" in ground_truth:
            ax_d.plot(ground_truth["time_s"], ground_truth["true_mean_radius_nm"], "--",
                      color=C_GT, lw=1.0, label="ground truth")
        tt = np.linspace(ts.min(), ts.max(), 300)
        if fit is not None:
            ax_d.plot(tt, coarsening_model(tt, fit["R0_nm"], fit["k"], fit["n"]), "-",
                      color=C_FIT, lw=1.6, label=f"power-law fit (n = {fit['n']:.2f})")
            txt = (r"$R^n - R_0^n = kt$" + "\n"
                   + f"n = {fit['n']:.2f} ± {fit['n_err']:.2f}\n"
                   + f"k = {fit['k']:.3g} nm$^{{n}}$ s$^{{-1}}$\n"
                   + f"$R_0$ = {fit['R0_nm']:.2f} nm\n"
                   + f"$R^2$ = {fit['r_squared']:.3f}")
            ax_d.text(0.97, 0.04, txt, transform=ax_d.transAxes, ha="right", va="bottom",
                      fontsize=7, linespacing=1.35,
                      bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.75", lw=0.6))
        if fit_lsw is not None:
            ax_d.plot(tt, coarsening_model(tt, fit_lsw["R0_nm"], fit_lsw["k"], 3.0), ":",
                      color=C_Y, lw=1.2, label="LSW (n = 3)")
        ax_d.set_xlabel("Time (s)")
        ax_d.set_ylabel(r"Mean radius $\langle R \rangle$ (nm)")
        ax_d.set_title("Coarsening kinetics", pad=3)
        ax_d.legend(loc="upper left", handlelength=1.8)
        lo = np.nanmin(r) if ground_truth is None else min(np.nanmin(r), ground_truth["true_mean_radius_nm"].min())
        hi = np.nanmax(r) if ground_truth is None else max(np.nanmax(r), ground_truth["true_mean_radius_nm"].max())
        pad = (hi - lo) * 0.15
        ax_d.set_ylim(lo - pad * 2.2, hi + pad * 2.6)
        _panel_label(ax_d, "D")

        fig.savefig(out_path, dpi=300, facecolor="white")
        plt.close(fig)
    return out_path
