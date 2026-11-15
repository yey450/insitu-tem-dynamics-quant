"""In-situ TEM 动力学定量分析 —— Streamlit 交互式演示界面。

运行方式：
    streamlit run app.py

两种运行模式：
    A. 默认演示：加载仓库自带的 data/synthetic_insitu_stack.tif 并运行完整流水线；
    B. 自定义：通过滑块调节剂量（噪声水平）/ 粗化速率等参数实时重新仿真，
       或上传本地多帧 .tif 序列进行分析。

本界面只复用 src/ 中的函数，所有结果在内存 / 临时目录中生成，
不会覆盖仓库中的 data/ 与 results/，也不影响 `python main.py` 命令行流程。
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
import tifffile

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.quantifier import SegmentationParams, fit_coarsening_kinetics, quantify_stack  # noqa: E402
from src.registration import drift_comparison_table, drift_rmse, register_stack  # noqa: E402
from src.simulator import SimulationConfig, load_stack, simulate_insitu_series  # noqa: E402
from src.visualizer import plot_summary  # noqa: E402

DEMO_TIF = ROOT / "data" / "synthetic_insitu_stack.tif"
DEMO_GT = ROOT / "data" / "ground_truth.csv"

MODE_A = "A · Demo dataset"
MODE_B = "B · Custom simulation / upload"
SRC_SIM = "Re-simulate with custom parameters"
SRC_UPLOAD = "Upload multi-frame TIFF"

CSS = """
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1320px;}
h1 {font-weight: 650; letter-spacing: -0.01em; margin-bottom: 0.1rem;}
[data-testid="stMetric"] {
    background: #f6f8fa; border: 1px solid #d0d7de; border-left: 4px solid #0072B2;
    border-radius: 6px; padding: 0.75rem 1rem 0.6rem 1rem;
}
[data-testid="stMetricLabel"] p {
    font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; color: #57606a;
}
[data-testid="stMetricValue"] {
    font-family: "JetBrains Mono", "Consolas", "SFMono-Regular", monospace; font-size: 1.55rem;
}
.status-bar {
    font-family: "JetBrains Mono", "Consolas", monospace; font-size: 0.78rem; color: #57606a;
    border-top: 1px solid #d0d7de; border-bottom: 1px solid #d0d7de;
    padding: 0.35rem 0.1rem; margin: 0.4rem 0 1rem 0;
}
section[data-testid="stSidebar"] h2 {font-size: 1.0rem; text-transform: uppercase; letter-spacing: 0.06em;}
</style>
"""


# ----------------------------------------------------------------------------
# 核心流水线（与 main.py 相同的步骤，结果全部返回为内存对象）
# ----------------------------------------------------------------------------
def _to_tyx(stack: np.ndarray) -> np.ndarray:
    """将上传的 TIFF 规范化为 (T, Y, X) 灰度序列。"""
    stack = np.squeeze(np.asarray(stack))
    if stack.ndim == 4 and stack.shape[-1] in (3, 4):  # (T, Y, X, RGB[A]) → 灰度
        stack = stack[..., :3].mean(axis=-1)
    if stack.ndim != 3:
        raise ValueError(f"Expected a multi-frame (T, Y, X) image stack, got shape {stack.shape}.")
    if stack.shape[0] < 3:
        raise ValueError("At least 3 frames are required for kinetic fitting.")
    return stack


def _fit_safely(times: np.ndarray, radius: np.ndarray, sem: np.ndarray, fixed_n=None):
    """先以 SEM 加权拟合，失败时退化为不加权拟合；仍失败则返回 None。"""
    for sigma in (sem, None):
        try:
            return fit_coarsening_kinetics(times, radius, sigma=sigma, fixed_n=fixed_n)
        except (ValueError, RuntimeError):
            continue
    return None


def run_analysis(
    stack: np.ndarray,
    times: np.ndarray,
    pixel_size: float,
    ground_truth: Optional[pd.DataFrame] = None,
    upsample: int = 50,
) -> Dict:
    """配准 → 分割定量 → 动力学拟合 → 绘图，返回可缓存的结果字典。"""
    t0 = time.perf_counter()
    reg = register_stack(stack, pixel_size, upsample_factor=upsample)
    drift_df = drift_comparison_table(reg, pixel_size, times, ground_truth)

    quant = quantify_stack(reg.aligned, times, pixel_size, reg.valid_mask, SegmentationParams())
    summary = quant.summary
    if ground_truth is not None:
        cols = [c for c in ("frame", "true_mean_radius_nm", "true_n_particles_fov") if c in ground_truth]
        summary = summary.merge(ground_truth[cols], on="frame", how="left")

    radius = summary["mean_radius_nm"].to_numpy()
    sem = summary["sem_radius_nm"].to_numpy()
    fit = _fit_safely(times, radius, sem)
    fit_lsw = _fit_safely(times, radius, sem, fixed_n=3.0)
    fits = {"free_exponent": fit, "lsw_n3": fit_lsw}

    with tempfile.TemporaryDirectory() as tmp:
        fig_path = plot_summary(
            raw_stack=stack, times_s=times, pixel_size_nm=pixel_size, drift_table=drift_df,
            aligned_last=reg.aligned[-1], labels_last=quant.labels[-1], summary=summary,
            fit=fit, fit_lsw=fit_lsw, ground_truth=ground_truth, valid_mask=reg.valid_mask,
            out_path=Path(tmp) / "insitu_quant_summary.png",
        )
        figure_png = fig_path.read_bytes()

    return dict(
        figure_png=figure_png,
        drift_table=drift_df,
        summary=summary,
        particles=quant.particles,
        fits=fits,
        drift_px=reg.drift_px,
        rmse_px=drift_rmse(drift_df),
        n_frames=int(stack.shape[0]),
        shape=tuple(int(s) for s in stack.shape[1:]),
        pixel_size=float(pixel_size),
        runtime_s=time.perf_counter() - t0,
    )


@st.cache_data(show_spinner=False, max_entries=4)
def analyze_demo(pixel_size: float, upsample: int, file_mtime: float) -> Dict:
    """模式 A：分析仓库自带的示例序列（file_mtime 仅用于在文件更新后使缓存失效）。"""
    stack = load_stack(DEMO_TIF)
    gt = pd.read_csv(DEMO_GT) if DEMO_GT.exists() else None
    times = gt["time_s"].to_numpy() if gt is not None else np.arange(stack.shape[0]) * 5.0
    res = run_analysis(stack, times, pixel_size, gt, upsample)
    res["source"] = f"demo · {DEMO_TIF.relative_to(ROOT).as_posix()}"
    return res


@st.cache_data(show_spinner=False, max_entries=8)
def analyze_simulation(n_frames: int, size: int, pixel_size: float, interval: float, dose: float,
                       rate: float, n_particles: int, seed: int, upsample: int) -> Dict:
    """模式 B（仿真）：按自定义参数重新仿真并分析。"""
    cfg = SimulationConfig(
        n_frames=n_frames, image_size=size, pixel_size_nm=pixel_size, frame_interval_s=interval,
        dose_e_per_px=dose, coarsening_rate=rate, n_particles=n_particles, seed=seed,
    )
    sim = simulate_insitu_series(cfg)
    res = run_analysis(sim.stack, sim.times_s, pixel_size, sim.ground_truth, upsample)
    buf = io.BytesIO()
    tifffile.imwrite(buf, sim.stack, imagej=True, resolution=(1 / pixel_size, 1 / pixel_size),
                     metadata={"axes": "TYX", "unit": "nm", "finterval": interval})
    res["stack_tif"] = buf.getvalue()
    res["ground_truth"] = sim.ground_truth
    res["source"] = f"simulation · dose={dose:g} e/px · K={rate:g} nm³/s · seed={seed}"
    return res


@st.cache_data(show_spinner=False, max_entries=4)
def analyze_upload(file_bytes: bytes, file_name: str, pixel_size: float, interval: float,
                   upsample: int) -> Dict:
    """模式 B（上传）：分析用户上传的多帧 TIFF（无真实值，RMSE 不可用）。"""
    stack = _to_tyx(tifffile.imread(io.BytesIO(file_bytes)))
    times = np.arange(stack.shape[0]) * interval
    res = run_analysis(stack, times, pixel_size, None, upsample)
    res["source"] = f"upload · {file_name}"
    return res


# ----------------------------------------------------------------------------
# 界面组件
# ----------------------------------------------------------------------------
def show_image(data: bytes, caption: str) -> None:
    """兼容新旧 Streamlit 版本的全宽图片显示。"""
    try:
        st.image(data, caption=caption, width="stretch")
    except Exception:  # 旧版本不支持 width="stretch"
        st.image(data, caption=caption, use_container_width=True)


def render_metrics(res: Dict) -> None:
    px = res["pixel_size"]
    summary = res["summary"]
    fit = res["fits"]["free_exponent"]
    drift_end = float(np.hypot(*res["drift_px"][-1]))
    n0, n1 = int(summary["n_particles"].iloc[0]), int(summary["n_particles"].iloc[-1])
    rmse = res["rmse_px"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cumulative drift", f"{drift_end:.2f} px", f"{drift_end * px:.2f} nm", delta_color="off")
    if np.isfinite(rmse):
        c2.metric("Registration RMSE", f"{rmse:.3f} px", f"{rmse * px * 1e3:.0f} pm", delta_color="off")
    else:
        c2.metric("Registration RMSE", "—", "no ground truth", delta_color="off")
    c3.metric("Particles · last frame", f"{n1}", f"{n1 - n0:+d} vs. t₀", delta_color="off")
    if fit is not None:
        c4.metric("Coarsening exponent n", f"{fit['n']:.2f} ± {fit['n_err']:.2f}",
                  f"R² = {fit['r_squared']:.3f}", delta_color="off")
    else:
        c4.metric("Coarsening exponent n", "—", "fit failed", delta_color="off")


def build_zip(res: Dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("insitu_quant_summary.png", res["figure_png"])
        zf.writestr("drift_trajectory.csv", res["drift_table"].to_csv(index=False))
        zf.writestr("frame_statistics.csv", res["summary"].to_csv(index=False))
        zf.writestr("particle_measurements.csv", res["particles"].to_csv(index=False))
        zf.writestr("kinetics_fit.json", json.dumps(res["fits"], indent=2))
    return buf.getvalue()


def render_downloads(res: Dict) -> None:
    st.markdown("##### Export")
    cols = st.columns(6 if "stack_tif" in res else 5)
    cols[0].download_button("Summary figure (PNG)", res["figure_png"], "insitu_quant_summary.png",
                            "image/png", key="dl_png")
    cols[1].download_button("Frame statistics (CSV)", res["summary"].to_csv(index=False),
                            "frame_statistics.csv", "text/csv", key="dl_frames")
    cols[2].download_button("Drift trajectory (CSV)", res["drift_table"].to_csv(index=False),
                            "drift_trajectory.csv", "text/csv", key="dl_drift")
    cols[3].download_button("Particles (CSV)", res["particles"].to_csv(index=False),
                            "particle_measurements.csv", "text/csv", key="dl_particles")
    cols[4].download_button("All results (ZIP)", build_zip(res), "insitu_quant_results.zip",
                            "application/zip", key="dl_zip", type="primary")
    if "stack_tif" in res:
        cols[5].download_button("Simulated stack (TIFF)", res["stack_tif"],
                                "synthetic_insitu_stack.tif", "image/tiff", key="dl_tif")


def render_tables(res: Dict) -> None:
    tab1, tab2, tab3 = st.tabs(["Frame statistics", "Drift trajectory", "Kinetic fit"])
    with tab1:
        st.dataframe(res["summary"].round(4), hide_index=True)
    with tab2:
        st.dataframe(res["drift_table"].round(4), hide_index=True)
    with tab3:
        rows = [{"model": name, **fit} for name, fit in res["fits"].items() if fit is not None]
        if rows:
            st.dataframe(pd.DataFrame(rows).round(4), hide_index=True)
        else:
            st.info("Kinetic fit unavailable (too few valid frames).")
        st.latex(r"R(t)^n - R_0^n = k\,t")


# ----------------------------------------------------------------------------
# 主程序
# ----------------------------------------------------------------------------
def sidebar_controls() -> Dict:
    sb = st.sidebar
    sb.header("Run mode")
    mode = sb.radio("Mode", [MODE_A, MODE_B], index=0, label_visibility="collapsed")
    cfg: Dict = {"mode": mode}

    if mode == MODE_A:
        sb.caption("Loads the bundled `data/synthetic_insitu_stack.tif` and runs the full "
                   "registration → segmentation → kinetics pipeline.")
        cfg["pixel_size"] = sb.number_input("Pixel size (nm/px)", 0.01, 5.0, 0.25, 0.01, format="%.3f")
    else:
        sb.header("Data source")
        source = sb.radio("Source", [SRC_SIM, SRC_UPLOAD], label_visibility="collapsed")
        cfg["source"] = source
        if source == SRC_SIM:
            sb.caption("Each parameter change triggers a new simulation (results are cached).")
            cfg["dose"] = sb.slider("Electron dose (e⁻/px/frame) · lower = noisier", 5, 200, 40, 5)
            cfg["rate"] = sb.slider("Coarsening rate K (nm³/s)", 0.1, 3.0, 0.9, 0.1)
            cfg["n_frames"] = sb.slider("Number of frames", 6, 40, 20, 1)
            cfg["n_particles"] = sb.slider("Initial particle count", 30, 200, 95, 5)
            cfg["size"] = sb.select_slider("Image size (px)", [256, 384, 512], value=512)
            c1, c2 = sb.columns(2)
            cfg["pixel_size"] = c1.number_input("Pixel size (nm)", 0.05, 2.0, 0.25, 0.05, format="%.2f")
            cfg["interval"] = c2.number_input("Frame Δt (s)", 0.5, 120.0, 5.0, 0.5)
            cfg["seed"] = int(sb.number_input("Random seed", 0, 99999, 42, 1))
        else:
            cfg["upload"] = sb.file_uploader("Multi-frame TIFF (T, Y, X)", type=["tif", "tiff"])
            c1, c2 = sb.columns(2)
            cfg["pixel_size"] = c1.number_input("Pixel size (nm)", 0.01, 5.0, 0.25, 0.01, format="%.3f")
            cfg["interval"] = c2.number_input("Frame Δt (s)", 0.01, 3600.0, 5.0, 0.5)

    sb.header("Registration")
    cfg["upsample"] = sb.select_slider("Phase-correlation upsampling", [10, 20, 50, 100], value=50,
                                       help="Sub-pixel precision = 1 / upsampling factor (px).")
    sb.divider()
    sb.caption("insitu-tem-dynamics-quant · CLI: `python main.py`")
    return cfg


def execute(cfg: Dict) -> Optional[Dict]:
    if cfg["mode"] == MODE_A:
        if not DEMO_TIF.exists():
            st.error(f"Demo stack not found: `{DEMO_TIF}`. Run `python main.py` once to generate it.")
            return None
        return analyze_demo(cfg["pixel_size"], cfg["upsample"], DEMO_TIF.stat().st_mtime)
    if cfg["source"] == SRC_SIM:
        return analyze_simulation(cfg["n_frames"], cfg["size"], cfg["pixel_size"], cfg["interval"],
                                  float(cfg["dose"]), float(cfg["rate"]), cfg["n_particles"],
                                  cfg["seed"], cfg["upsample"])
    upload = cfg.get("upload")
    if upload is None:
        st.info("Upload a multi-frame `.tif` stack in the sidebar to start the analysis.")
        return None
    return analyze_upload(upload.getvalue(), upload.name, cfg["pixel_size"], cfg["interval"],
                          cfg["upsample"])


def main() -> None:
    st.set_page_config(page_title="In-situ TEM Dynamics Quant", page_icon="🔬", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("In-situ TEM Dynamics Quant")
    st.caption("Sub-pixel drift registration · nanoparticle segmentation · coarsening kinetics")

    cfg = sidebar_controls()
    try:
        with st.spinner("Running registration, segmentation and kinetic fitting …"):
            res = execute(cfg)
    except Exception as exc:  # 将异常展示在界面而非中断应用
        st.error(f"Analysis failed: {exc}")
        return
    if res is None:
        return

    h, w = res["shape"]
    st.markdown(
        f"<div class='status-bar'>SOURCE {res['source']} &nbsp;│&nbsp; {res['n_frames']} frames × "
        f"{h}×{w} px &nbsp;│&nbsp; {res['pixel_size']:g} nm/px &nbsp;│&nbsp; "
        f"pipeline {res['runtime_s']:.1f} s</div>",
        unsafe_allow_html=True,
    )
    render_metrics(res)
    st.write("")
    show_image(res["figure_png"], "A · raw t₀ / t_end · B · drift trajectory · "
                                  "C · segmentation at t_end · D · coarsening kinetics")
    render_downloads(res)
    with st.expander("Data tables", expanded=False):
        render_tables(res)


if __name__ == "__main__":
    main()
