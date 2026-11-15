"""一键运行完整流水线：仿真 → 亚像素漂移校正 → 定量分割 → 动力学拟合 → 可视化。

示例：
    python main.py
    python main.py --frames 30 --pixel-size 0.2 --seed 7
    python main.py --skip-simulation          # 复用 data/ 下已有的序列
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.quantifier import SegmentationParams, fit_coarsening_kinetics, quantify_stack
from src.registration import drift_comparison_table, drift_rmse, register_stack
from src.simulator import SimulationConfig, load_stack, save_simulation, simulate_insitu_series
from src.visualizer import plot_summary


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="In-situ TEM nanoparticle coarsening: simulation, sub-pixel drift "
                    "registration and quantitative kinetics analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--frames", type=int, default=20, help="number of frames in the series")
    p.add_argument("--size", type=int, default=512, help="image size in pixels (square)")
    p.add_argument("--pixel-size", type=float, default=0.25, help="pixel size in nm/pixel")
    p.add_argument("--frame-interval", type=float, default=5.0, help="time between frames (s)")
    p.add_argument("--n-particles", type=int, default=95, help="initial number of particles")
    p.add_argument("--dose", type=float, default=40.0, help="electron dose per pixel per frame")
    p.add_argument("--seed", type=int, default=42, help="random seed (reproducibility)")
    p.add_argument("--upsample", type=int, default=50,
                   help="phase-correlation upsampling factor (precision = 1/upsample px)")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--skip-simulation", action="store_true",
                   help="reuse existing data/synthetic_insitu_stack.tif and ground_truth.csv")
    return p.parse_args(argv)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.frames < 3:
        raise SystemExit("--frames must be >= 3 for kinetic fitting.")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    # 1) 仿真 ---------------------------------------------------------------
    tif_path = args.data_dir / "synthetic_insitu_stack.tif"
    gt_path = args.data_dir / "ground_truth.csv"
    if args.skip_simulation:
        log(f"Loading existing series from {tif_path}")
        stack = load_stack(tif_path)
        ground_truth = pd.read_csv(gt_path) if gt_path.exists() else None
        times = (ground_truth["time_s"].to_numpy() if ground_truth is not None
                 else np.arange(stack.shape[0]) * args.frame_interval)
    else:
        cfg = SimulationConfig(
            n_frames=args.frames, image_size=args.size, pixel_size_nm=args.pixel_size,
            frame_interval_s=args.frame_interval, n_particles=args.n_particles,
            dose_e_per_px=args.dose, seed=args.seed,
        )
        log(f"Simulating {cfg.n_frames} frames, {cfg.image_size}x{cfg.image_size} px, "
            f"{cfg.pixel_size_nm} nm/px ...")
        sim = simulate_insitu_series(cfg)
        tif_path, gt_path = save_simulation(sim, args.data_dir)
        stack, ground_truth, times = sim.stack, sim.ground_truth, sim.times_s
        d_end = np.hypot(*sim.drift_px[-1])
        log(f"  saved {tif_path} and {gt_path} (cumulative drift {d_end:.1f} px)")

    # 2) 亚像素漂移校正 ------------------------------------------------------
    log("Estimating sub-pixel drift by phase correlation ...")
    reg = register_stack(stack, args.pixel_size, upsample_factor=args.upsample)
    drift_df = drift_comparison_table(reg, args.pixel_size, times, ground_truth)
    drift_df.to_csv(args.results_dir / "drift_trajectory.csv", index=False, float_format="%.5f")
    rmse = drift_rmse(drift_df)
    log(f"  drift RMSE vs ground truth: {rmse:.3f} px ({rmse * args.pixel_size:.4f} nm)")

    # 3) 定量分割 ------------------------------------------------------------
    log("Segmenting particles (NLM denoising, adaptive threshold, watershed) ...")
    quant = quantify_stack(reg.aligned, times, args.pixel_size, reg.valid_mask, SegmentationParams())
    quant.particles.to_csv(args.results_dir / "particle_measurements.csv", index=False,
                           float_format="%.5g")
    summary = quant.summary
    if ground_truth is not None:
        summary = summary.merge(
            ground_truth[["frame", "true_mean_radius_nm", "true_n_particles_fov"]], on="frame")
    summary.to_csv(args.results_dir / "frame_statistics.csv", index=False, float_format="%.5g")
    s0, s1 = summary.iloc[0], summary.iloc[-1]
    log(f"  N: {int(s0.n_particles)} -> {int(s1.n_particles)}, <D>: {s0.mean_diameter_nm:.2f} -> "
        f"{s1.mean_diameter_nm:.2f} nm, circularity(end) = {s1.mean_circularity:.3f}")

    # 4) 动力学拟合 ----------------------------------------------------------
    log("Fitting coarsening kinetics R^n - R0^n = k t ...")
    sem = summary["sem_radius_nm"].to_numpy()
    fit = fit_coarsening_kinetics(times, summary["mean_radius_nm"].to_numpy(), sigma=sem)
    fit_lsw = fit_coarsening_kinetics(times, summary["mean_radius_nm"].to_numpy(), sigma=sem,
                                      fixed_n=3.0)
    fits = {"free_exponent": fit, "lsw_n3": fit_lsw}
    if ground_truth is not None:
        fits["ground_truth_free_exponent"] = fit_coarsening_kinetics(
            times, ground_truth["true_mean_radius_nm"].to_numpy())
    with open(args.results_dir / "kinetics_fit.json", "w", encoding="utf-8") as fh:
        json.dump(fits, fh, indent=2)
    log(f"  n = {fit['n']:.2f} ± {fit['n_err']:.2f}, k = {fit['k']:.3g} nm^n/s, "
        f"R0 = {fit['R0_nm']:.2f} nm, R^2 = {fit['r_squared']:.3f}")

    # 5) 可视化 --------------------------------------------------------------
    fig_path = plot_summary(
        raw_stack=stack, times_s=times, pixel_size_nm=args.pixel_size, drift_table=drift_df,
        aligned_last=reg.aligned[-1], labels_last=quant.labels[-1], summary=summary,
        fit=fit, fit_lsw=fit_lsw, ground_truth=ground_truth, valid_mask=reg.valid_mask,
        out_path=args.results_dir / "insitu_quant_summary.png",
    )
    log(f"Summary figure saved to {fig_path}")
    log(f"Pipeline finished in {time.perf_counter() - t_start:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
