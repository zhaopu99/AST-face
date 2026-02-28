#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_icp.py — ICP alignment for AST-Face pipeline

This script aligns a source OBJ mesh to a target OBJ mesh using a similarity transform
(rotation + translation + optional scale) estimated by ICP.

Typical usage (single pair):
  python 01_icp.py --source path/to/src.obj --target path/to/tgt.obj --out aligned.obj --estimate_scale

Batch usage (align all .obj in a folder to one target):
  python 01_icp.py --source_dir path/to/folder --target path/to/tgt.obj --inplace

Notes
- This script replaces only vertex positions ("v ...") and keeps all other OBJ lines.
- ICP runs on point clouds formed by the mesh vertices (no normals, no faces used for matching).
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree


# -----------------------------
# Data structures
# -----------------------------
@dataclass(frozen=True)
class SimilarityTransform:
    R: np.ndarray  # (3,3)
    T: np.ndarray  # (3,)
    s: float       # scalar


@dataclass(frozen=True)
class ICPSolution:
    converged: bool
    rmse: float
    Xt: np.ndarray               # transformed X (in the same normalized space as Y)
    RTs: SimilarityTransform
    t_history: List[SimilarityTransform]


# -----------------------------
# OBJ utilities
# -----------------------------
def parse_obj_vertices(obj_path: str) -> np.ndarray:
    """Parse vertices from an OBJ file (lines starting with 'v ')."""
    verts: List[List[float]] = []
    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not verts:
        raise ValueError(f"No vertices found in OBJ: {obj_path}")
    return np.asarray(verts, dtype=np.float32)


def save_aligned_obj(original_obj_path: str, new_vertices: np.ndarray, output_path: str) -> None:
    """
    Write a new OBJ by replacing vertex positions in `original_obj_path` with `new_vertices`,
    while preserving all other lines (faces, materials, vt/vn, etc.).
    """
    vertex_index = 0
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(original_obj_path, "r", encoding="utf-8", errors="ignore") as f_in, \
         open(out_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if line.startswith("v "):
                if vertex_index >= new_vertices.shape[0]:
                    raise ValueError(
                        f"Vertex count mismatch: file has more vertices than provided array "
                        f"({new_vertices.shape[0]})."
                    )
                x, y, z = new_vertices[vertex_index]
                f_out.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
                vertex_index += 1
            else:
                f_out.write(line)

    if vertex_index != new_vertices.shape[0]:
        raise ValueError(
            f"Vertex count mismatch: provided {new_vertices.shape[0]} vertices, "
            f"but wrote {vertex_index} vertices."
        )


# -----------------------------
# ICP core
# -----------------------------
def normalize(points: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, float]:
    """Normalize point cloud to unit sphere (centered, scaled)."""
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    scale = float(np.max(np.linalg.norm(centered, axis=1)) + eps)
    return centered / scale, centroid, scale


def denormalize(points_norm: np.ndarray, centroid: np.ndarray, scale: float) -> np.ndarray:
    """Invert normalize()."""
    return points_norm * scale + centroid


def corresponding_points_alignment(
    X: np.ndarray,
    Y: np.ndarray,
    weights: Optional[np.ndarray] = None,
    estimate_scale: bool = False,
    allow_reflection: bool = False,
    eps: float = 1e-9
) -> SimilarityTransform:
    """
    Estimate similarity transform that maps X -> Y (in a least-squares sense).
    This uses a weighted Procrustes solution with optional uniform scale.
    """
    if X.shape != Y.shape:
        raise ValueError(f"X and Y must have same shape, got {X.shape} vs {Y.shape}")

    N, dim = X.shape
    w = weights if weights is not None else np.ones((N,), dtype=np.float64)
    w = w.astype(np.float64)
    total_w = float(np.clip(w.sum(), eps, None))

    X_mean = (X.T @ w) / total_w
    Y_mean = (Y.T @ w) / total_w

    Xc = X - X_mean
    Yc = Y - Y_mean

    cov_xy = (Xc.T * w) @ Yc / total_w
    U, S, Vt = np.linalg.svd(cov_xy)

    E = np.eye(dim)
    if not allow_reflection:
        if np.linalg.det(U @ Vt) < 0:
            E[-1, -1] = -1

    R = U @ E @ Vt

    if estimate_scale:
        x_var = float(np.sum(w * np.sum(Xc * Xc, axis=1)) / total_w)
        scale = float((S * np.diag(E)).sum() / (x_var + eps))
    else:
        scale = 1.0

    T = Y_mean - scale * (R @ X_mean)
    return SimilarityTransform(R=R, T=T.astype(np.float64), s=scale)


def iterative_closest_point(
    X: np.ndarray,  # (N,3)
    Y: np.ndarray,  # (M,3)
    init_transform: Optional[SimilarityTransform] = None,
    max_iterations: int = 100,
    relative_rmse_thr: float = 1e-6,
    estimate_scale: bool = False,
    allow_reflection: bool = False,
    verbose: bool = False,
) -> ICPSolution:
    """
    ICP (single-pair). We solve for a similarity transform from X to Y using nearest neighbors.
    """
    if X.ndim != 2 or Y.ndim != 2 or X.shape[1] != Y.shape[1]:
        raise ValueError(f"Expected X,Y shapes (N,3) and (M,3). Got {X.shape}, {Y.shape}")

    Xt = X.copy()
    t_history: List[SimilarityTransform] = []
    prev_rmse: Optional[float] = None

    # init
    if init_transform is not None:
        R0, T0, s0 = init_transform.R, init_transform.T, init_transform.s
        Xt = s0 * (Xt @ R0) + T0
    tree = cKDTree(Y)

    converged = False
    rmse = float("inf")

    for it in range(max_iterations):
        _, nn_idx = tree.query(Xt)
        Y_nn = Y[nn_idx]

        tform = corresponding_points_alignment(
            X, Y_nn,
            estimate_scale=estimate_scale,
            allow_reflection=allow_reflection
        )

        Xt = tform.s * (X @ tform.R) + tform.T
        t_history.append(tform)

        residual = np.linalg.norm(Xt - Y_nn, axis=1)
        rmse = float(np.sqrt(np.mean(residual * residual)))

        if prev_rmse is not None:
            rel = abs(prev_rmse - rmse) / max(prev_rmse, 1e-12)
            if rel < relative_rmse_thr:
                converged = True
                break

        prev_rmse = rmse
        if verbose:
            print(f"[ICP] iter {it+1:03d}: rmse={rmse:.6f}")

    # compose final transform as the last one in history
    RTs = t_history[-1] if t_history else SimilarityTransform(R=np.eye(3), T=np.zeros(3), s=1.0)
    return ICPSolution(converged=converged, rmse=rmse, Xt=Xt, RTs=RTs, t_history=t_history)


# -----------------------------
# CLI
# -----------------------------
def _require(path: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"Not found: {Path(path).resolve()}")


def align_one(
    source_obj: str,
    target_obj: str,
    out_obj: str,
    estimate_scale: bool,
    max_iterations: int,
    relative_rmse_thr: float,
    verbose: bool
) -> ICPSolution:
    _require(source_obj)
    _require(target_obj)

    X = parse_obj_vertices(source_obj)
    Y = parse_obj_vertices(target_obj)

    Xn, _, _ = normalize(X)
    Yn, Yc, Ys = normalize(Y)

    sol = iterative_closest_point(
        Xn, Yn,
        estimate_scale=estimate_scale,
        max_iterations=max_iterations,
        relative_rmse_thr=relative_rmse_thr,
        verbose=verbose
    )

    # denormalize to target space
    X_aligned = denormalize(sol.Xt, Yc, Ys)

    save_aligned_obj(source_obj, X_aligned, out_obj)
    return sol


def main() -> None:
    ap = argparse.ArgumentParser(
        description="ICP align OBJ meshes (AST-Face step 01).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    g1 = ap.add_argument_group("single-pair mode")
    g1.add_argument("--source", type=str, default="", help="Source OBJ to be aligned.")
    g1.add_argument("--target", type=str, default="", help="Target OBJ (reference).")
    g1.add_argument("--out", type=str, default="", help="Output OBJ path (single-pair).")

    g2 = ap.add_argument_group("batch mode")
    g2.add_argument("--source_dir", type=str, default="", help="Folder containing OBJ files to align.")
    g2.add_argument("--out_dir", type=str, default="", help="Output folder for aligned OBJs.")
    g2.add_argument("--inplace", action="store_true", help="Overwrite OBJs in source_dir (batch mode).")

    ap.add_argument("--estimate_scale", action="store_true", help="Estimate uniform scale during ICP.")
    ap.add_argument("--max_iterations", type=int, default=100, help="Max ICP iterations.")
    ap.add_argument("--relative_rmse_thr", type=float, default=1e-6, help="Relative RMSE threshold for convergence.")
    ap.add_argument("--verbose", action="store_true", help="Print per-iteration RMSE.")

    args = ap.parse_args()

    # Determine mode
    if args.source_dir:
        if not args.target:
            raise ValueError("--target is required for --source_dir mode.")
        _require(args.target)
        src_dir = Path(args.source_dir)
        if not src_dir.exists():
            raise FileNotFoundError(f"--source_dir not found: {src_dir.resolve()}")

        if args.inplace:
            out_dir = src_dir
        else:
            if not args.out_dir:
                raise ValueError("--out_dir is required unless --inplace is set.")
            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

        obj_files = sorted([p for p in src_dir.rglob("*.obj")])
        if not obj_files:
            raise ValueError(f"No .obj files found under: {src_dir.resolve()}")

        print(f"[ICP] batch mode: {len(obj_files)} files")
        for p in obj_files:
            rel = p.relative_to(src_dir)
            out_path = (out_dir / rel) if not args.inplace else p
            out_path.parent.mkdir(parents=True, exist_ok=True)

            t0 = time.time()
            sol = align_one(
                source_obj=str(p),
                target_obj=args.target,
                out_obj=str(out_path),
                estimate_scale=args.estimate_scale,
                max_iterations=args.max_iterations,
                relative_rmse_thr=args.relative_rmse_thr,
                verbose=args.verbose
            )
            dt = time.time() - t0
            print(f"[ICP] {rel} -> {out_path.name} | rmse={sol.rmse:.6f} | converged={sol.converged} | {dt:.2f}s")
        return

    # single pair
    if not (args.source and args.target and args.out):
        raise ValueError("Single-pair mode requires --source, --target, and --out (or use --source_dir batch mode).")

    t0 = time.time()
    sol = align_one(
        source_obj=args.source,
        target_obj=args.target,
        out_obj=args.out,
        estimate_scale=args.estimate_scale,
        max_iterations=args.max_iterations,
        relative_rmse_thr=args.relative_rmse_thr,
        verbose=args.verbose
    )
    dt = time.time() - t0
    print(f"[ICP] done | rmse={sol.rmse:.6f} | converged={sol.converged} | {dt:.2f}s")


if __name__ == "__main__":
    main()
