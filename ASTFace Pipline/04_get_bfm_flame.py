#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_get_bfm_flame_cli.py — FLAME-assisted projection to BFM topology (AST-Face Step 04)

This script turns a fitted FLAME mesh into a mesh with **BFM topology** by using a precomputed
BFM-to-FLAME vertex index mapping (produced once, offline).

Two modes are provided:

1) reconstruct (recommended for AST-Face pipeline)
   Reconstruct a BFM-topology mesh by selecting vertices from a subdivided FLAME mesh
   using the mapping indices.

2) optimize_indices (optional / research mode)
   Given a BFM template mesh and a fitted FLAME mesh, solve a discrete assignment problem
   (via Gumbel-Softmax) to estimate the BFM->FLAME mapping indices, then save:
     - optimized_mesh.obj (BFM topology)
     - bfm_to_flame_indices.txt

Notes
- This step depends on PyTorch3D and requires GPU for the optimize_indices mode.
- For reproducibility in AST-Face, we **fix** the mapping indices and reuse them for all scans.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.ops import knn_points, SubdivideMeshes


# -----------------------------
# Mesh helpers
# -----------------------------
def subdivide_mesh(verts: torch.Tensor, faces: torch.Tensor, num_subdivisions: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Subdivide mesh (PyTorch3D) to increase vertex density.
    Returns (verts_packed, faces_packed).
    """
    mesh = Meshes(verts=[verts], faces=[faces]).cpu()
    subdivide = SubdivideMeshes()
    for _ in range(num_subdivisions):
        mesh = subdivide(mesh)
    return mesh.verts_packed(), mesh.faces_packed()


# -----------------------------
# Optimizer (optional mode)
# -----------------------------
class DiscreteVertexOptimizer:
    """
    Discrete assignment optimizer:
    For each BFM vertex, choose one candidate FLAME vertex from a KNN pool.
    Uses Gumbel-Softmax hard sampling for a differentiable discrete selection.
    """
    def __init__(
        self,
        bfm_mesh: Meshes,
        flame_verts: torch.Tensor,
        k_neighbors: int = 10,
        device: str = "cuda",
        lambda_dist: float = 0.8,
        lambda_normal: float = 0.0,
        lr: float = 0.15,
    ):
        self.device = torch.device(device)
        self.bfm_verts = bfm_mesh.verts_packed().float().to(self.device)
        self.flame_verts = flame_verts.float().to(self.device)

        with torch.no_grad():
            _, knn_idx, _ = knn_points(
                self.bfm_verts.unsqueeze(0),
                self.flame_verts.unsqueeze(0),
                K=k_neighbors
            )
            self.global_indices = knn_idx[0].clone()  # (V_bfm, K)
            self.candidate_pool = self.flame_verts[self.global_indices]  # (V_bfm, K, 3)

        self.logits = nn.Parameter(torch.zeros(len(self.bfm_verts), k_neighbors, device=self.device))
        self.optimizer = optim.Adam([self.logits], lr=lr)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == "cuda"))

        self.edges = bfm_mesh.edges_packed().to(self.device)
        with torch.no_grad():
            edge_vec = self.bfm_verts[self.edges[:, 0]] - self.bfm_verts[self.edges[:, 1]]
            self.orig_edge_len = torch.norm(edge_vec, dim=1)

        self.bfm_faces = bfm_mesh.faces_packed().to(self.device)
        with torch.no_grad():
            self.original_normals = self._compute_face_normals(self.bfm_verts, self.bfm_faces)

        self.lambda_normal = float(lambda_normal)
        self.lambda_dist = float(lambda_dist)

    def select_vertices(self, temperature: float = 0.1) -> torch.Tensor:
        weights = torch.nn.functional.gumbel_softmax(self.logits, tau=temperature, hard=True)
        return torch.einsum("vk,vkc->vc", weights, self.candidate_pool)

    def compute_loss(self, selected_verts: torch.Tensor) -> torch.Tensor:
        diff = selected_verts - self.bfm_verts
        data_loss = torch.mean(torch.sum(diff * diff, dim=1))

        edge_vec = selected_verts[self.edges[:, 0]] - selected_verts[self.edges[:, 1]]
        current_len = torch.norm(edge_vec, dim=1)
        dist_loss = torch.mean(torch.abs(current_len - self.orig_edge_len))

        if self.lambda_normal > 0:
            current_normals = self._compute_face_normals(selected_verts, self.bfm_faces)
            normal_cos = 1 - torch.cosine_similarity(current_normals, self.original_normals, dim=1)
            normal_loss = torch.mean(normal_cos)
        else:
            normal_loss = torch.tensor(0.0, device=self.device)

        return data_loss + self.lambda_dist * dist_loss + self.lambda_normal * normal_loss

    def optimize(self, iterations: int = 800, temp_start: float = 0.5, temp_end: float = 0.05, log_every: int = 50):
        for it in range(iterations):
            self.optimizer.zero_grad(set_to_none=True)

            # temperature annealing
            t = it / max(iterations - 1, 1)
            temp = max(temp_end, temp_start * (1 - t))

            with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
                selected = self.select_vertices(temp)
                loss = self.compute_loss(selected)

            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_([self.logits], 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if (it % log_every) == 0:
                print(f"[opt] iter {it:04d}/{iterations} | loss={loss.item():.6f} | temp={temp:.3f}")
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        with torch.no_grad():
            final_choices = torch.argmax(self.logits, dim=1)  # (V_bfm,)
            selected_verts = self.candidate_pool[torch.arange(len(final_choices), device=self.device), final_choices]
            global_idx = self.global_indices[torch.arange(len(final_choices), device=self.device), final_choices]
            return selected_verts.float(), global_idx

    @staticmethod
    def _compute_face_normals(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        e1 = v1 - v0
        e2 = v2 - v0
        normals = torch.cross(e1, e2, dim=1)
        return nn.functional.normalize(normals, dim=1)


def validate_indices(optimized_verts: torch.Tensor, flame_verts: torch.Tensor, indices: torch.Tensor, eps: float = 1e-6) -> bool:
    selected = flame_verts[indices]
    mismatch = torch.any(torch.abs(optimized_verts - selected) > eps, dim=1)
    if torch.any(mismatch):
        print(f"[warn] mismatched vertices: {int(torch.sum(mismatch).item())}")
        return False
    return True


# -----------------------------
# Reconstruct (pipeline mode)
# -----------------------------
def reconstruct_bfm_from_indices(
    original_flame_obj: str,
    bfm_faces_obj: str,
    indices_txt: str,
    output_obj: str,
    subdivisions: int = 6,
) -> None:
    """
    Reconstruct a BFM-topology mesh from a FLAME mesh using a precomputed BFM->FLAME index mapping.

    Args:
        original_flame_obj: fitted FLAME mesh OBJ
        bfm_faces_obj: an OBJ that provides the BFM face topology (faces only; vertices can be anything)
        indices_txt: text file with length V_bfm indices into the subdivided FLAME vertex array
        output_obj: output OBJ path (BFM topology)
        subdivisions: number of SubdivideMeshes steps applied to FLAME before indexing
    """
    for p in [original_flame_obj, bfm_faces_obj, indices_txt]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Not found: {Path(p).resolve()}")

    # Load FLAME
    flame_verts, flame_faces, _ = load_obj(original_flame_obj)
    t0 = time.time()
    flame_verts_sub, _ = subdivide_mesh(flame_verts, flame_faces.verts_idx, num_subdivisions=subdivisions)
    print(f"[reconstruct] subdivisions={subdivisions} | time={time.time()-t0:.2f}s")

    flame_np = flame_verts_sub.detach().cpu().numpy()

    # Load BFM faces
    _, bfm_faces, _ = load_obj(bfm_faces_obj)
    bfm_faces_idx = bfm_faces.verts_idx

    # Load indices
    idx = np.loadtxt(indices_txt, dtype=np.int64)
    if idx.ndim != 1:
        idx = idx.reshape(-1)
    if idx.min() < 0 or idx.max() >= len(flame_np):
        raise ValueError(f"Index out of range: min={idx.min()}, max={idx.max()}, flameV={len(flame_np)}")

    recon = flame_np[idx]  # (V_bfm, 3)
    recon_t = torch.tensor(recon, dtype=torch.float32)

    out_path = Path(output_obj)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_obj(str(out_path), recon_t, bfm_faces_idx)
    print(f"[reconstruct] saved: {out_path.resolve()} | V={recon_t.shape[0]} F={bfm_faces_idx.shape[0]}")


# -----------------------------
# CLI
# -----------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AST-Face Step 04: FLAME-assisted projection to BFM topology.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # reconstruct
    pr = sub.add_parser("reconstruct", help="Reconstruct BFM-topology mesh from FLAME mesh using precomputed indices.")
    pr.add_argument("--flame_obj", required=True, help="Input fitted FLAME mesh OBJ.")
    pr.add_argument("--bfm_faces_obj", required=True, help="OBJ providing BFM faces (topology).")
    pr.add_argument("--indices_txt", required=True, help="bfm_to_flame_indices.txt")
    pr.add_argument("--out_obj", required=True, help="Output BFM-topology OBJ.")
    pr.add_argument("--subdivisions", type=int, default=6, help="Number of SubdivideMeshes steps for FLAME.")

    # optimize_indices (optional)
    po = sub.add_parser("optimize_indices", help="(Optional) Estimate BFM->FLAME mapping indices by optimization.")
    po.add_argument("--bfm_obj", required=True, help="BFM template OBJ (verts+faces).")
    po.add_argument("--flame_obj", required=True, help="Fitted FLAME mesh OBJ.")
    po.add_argument("--out_dir", required=True, help="Output directory.")
    po.add_argument("--subdivisions", type=int, default=6, help="Subdivide FLAME steps before building candidates.")
    po.add_argument("--k_neighbors", type=int, default=10, help="KNN candidate pool size.")
    po.add_argument("--iterations", type=int, default=800, help="Optimization iterations.")
    po.add_argument("--device", type=str, default="cuda", help="cuda or cpu (opt mode is best on cuda).")
    po.add_argument("--lr", type=float, default=0.15, help="Optimizer learning rate.")
    po.add_argument("--lambda_dist", type=float, default=0.8, help="Edge-length regularization weight.")
    po.add_argument("--lambda_normal", type=float, default=0.0, help="Normal-consistency weight.")
    po.add_argument("--log_every", type=int, default=50, help="Log interval.")

    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.cmd == "reconstruct":
        reconstruct_bfm_from_indices(
            original_flame_obj=args.flame_obj,
            bfm_faces_obj=args.bfm_faces_obj,
            indices_txt=args.indices_txt,
            output_obj=args.out_obj,
            subdivisions=args.subdivisions,
        )
        return

    # optimize_indices
    dev = torch.device(args.device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load BFM mesh
    bfm_verts, bfm_faces, _ = load_obj(args.bfm_obj, device=dev)
    bfm_mesh = Meshes(verts=[bfm_verts], faces=[bfm_faces.verts_idx]).to(dev)

    # Load and subdivide FLAME
    flame_verts, flame_faces, _ = load_obj(args.flame_obj, device=dev)
    t0 = time.time()
    flame_verts_sub, _ = subdivide_mesh(flame_verts, flame_faces.verts_idx, num_subdivisions=args.subdivisions)
    print(f"[opt] subdivisions={args.subdivisions} | time={time.time()-t0:.2f}s")

    optimizer = DiscreteVertexOptimizer(
        bfm_mesh=bfm_mesh,
        flame_verts=flame_verts_sub,
        k_neighbors=args.k_neighbors,
        device=str(dev),
        lambda_dist=args.lambda_dist,
        lambda_normal=args.lambda_normal,
        lr=args.lr,
    )

    print("[opt] start optimizing indices...")
    optimized_verts, bfm_to_flame_indices = optimizer.optimize(
        iterations=args.iterations,
        log_every=args.log_every
    )

    ok = validate_indices(optimized_verts, flame_verts_sub, bfm_to_flame_indices)
    print(f"[opt] validate: {ok}")

    # Save
    indices_path = out_dir / "bfm_to_flame_indices.txt"
    np.savetxt(str(indices_path), bfm_to_flame_indices.detach().cpu().numpy().astype(np.int64), fmt="%d")
    print(f"[opt] saved indices: {indices_path.resolve()}")

    out_mesh_path = out_dir / "optimized_mesh.obj"
    save_obj(str(out_mesh_path), optimized_verts.detach().cpu(), bfm_mesh.faces_packed().detach().cpu())
    print(f"[opt] saved mesh: {out_mesh_path.resolve()}")


if __name__ == "__main__":
    main()
