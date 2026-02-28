# 05_edge_nicp — Edge-NICP topology unification (AST-Face)

This step performs **non-rigid registration** to unify topology and establish **dense vertex correspondence** across subjects and expressions in AST-Face.

We provide a runnable entry script:
- `run_edge_nicp.py` — registers a BFM-topology template mesh to a target mesh and writes the registered mesh as `.obj`.

> Note: This step requires a CUDA-enabled environment with **PyTorch3D**.

---

## Environment setup (tested)

```bash
conda create -n enicp python=3.9
conda activate enicp
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
pip install seaborn
```

---

## Required inputs

- **Template mesh (BFM topology)**: e.g., `testdata/astface_show/bfm_show_au27.obj`
- **Target mesh**: the scan/expression mesh to register, e.g., `testdata/astface_show/show_au27.obj`
- **Target landmarks (txt)**: `N x 3` 3D landmarks for the target mesh (used for rigid initialization)
- **Config file (json)**: e.g., `config/fine_grain.json` (inner/outer iterations, weights, schedules)

---

## Run (example)

```bash
python run_edge_nicp.py \
  --template_mesh result/flame_to_bfm.obj \
  --target_mesh testdata/astface_show/show_au27.obj \
  --target_landmarks_txt testdata/astface_show/show_au27_landmarks.txt \
  --config config/fine_grain.json \
  --out_mesh result/out_mesh.obj \
  --device cuda:0
```

Output:
- `result/out_mesh.obj` — registered mesh in the same topology as the template.

---

## Notes

- This script assumes meshes are roughly in canonical coordinates (see Step 01 alignment).
- Hyperparameters in `config/fine_grain.json` are fixed across subjects/expressions in AST-Face to ensure reproducibility.
