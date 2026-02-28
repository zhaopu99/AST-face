# 03_flame_fitting — FLAME model fitting (flame-fitting)

This step fits a **FLAME** face model to each scan to obtain a robust parametric initialization for later topology-unification.

We use the official open-source implementation:

- flame-fitting: https://github.com/Rubikplayer/flame-fitting

> This repository does **not** redistribute FLAME assets or the flame-fitting source code.  
> Please obtain them from the official sources and follow the instructions below.

---

## Input / Output

**Input:** aligned meshes from Step 01 (canonical pose) + corresponding 3D landmarks from Step 02  
**Output:** fitted FLAME meshes / parameters used for subsequent projection (Step 04)

---

## 1) Get flame-fitting

```bash
git clone https://github.com/Rubikplayer/flame-fitting.git
cd flame-fitting
pip install -r requirements.txt
```

⚠️ **FLAME model files are not included** in the repository and require a separate license/registration from the FLAME authors.

---

## 2) Run FLAME fitting (AST-Face setting)

We use the default configuration provided by `flame-fitting` and perform fitting in two stages:
(i) rigid landmark-based alignment, then (ii) non-rigid fitting with identity/expression parameters.

Please follow the upstream repository instructions for the exact command line and required paths (FLAME model, landmark embeddings, and scan/landmark inputs).

---

## Notes

- This step is CPU-based (Chumpy optimization) and may take time depending on mesh resolution.
- Ensure input meshes are roughly in canonical coordinates (handled by Step 01).
- The fitted FLAME result is used as an initialization for the FLAME-assisted projection in Step 04 (`04_get_bfm_flame.py`).

---

## Citation / License

- FLAME is distributed under its own license (see the FLAME project website).
- If you use `flame-fitting`, please cite the FLAME paper and/or the flame-fitting repository as appropriate.
