# 02_landmarks — 84-point 3D facial landmarks (Deep-MVLM)

This step extracts **84 3D facial landmarks** for AST-Face meshes using the official **Deep-MVLM** implementation:

- Deep-MVLM: https://github.com/RasmusRPaulsen/Deep-MVLM

> This repository does **not** redistribute Deep-MVLM source code.  
> We provide a minimal usage guide to reproduce the landmark annotations used in AST-Face.

---

## Input / Output

**Input:** aligned face meshes from Step 01 (canonical pose)  
**Output:** per-mesh landmark files (`.txt` and `.vtk`, 84 points)

(Optionally) convert to `*.npy` with shape `(84, 3)` for downstream processing.

---

## 1) Get Deep-MVLM

```bash
git clone https://github.com/RasmusRPaulsen/Deep-MVLM.git
cd Deep-MVLM
pip install -r requirements.txt
```

---

## 2) Predict landmarks (AST-Face setting)

We use the **BU-3DFE 84-landmark** configuration provided by Deep-MVLM:

```bash
python predict.py --c configs/BU_3DFE-RGB+depth.json --n myscan
```

`--n` can be:
- a single mesh file (e.g., `xxx.obj`)
- a directory containing meshes
- a text file listing mesh paths

Example (directory mode):

```bash
python predict.py --c configs/BU_3DFE-RGB+depth.json --n /path/to/astface_meshes/
```

---

## Notes

- If a mesh has **no texture**, consider using a config with `geometry+depth` or `depth` renderings (see Deep-MVLM docs).
- Deep-MVLM assumes a roughly canonical coordinate system (face centered near origin; nose roughly toward +z).  
  In AST-Face, this is handled by Step 01 (`01_icp.py`).

---

## Citation

If you use Deep-MVLM, please cite:

```bibtex
@inproceedings{paulsen2018multi,
  title={Multi-view Consensus CNN for 3D Facial Landmark Placement},
  author={Paulsen, Rasmus R and Juhl, Kristine Aavild and Haspang, Thilde Marie and Hansen, Thomas and Ganz, Melanie and Einarsson, Gudmundur},
  booktitle={Asian Conference on Computer Vision},
  pages={706--719},
  year={2018},
  organization={Springer}
}
```
