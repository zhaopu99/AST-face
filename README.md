# AST-Face Processing Pipeline

This repository provides the processing pipeline code for the **AST-Face dataset**, covering the complete workflow from raw 3D face scans to topology-unified meshes with consistent vertex correspondence, as well as the generation of deformation fields for downstream applications.  

- **Dataset (public subset)**: [OSF Repository](https://osf.io/xk4f6/)  
- **Restricted subset (textures & RGB images)**: requires signing a Data Usage Agreement (DUA). See **Applying for Restricted Data** below.  

---

## 🚀 Pipeline Overview
The pipeline corresponds to the workflow described in the paper. Each script/module implements one major step:

- `01_icp.py` – ICP alignment of raw scans to a canonical pose  
- `02_landmarks/` – Extract 84 facial landmarks using [Deep-MVLM](https://github.com/RasmusRPaulsen/Deep-MVLM)  
- `03_flame_fitting/` – Fit FLAME models using [flame-fitting](https://github.com/Rubikplayer/flame-fitting)  
- `04_get_bfm_flame.py` – **FLAME-assisted Projection**: convert fitted FLAME meshes to BFM topology through subdivision & projection  
- `05_edge_nicp/` – Perform non-rigid registration with [edge-nicp](https://github.com/zhaopu99/edge-nicp) to unify topology and vertex count  
- `06_bfm_cropper.py` – Crop processed meshes
- `07_deformation_fields.py` – Compute per-vertex deformation fields relative to neutral scans  

---

## 📦 Final Outputs
- **Standardized Meshes** (unified topology, identical vertex count)  
- **84 Landmark annotations**  
- **Deformation fields**  

---

## ⚙️ Dependencies & Third-Party Code
- Python 3.9+  
- Libraries: `numpy`, `scipy`, `pytorch3d`  

---

## 🔒 Applying for Restricted Data (Textures & RGB Images)
The AST-Face dataset includes a restricted subset containing **facial textures** and **multi-view RGB images** (frontal, left, right) for the **52 participants** who explicitly consented to texture collection.  

To access this restricted subset:  
1. Download the **Data Usage Agreement (DUA)**.  
2. Read, complete, and sign the form.  
3. Send the signed DUA to **zhaoyaopu@buaa.edu.cn**  
   - Subject line: `AST-Face Restricted Data Access Request`  
4. Upon approval, you will receive instructions to download the restricted data.  

⚠️ **Note:** Public data (raw meshes without textures, standardized meshes, deformation fields, and landmarks) is freely available on [OSF](https://osf.io/xk4f6/). Only the restricted subset requires a DUA.  

---

## 🔗 External Repositories
- **Landmarks (84 pts):** [Deep-MVLM](https://github.com/RasmusRPaulsen/Deep-MVLM)  
- **FLAME fitting:** [flame-fitting](https://github.com/Rubikplayer/flame-fitting)  
- **Edge-NICP (our implementation):** [edge-nicp](https://github.com/zhaopu99/edge-nicp)  

---

## 🙏 Acknowledgements
We sincerely thank the authors of the following open-source projects for their contributions to the community, which were essential for building the AST-Face processing pipeline:  

- **Deep-MVLM** – for multi-view landmark detection and extraction of 84 anatomical facial landmarks.  
- **FLAME-fitting** – for the implementation of FLAME model fitting, enabling robust expression capture and geometry alignment.  

Their excellent work provided the foundation for key components of AST-Face.  

---

## ⚠️ Disclaimer
Code is provided **for academic research and reproducibility** of the paper.  
Users are responsible for compliance with data licenses and ethical guidelines.  
Any misuse of the dataset or code is solely the responsibility of the user.  
