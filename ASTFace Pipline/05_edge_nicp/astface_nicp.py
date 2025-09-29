import torch
import io3d
import os
import numpy as np
import scipy.io as sio
import json
from landmark import get_landmarks_3d
from nicp import non_rigid_icp_mesh2pcl, non_rigid_icp_mesh2mesh, tranformAtoB, non_rigid_icp_mesh2mesh_2, tranformAtoB2, non_rigid_icp_mesh2mesh_edge
from pytorch3d.structures import Meshes
from utils import batch_vertex_sample

import time

 
device = torch.device('cuda:1')
fine_config = json.load(open('/pytorch-nicp/config/fine_grain.json'))
path_config = json.load(open('/pytorch-nicp/config/path.json'))

source_ld_index = np.array([21,25,35,31,83,40,41,43,4,3,1,0,7,5,8,9,11,12,13,15,60,49,50,51,52,53,64,55,57,59,61,62,63,65,66,67])
ldindex = np.array([17,21,22,26,30,31,33,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,57,59,61,62,63,65,66,67])

bfm_lm_index_np = np.loadtxt("/pytorch-nicp/BFM/bfm_landmarks_withear.txt").astype(np.int64)[ldindex]

''' '''


id = 'show'
kind = 'au27'


kpt = np.load(f'ldmks_{id}_{kind}.npy')
obj_file_name=f'scan_after_{id}_{kind}.obj'

out_path = 'astface_nicp'

print(obj_file_name)

target_lm = torch.unsqueeze(torch.tensor(kpt,dtype=torch.float32).to(device), 0)
meshes = io3d.load_objs_as_meshes([obj_file_name]).to(device)
ldindex = np.array([17,21,22,26,30,31,33,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,57,59,61,62,63,65,66,67])

with torch.no_grad():

    
    bfm_path = 'pytorch-nicp/BFM/template_5W.obj'
    bfm_meshes = io3d.load_obj_as_mesh(bfm_path, device = device)
    
    verts2 = bfm_meshes.verts_list()[0]
    # 计算每个坐标轴的最小值
    min_vals, _ = torch.min(bfm_meshes.verts_list()[0], dim=0)
    # 计算每个坐标轴的最大值
    max_vals, _ = torch.max(bfm_meshes.verts_list()[0], dim=0)
    print(min_vals)
    print(max_vals)

    bfm_lm_index_np = np.loadtxt("BFM/bfm_landmarks_withear.txt").astype(np.int64)[ldindex]
    

    bfm_lm_index = torch.unsqueeze(torch.from_numpy(bfm_lm_index_np), 0).to(device)    
    bfm_lm = torch.tensor(batch_vertex_sample(bfm_lm_index, bfm_meshes.verts_padded()), dtype=torch.float32).to(device)
    numpy_bfm_lm = bfm_lm.detach().cpu().numpy()[0]

    after_bfm, after_bfm_lm = tranformAtoB2(meshes, bfm_meshes, target_lm, bfm_lm, device)
    

    # 开始计时
start_time = time.time()
registered_mesh = non_rigid_icp_mesh2mesh_edge(after_bfm, meshes, bfm_lm_index, target_lm, fine_config, device, with_edge = True, with_ld = True)


end_time = time.time()
# 计算并打印时间差
elapsed_time = end_time - start_time
print(f"代码执行时间: {elapsed_time:.5f} 秒")
path_out_mesh = f"{out_path}/edge_{id}_{kind}.obj"
io3d.save_meshes_as_objs([path_out_mesh], registered_mesh, save_textures = False)

