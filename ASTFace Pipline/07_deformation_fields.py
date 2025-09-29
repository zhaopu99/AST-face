import torch
import numpy as np
from pytorch3d.io import load_objs_as_meshes




device = torch.device('cuda')

au0_obj_path = ''
express_obj_path = ''
save_path = ''

meshes_au0 = load_objs_as_meshes([au0_obj_path], device = device)
point_au0 = meshes_au0.verts_packed()


meshes = load_objs_as_meshes([express_obj_path], device = device)
points = meshes.verts_packed()

deformation_fields = (points - point_au0).cpu().numpy()
np.save(save_path, deformation_fields)
