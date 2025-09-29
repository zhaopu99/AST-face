import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.ops import knn_points
from pytorch3d.ops import sample_points_from_meshes, knn_points, SubdivideMeshes
import time
import os
def subdivide_mesh(verts, faces, num_subdivisions=2):
    """
    对模型进行细分，增加顶点密度
    """
    mesh = Meshes(verts=[verts], faces=[faces]).cpu()
    subdivide = SubdivideMeshes()
    for _ in range(num_subdivisions):
        mesh = subdivide(mesh)
    return mesh.verts_packed(), mesh.faces_packed()

def deterministic_subdivide_mesh(verts, faces, num_subdivisions=2):
    """
    终极确定性细分：每次细分后执行顶点重排序
    """
    device = verts.device if isinstance(verts, torch.Tensor) else 'cpu'
    
    # 初始化网格
    mesh = Meshes(
        verts=[torch.as_tensor(verts, device=device)], 
        faces=[torch.as_tensor(faces, device=device)]
    )
    
    subdivide = SubdivideMeshes()
    
    for _ in range(num_subdivisions):
        # 执行细分
        mesh = subdivide(mesh)
        
        # 获取当前顶点和面
        verts_new = mesh.verts_packed()
        faces_new = mesh.faces_packed()
        
        # 生成动态排序键（防止浮点精度问题）
        scaled_verts = verts_new * 1e6  # 放大坐标值避免精度丢失
        compound_key = (scaled_verts[:, 0].long() << 40 | 
                        scaled_verts[:, 1].long() << 20 | 
                        scaled_verts[:, 2].long())
        
        # 获取排序索引
        sorted_idx = torch.argsort(compound_key)
        
        # 创建顶点映射表
        inv_mapping = torch.zeros_like(sorted_idx)
        inv_mapping[sorted_idx] = torch.arange(len(sorted_idx), device=device)
        
        # 重新排列顶点和面
        ordered_verts = verts_new[sorted_idx]
        ordered_faces = inv_mapping[faces_new]
        
        # 重建网格对象
        mesh = Meshes(verts=[ordered_verts], faces=[ordered_faces])
    
    return mesh.verts_packed(), mesh.faces_packed()

class DiscreteVertexOptimizer:
    def __init__(self, bfm_mesh, flame_verts, k_neighbors=30, device='cuda'):
        """
        改进版离散顶点优化器
        :param bfm_mesh: BFM网格 (PyTorch3D Meshes)
        :param flame_verts: FLAME顶点坐标 (N,3)
        :param k_neighbors: 每个顶点的候选邻居数
        :param device: 计算设备
        """
        self.device = device
        
        # 使用半精度存储候选池
        self.bfm_verts = bfm_mesh.verts_packed().float().to(device)
        flame_verts = flame_verts.to(device).float()  # 使用float16
        
        # 构建候选池
        with torch.no_grad():
            _, knn_idx, _ = knn_points(
                self.bfm_verts.unsqueeze(0), 
                flame_verts.unsqueeze(0), 
                K=k_neighbors
            )
            self.candidate_pool = flame_verts[knn_idx[0]]  # [B, K, 3]
        
        # 初始化选择概率
        self.logits = nn.Parameter(
            torch.zeros(len(self.bfm_verts), k_neighbors, device=device)
        )
        
        # 预计算边信息
        self.edges = bfm_mesh.edges_packed().to(device)
        with torch.no_grad():
            edge_vec = self.bfm_verts[self.edges[:,0]] - self.bfm_verts[self.edges[:,1]]
            self.orig_edge_len = torch.norm(edge_vec, dim=1)
        
        self.optimizer = optim.Adam([self.logits], lr=0.15)
        self.scaler = torch.cuda.amp.GradScaler()
        self.global_indices = knn_idx[0].clone()  # [num_bfm, k_neighbors]

        self.bfm_faces = bfm_mesh.faces_packed().to(device)
        with torch.no_grad():
            self.original_normals = self._compute_face_normals(bfm_mesh.verts_packed(), self.bfm_faces) 
        self.lambda_normal = 0
        self.lambda_dist=0.8

    def select_vertices(self, temperature=0.1):
        """改进的顶点选择方法"""
        weights = torch.nn.functional.gumbel_softmax(
            self.logits, 
            tau=temperature, 
            hard=True
        )
        return torch.einsum('bk,bkc->bc', weights, self.candidate_pool.float())  # 转回float32计算

    def compute_loss(self, selected_verts):
        """改进的损失计算"""
        # 数据项：最小化到最近候选点的距离
        with torch.cuda.amp.autocast():
            diff = selected_verts - self.bfm_verts
            data_loss = torch.mean(torch.sum(diff**2, dim=1))
            
            # 边长度约束项
            edge_vec = selected_verts[self.edges[:,0]] - selected_verts[self.edges[:,1]]
            current_len = torch.norm(edge_vec, dim=1)
            dist_loss = torch.mean(torch.abs(current_len - self.orig_edge_len))
            
            # 新增法线一致性损失项  <--- 新增开始
            current_normals = self._compute_face_normals(selected_verts, self.bfm_faces)
            normal_cos = 1 - torch.cosine_similarity(current_normals, self.original_normals, dim=1)
            normal_loss = torch.mean(normal_cos)

            return data_loss + self.lambda_dist * dist_loss + self.lambda_normal * normal_loss

    def optimize(self, iterations=800):
        """内存优化的训练循环"""
        for epoch in range(iterations):
            self.optimizer.zero_grad()
            
            # 退火温度策略
            temp = max(0.05, 0.5 * (1 - epoch/max(iterations,1)))
            selected_verts = self.select_vertices(temp)
            
            # 混合精度计算
            with torch.cuda.amp.autocast():
                loss = self.compute_loss(selected_verts)
            
            # 梯度管理
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_([self.logits], 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # 定期清理缓存
            if epoch % 50 == 0:
                torch.cuda.empty_cache()
                print(f"Iter {epoch:4d} | Loss: {loss.item():.6f} | Temp: {temp:.3f}")

        # 最终硬选择
        '''
        with torch.no_grad():
            final_choices = torch.argmax(self.logits, dim=1)
            return self.candidate_pool[torch.arange(len(final_choices)), final_choices].float()
        '''
        # 修改最后的返回部分
        with torch.no_grad():
            final_choices = torch.argmax(self.logits, dim=1)
            selected_verts = self.candidate_pool[torch.arange(len(final_choices)), final_choices].float()
            global_idx = self.global_indices[torch.arange(len(final_choices)), final_choices]
            return selected_verts, global_idx

    def _compute_face_normals(self, verts, faces):
        """计算面法线"""
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        e1 = v1 - v0
        e2 = v2 - v0
        normals = torch.cross(e1, e2, dim=1)
        return nn.functional.normalize(normals, dim=1)

def load_data(bfm_path, flame_path, device='cuda'):
    """改进的数据加载方法"""
    # 加载BFM模型
    bfm_verts, bfm_faces, _ = load_obj(bfm_path, device=device)
    bfm_mesh = Meshes(verts=[bfm_verts], faces=[bfm_faces.verts_idx])
    
    # 加载FLAME模型并去重
    flame_verts, flame_faces, _ = load_obj(flame_path, device=device)
    t1 = time.time()
    flame_verts, flame_faces_after = subdivide_mesh(flame_verts, flame_faces[0], num_subdivisions=3)
    t2 = time.time()
    flame_faces_after_np = flame_faces_after.detach().cpu().numpy()
    print(t2-t1)
    
    return bfm_mesh.to(device), flame_verts.to(device)

def validate_result(optimized_verts, flame_verts, indices):
    """更精确的索引验证方法"""
    selected = flame_verts[indices]
    mismatch = torch.any(torch.abs(optimized_verts - selected) > 1e-6, dim=1)
    if torch.any(mismatch):
        print(f"警告：发现{torch.sum(mismatch)}个顶点不匹配")
        return False
    return True

def main():
    # 配置环境
    torch.backends.cudnn.benchmark = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载数据
    #bfm_path = "/home/zhaoyaopu/face_reconstruction/BFM_to_FLAME/data/template.obj"
    bfm_path = "/home/zhaoyaopu/face_reconstruction/flame-fitting/data/openmouth11/after_scan.obj"
    bfm_verts, bfm_faces, _ = load_obj(bfm_path, device=device)
    bfm_mesh = Meshes(verts=[bfm_verts], faces=[bfm_faces.verts_idx]).to(device)
    
    #flame_path = "/home/zhaoyaopu/face_reconstruction/BFM_to_FLAME/output/template_flame.obj"
    flame_path = "/home/zhaoyaopu/face_reconstruction/flame-fitting/output/openmouth11/fit_scan_result.obj"
    # 加载FLAME模型并去重
    flame_verts, flame_faces, _ = load_obj(flame_path, device=device)
    t1 = time.time()
    flame_verts, flame_faces_after = subdivide_mesh(flame_verts, flame_faces[0], num_subdivisions=6)
    t2 = time.time()
    print('subdivisions time:', t2-t1)
    flame_verts = flame_verts.to(device)
    

    # 初始化优化器
    optimizer = DiscreteVertexOptimizer(
        bfm_mesh=bfm_mesh,
        flame_verts=flame_verts,
        k_neighbors=10,
        device=device
    )
    
    # 执行优化
    print("开始优化...")
    optimized_verts, bfm_to_flame_indices = optimizer.optimize(iterations=800)
    
    
    # 修改验证方式
    print("验证结果...")
    validate_result(optimized_verts, flame_verts, bfm_to_flame_indices)
    
    np.savetxt("bfm_to_flame_indices.txt", 
               bfm_to_flame_indices.cpu().numpy(),
               fmt='%d')
    
    print("索引文件已保存为 bfm_to_flame_indices.txt")
    # 保存结果
    save_obj(
        "optimized_mesh.obj",
        optimized_verts.cpu(),
        bfm_mesh.faces_packed().cpu()
    )
    print("优化完成，结果已保存")

def reconstruct_bfm_from_indices(original_flame_path, 
                                original_bfm_faces_path,
                                indices_txt_path,
                                output_obj_path):
    """
    根据索引文件从原始FLAME重建BFM模型
    
    参数：
    original_flame_path: 原始FLAME模型.obj路径
    original_bfm_faces_path: 原始BFM面结构.obj路径
    indices_txt_path: 索引文件路径
    output_obj_path: 输出文件路径
    
    返回：
    重建后的BFM网格（PyTorch3D Meshes）
    """
    # 加载原始FLAME顶点数据
    flame_verts, flame_faces, _ = load_obj(original_flame_path)
    t1 = time.time()
    flame_verts, faces = subdivide_mesh(flame_verts, flame_faces[0], num_subdivisions=6)
    t2 = time.time()
    print('subdivisions time:', t2-t1)
    flame_verts = flame_verts.numpy()  # 转换为numpy数组方便索引
    
    # 加载原始BFM面结构
    _, bfm_faces, _ = load_obj(original_bfm_faces_path)
    
    # 读取索引文件
    bfm_to_flame_indices = np.loadtxt(indices_txt_path, dtype=np.int64)
    
    # 验证索引有效性
    valid_mask = (bfm_to_flame_indices >= 0) & (bfm_to_flame_indices < len(flame_verts))
    if not np.all(valid_mask):
        invalid_indices = np.where(~valid_mask)[0]
        raise ValueError(
            f"发现{len(invalid_indices)}个无效索引，例如："
            f"顶点{invalid_indices[:5]}的索引值为"
            f"{bfm_to_flame_indices[invalid_indices[:5]]} "
            f"(有效范围0-{len(flame_verts)-1})"
        )
    
    # 根据索引获取顶点坐标
    reconstructed_verts = flame_verts[bfm_to_flame_indices]
    
    # 转换为PyTorch张量
    reconstructed_verts = torch.tensor(reconstructed_verts, dtype=torch.float32)
    
    # 保存重建后的模型
    save_obj(
        output_obj_path,
        reconstructed_verts,
        bfm_faces.verts_idx
    )
    
    # 返回PyTorch3D网格对象
    return Meshes(
        verts=[reconstructed_verts],
        faces=[bfm_faces.verts_idx]
    )



if __name__ == "__main__":
    original_flame_path = ''
    out_obj_path = ''

    if not os.path.exists(out_obj_path):
        reconstructed_mesh = reconstruct_bfm_from_indices(
            original_flame_path = original_flame_path,
            original_bfm_faces_path = "/home/zhaoyaopu/face_reconstruction/BFM_to_FLAME/data/template.obj",
            indices_txt_path="data/bfm_to_flame_indices.txt",
            output_obj_path=out_obj_path
        )
        print(f"BFM模型完成, 结果已保存为{out_obj_path}")


