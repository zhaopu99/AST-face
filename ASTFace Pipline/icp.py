import os
import time
import numpy as np
from scipy.spatial import cKDTree
from collections import namedtuple

# 定义数据结构
SimilarityTransform = namedtuple('SimilarityTransform', ['R', 'T', 's'])
ICPSolution = namedtuple('ICPSolution', ['converged', 'rmse', 'Xt', 'RTs', 't_history'])

def corresponding_points_alignment(
    X: np.ndarray,
    Y: np.ndarray,
    weights: np.ndarray = None,
    estimate_scale: bool = False,
    allow_reflection: bool = False,
    eps: float = 1e-9
) -> SimilarityTransform:
    N, dim = X.shape
    weights = weights if weights is not None else np.ones(N)
    total_weight = np.clip(weights.sum(), eps, None)
    
    # 加权质心
    X_mean = (X.T @ weights) / total_weight
    Y_mean = (Y.T @ weights) / total_weight
    
    # 中心化
    X_centered = X - X_mean
    Y_centered = Y - Y_mean
    
    # 协方差矩阵（使用加权均值）
    cov_xy = (X_centered.T * weights) @ Y_centered / total_weight
    
    # SVD分解
    U, S, Vt = np.linalg.svd(cov_xy)
    
    # 处理反射
    E = np.eye(dim)
    if not allow_reflection:
        if np.linalg.det(U @ Vt) < 0:
            E[-1, -1] = -1
    
    # 关键修正点：计算迹的方式
    if estimate_scale:
        x_var = np.sum(weights * np.linalg.norm(X_centered, axis=1)**2) / total_weight
        scale = (np.sum(S * np.diag(E))) / (x_var + eps)  # 修正后的迹计算

    else:
        scale = 1.0
    
    # 计算旋转和平移
    R = U @ E @ Vt
    T = Y_mean - scale * (R @ X_mean)
    
    return SimilarityTransform(R, T, scale)

def iterative_closest_point(
    X: np.ndarray,  # (N, dim)
    Y: np.ndarray,  # (M, dim)
    init_transform: SimilarityTransform = None,
    max_iterations: int = 100,
    relative_rmse_thr: float = 1e-6,
    estimate_scale: bool = False,
    allow_reflection: bool = False,
    verbose: bool = False,
) -> ICPSolution:
    """
    改进后的ICP实现（单样本版本）
    """
    Xt = X.copy()
    t_history = []
    prev_rmse = None
    
    # 初始化变换参数
    if init_transform:
        R = init_transform.R
        T = init_transform.T
        s = init_transform.s
        Xt = s * (Xt @ R) + T
    else:
        R = np.eye(X.shape[1])
        T = np.zeros(X.shape[1])
        s = 1.0
    
    converged = False
    for iter in range(max_iterations):
        # 最近邻搜索
        tree = cKDTree(Y)
        _, nn_indices = tree.query(Xt)
        Y_nn = Y[nn_indices]
        
        # 点云对齐
        transform = corresponding_points_alignment(
            X, Y_nn,
            estimate_scale=estimate_scale,
            allow_reflection=allow_reflection
        )
        
        # 更新变换参数
        R = transform.R
        T = transform.T
        s = transform.s
        
        # 应用变换到原始点云
        Xt = s * (X @ R) + T
        t_history.append(transform)
        
        # 计算RMSE
        residuals = np.linalg.norm(Xt - Y_nn, axis=1)
        rmse = np.sqrt(np.mean(residuals**2))
        
        # 收敛判断
        if prev_rmse is not None:
            relative_rmse = abs(prev_rmse - rmse) / prev_rmse
            if relative_rmse < relative_rmse_thr:
                converged = True
                break
        
        prev_rmse = rmse
        if verbose:
            print(f"Iteration {iter+1}: RMSE = {rmse:.6f}")
    
    return ICPSolution(
        converged=converged,
        rmse=rmse,
        Xt=Xt,
        RTs=SimilarityTransform(R, T, s),
        t_history=t_history
    )

def normalize(points):
    """点云归一化到单位球内"""
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    scale = np.max(np.linalg.norm(centered, axis=1)) + 1e-8
    return centered / scale, centroid, scale

def denormalize(normalized_points: np.ndarray, 
                centroid: np.ndarray, 
                scale: float) -> np.ndarray:
    """
    将归一化后的点云还原到原始坐标系
    参数:
        normalized_points: 归一化后的点云 (N,3)
        centroid: 原始点云质心 (3,)
        scale: 原始点云缩放因子 (标量)
    返回:
        原始点云 (N,3)
    """
    # 1. 恢复缩放: 乘以 scale
    scaled_points = normalized_points * scale
    # 2. 恢复平移: 加上质心
    original_points = scaled_points + centroid
    return original_points


def save_aligned_obj(
    original_obj_path: str,      # 原始OBJ文件路径
    new_vertices: np.ndarray,    # 对齐后的顶点数组 (N,3)
    output_path: str             # 输出OBJ文件路径
) -> None:
    """
    将配准后的顶点坐标写入新OBJ文件，保留原有面、纹理、材质信息
    """
    vertex_index = 0  # 当前处理的顶点索引
    
    with open(original_obj_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            # 处理顶点行：替换为新坐标
            if line.startswith('v '):
                if vertex_index >= new_vertices.shape[0]:
                    raise ValueError("顶点数量不匹配")
                
                x, y, z = new_vertices[vertex_index]
                f_out.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
                vertex_index += 1
            
            # 其他行直接复制（材质、纹理、面等）
            else:
                f_out.write(line)
    
    print(f"已保存配准后的OBJ文件至: {os.path.abspath(output_path)}")


def parse_obj_manually(obj_path: str) -> dict:
    """
    手动解析OBJ文件，严格保留顶点、纹理、面的原始顺序
    返回:
        {
            'vertices': np.array (N,3),    # 顶点坐标（顺序与文件一致）
            'texcoords': np.array (M,2),   # 纹理坐标
            'faces': list of face_defs     # 面定义（原始字符串）
        }
    """
    vertices = []
    texcoords = []
    faces = []
    
    with open(obj_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            
            if parts[0] == 'v':
                # 顶点坐标: v x y z
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'vt':
                # 纹理坐标: vt u v
                texcoords.append([float(parts[1]), float(parts[2])])
            elif parts[0] == 'f':
                # 面定义: f v1/vt1 v2/vt2 v3/vt3
                faces.append(' '.join(parts[1:]))
    
    return {
        'vertices': np.array(vertices, dtype=np.float32),
        'texcoords': np.array(texcoords, dtype=np.float32),
        'faces': faces
    }



def icp_folder(folder_path, target_scan_path):
    # 加载模型A和B的顶点（假设B为目标模型）
    target = parse_obj_manually(target_scan_path)['vertices']
    target_norm, target_centroid, target_scale = normalize(target)

    for obj_scan_path in os.listdir(folder_path):
        if obj_scan_path.endswith('.obj'):
            print(obj_scan_path)
            t1 = time.time()
            scan_path = os.path.join(folder_path, obj_scan_path)
            X = parse_obj_manually(scan_path)['vertices']
            X_norm, scan_centroid, scan_scale = normalize(X)
            result = iterative_closest_point(X_norm, target_norm, estimate_scale=True)
            print("\n对齐结果:")
            print(f"收敛状态: {result.converged}")
            print(f"最终RMSE: {result.rmse:.6f}")
            print("估计旋转矩阵:\n", result.RTs.R.round(2))
            print("估计平移向量:", result.RTs.T.round(2))
            print("估计缩放因子:", result.RTs.s.round(3))

            restored_points = denormalize(result.Xt, target_centroid, target_scale)

            t2 = time.time()
            print(t2-t1)
            save_aligned_obj(
                original_obj_path=scan_path,
                new_vertices=restored_points,
                output_path=scan_path
            )
            t3 = time.time()
            print(t3-t2)

        

# 使用示例
if __name__ == "__main__":
    
    path1 = ''  ## obj path
    path2 = ''  ## target obj
    t0 = time.time()
    X = parse_obj_manually(path1)['vertices']
    Y = parse_obj_manually(path2)['vertices']
    print('check')
    t1 = time.time()
    print(t1-t0)
    X_norm, scan_centroid, scan_scale = normalize(X)
    Y_norm, target_centroid, target_scale = normalize(Y)

    # 运行ICP
    result = iterative_closest_point(X_norm, Y_norm, estimate_scale=True)

    print("\n对齐结果:")
    print(f"收敛状态: {result.converged}")
    print(f"最终RMSE: {result.rmse:.6f}")
    print("估计旋转矩阵:\n", result.RTs.R.round(2))
    print("估计平移向量:", result.RTs.T.round(2))
    print("估计缩放因子:", result.RTs.s.round(3))

    transformed_X = (X_norm @ result.RTs.R.T) * result.RTs.s + result.RTs.T 
    restored_points = denormalize(result.Xt, target_centroid, target_scale)
    
    t2 = time.time()
    print(t2-t1)
    save_aligned_obj(
        original_obj_path=path1,
        new_vertices=restored_points,
        output_path="aligned_result.obj"
    )
    t3 = time.time()
    print(t3-t2)

    '''
    path1 = '' ## obj folder
    path2 = '' ## target obj
   
    icp_folder(path1, path2)
    '''
