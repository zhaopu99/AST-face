import os
import numpy as np


def reconstruct_correct(original_obj, mapping_txt, output_obj):
    # 读取原始OBJ顶点
    orig_vertices = []
    with open(original_obj, 'r') as f:
        for line in f:
            if line.startswith('v '):
                orig_vertices.append(line.strip())
    
    # 解析描述文件
    kept_indices = []
    original_faces = []
    
    with open(mapping_txt, 'r') as f:
        # 第一行是顶点索引
        kept_indices = list(map(int, f.readline().split(':')[1].strip().split(',')))
        
        # 后续行是面信息（使用原始索引）
        for line in f:
            if line.startswith('f '):
                original_faces.append(line.strip())
    
    # 创建索引映射：原始索引 -> 新索引
    index_map = {}
    new_index = 1
    
    # 首先处理保留的顶点
    for orig_idx in kept_indices:
        if orig_idx > 0:  # 有效的原始索引
            index_map[orig_idx] = new_index
            new_index += 1
    
    # 写入裁剪后的OBJ
    with open(output_obj, 'w') as f:
        # 写入保留的顶点
        for orig_idx in kept_indices:
            if orig_idx > 0:
                f.write(orig_vertices[orig_idx-1] + "\n")
        
        # 写入面（将原始索引映射为新索引）
        for face in original_faces:
            parts = face.split()
            new_face = "f"
            
            # 处理每个顶点索引
            for part in parts[1:]:
                orig_idx = int(part)
                if orig_idx in index_map:
                    new_face += " " + str(index_map[orig_idx])
                else:
                    # 如果索引不在映射中，保留原始值
                    new_face += " " + str(orig_idx)
            
            f.write(new_face + "\n")

bfm_all_path = ''
out_obj_path = ''
reconstruct_correct(bfm_all_path, "data/cut_bfm.txt", out_obj_path)