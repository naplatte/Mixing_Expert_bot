"""
节点嵌入工具模块

由于节点特征已经由官方预处理好（4个pt文件），本模块仅提供：
1. 加载和验证4个embedding文件的工具函数
2. GraphExpert会在初始化时自动加载这些文件

文件路径: ../../autodl-fs/node_embedding/
- node_cat_embeddings.pt  [num_nodes, 32]
- node_num_embeddings.pt  [num_nodes, 32]
- node_post_embeddings.pt [num_nodes, 32]
- node_des_embeddings.pt  [num_nodes, 32]
"""

import torch
import os


def load_node_embeddings(embedding_dir='../../autodl-fs/node_embedding'):
    """
    加载4个预处理好的节点embedding文件并拼接
    
    Args:
        embedding_dir: embedding文件所在目录
        
    Returns:
        node_features: [num_nodes, 128] - 拼接后的节点特征
        info: dict - 各embedding的形状信息
    """
    cat_emb = torch.load(os.path.join(embedding_dir, 'node_cat_embeddings.pt'), map_location='cpu')
    num_emb = torch.load(os.path.join(embedding_dir, 'node_num_embeddings.pt'), map_location='cpu')
    post_emb = torch.load(os.path.join(embedding_dir, 'node_post_embeddings.pt'), map_location='cpu')
    des_emb = torch.load(os.path.join(embedding_dir, 'node_des_embeddings.pt'), map_location='cpu')
    
    # 验证形状一致性
    num_nodes = cat_emb.shape[0]
    assert num_emb.shape[0] == num_nodes, f"num_emb节点数不匹配: {num_emb.shape[0]} vs {num_nodes}"
    assert post_emb.shape[0] == num_nodes, f"post_emb节点数不匹配: {post_emb.shape[0]} vs {num_nodes}"
    assert des_emb.shape[0] == num_nodes, f"des_emb节点数不匹配: {des_emb.shape[0]} vs {num_nodes}"
    
    # 拼接
    node_features = torch.cat([cat_emb, num_emb, post_emb, des_emb], dim=1).float()
    
    info = {
        'num_nodes': num_nodes,
        'cat_shape': cat_emb.shape,
        'num_shape': num_emb.shape,
        'post_shape': post_emb.shape,
        'des_shape': des_emb.shape,
        'total_shape': node_features.shape
    }
    
    return node_features, info


def verify_embeddings(embedding_dir='../../autodl-fs/node_embedding'):
    """
    验证4个embedding文件是否存在且格式正确
    
    Args:
        embedding_dir: embedding文件所在目录
        
    Returns:
        bool: 是否全部验证通过
    """
    files = [
        'node_cat_embeddings.pt',
        'node_num_embeddings.pt', 
        'node_post_embeddings.pt',
        'node_des_embeddings.pt'
    ]
    
    print(f"验证embedding文件目录: {embedding_dir}")
    print("=" * 60)
    
    all_valid = True
    shapes = []
    
    for f in files:
        path = os.path.join(embedding_dir, f)
        if not os.path.exists(path):
            print(f"  ✗ {f} - 文件不存在")
            all_valid = False
            continue
            
        try:
            emb = torch.load(path, map_location='cpu')
            shapes.append(emb.shape)
            print(f"  ✓ {f} - {emb.shape}")
        except Exception as e:
            print(f"  ✗ {f} - 加载失败: {e}")
            all_valid = False
    
    if all_valid and len(shapes) == 4:
        # 检查节点数是否一致
        num_nodes = shapes[0][0]
        if all(s[0] == num_nodes for s in shapes):
            print(f"\n节点数: {num_nodes}")
            print(f"总特征维度: {sum(s[1] for s in shapes)}")
        else:
            print(f"\n✗ 节点数不一致: {[s[0] for s in shapes]}")
            all_valid = False
    
    print("=" * 60)
    print(f"验证结果: {'✓ 通过' if all_valid else '✗ 失败'}")
    
    return all_valid


if __name__ == '__main__':
    # 运行验证
    verify_embeddings()
