"""
门控网络和图分析工具
用于分析异质图中的节点度数和孤立节点
"""
import torch
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.dataset import Twibot20


def analyze_node_degrees(edge_index, num_nodes=None, labeled_nodes_count=None):
    """
    分析图中节点的度数分布

    Args:
        edge_index: torch.Tensor, shape [2, num_edges], 边索引
        num_nodes: int, 节点总数（如果为None，则从edge_index推断）
        labeled_nodes_count: int, 有标签的节点数量（如果指定，则只统计这些节点）

    Returns:
        degree: torch.Tensor, 每个节点的度数（出度+入度）
        degree_stats: dict, 度为0-5的节点索引列表字典
        out_degree: torch.Tensor, 每个节点的出度
        in_degree: torch.Tensor, 每个节点的入度
    """
    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1

    # 初始化度数数组
    out_degree = torch.zeros(num_nodes, dtype=torch.long)
    in_degree = torch.zeros(num_nodes, dtype=torch.long)

    # 计算出度（从源节点出发）
    source_nodes = edge_index[0]
    for node in source_nodes:
        out_degree[node] += 1

    # 计算入度（到达目标节点）
    target_nodes = edge_index[1]
    for node in target_nodes:
        in_degree[node] += 1

    # 总度数 = 出度 + 入度
    degree = out_degree + in_degree

    # 如果指定了有标签节点的数量，只统计前labeled_nodes_count个节点
    if labeled_nodes_count is not None:
        degree_subset = degree[:labeled_nodes_count]
    else:
        degree_subset = degree

    # 统计度为0-5的节点
    degree_stats = {}
    for d in range(6):
        degree_stats[d] = (degree_subset == d).nonzero(as_tuple=True)[0].tolist()

    return degree, degree_stats, out_degree, in_degree


def print_degree_statistics(degree, degree_stats, labeled_only=False, labeled_count=None):
    """
    打印度数统计信息

    Args:
        degree: torch.Tensor, 每个节点的度数
        degree_stats: dict, 度为0-5的节点索引列表字典
        labeled_only: bool, 是否只统计有标签节点
        labeled_count: int, 有标签节点的数量
    """
    if labeled_only and labeled_count is not None:
        num_nodes = labeled_count
        degree_subset = degree[:labeled_count]
    else:
        num_nodes = len(degree)
        degree_subset = degree

    print("\n" + "="*60)
    if labeled_only:
        print("图节点度数分析报告（仅有标签节点）")
    else:
        print("图节点度数分析报告（全部节点）")
    print("="*60)
    print(f"总节点数: {num_nodes}")
    print()

    # 打India为0-5的节点统计
    print("度数分布统计：")
    for d in range(6):
        count = len(degree_stats[d])
        percentage = count/num_nodes*100
        print(f"  度为{d}的节点数: {count:6d} ({percentage:5.2f}%)")

    print(f"\n总体统计：")
    print(f"  平均度数: {degree_subset.float().mean().item():.2f}")
    print(f"  最大度数: {degree_subset.max().item()}")
    print(f"  最小度数: {degree_subset.min().item()}")
    print("="*60 + "\n")


def get_node_id_from_index(dataset, node_index):
    """
    通过节点索引获取对应的节点ID

    Args:
        dataset: Twibot20对象，包含节点数据
        node_index: int，节点索引

    Returns:
        node_id: int或None，对应的节点ID，如果索引不存在则返回None
    """
    # 使用df_data而不是df_data_labeled，因为df_data包含所有节点（有标签+支持集）
    if hasattr(dataset, 'df_data') and node_index < len(dataset.df_data):
        return dataset.df_data.iloc[node_index]['ID']
    return None


def get_node_neighbors(edge_index, node_index, dataset=None):
    """
    获取节点的所有邻居信息

    Args:
        edge_index: torch.Tensor, shape [2, num_edges]
        node_index: int, 节点索引
        dataset: Twibot20对象（可选，用于查询ID）

    Returns:
        neighbors_info: dict，包含出边和入边的邻居信息
    """
    # 找出以node_index为源节点的边（出边）
    out_edges_mask = edge_index[0] == node_index
    out_neighbors = edge_index[1, out_edges_mask].tolist()

    # 找出以node_index为目标节点的边（入边）
    in_edges_mask = edge_index[1] == node_index
    in_neighbors = edge_index[0, in_edges_mask].tolist()

    info = {
        'out_neighbors': out_neighbors,
        'in_neighbors': in_neighbors,
        'out_degree': len(out_neighbors),
        'in_degree': len(in_neighbors),
        'total_degree': len(out_neighbors) + len(in_neighbors)
    }

    if dataset is not None:
        info['out_neighbor_ids'] = [get_node_id_from_index(dataset, idx) for idx in out_neighbors]
        info['in_neighbor_ids'] = [get_node_id_from_index(dataset, idx) for idx in in_neighbors]

    return info


def find_isolated_nodes(device='cuda', labeled_only=False):
    """
    主函数：加载图数据并分析孤立节点

    Args:
        device: str, 计算设备 ('cuda' 或 'cpu')
        labeled_only: bool, 是否只统计有标签节点

    Returns:
        dataset: Twibot20对象，包含节点数据
        degree: torch.Tensor, 每个节点的度数
        degree_stats: dict, 度为0-5的节点索引列表字典
    """
    print("开始加载数据集...")

    # 初始化数据集（process=True表示加载原始数据以获取节点ID信息）
    dataset = Twibot20(root='./processed_data', device=device, process=True, save=True)

    # 构建图（如果已存在会直接加载）
    edge_index, edge_type = dataset.build_graph()

    print(f"图加载完成！")
    print(f"边数量: {edge_index.shape[1]}")
    print(f"边类型数量: {len(torch.unique(edge_type))}")

    # 获取节点总数（从数据集推断）
    if hasattr(dataset, 'df_data'):
        num_nodes = len(dataset.df_data)
    else:
        num_nodes = int(edge_index.max()) + 1

    # 有标签节点数量: train(8278) + val(2365) + test(1183) = 11826
    labeled_nodes_count = 8278 + 2365 + 1183

    print(f"推断的节点总数: {num_nodes}")
    print(f"有标签节点数量: {labeled_nodes_count}")

    if labeled_only:
        # 只统计有标签节点
        print("\n正在分析有标签节点的度数...")
        degree, degree_stats, out_degree, in_degree = analyze_node_degrees(
            edge_index, num_nodes, labeled_nodes_count=labeled_nodes_count
        )
        print_degree_statistics(degree, degree_stats,
                               labeled_only=True, labeled_count=labeled_nodes_count)
    else:
        # 统计全部节点
        print("\n正在分析全部节点的度数...")
        degree_all, degree_stats_all, out_degree_all, in_degree_all = analyze_node_degrees(
            edge_index, num_nodes, labeled_nodes_count=None
        )
        print_degree_statistics(degree_all, degree_stats_all,
                               labeled_only=False, labeled_count=None)

        # 同时统计有标签节点
        print("\n正在分析有标签节点的度数...")
        degree_labeled, degree_stats_labeled, out_degree_labeled, in_degree_labeled = analyze_node_degrees(
            edge_index, num_nodes, labeled_nodes_count=labeled_nodes_count
        )
        print_degree_statistics(degree_labeled, degree_stats_labeled,
                               labeled_only=True, labeled_count=labeled_nodes_count)

    # 打印度为0-5的节点索引和ID示例（每种度打印前3个）
    if labeled_only:
        target_stats = degree_stats
        target_degree = degree
        target_out_degree = out_degree
        target_in_degree = in_degree
    else:
        target_stats = degree_stats_labeled
        target_degree = degree_labeled
        target_out_degree = out_degree_labeled
        target_in_degree = in_degree_labeled

    print("\n" + "="*60)
    print("节点索引和ID示例（每种度前3个，含邻居详情）")
    print("="*60)

    for d in range(6):
        indices = target_stats[d][:3]  # 获取前3个节点索引
        if len(indices) > 0:
            print(f"\n度为{d}的节点:")
            for idx in indices:
                node_id = get_node_id_from_index(dataset, idx)
                out_deg = target_out_degree[idx].item()
                in_deg = target_in_degree[idx].item()
                total_deg = target_degree[idx].item()

                # 获取邻居信息
                neighbor_info = get_node_neighbors(edge_index, idx, dataset)

                print(f"  索引: {idx:6d} -> ID: {node_id}")
                print(f"    度数: 总={total_deg} (出度={out_deg}, 入度={in_deg})")
                print(f"    出边邻居索引: {neighbor_info['out_neighbors'][:5]}{'...' if len(neighbor_info['out_neighbors']) > 5 else ''}")
                print(f"    出边邻居ID: {neighbor_info['out_neighbor_ids'][:5]}{'...' if len(neighbor_info['out_neighbor_ids']) > 5 else ''}")
                print(f"    入边邻居索引: {neighbor_info['in_neighbors'][:5]}{'...' if len(neighbor_info['in_neighbors']) > 5 else ''}")
                print(f"    入边邻居ID: {neighbor_info['in_neighbor_ids'][:5]}{'...' if len(neighbor_info['in_neighbor_ids']) > 5 else ''}")
        else:
            print(f"\n度为{d}的节点: 无")

    print("="*60 + "\n")

    if labeled_only:
        return dataset, degree, degree_stats
    else:
        return dataset, degree_labeled, degree_stats_labeled


if __name__ == "__main__":
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 运行分析（labeled_only=False 会同时显示全部节点和有标签节点的统计）
    dataset, degree, degree_stats = find_isolated_nodes(device=device, labeled_only=False)

