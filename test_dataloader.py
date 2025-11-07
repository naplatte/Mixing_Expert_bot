"""
检测 dataset.py 中 dataloader 方法中的三个函数是否生效：
1. load_labels()
2. Des_preprocess() (当 process=True 时)
3. Des_embedding()
"""
import torch
import numpy as np
import os
from dataset import Twibot20

def test_load_labels():
    """测试 load_labels() 函数"""
    print("=" * 60)
    print("测试 1: load_labels() 函数")
    print("=" * 60)
    
    try:
        # 初始化数据集
        print("\n初始化数据集...")
        dataset = Twibot20(root='./processed_data', device='cpu', process=True, save=True)
        print("✓ 数据集初始化成功")
        
        # 检查必要的数据框是否存在
        if not hasattr(dataset, 'df_data_labeled'):
            print("✗ 错误: df_data_labeled 不存在")
            return False
        
        print(f"   - 数据集形状: {dataset.df_data_labeled.shape}")
        print(f"   - Label 列是否存在: {'label' in dataset.df_data_labeled.columns}")
        
        # 调用 load_labels
        print("\n调用 load_labels()...")
        labels = dataset.load_labels()
        
        # 验证结果
        print("\n验证返回结果:")
        if not isinstance(labels, torch.Tensor):
            print(f"✗ 错误: 返回的不是 torch.Tensor，而是 {type(labels)}")
            return False
        
        print(f"✓ 返回类型正确: torch.Tensor")
        print(f"   - 形状: {labels.shape}")
        print(f"   - 数据类型: {labels.dtype}")
        print(f"   - 设备: {labels.device}")
        print(f"   - 唯一值: {torch.unique(labels)}")
        print(f"   - 值范围: [{labels.min().item()}, {labels.max().item()}]")
        
        # 验证数量匹配
        expected_count = len(dataset.df_data_labeled)
        actual_count = len(labels)
        if actual_count == expected_count:
            print(f"✓ 标签数量 ({actual_count}) 与数据数量 ({expected_count}) 匹配")
        else:
            print(f"✗ 警告: 标签数量 ({actual_count}) 与数据数量 ({expected_count}) 不匹配")
            return False
        
        # 检查文件是否保存
        label_path = './processed_data/label.pt'
        if os.path.exists(label_path):
            print(f"✓ 标签文件已保存: {label_path}")
            # 验证可以重新加载
            loaded_labels = torch.load(label_path)
            if torch.equal(labels.cpu(), loaded_labels):
                print(f"✓ 保存的文件可以正确加载，内容一致")
            else:
                print(f"✗ 警告: 保存的文件内容与原始数据不一致")
                return False
        
        print("\n✓ load_labels() 测试通过！")
        return True
        
    except Exception as e:
        print(f"\n✗ load_labels() 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_des_preprocess():
    """测试 Des_preprocess() 函数"""
    print("\n" + "=" * 60)
    print("测试 2: Des_preprocess() 函数")
    print("=" * 60)
    
    try:
        # 初始化数据集
        print("\n初始化数据集...")
        dataset = Twibot20(root='./processed_data', device='cpu', process=True, save=True)
        print("✓ 数据集初始化成功")
        
        # 调用 Des_preprocess
        print("\n调用 Des_preprocess()...")
        description = dataset.Des_preprocess()
        
        # 验证结果
        print("\n验证返回结果:")
        if not isinstance(description, np.ndarray):
            print(f"✗ 错误: 返回的不是 numpy.ndarray，而是 {type(description)}")
            return False
        
        print(f"✓ 返回类型正确: numpy.ndarray")
        print(f"   - 形状: {description.shape}")
        print(f"   - 数据类型: {description.dtype}")
        
        # 验证数量匹配
        expected_count = len(dataset.df_data_labeled)
        actual_count = len(description)
        if actual_count == expected_count:
            print(f"✓ Description 数量 ({actual_count}) 与数据数量 ({expected_count}) 匹配")
        else:
            print(f"✗ 警告: Description 数量 ({actual_count}) 与数据数量 ({expected_count}) 不匹配")
            return False
        
        # 统计 None 的数量
        none_count = np.sum(description == 'None')
        print(f"   - 'None' 的数量: {none_count} / {len(description)} ({none_count/len(description)*100:.2f}%)")
        
        # 显示前几个样本
        print(f"\n前5个样本:")
        for i in range(min(5, len(description))):
            desc_str = str(description[i])[:80]  # 只显示前80个字符
            print(f"   [{i}]: {desc_str}...")
        
        # 检查文件是否保存
        desc_path = './processed_data/description.npy'
        if os.path.exists(desc_path):
            print(f"\n✓ Description 文件已保存: {desc_path}")
            # 验证可以重新加载
            loaded_desc = np.load(desc_path, allow_pickle=True)
            if np.array_equal(description, loaded_desc):
                print(f"✓ 保存的文件可以正确加载，内容一致")
            else:
                print(f"✗ 警告: 保存的文件内容与原始数据不一致")
                return False
        
        print("\n✓ Des_preprocess() 测试通过！")
        return True
        
    except Exception as e:
        print(f"\n✗ Des_preprocess() 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_des_embedding():
    """测试 Des_embedding() 函数"""
    print("\n" + "=" * 60)
    print("测试 3: Des_embedding() 函数")
    print("=" * 60)
    
    try:
        # 初始化数据集
        print("\n初始化数据集...")
        dataset = Twibot20(root='./processed_data', device='cpu', process=True, save=True)
        print("✓ 数据集初始化成功")
        
        # 确保 description.npy 存在
        desc_npy_path = './processed_data/description.npy'
        if not os.path.exists(desc_npy_path):
            print(f"\n警告: {desc_npy_path} 不存在，先运行 Des_preprocess()...")
            dataset.Des_preprocess()
            print("✓ Des_preprocess() 完成")
        
        # 调用 Des_embedding
        print("\n调用 Des_embedding()...")
        print("注意: 如果 des_tensor.pt 不存在，将需要下载模型并计算嵌入，可能需要较长时间")
        des_tensor = dataset.Des_embedding()
        
        # 验证结果
        print("\n验证返回结果:")
        if not isinstance(des_tensor, torch.Tensor):
            print(f"✗ 错误: 返回的不是 torch.Tensor，而是 {type(des_tensor)}")
            return False
        
        print(f"✓ 返回类型正确: torch.Tensor")
        print(f"   - 形状: {des_tensor.shape}")
        print(f"   - 数据类型: {des_tensor.dtype}")
        print(f"   - 设备: {des_tensor.device}")
        
        # 验证维度
        if len(des_tensor.shape) != 2:
            print(f"✗ 错误: Tensor 形状不符合预期，应该是2维，实际是 {len(des_tensor.shape)} 维")
            return False
        
        num_samples, embedding_dim = des_tensor.shape
        print(f"   - 样本数量: {num_samples}")
        print(f"   - 嵌入维度: {embedding_dim}")
        
        # 验证数量匹配
        expected_count = len(dataset.df_data_labeled)
        if num_samples == expected_count:
            print(f"✓ 嵌入数量 ({num_samples}) 与数据数量 ({expected_count}) 匹配")
        else:
            print(f"✗ 警告: 嵌入数量 ({num_samples}) 与数据数量 ({expected_count}) 不匹配")
            return False
        
        # 检查嵌入向量的统计信息
        print(f"\n嵌入向量统计信息:")
        print(f"   - 均值: {des_tensor.mean().item():.6f}")
        print(f"   - 标准差: {des_tensor.std().item():.6f}")
        print(f"   - 最小值: {des_tensor.min().item():.6f}")
        print(f"   - 最大值: {des_tensor.max().item():.6f}")
        
        # 检查前几个样本
        print(f"\n前3个样本的统计:")
        for i in range(min(3, des_tensor.shape[0])):
            sample = des_tensor[i]
            print(f"   样本[{i}]: 形状={sample.shape}, 均值={sample.mean().item():.4f}, 标准差={sample.std().item():.4f}")
        
        # 检查文件是否保存
        tensor_path = './processed_data/des_tensor.pt'
        if os.path.exists(tensor_path):
            print(f"\n✓ 嵌入文件已保存: {tensor_path}")
            # 验证可以重新加载
            loaded_tensor = torch.load(tensor_path)
            if torch.equal(des_tensor.cpu(), loaded_tensor):
                print(f"✓ 保存的文件可以正确加载，内容一致")
            else:
                # 允许小的数值误差
                if torch.allclose(des_tensor.cpu(), loaded_tensor, atol=1e-6):
                    print(f"✓ 保存的文件可以正确加载，内容基本一致（允许数值误差）")
                else:
                    print(f"✗ 警告: 保存的文件内容与原始数据不一致")
                    return False
        
        print("\n✓ Des_embedding() 测试通过！")
        return True
        
    except Exception as e:
        print(f"\n✗ Des_embedding() 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader():
    """测试完整的 dataloader() 方法"""
    print("\n" + "=" * 60)
    print("测试 4: dataloader() 完整流程")
    print("=" * 60)
    
    try:
        # 初始化数据集
        print("\n初始化数据集...")
        dataset = Twibot20(root='./processed_data', device='cpu', process=True, save=True)
        print("✓ 数据集初始化成功")
        
        # 调用 dataloader
        print("\n调用 dataloader()...")
        des_tensor, train_idx, val_idx, test_idx = dataset.dataloader()
        
        # 验证返回结果
        print("\n验证返回结果:")
        
        # 验证 des_tensor
        if not isinstance(des_tensor, torch.Tensor):
            print(f"✗ 错误: des_tensor 不是 torch.Tensor，而是 {type(des_tensor)}")
            return False
        print(f"✓ des_tensor 类型正确: {type(des_tensor)}")
        print(f"   - 形状: {des_tensor.shape}")
        
        # 验证索引
        print(f"✓ train_idx: 范围 {train_idx[0]} ~ {train_idx[-1]}, 数量: {len(train_idx)}")
        print(f"✓ val_idx: 范围 {val_idx[0]} ~ {val_idx[-1]}, 数量: {len(val_idx)}")
        print(f"✓ test_idx: 范围 {test_idx[0]} ~ {test_idx[-1]}, 数量: {len(test_idx)}")
        
        # 验证索引总数
        total_idx = len(train_idx) + len(val_idx) + len(test_idx)
        expected_count = len(dataset.df_data_labeled)
        if total_idx == expected_count:
            print(f"✓ 索引总数 ({total_idx}) 与数据数量 ({expected_count}) 匹配")
        else:
            print(f"✗ 警告: 索引总数 ({total_idx}) 与数据数量 ({expected_count}) 不匹配")
            return False
        
        # 验证嵌入数量
        if des_tensor.shape[0] == expected_count:
            print(f"✓ 嵌入数量 ({des_tensor.shape[0]}) 与数据数量 ({expected_count}) 匹配")
        else:
            print(f"✗ 警告: 嵌入数量 ({des_tensor.shape[0]}) 与数据数量 ({expected_count}) 不匹配")
            return False
        
        print("\n✓ dataloader() 完整流程测试通过！")
        return True
        
    except Exception as e:
        print(f"\n✗ dataloader() 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("=" * 60)
    print("检测 dataloader 方法中的三个函数是否生效")
    print("=" * 60)
    
    results = {}
    
    # 测试 1: load_labels()
    results['load_labels'] = test_load_labels()
    
    # 测试 2: Des_preprocess()
    results['des_preprocess'] = test_des_preprocess()
    
    # 测试 3: Des_embedding()
    results['des_embedding'] = test_des_embedding()
    
    # 测试 4: 完整的 dataloader()
    results['dataloader'] = test_dataloader()
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"1. load_labels():      {'✓ 通过' if results['load_labels'] else '✗ 失败'}")
    print(f"2. Des_preprocess():   {'✓ 通过' if results['des_preprocess'] else '✗ 失败'}")
    print(f"3. Des_embedding():    {'✓ 通过' if results['des_embedding'] else '✗ 失败'}")
    print(f"4. dataloader():       {'✓ 通过' if results['dataloader'] else '✗ 失败'}")
    
    if all(results.values()):
        print("\n✓ 所有测试通过！dataloader 中的三个函数都正常工作。")
    else:
        print("\n✗ 部分测试失败，请检查错误信息")
        failed = [name for name, result in results.items() if not result]
        print(f"   失败的测试: {', '.join(failed)}")


if __name__ == '__main__':
    main()

