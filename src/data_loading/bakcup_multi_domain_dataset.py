import numpy as np
import os
import tempfile
from typing import List, Dict, Union, Optional, Tuple, Any
import pickle


class MultiDomainItemDataset:
    """
    用于语义ID码本训练的数据集。
    加载多域物品的CLIP embedding及对应的域ID，供RQ-VAE或分层RQ-VAE训练。
    """
    def __init__(
        self,
        domain_configs: List[Dict[str, str]],
        expected_embed_dim: Optional[int] = None,
        return_item_id: bool = False,
    ):
        """
        参数:
            domain_configs: 每个域配置字典，必须包含:
                - 'name': 域名 (字符串)
                - 'embed_path': CLIP embedding .npy 文件路径
                - 'id_map_path': 物品ID映射 .npy 文件路径 (可选，若不需要ID可设为None)
                - (可选) 'user_seq_path' 此处不需要
            expected_embed_dim: 期望的embedding维度，用于验证
            return_item_id: 是否在__getitem__中返回全局物品ID
        """
        self.domain_names = []
        self.item_embeds_list = []          # 暂存各域的embedding，最后合并
        self.domain_ids_list = []            # 暂存各域的domain标签
        self.item_global_ids = []             # 所有全局物品ID列表
        self.return_item_id = return_item_id

        global_idx_start = 0
        for cfg in domain_configs:
            domain_name = cfg['name']
            embed_path = cfg['embed_path']
            id_map_path = cfg.get('id_map_path')  # 可能为None

            # 1. 加载embedding
            if not os.path.exists(embed_path):
                raise FileNotFoundError(f"Embedding file not found: {embed_path}")
            embeds = np.load(embed_path)                # shape (num_items, embed_dim)
            if expected_embed_dim is not None:
                assert embeds.shape[1] == expected_embed_dim, \
                    f"Domain {domain_name} embedding dim {embeds.shape[1]} != expected {expected_embed_dim}"
            num_items = embeds.shape[0]

            # 2. 加载ID映射 (如果提供)
            if id_map_path is not None and os.path.exists(id_map_path):
                raw_ids = np.load(id_map_path, allow_pickle=True)
                # 处理可能的字节串
                if raw_ids.dtype.type is np.bytes_:
                    raw_ids = np.array([id_.decode('utf-8') for id_ in raw_ids])
                elif raw_ids.dtype.type is np.str_:
                    raw_ids = raw_ids.astype(str)
                if raw_ids.ndim > 1:
                    raw_ids = raw_ids.reshape(-1)
                assert len(raw_ids) == num_items, \
                    f"Domain {domain_name}: #embeds {num_items} != #ids {len(raw_ids)}"
                global_ids = raw_ids
            else:
                # 如果没有ID映射，使用域内索引作为ID
                global_ids = [i for i in range(num_items)]

            # 3. 存储当前域的数据
            self.domain_names.append(domain_name)
            self.item_embeds_list.append(embeds)
            self.domain_ids_list.append(np.full(num_items, len(self.domain_names)-1, dtype=int))
            self.item_global_ids.extend(global_ids)

        # 合并所有embedding和domain标签
        self.item_embeds = np.vstack(self.item_embeds_list)               # (total_items, embed_dim)
        self.domain_ids = np.concatenate(self.domain_ids_list)            # (total_items,)

        # 验证一致性
        assert len(self.item_global_ids) == self.item_embeds.shape[0] == len(self.domain_ids)

    def __len__(self) -> int:
        return self.item_embeds.shape[0]

    def __getitem__(self, idx: int) -> Union[Tuple[np.ndarray, int], Tuple[np.ndarray, int, str]]:
        """
        返回:
            如果 return_item_id=False: (embed, domain_id)
            如果 return_item_id=True:  (embed, domain_id, global_item_id)
        """
        embed = self.item_embeds[idx]
        domain_id = self.domain_ids[idx]
        if self.return_item_id:
            return embed, domain_id, self.item_global_ids[idx]
        else:
            return embed, domain_id

    def get_domain_stats(self) -> Dict[str, int]:
        """返回各域物品数量统计"""
        stats = {}
        for i, name in enumerate(self.domain_names):
            stats[name] = np.sum(self.domain_ids == i)
        return stats


class MultiDomainSeqDataset:
    """
    用于下游生成式推荐训练（T5）的数据集。
    加载预处理好的语义ID序列数据，每个样本包含输入token序列和目标token序列。
    数据格式假设为pickle文件，包含列表，每个元素为字典：
        {
            'input_ids': List[int],      # 输入token序列
            'target_ids': List[int],      # 目标token序列
            'domain': str,                 # 域名（可选）
            'user_id': str                  # 用户ID（可选）
        }
    """
    def __init__(self, data_path: str):
        """
        参数:
            data_path: 预处理后的数据文件路径（pickle格式）
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        with open(data_path, 'rb') as f:
            self.samples = pickle.load(f)
        # 确保是列表
        assert isinstance(self.samples, list), "Data file should contain a list of samples."

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        返回一个样本的字典，至少包含 'input_ids' 和 'target_ids'。
        """
        return self.samples[idx]


def test_item_dataset_with_real_data():
    """使用真实数据测试 MultiDomainItemDataset"""
    # 定义真实数据路径（请根据实际环境调整）
    real_data_configs = [
        {
            'name': 'QB',
            'embed_path': '/mnt/data_2/data_oleg/recsys/ninerec/Downstream_datasets/QB/CLIP_emb/shard_00000_fused.npy',
            'id_map_path': '/mnt/data_2/data_oleg/recsys/ninerec/Downstream_datasets/QB/CLIP_emb/shard_00000_ids.npy',
        },
        {
            'name': 'TN',
            'embed_path': '/mnt/data_2/data_oleg/recsys/ninerec/Downstream_datasets/TN/CLIP_emb/shard_00000_fused.npy',
            'id_map_path': '/mnt/data_2/data_oleg/recsys/ninerec/Downstream_datasets/TN/CLIP_emb/shard_00000_ids.npy',  # 假设TN也有id映射文件
        }
    ]

    # 检查文件是否存在
    for cfg in real_data_configs:
        for key in ['embed_path', 'id_map_path']:
            if not os.path.exists(cfg[key]):
                print(f"警告：文件 {cfg[key]} 不存在，将跳过该域的加载。")
                # 这里可以选择跳过或退出，为演示我们仅打印警告

    # 创建数据集（期望embedding维度为768）
    expected_dim = 768
    try:
        item_dataset = MultiDomainItemDataset(real_data_configs, expected_embed_dim=expected_dim, return_item_id=True)
    except Exception as e:
        print(f"加载 MultiDomainItemDataset 失败：{e}")
        return

    # 验证基本属性
    print("\n=== MultiDomainItemDataset 测试结果 ===")
    print(f"总物品数: {len(item_dataset)}")
    print(f"物品embedding矩阵形状: {item_dataset.item_embeds.shape}")
    print(f"域ID数组形状: {item_dataset.domain_ids.shape}")
    print(f"全局物品ID数量: {len(item_dataset.item_global_ids)}")
    print(f"各域统计: {item_dataset.get_domain_stats()}")
    print(f"域名列表: {item_dataset.domain_names}")

    # 测试索引访问
    idx = 0
    embed, domain_id, gid = item_dataset[idx]
    print(f"\n第一个样本 (索引 {idx}):")
    print(f"  全局物品ID: {gid}")
    print(f"  域ID: {domain_id} ({item_dataset.domain_names[domain_id]})")
    print(f"  embedding 前5个元素: {embed[:5]}")

    # 检查embedding数据类型和维度
    assert embed.shape == (expected_dim,), f"embedding维度错误，应为{expected_dim}，实际为{embed.shape}"
    assert isinstance(domain_id, (int, np.integer)), "域ID应为整数"

    print("\n✅ MultiDomainItemDataset 测试通过。\n")

if __name__ == "__main__":
    test_item_dataset_with_real_data()