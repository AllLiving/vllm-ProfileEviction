import torch
import torch.nn.functional as F
import math 
import time
import os

class KVCachePruner:
    def __init__(self, cache_prune_type, evict_method, block_size, cache_budget, initial_blocks, evict_metric="value_l2"):
        self.cache_prune_type = cache_prune_type
        self.evict_method = evict_method
        self.orig_block_size = block_size
        assert self.cache_prune_type == "budget"
        self.cache_budget = cache_budget
        # self.sub_evict_method = "value_l2"  # for global & local
        self.sub_evict_method = evict_metric
        self.initial_blocks = initial_blocks   # Number of compressed blocks to keep unpruned
        # print(f"\n[DEBUG] PagedEviction initialized with Metric: {self.sub_evict_method}\n")
        assert self.cache_budget >= 3 * self.orig_block_size, "Cache budget must be at least 3 times the original block size"
        
    def key_l2(self, key_block):
        return torch.norm(key_block, p=2, dim = -1)  # mean over the heads

    def value_l2(self, value_block):
        return torch.norm(value_block, p=2, dim = -1)
    
    def inverse_key_l2(self, key_block, value_block):
        return torch.div(1, self.key_l2(key_block) + 1e-8)
    
    def value_l2_plus_key_l2(self, key_block, value_block):
        return self.value_l2(key_block) + self.key_l2(value_block)

    def value_div_key(self, key_block, value_block):
        # 论文核心公式: ||V||_2 / ||K||_2
        return self.value_l2(value_block) / (self.key_l2(key_block) + 1e-8)
    # [Task 1: Helper Function - Hoyer Sparsity]
    def compute_hoyer_sparsity(self, token_tensor):
        """
        Compute Hoyer Sparsity (Hoyer, 2004) for tokens.
        Measure of sparsity: 0 (dense) to 1 (sparse).
        Formula: (sqrt(d) - L1/L2) / (sqrt(d) - 1)
        Input: [..., head_dim]
        Output: [...]
        """
        # token_tensor: [batch, seq, num_heads, head_dim] or [N, num_heads, head_dim]
        d = token_tensor.shape[-1]
        sqrt_d = math.sqrt(d)

        # Calculate L1 and L2 norms along the feature dimension
        l1_norm = torch.norm(token_tensor, p=1, dim=-1)
        l2_norm = torch.norm(token_tensor, p=2, dim=-1)

        # Avoid division by zero with epsilon
        sparsity = (sqrt_d - (l1_norm / (l2_norm + 1e-8))) / (sqrt_d - 1)
        return sparsity

    def _profile_sparsity_cascade(self, keys, values):
        k_l1_norm = torch.norm(keys, p=1, dim=-1)
        k_l2_norm = torch.norm(keys, p=2, dim=-1)
        v_l1_norm = torch.norm(values, p=1, dim=-1)
        v_l2_norm = torch.norm(values, p=2, dim=-1)
        devoto_score = 1.0 / (k_l2_norm + 1e-8)
        seq_len = keys.shape[0]
        if seq_len < self.cache_budget:
            scores = devoto_score
        else:
            gamma = 0.4
            alpha = 0.3
            rescue_ratio = 0.04
            k_sparsity = k_l2_norm / (k_l1_norm + 1e-8)
            v_sparsity = v_l2_norm / (v_l1_norm + 1e-8)
            hoper = alpha * k_sparsity + (1 - alpha) * v_sparsity
            hoper = torch.maximum(alpha * k_sparsity, v_sparsity)
            # hoper = (k_sparsity ** gamma) * (v_sparsity ** (1 - gamma))
            # 只有能量大于中位数的 Token 才有资格申请免死金牌
            quantile_ratio = 0.4
            devoto_mean = devoto_score.mean(dim=0, keepdim=True) * 0.85
            # k_l2_median = torch.median(k_l2_norm, dim=0, keepdim=True).values
            k_l2_median = torch.kthvalue(k_l2_norm, max(1, int(seq_len * quantile_ratio)), dim=0, keepdim=True).values
            valid_mask = devoto_score < devoto_mean
            valid_mask = k_l2_norm < k_l2_median
            indicator = hoper * valid_mask.float()
            # 级联设计：提取极性最强的前 15% Token 进行全局豁免
            rescue_budget_tokens = int(self.cache_budget * rescue_ratio)
            sparsity_kth = max(1, seq_len - rescue_budget_tokens)
            if sparsity_kth < seq_len:
                threshold = torch.kthvalue(indicator, sparsity_kth, dim=0, keepdim=True).values
                is_highly_sparse = (indicator >= threshold) & (indicator > 0)
                max_devoto = devoto_score.max(dim=0, keepdim=True).values
                scores = devoto_score + (is_highly_sparse.float() * max_devoto * 2.0)
            else:
                scores = devoto_score
        return scores

    def _profile_sparsity_miDualTrack(self, keys, values):
        """
        [范式 A: 统一微观双轨制]
        完全对齐 Block 级的 DualTrack 逻辑，下放到 Token 级别。
        """
        # 1. 提取微观 K 极性 (Hoyer变体: L2/L1) -> [S, H]
        k_l2 = torch.linalg.vector_norm(keys, ord=2, dim=-1)
        k_l1 = torch.linalg.vector_norm(keys, ord=1, dim=-1)
        k_polarity = k_l2 / (k_l1 + 1e-8)  # [S, H]
        # 2. 轨道一：Token特征容量 (Token Feature Volume)
        # 绝对线性保留 V，使用 L1 范数聚合 Head 维度，评估该 Token 的纯粹信息量
        v_abs = torch.abs(values) # [S, H, D]
        # 注意：此处先在 D 维度求 L1 得到每个 Head 的信息量，再在 H 维度求和/平均
        v_volume_per_head = torch.linalg.vector_norm(v_abs, ord=1, dim=-1) # [S, H]
        # 3. 轨道二：Token 检索概率 (Token Addressability)
        # 引入 Kappa 放大极性差异 (可与 Block 阶段保持一致)
        kappa = float(os.environ.get("PROFILE_KAPPA_TOKEN", "1.0"))
        k_weight = k_polarity ** kappa # [S, H]
        # 4. 微观绑定与几何融合
        gamma = float(os.environ.get("PROFILE_GAMMA_TOKEN", "0.2"))
        token_scores_per_head = (v_volume_per_head ** gamma) * (k_weight ** (1.0 - gamma)) # [S, H]
        # 5. 跨 Head 聚合 (至关重要，防止张量错位)
        # 对齐到 seq_len 维度，保证一个 Token 只要在某几个 Head 上极强，就能活下来
        final_token_scores = token_scores_per_head.mean(dim=1) # [S]
        final_token_scores = token_scores_per_head
        return final_token_scores

    def _profile_sparsity_DevotoKGate(self, keys, values):
        """
        [范式 B: 动态动能门控]
        融合了 Devoto 定律与极性门控，不使用复杂的幂级数，直接依靠微观物理量。
        """
        # 1. 提取 K 的绝对潜能 (Devoto定律：1/L2)
        k_l2 = torch.linalg.vector_norm(keys, ord=2, dim=-1)
        devoto_potential = 1.0 / (k_l2 + 1e-8) # [S, H]
        # 2. 提取 K 的相对极性 (检索锋利度)
        k_l1 = torch.linalg.vector_norm(keys, ord=1, dim=-1)
        k_polarity = k_l2 / (k_l1 + 1e-8) # [S, H]
        # 3. 综合检索动能 (Kinetic Energy of Retrieval)
        # 既要绝对值小(容易被选中)，又要分布锋利
        retrieval_kinetic = devoto_potential * k_polarity # [S, H]
        # 4. 动能加权 V 容量
        v_abs = torch.abs(values)
        v_volume = torch.linalg.vector_norm(v_abs, ord=1, dim=-1) # [S, H]
        # 线性绑定
        token_energy = v_volume * retrieval_kinetic # [S, H]
        # 5. 跨 Head 聚合，保障序列对齐
        # 采用 sum 还是 mean 都可以，因为是对特定长度排序
        final_scores = token_energy.mean(dim=1) # [S]
        final_scores = token_energy
        return final_scores

    def _profile_sparsity_smoothWindow(self, keys, values):
        """
        [范式 2: 局部语境平滑]
        要素：引入 n-gram 感受野，保护连续语义片段，对抗 RoPE 破坏。
        """
        k_l1 = torch.linalg.vector_norm(keys, ord=1, dim=-1)
        retrieval_kinetic = 1.0 / (k_l1 + 1e-8) # [S, H]
        v_volume = torch.linalg.vector_norm(values, ord=1, dim=-1) # [S, H]
        raw_energy = v_volume * retrieval_kinetic # [S, H]
        # 先在 Head 维度求均值或和 (降维到 [S])
        seq_energy = raw_energy.sum(dim=1) # [S]
        # 引入平滑窗口超参 (建议 3, 5, 7，代表保护 3-gram 到 7-gram)
        # window_size = int(os.environ.get("PROFILE_WINDOW_SIZE", "3"))
        window_size = int(os.environ.get("PROFILE_GAMMA_TOKEN", "3"))
        if window_size > 1:
            # 增加 padding 以保持长度不变
            pad_size = window_size // 2
            # 1. 先升维到 [1, 1, S]，满足 pad 对于 mode='replicate' 的最小 3D 维度要求
            seq_energy_3d = seq_energy.unsqueeze(0).unsqueeze(0)
            # 2. 对 3D 张量进行边缘复制 padding
            padded_energy = torch.nn.functional.pad(seq_energy_3d, (pad_size, pad_size), mode='replicate')
            # 3. 滑动平均滤波，随后降维回 1D [S]
            smoothed_energy = torch.nn.functional.avg_pool1d(
                padded_energy,
                kernel_size=window_size,
                stride=1
            ).squeeze(0).squeeze(0)
        else:
            smoothed_energy = seq_energy
        return smoothed_energy.unsqueeze(-1) # [S, 1] 适配外部代码

    def _profile_sparsity_HeadSpec(self, keys, values):
        """
        [范式 1: Head 特化感知]
        要素：不再求和，只要一个 Token 在任意一个 Head 上是不可或缺的，就赋予高分。
        """
        # 1. 微观双轨 (使用最优的 DevotoKGate 即 1/L1)
        k_l1 = torch.linalg.vector_norm(keys, ord=1, dim=-1)
        retrieval_kinetic = 1.0 / (k_l1 + 1e-8) # [S, H]
        v_volume = torch.linalg.vector_norm(values, ord=1, dim=-1) # [S, H]
        # 2. Token 能量
        token_energy = v_volume * retrieval_kinetic # [S, H]
        # 3. [关键改动] 提取特化极值！
        # 代表这个 Token 在它最擅长的那个 Head 里有多重要
        final_scores = torch.max(token_energy, dim=1).values # [S]
        # 注意：为了适配你的 prune_prompt_inplace (那里有 sum(dim=1)), 
        # 我们返回 [S, 1]，这样外部 sum 后依然是 max 的结果
        return final_scores.unsqueeze(-1)

    def _profile_sparsity_miDualTrackHyper(self, keys, values):
        """
        [范式 A: 统一微观双轨制]
        完全对齐 Block 级的 DualTrack 逻辑，下放到 Token 级别。
        """
        # 1. 提取微观 K 极性 (Hoyer变体: L2/L1) -> [S, H]
        k_l2 = torch.linalg.vector_norm(keys, ord=2, dim=-1)
        k_l1 = torch.linalg.vector_norm(keys, ord=1, dim=-1)
        k_polarity = k_l2 / (k_l1 + 1e-8)  # [S, H]
        # 2. 轨道一：Token特征容量 (Token Feature Volume)
        # 绝对线性保留 V，使用 L1 范数聚合 Head 维度，评估该 Token 的纯粹信息量
        v_abs = torch.abs(values) # [S, H, D]
        # 注意：此处先在 D 维度求 L1 得到每个 Head 的信息量，再在 H 维度求和/平均
        v_volume_per_head = torch.linalg.vector_norm(v_abs, ord=1, dim=-1) # [S, H]
        # 3. 轨道二：Token 检索概率 (Token Addressability)
        # 引入 Kappa 放大极性差异 (可与 Block 阶段保持一致)
        # kappa = float(os.environ.get("PROFILE_KAPPA_TOKEN", "1.0"))
        kappa = 1.0
        k_weight = k_polarity ** kappa # [S, H]
        # 4. 微观绑定与几何融合
        gamma = float(os.environ.get("PROFILE_GAMMA_TOKEN", "0.2"))
        alpha = 0.5 + gamma
        beta = 1.5 - gamma
        token_scores_per_head = (v_volume_per_head ** alpha) * (k_weight ** beta) # [S, H]
        # 5. 跨 Head 聚合 (至关重要，防止张量错位)
        # 对齐到 seq_len 维度，保证一个 Token 只要在某几个 Head 上极强，就能活下来
        final_token_scores = token_scores_per_head.mean(dim=1) # [S]
        final_token_scores = token_scores_per_head
        return final_token_scores

    def _profile_sparsity_hyperDevotoKGate(self, keys, values):
        """
        [范式 3: 纯粹锐度隔离]
        要素：摒弃幅值影响，纯粹依靠分布的锋利度。加入 Kappa 超参测试隔离后的放缩。
        """
        # 1. 提取 K 的绝对潜能 (Devoto定律：1/L2)
        k_l2 = torch.linalg.vector_norm(keys, ord=2, dim=-1)
        devoto_potential = 1.0 / (k_l2 + 1e-8) # [S, H]
        # 2. 提取 K 的相对极性 (检索锋利度)
        k_l1 = torch.linalg.vector_norm(keys, ord=1, dim=-1)
        k_polarity = k_l2 / (k_l1 + 1e-8) # [S, H]
        # 3. 综合检索动能 (Kinetic Energy of Retrieval)
        # 既要绝对值小(容易被选中)，又要分布锋利
        gamma = float(os.environ.get("PROFILE_GAMMA_TOKEN", "0.5"))
        alpha = 1.5 - gamma
        beta = 0.5 + gamma
        # retrieval_kinetic = (devoto_potential ** gamma ) * (k_polarity ** (1-gamma)) # [S, H]
        retrieval_kinetic = (devoto_potential ** beta ) * (k_polarity ** alpha) # [S, H]
        # 4. 动能加权 V 容量
        v_abs = torch.abs(values)
        v_volume = torch.linalg.vector_norm(v_abs, ord=1, dim=-1) # [S, H]
        # 线性绑定
        token_energy = v_volume * retrieval_kinetic # [S, H]
        # 5. 跨 Head 聚合，保障序列对齐
        # 采用 sum 还是 mean 都可以，因为是对特定长度排序
        final_scores = token_energy
        return final_scores

    def _profile_sparsity_vPolarDevoto(self, keys, values):
        """
        [范式 A: V极性 + 跨头 L2 聚合]
        利用 L2 范数完美解决 Head 稀释与孤岛问题，同时引入 V 极性筛选高信息密度 Token。
        """
        # 1. 计算微观标量 (Token & Head 级别)
        v_l2 = torch.linalg.vector_norm(values, ord=2, dim=-1)
        v_l1 = torch.linalg.vector_norm(values, ord=1, dim=-1)
        v_polarity = v_l2 / (v_l1 + 1e-8)  # [S, H] -> 区分废话和硬信息
        k_l2 = torch.linalg.vector_norm(keys, ord=2, dim=-1)
        devoto_potential = 1.0 / (k_l2 + 1e-8) # [S, H]
        # 2. 动能绑定 (此处保留最稳定的 Devoto 作为 K 的唯一代理)
        token_energy = (v_l2 * v_polarity) * devoto_potential # [S, H]
        # 3. 跨 Head 维度的 L2 聚合 (核心改动！！！)
        # 既奖励在少数 Head 上极高(特化)的 Token，也惩罚在所有 Head 上平庸的 Token
        # 输出 [S]，外部即使调用 sum(dim=1) (对标量无效) 也不会改变其分布
        final_scores = torch.linalg.vector_norm(token_energy, ord=2, dim=1).unsqueeze(-1) # [S, 1]
        return final_scores

    def _profile_sparsity_basicV2divK1(self, keys, values):
        """
        [Token 范式 1: 统一物理能量场]
        结合 V 的 L2 幅值(自带隐藏极性) 与 K 的 L1 检索难度(融合 Devoto 与锋利度)。
        数学极其干净，无任何硬编码超参。
        """
        # 1. V 的特征有效能量 (L2 范数自带对锋利分布的隐性奖励)
        v_energy = torch.linalg.vector_norm(values, ord=2, dim=-1) # [S, H]
        # 2. K 的终极检索锋利度 (1 / L1)
        # 完美融合绝对幅值限制与相对极性惩罚
        k_l1 = torch.linalg.vector_norm(keys, ord=1, dim=-1)
        retrieval_sharpness = 1.0 / (k_l1 + 1e-8) # [S, H]
        # 3. 线性绑定：这代表了该 Token 在此 Head 上的“有效期望信息量”
        token_scores = v_energy * retrieval_sharpness # [S, H]
        return torch.linalg.vector_norm(token_scores, ord=2, dim=1).unsqueeze(-1) # [S, 1]

    def _profile_sparsity_v1_0410(self, keys, values):
        """
        [终极范式: Unified Polarity Ratio (UPR)]
        核心理论：
        1. V 的容量 * V 的极性 = L2(V)
        2. K 的潜能 * K 的极性 = 1 / L1(K)
        3. 跨 Head 聚合采用 L2 范数，完美平衡 "Sum的特征稀释" 与 "Max的语义孤岛"。
        """
        # ==========================================
        # 步骤 1：Token 级微观物理量坍缩计算
        # ==========================================
        # 1. 提取 V 的综合载荷量 (天然包含 V容量 与 V极性) -> [S, H]
        v_payload = torch.linalg.vector_norm(values, ord=2, dim=-1)
        # 2. 提取 K 的综合检索力 (天然包含 Devoto 与 K极性) -> [S, H]
        # 注意：此处使用 L1 范数
        k_l1 = torch.linalg.vector_norm(keys, ord=1, dim=-1)
        k_retrieval = 1.0 / (k_l1 + 1e-8)
        # ==========================================
        # 步骤 2：微观特征绑定与宏观调控
        # ==========================================
        # 允许通过 gamma 宏观调节 V 和 K 的倾斜度
        # 默认 gamma=0.5 时，即完全等价于 (v_payload / k_l1)^0.5
        gamma = float(os.environ.get("PROFILE_GAMMA_TOKEN", "0.5"))
        # Token 在各 Head 上的独立能量 -> [S, H]
        alpha = 0.5 + gamma
        beta = 1.5 - gamma
        token_energy_per_head = (v_payload ** gamma) * (k_retrieval ** (1.0 - gamma))
        # ==========================================
        # 步骤 3：跨 Head 防坍缩聚合 (L2 聚合)
        # ==========================================
        # 替代 sum(dim=1) 和 max(dim=1)
        # L2 范数天然奖励锋利度，同时兼顾广度，完美克服 Qasper 的孤岛效应
        final_scores = torch.linalg.vector_norm(token_energy_per_head, ord=2, dim=1) # [S]
        # 适配原代码的输入格式要求，升维 [S, 1] 使得外部如果强制 sum 不受影响
        return final_scores.unsqueeze(-1)

    def _profile_sparsity_v2_0410(self, keys, values):
        """
        [范式 C: 全息正交投影]
        将 [Num_Heads, Head_Dim] 展平为单一的超高维向量进行评估，从根本上消灭 Head 维度的聚合难题。
        """
        S, H, D = keys.shape
        # 将每个 Token 视为一个极其高维 (H * D) 的单个向量
        keys_flat = keys.view(S, H * D)
        values_flat = values.view(S, H * D)
        # 1. 宏观 V 幅值与 V 极性 (基于全特征空间)
        v_l1 = torch.linalg.vector_norm(values_flat, ord=1, dim=-1) # [S]
        v_l2 = torch.linalg.vector_norm(values_flat, ord=2, dim=-1) # [S]
        v_polarity_macro = v_l2 / (v_l1 + 1e-8) # [S]
        # 2. 宏观 K 潜能
        k_l2 = torch.linalg.vector_norm(keys_flat, ord=2, dim=-1) # [S]
        devoto_macro = 1.0 / (k_l2 + 1e-8) # [S]
        # 3. 正交组合：V的幅值(L1) × V的纯度(极性) × K的引力(Devoto)
        # 这个组合数学上极其干净，没有任何抵消现象
        final_scores = v_l1 * v_polarity_macro * devoto_macro # [S]
        # 适配外部的 shape 要求
        return final_scores.unsqueeze(-1) # [S, 1]

    def _profile_sparsity_v3_0410(self, keys, values):
        """
        [范式 B: K 极性非线性隔离]
        防止 1/L2 和 L2/L1 抵消，将 K 极性通过高次幂彻底独立为“免死金牌”权重。
        """
        # 1. 基础载荷与潜能
        v_l2 = torch.linalg.vector_norm(values, ord=2, dim=-1) # [S, H]
        k_l2 = torch.linalg.vector_norm(keys, ord=2, dim=-1)
        base_energy = v_l2 / (k_l2 + 1e-8) # 经典的 paged_ratio [S, H]
        # 2. 提取独立的 K 极性并进行指数级放大
        k_l1 = torch.linalg.vector_norm(keys, ord=1, dim=-1)
        k_polarity = k_l2 / (k_l1 + 1e-8) # [S, H]
        # 使用 kappa = 2.0 强行拉开钩子锐度的差距 (不使用 gamma，防止数学污染)
        kappa = float(os.environ.get("PROFILE_KAPPA_TOKEN", "2.0"))
        sharpness_multiplier = k_polarity ** kappa # [S, H]
        # 3. 施加锐度奖励
        gated_energy = base_energy * sharpness_multiplier # [S, H]
        # 4. 跨 Head L2 聚合
        final_scores = torch.linalg.vector_norm(gated_energy, ord=2, dim=1).unsqueeze(-1) # [S, 1]
        return final_scores

    def _profile_sparsity_vkLoadHyperKappa(self, keys, values):
        """
        [范式 2: 双重 Hoyer 有界动能]
        无需调节复杂的指数超参，利用 Hoyer 稀疏度的 [0,1] 有界特性进行天然的品质门控。
        """
        d = keys.shape[-1]
        sqrt_d = math.sqrt(d)
        # 1. 基础特征提取
        k_l1 = torch.linalg.vector_norm(keys, ord=1, dim=-1)
        k_l2 = torch.linalg.vector_norm(keys, ord=2, dim=-1)
        v_l1 = torch.linalg.vector_norm(values, ord=1, dim=-1)
        v_l2 = torch.linalg.vector_norm(values, ord=2, dim=-1)
        # 2. 计算 Hoyer 稀疏度 (0代表完全均匀噪音，1代表完美独热尖峰)
        k_hoyer = (sqrt_d - (k_l1 / (k_l2 + 1e-8))) / (sqrt_d - 1.0) # [S, H]
        v_hoyer = (sqrt_d - (v_l1 / (v_l2 + 1e-8))) / (sqrt_d - 1.0) # [S, H]
        # 3. 提取 Devoto 潜能与 V 能量
        devoto_potential = 1.0 / (k_l2 + 1e-8)
        v_energy = v_l2
        # 4. 品质门控绑定
        # K 的综合动能 = 基础潜能 * Hoyer惩罚 (纯噪音K直接打0)
        kappa = float(os.environ.get("PROFILE_KAPPA_TOKEN", "1.0"))
        theta = 1.42
        k_kinetic = devoto_potential * (k_hoyer ** theta)
        # V 的综合载荷 = 基础能量 * Hoyer惩罚 (纯噪音V直接打0)
        v_payload = v_energy * v_hoyer
        # 5. 组合与多头共识
        token_scores_per_head = v_payload * (k_kinetic ** kappa) # [S, H]
        final_scores = token_scores_per_head.sum(dim=1).unsqueeze(-1) # [S]
        return final_scores

    def _profile_sparsity_pagedPolarBias(self, keys, values):
        """
        [范式 1: 正交极性修饰比]
        彻底解决代数约分问题。保留 V/K_L2 能量底座，将极性作为纯粹的品质缩放因子。
        """
        # 1. 基础能量与潜能 (L2 范数)
        k_l2 = torch.linalg.vector_norm(keys, ord=2, dim=-1)
        v_l2 = torch.linalg.vector_norm(values, ord=2, dim=-1)
        # 2. 绝对能量底座 (Paged Ratio)
        base_ratio = v_l2 / (k_l2 + 1e-8)  # [S, H]
        # 3. 提取极性 (L2/L1) -> 无量纲品质因子
        k_l1 = torch.linalg.vector_norm(keys, ord=1, dim=-1)
        v_l1 = torch.linalg.vector_norm(values, ord=1, dim=-1)
        k_polarity = k_l2 / (k_l1 + 1e-8)  # [S, H]
        v_polarity = v_l2 / (v_l1 + 1e-8)  # [S, H]
        # 4. 引入控制超参 (通常建议 gamma 设在 0.5 到 1.0 之间)
        gamma_k = float(os.environ.get("PROFILE_GAMMA_TOKEN", "1.0"))
        gamma_v = float(os.environ.get("PROFILE_KAPPA_TOKEN", "1.0"))
        # 5. 正交融合：底座 * (K极性增益) * (V极性增益)
        # 物理意义：能量大的 V 被保留；锐利的 V 被额外奖励；容易被命中的锐利 K 被双重奖励。
        token_scores_per_head = base_ratio * (k_polarity ** gamma_k) * (v_polarity ** gamma_v) # [S, H]
        # 6. 多头共识聚合 (防止孤岛效应)
        final_scores = token_scores_per_head.sum(dim=1) # [S]
        return final_scores.unsqueeze(-1)

    def _profile_sparsity_multi4HyperHoyer(self, keys, values):
        """
        [范式 4: 自适应平滑动能稀疏 (Adaptive L2-Smoothed Kinetic Hoyer)]
        要素：
        1. 保留 V_energy 和 K_devoto 的线性绝对量纲。
        2. 将 Hoyer Sparsity 作为无量纲门控。
        3. 根据上下文压缩率动态调节 K 和 V 的门控幂级数 (kappa, gamma)。
        """
        seq_len = keys.shape[0]
        budget = self.cache_budget
        # 1. 计算压缩率 (Compression Ratio)
        # c_ratio 越小，说明上下文越长，压缩压力越大
        c_ratio = budget / float(seq_len) if seq_len > budget else 1.0
        c_ratio /= 2
        # 2. 动态生成门控指数 (Dynamic Exponents)
        # 当压力极大 (c_ratio -> 0) 时：我们需要保住 Payload (gamma -> 0.5, 宽容 V)，
        # 同时不能被孤立的锋利 K 带偏，削弱寻址惩罚 (kappa -> 0.5)。
        # 当压力小 (c_ratio -> 1) 时：我们需要精确的寻址路标，增强 K 惩罚 (kappa -> 2.0)，
        # 并严格要求 V 的信息密度 (gamma -> 1.5)。
        # 这里的基准值可以根据实际情况微调，以下提供了线性映射逻辑：
        base_kappa_min = 0.1  # 极端压缩下对 K 的最低惩罚
        base_kappa_max = 2.0  # 宽裕预算下对 K 的最高惩罚
        kappa = base_kappa_min + (base_kappa_max - base_kappa_min) * c_ratio
        base_gamma_min = 0.9  # 极端压缩下对 V 的最低惩罚 (尽量保留实体)
        base_gamma_max = 2.0  # 宽裕预算下对 V 的最高惩罚
        gamma = base_gamma_min + (base_gamma_max - base_gamma_min) * (1 - c_ratio)
        # 3. 提取线性基础物理量 (Linear Physics)
        v_energy = torch.linalg.vector_norm(values, ord=2, dim=-1) # [S, H]
        k_energy = torch.linalg.vector_norm(keys, ord=2, dim=-1)   # [S, H]
        devoto_potential = 1.0 / (k_energy + 1e-8)                 # [S, H]
        # 4. 提取并应用动态对比度拉伸 (Contrast Stretching on [0,1] gates)
        v_hoyer_gate = self.compute_hoyer_sparsity(values) ** gamma  # [S, H]
        k_hoyer_gate = self.compute_hoyer_sparsity(keys) ** kappa    # [S, H]
        # 5. Token 级微观物理绑定 (Micro-binding)
        # V载荷 = 绝对能量 * V特异性门控
        v_payload = v_energy * v_hoyer_gate
        # K动能 = 基础存活率 * K特异性门控
        k_kinetic = devoto_potential * k_hoyer_gate
        # 最终微观评分矩阵
        token_score_per_head = v_payload * k_kinetic  # [S, H]
        # 6. 平滑跨 Head 聚合 (L2 Aggregation)
        # 避免 sum 的稀释和 max 的断崖
        final_scores = torch.linalg.vector_norm(token_score_per_head, ord=1, dim=1) # [S]
        final_scores = token_score_per_head.sum(dim=1)
        return final_scores.unsqueeze(-1)

    def _profile_sparsity_multi4HyperKpart(self, keys, values):
        """
        [范式 4: 自适应平滑动能稀疏 (Adaptive L2-Smoothed Kinetic Hoyer)]
        要素：
        1. 序列内归一化剥离 K 量纲，绝对线性保留 V 量纲 (解决量纲危机)。
        2. 基于压缩率 c_ratio，使极度压缩下 V 门控严苛/K 被屏蔽；宽裕预算下 K 门控严苛/V 宽容。
        3. 仅需 1 个外部超参控制全局振幅。
        """
        seq_len = keys.shape[0]
        budget = self.cache_budget
        # 1. 计算压缩率 c_ratio 
        c_ratio = min(1.0, float(budget) / max(seq_len, 1))
        # 2. 唯一的外部超参控制 (推荐调优区间: 1.0 ~ 2.0，默认 1.5)
        # 这一个参数同时决定了 K 放大器的上限和 V 过滤器的上限
        base_param = float(os.environ.get("PROFILE_KAPPA_TOKEN", "1.5"))
        # 动态门控生成 (核心相变逻辑)
        # 极端压缩(c=0.04): kappa = 0.3 (K基本失效), gamma = 1.44 (V过滤极严) -> 完美治愈 Qasper U型
        # 轻度压缩(c=0.50): kappa = 1.06 (K主导过滤), gamma = 0.75 (V宽容) -> 完美保住 QMSum 高budget
        kappa = base_param * math.sqrt(c_ratio)
        gamma = base_param * (1.0 - c_ratio)
        # 3. 提取线性基础物理量
        # V 保留绝对物理量纲！
        v_energy = torch.linalg.vector_norm(values, ord=2, dim=-1) # [S, H]
        # [破局点] 对 K 进行 Head 级局部归一化，强行剥离量纲！
        k_l2 = torch.linalg.vector_norm(keys, ord=2, dim=-1) # [S, H]
        k_l2_mean = k_l2.mean(dim=0, keepdim=True) # [1, H]
        k_l2_normalized = k_l2 / (k_l2_mean + 1e-8)
        # 此时的 Devoto 变为纯粹的无量纲动能乘子，对其求 kappa 次方绝对合法
        devoto_potential = 1.0 / (k_l2_normalized + 1e-8) # [S, H]
        # 4. 提取无量纲品质门控 (Hoyer: 0~1)
        v_hoyer_gate = self.compute_hoyer_sparsity(values) # [S, H]
        k_hoyer_gate = self.compute_hoyer_sparsity(keys)   # [S, H]
        # 5. Token 级微观物理绑定 (Micro-binding)
        # V载荷 = 绝对能量[量纲V] * (Hoyer特异性[0~1] ^ gamma) -> 保持量纲V
        v_payload = v_energy * (v_hoyer_gate ** gamma)
        # K动能 = (无量纲Devoto * 无量纲Hoyer) ^ kappa -> 纯粹无量纲门控
        k_kinetic = (devoto_potential * k_hoyer_gate) ** kappa
        # 最终得分 = V载荷 * K动能 (完美等同于带极性修饰的真实 Payload)
        token_score_per_head = v_payload * k_kinetic # [S, H]
        # 6. 平滑跨 Head 聚合 (Sum Aggregation)
        # 避免 QMSum 因为 Max 孤岛效应或 L2 惩罚背景词而发生分数坍缩
        final_scores = token_score_per_head.sum(dim=1) # [S]
        return final_scores.unsqueeze(-1)

    def _profile_sparsity_v2_0413Night(self, keys, values):
        """
        [范式 4: 自适应平滑动能稀疏 (Adaptive L2-Smoothed Kinetic Hoyer)]
        要素：
        1. 序列内归一化剥离 K 量纲，绝对线性保留 V 量纲 (解决量纲危机)。
        2. 基于压缩率 c_ratio，使极度压缩下 V 门控严苛/K 被屏蔽；宽裕预算下 K 门控严苛/V 宽容。
        3. 仅需 1 个外部超参控制全局振幅。
        """
        seq_len = keys.shape[0]
        budget = self.cache_budget
        # 1. 计算压缩率 c_ratio 
        c_ratio = min(1.0, float(budget) / max(seq_len, 1))
        # 2. 唯一的外部超参控制 (推荐调优区间: 1.0 ~ 2.0，默认 1.5)
        # 这一个参数同时决定了 K 放大器的上限和 V 过滤器的上限
        base_param = float(os.environ.get("PROFILE_KAPPA_TOKEN", "1.5"))
        # 动态门控生成 (核心相变逻辑)
        # 极端压缩(c=0.04): kappa = 0.3 (K基本失效), gamma = 1.44 (V过滤极严) -> 完美治愈 Qasper U型
        # 轻度压缩(c=0.50): kappa = 1.06 (K主导过滤), gamma = 0.75 (V宽容) -> 完美保住 QMSum 高budget
        kappa = base_param * c_ratio
        gamma = 0.5 + base_param * (1.0 - c_ratio)
        # 3. 提取线性基础物理量
        # V 保留绝对物理量纲！
        v_energy = torch.linalg.vector_norm(values, ord=2, dim=-1) # [S, H]
        # [破局点] 对 K 进行 Head 级局部归一化，强行剥离量纲！
        k_l2 = torch.linalg.vector_norm(keys, ord=2, dim=-1) # [S, H]
        k_l2_mean = k_l2.mean(dim=0, keepdim=True) # [1, H]
        k_l2_normalized = k_l2 / (k_l2_mean + 1e-8)
        # 此时的 Devoto 变为纯粹的无量纲动能乘子，对其求 kappa 次方绝对合法
        devoto_potential = 1.0 / (k_l2_normalized + 1e-8) # [S, H]
        # 4. 提取无量纲品质门控 (Hoyer: 0~1)
        v_hoyer_gate = self.compute_hoyer_sparsity(values) # [S, H]
        k_hoyer_gate = self.compute_hoyer_sparsity(keys)   # [S, H]
        # 5. Token 级微观物理绑定 (Micro-binding)
        # V载荷 = 绝对能量[量纲V] * (Hoyer特异性[0~1] ^ gamma) -> 保持量纲V
        v_payload = v_energy * (v_hoyer_gate ** gamma)
        # K动能 = (无量纲Devoto * 无量纲Hoyer) ^ kappa -> 纯粹无量纲门控
        k_kinetic = (devoto_potential * k_hoyer_gate) ** kappa
        k_kinetic = k_hoyer_gate ** kappa
        # 最终得分 = V载荷 * K动能 (完美等同于带极性修饰的真实 Payload)
        token_score_per_head = v_payload * k_kinetic # [S, H]
        # 6. 平滑跨 Head 聚合 (Sum Aggregation)
        # 避免 QMSum 因为 Max 孤岛效应或 L2 惩罚背景词而发生分数坍缩
        final_scores = token_score_per_head.sum(dim=1) # [S]
        return final_scores.unsqueeze(-1)

    def _profile_sparsity_v4Hoyer0414(self, keys, values):
        """
        [范式 4: 自适应平滑动能稀疏 (Adaptive L2-Smoothed Kinetic Hoyer)]
        优化点：
        1. V_payload 严格保持 1.0 次幂，誓死捍卫信息载荷的绝对量纲与信噪比。
        2. K_kinetic 采用无量纲局部归一化。
        3. 放弃根号，使用 2*c_ratio 的线性映射。
        """
        seq_len = keys.shape[0]
        budget = self.cache_budget
        
        # 1. 计算压缩率 (上限为1.0)
        c_ratio = min(1.0, float(budget) / max(seq_len, 1))
        # 2. 唯一的外部超参控制 (决定 4096 处 K 的主导地位，建议调优区间: 1.0 ~ 1.5)
        base_kappa = float(os.environ.get("PROFILE_KAPPA_TOKEN", "1.2"))
        # 3. 动态门控生成 (线性相变)
        # 乘 2.0 的目的是让 c_ratio=0.5 (即 4096/8192) 时，kappa 恰好等于 base_kappa
        # 极端压缩(c=0.03): kappa 极小, K 乘子退化为 1, V 绝对主导
        # 轻度压缩(c=0.50): kappa 达到 base_kappa, 逼近/略超越 四项连乘
        kappa = base_kappa * (2.0 * c_ratio)
        kappa = 1.0 * (2.0 * c_ratio)
        # --- 轨道一：提取绝对线性 V 载荷 ---
        # 严禁在此处添加任何幂级数，保持真实物理量纲！
        v_energy = torch.linalg.vector_norm(values, ord=2, dim=-1) # [S, H]
        v_hoyer_gate = self.compute_hoyer_sparsity(values)         # [S, H], 无量纲
        v_payload = v_energy * v_hoyer_gate                        # [S, H], 保持 V 能量纲
        # --- 轨道二：提取无量纲 K 动能 ---
        k_l2 = torch.linalg.vector_norm(keys, ord=2, dim=-1)       # [S, H]
        k_l2_mean = k_l2.mean(dim=0, keepdim=True)                 # [1, H]
        # 局部归一化，彻底剥离 K 量纲
        k_l2_normalized = k_l2 / (k_l2_mean + 1e-8)                # 无量纲
        devoto_potential = 1.0 / (k_l2_normalized + 1e-8)          # 无量纲
        k_hoyer_gate = self.compute_hoyer_sparsity(keys)           # [S, H], 无量纲
        # K动能 = (无量纲Devoto * 无量纲Hoyer) ^ kappa
        # 纯粹对比度拉伸，物理上绝对安全
        k_kinetic = (devoto_potential * k_hoyer_gate) ** kappa     # [S, H]
        # --- 微观绑定与聚合 ---
        # 最终得分 = V线性载荷 * 动态K动能
        token_score_per_head = v_payload * k_kinetic               # [S, H]
        token_score_per_head = v_payload  # [S, H]
        # Sum 平滑跨 Head 聚合 (保护语境连贯性)
        final_scores = token_score_per_head.sum(dim=1)             # [S]
        return final_scores.unsqueeze(-1)

    def _profile_sparsity_multi4HyperKsparsity(self, keys, values):
        """
        [范式: L2-Smoothed Kinetic Hoyer (L2平滑动能稀疏范式)]
        要素：引入 V 极性，使用 Hoyer 解耦 K 形状与能量，并用 L2 范数替代 sum/max 解决 Head 稀释问题。
        """
        # 1. 提取 V 综合容量 (Payload Richness)
        # 能量 * 信息密度(极性)
        v_energy = torch.linalg.vector_norm(values, ord=2, dim=-1) # [S, H]
        v_sparsity = self.compute_hoyer_sparsity(values)           # [S, H]
        # V 的有效载荷 = 能量基数放大极性
        v_richness = v_energy * v_sparsity                         # [S, H]
        # 2. 提取 K 检索动能 (Retrieval Kinetic)
        # 易命中性 (Devoto) * 检索专属度 (Hoyer)
        k_energy = torch.linalg.vector_norm(keys, ord=2, dim=-1)   # [S, H]
        devoto_potential = 1.0 / (k_energy + 1e-8)                 # [S, H]
        k_sparsity = self.compute_hoyer_sparsity(keys)             # [S, H]
        # 引入温和的调节系数 (默认1.0)，避免极端放大
        kappa = float(os.environ.get("PROFILE_KAPPA_TOKEN", "1.0"))
        theta = 1.42
        # retrieval_kinetic = devoto_potential * (k_sparsity ** kappa) # [S, H]
        retrieval_kinetic = devoto_potential * (k_sparsity ** theta) # [S, H]
        # 3. Token 级微观物理绑定
        # 此时的 token_score 完美融合了 V的质量、K的命中率、K的专属度
        token_score_per_head = v_richness * (retrieval_kinetic ** kappa)     # [S, H]
        token_score_per_head = v_richness * devoto_potential * (k_sparsity ** kappa)     # [S, H]
        # 4. [破局点] 平滑聚合 (L2 Aggregation over Heads)
        # 完美解决 Sum 的稀释效应 和 Max 的孤岛效应
        # 注意：这里是在 dim=1 (num_heads) 上求范数！
        final_scores = torch.linalg.vector_norm(token_score_per_head, ord=2, dim=1) # [S]
        # return final_scores.unsqueeze(-1)
        return token_score_per_head

    def _profile_sparsity_v1(self, keys, values):
        """
        [正交消融测试专用函数]
        解耦 K 和 V 的所有底层特征，控制变量进行单项测试。
        """
        # --- 1. 计算所有基础底层组件 ---
        v_l2 = torch.linalg.vector_norm(values, ord=2, dim=-1)       # V 能量
        v_l1 = torch.linalg.vector_norm(values, ord=1, dim=-1)       # V 体积
        v_hoyer = self.compute_hoyer_sparsity(values)                # V 极性
        k_l2 = torch.linalg.vector_norm(keys, ord=2, dim=-1)         # K 能量
        k_l1 = torch.linalg.vector_norm(keys, ord=1, dim=-1)         # K 综合量纲
        k_hoyer = self.compute_hoyer_sparsity(keys)                  # K 极性
        devoto = 1.0 / (k_l2 + 1e-8)                                 # 标准 Devoto 
        # 序列内归一化的 Devoto (解决量纲问题)
        k_l2_mean = k_l2.mean(dim=0, keepdim=True)
        norm_devoto = 1.0 / ((k_l2 / (k_l2_mean + 1e-8)) + 1e-8)
        # --- 2. 获取测试目标指标 ---
        metric_name = os.environ.get("ABLATION_METRIC", "V_L2")
        if metric_name == "V_L2":
            token_scores_per_head = v_l2
        elif metric_name == "V_Hoyer":
            token_scores_per_head = v_hoyer
        elif metric_name == "V_L2_Hoyer":
            token_scores_per_head = v_l2 * v_hoyer
        elif metric_name == "V_L1":
            token_scores_per_head = v_l1
        elif metric_name == "V_L1_Hoyer":
            token_scores_per_head = v_l1 * v_hoyer
        elif metric_name == "1_K_L2":
            token_scores_per_head = devoto
        elif metric_name == "K_Hoyer":
            token_scores_per_head = k_hoyer
        elif metric_name == "1_K_L2_Hoyer":
            token_scores_per_head = devoto * k_hoyer
        elif metric_name == "1_NormK_Hoyer":
            token_scores_per_head = norm_devoto * k_hoyer
        elif metric_name == "1_K_L1":
            token_scores_per_head = 1.0 / (k_l1 + 1e-8)
        else:
            raise ValueError(f"Unknown ABLATION_METRIC: {metric_name}")
        # --- 3. 获取 Head 聚合方式 ---
        aggr_method = os.environ.get("ABLATION_AGGR", "sum")
        if aggr_method == "sum":
            # Sum 聚合 (L1 空间)：保护均势特征，防止语境断裂
            final_scores = token_scores_per_head.sum(dim=1)
        elif aggr_method == "l2":
            # L2 聚合：保护极端特征，允许偏科 (玻璃大炮) 存活
            final_scores = torch.linalg.vector_norm(token_scores_per_head, ord=2, dim=1)
        else:
            raise ValueError(f"Unknown ABLATION_AGGR: {aggr_method}")
        return final_scores.unsqueeze(-1)

    def _profile_sparsity_v2(self, keys, values):
        """
        [终极范式：解耦聚合与压缩率相变 (Decoupled Aggregation Phase Transition)]
        1. V 侧：使用 L1 体积，跨 Head L2 聚合 -> 提取并保护“偏科”的事实实体 (治愈 Qasper)。
        2. K 侧：使用归一化 Devoto * Hoyer，跨 Head Sum 聚合 -> 保护“均匀”的语境骨架 (稳住 QMSum)。
        3. 融合：根据压缩率动态调节两者的主导权。
        """
        seq_len = keys.shape[0]
        budget = self.cache_budget
        # 1. 计算压缩率 c_ratio
        c_ratio = min(1.0, float(budget) / max(seq_len, 1))
        # 2. 外部调控超参 (推荐搜索范围: 1.0 ~ 2.5)
        base_param = float(os.environ.get("PROFILE_KAPPA_TOKEN", "1.0"))
        # 动态相变逻辑
        # 极度压缩 (c -> 0): gamma 大 (强化实体保留)，kappa 小 (削弱路由惩罚)
        # 宽裕预算 (c -> 0.5+): kappa 大 (强化路由筛选)，gamma 趋近基础值
        gamma = 1.0 + (1.0 - c_ratio) * base_param * 0.5  # V 的权重
        kappa = c_ratio * base_param * 2.0                # K 的权重
        kappa = 0.5 + c_ratio * 1.0 * base_param
        # ==========================================
        # 轨道一：V 侧 (Payload & Entities)
        # ==========================================
        # 提取 dim=-1 上的 L1 体积 (最强的信息容量指标)
        v_l1_volume = torch.linalg.vector_norm(values, ord=1, dim=-1) # [S, H]
        # [破局点] 跨 Head L2 聚合：放大偏科极值，保护孤立实体词 (Qasper 救星)
        v_score = torch.linalg.vector_norm(v_l1_volume, ord=2, dim=1) # [S]
        # ==========================================
        # 轨道二：K 侧 (Routing & Context)
        # ==========================================
        k_l2 = torch.linalg.vector_norm(keys, ord=2, dim=-1) # [S, H]
        # 序列内 Head 级归一化，安全剥离 Devoto 的量纲
        k_l2_mean = k_l2.mean(dim=0, keepdim=True)
        norm_devoto = 1.0 / ((k_l2 / (k_l2_mean + 1e-8)) + 1e-8) # [S, H]
        # K 极性 (仅在低 budget 时有微弱提升，此处保留作为品质门控)
        k_hoyer = self.compute_hoyer_sparsity(keys) # [S, H]
        k_kinetic = norm_devoto * k_hoyer # [S, H]
        # [破局点] 跨 Head Sum 聚合：保护均匀激活的背景逻辑词 (QMSum 定海神针)
        k_score = k_kinetic.sum(dim=1) # [S]
        # ==========================================
        # 终极融合 (一维标量空间的几何加权)
        # ==========================================
        # 避免 0 的次幂导致 NaN
        final_scores = (v_score ** gamma) * ((k_score + 1e-6) ** kappa) # [S]
        return final_scores.unsqueeze(-1)

    def _profile_sparsity_v3(self, keys, values):
        pass

    def get_token_score(self, keys, values, evict_method):
        scores = 0 
        if evict_method in ['global', 'local']:
            if self.sub_evict_method == "value_l2":
                # use value_l2 norm for global and local eviction
                scores = self.value_l2(values)
            elif self.sub_evict_method == "key_l2":
                # use key_l2 norm for global and local eviction
                scores = self.key_l2(keys)
            elif self.sub_evict_method == "paged_ratio":
                scores = self.value_div_key(keys, values)
            elif self.sub_evict_method == "value_l2_plus_key_l2":
                # use value_l2 + key_l2 norm for global and local eviction
                scores = self.value_l2_plus_key_l2(keys, values)
            elif self.sub_evict_method == "inverse_key_l2":
                # use inverse key_l2 norm for global and local eviction
                scores = self.inverse_key_l2(keys, values)
            elif self.sub_evict_method == "profile_richness":
                profile_sparsity_version = os.environ.get("PROFILE_SPARSITY_VER", "1")
                if profile_sparsity_version == "1":
                    scores = self._profile_sparsity_v1(keys, values)
                elif profile_sparsity_version == "2":
                    scores = self._profile_sparsity_v2(keys, values)
                elif profile_sparsity_version == "3":
                    scores = self._profile_sparsity_v3(keys, values)
                elif profile_sparsity_version == "4":
                    scores = self._profile_sparsity_v4(keys, values)
                else:
                    raise ValueError(f"Unknown PROFILE_VER: {profile_sparsity_version}")
        elif evict_method == "inverse_key_l2":
            # use inverse key_l2 norm for inverse_key_l2 eviction
            scores = self.inverse_key_l2(keys, values)
        else:
            raise ValueError(f"Unknown eviction method: {evict_method}")
        return scores

    def _profile_maxpool_Vmaxpool(self, block_keys, block_values):
        """版本 v1: 待调参或修改逻辑的预留位置"""
        # Step 1: Meso-level Aggregation (Max Pooling)
        # 直接在 dim=0 (Token维度) 上取 Max。
        # 含义：在这个 Block 范围内，每个特征维度的最大激活值是多少？
        # Output Shape: [num_heads, head_dim]
        profile_vector = torch.max(torch.abs(block_values), dim=0).values
        # 放弃融合算子优化，经验数据是唯一真理 (Empirical data is king)
        # profile_vector = torch.amax(block_values.abs(), dim=0)

        # Step 2: Scoring (L2 Norm)
        # 计算 Profile 向量的模长，代表特征丰富度/能量
        # Output Shape: [num_heads]
        profile_norm = torch.norm(profile_vector, p=2, dim=-1)
        # profile_norm = torch.linalg.vector_norm(profile_vector, ord=2, dim=-1)

        # Step 3: Aggregate over heads (Mean)
        # Output Shape: Scalar
        return profile_norm.mean().item()

    def _profile_maxpool_v2(self, block_keys, block_values):
        """版本 v2: 待调参或修改逻辑的预留位置"""
        pass

    def _profile_maxpool_v3(self, block_keys, block_values):
        """
        [提纯版双轨制：峰值特征包络 × 检索连通密度]
        逻辑极其清晰，专为学术论文公式对应而设计。完美复现 Lab 3.2 优势，消灭 broadcasting 隐患。
        """
        # 1. 基础极性定义 (Token级)
        k_l2 = torch.linalg.vector_norm(block_keys, ord=2, dim=-1)
        k_l1 = torch.linalg.vector_norm(block_keys, ord=1, dim=-1)
        k_polarity = k_l2 / (k_l1 + 1e-8)  # [S, H]
        # 轨道一：Peak Feature Envelope (峰值特征包络)
        # 寻找被最锐利的 K 锚定的最大 V 载荷 (微观线性绑定 -> Max -> L1)
        v_abs = torch.abs(block_values)    # [S, H, D]
        weighted_v = v_abs * k_polarity.unsqueeze(-1)  # [S, H, D]
        gated_v_maxpool = torch.max(weighted_v, dim=0).values # [H, D]
        # L1 衡量包络的信息容量
        envelope_volume = torch.linalg.vector_norm(gated_v_maxpool, ord=1, dim=-1) # [H]
        # 轨道二：Addressability Density (检索连通密度)
        # 评估该 Block 作为一个整体的“防碎化”能力 (K极性的宏观平均)
        # [可选] 引入 Kappa 强化极性，根据 Lab 4 的结论，KAPPA=2.0 左右能放大差异
        kappa = float(os.environ.get("PROFILE_MAXPOOL_KAPPA", "1.1"))
        hook_density = (k_polarity ** kappa).mean(dim=0) # [H]
        # 宏观几何融合 (Geometric Fusion)
        gamma = float(os.environ.get("PROFILE_MAXPOOL_GAMMA", "0.1"))
        # 两个指标均在 [H] 维度，几何加权后求均值，物理意义绝对严谨
        final_score = ((envelope_volume ** gamma) * (hook_density ** (1.0 - gamma))).mean().item()
        return final_score

    def get_block_score(self, block_keys, block_values, evict_method):
        #### only used for global and local eviction
        assert evict_method in ["global", "local"], "Only global and local eviction methods are supported for block score calculation"
        if self.sub_evict_method == "value_l2":
            value_norms = torch.norm(block_values, p=2, dim=-1).mean(dim=1).sum(dim=0)
            return value_norms
        elif self.sub_evict_method == "key_l2":
            key_norms = torch.norm(block_keys, p=2, dim=-1).mean(dim=1).sum(dim=0)
            return key_norms
        elif self.sub_evict_method == "value_l2_plus_key_l2":
            value_norms = torch.norm(block_values, p=2, dim=-1).mean(dim=1).sum(dim=0)
            key_norms = torch.norm(block_keys, p=2, dim=-1).mean(dim=1).sum(dim=0)
            return value_norms + key_norms
        elif self.sub_evict_method == "inverse_key_l2":
            # 遵循先个体后聚合的泛化原则
            k_norm = torch.linalg.vector_norm(block_keys, ord=2, dim=-1)
            inverse_k = 1.0 / (k_norm + 1e-8)
            # 先对 heads 求平均/求和(对齐Token打分)，再对 seq 求平均
            return inverse_k.sum(dim=1).mean(dim=0).item()
        elif self.sub_evict_method == "paged_ratio":
            # 单个 Token & Head 的 Norm
            v_norm = torch.linalg.vector_norm(block_values, ord=2, dim=-1) # shape: [seq_len_in_block, num_heads]
            k_norm = torch.linalg.vector_norm(block_keys, ord=2, dim=-1)
            # 每个 Token 的 Ratio (先相除)
            ratio = v_norm / (k_norm + 1e-8)
            # 对 heads 进行聚合 (对齐 Token 打分的 sum 逻辑)，然后对 block 内的 tokens (dim=0) 求平均
            return ratio.sum(dim=1).mean(dim=0).item()
        elif self.sub_evict_method == "profile_richness":
            # Input: block_values [num_tokens, num_heads, head_dim]
            # 这里的 num_tokens 被视为属于同一个 Block (Scope)
            profile_maxpool_version = os.environ.get("PROFILE_MAXPOOL_VER", "3")
            if profile_maxpool_version == "1":
                scores = self._profile_maxpool_v1(block_keys, block_values)
            elif profile_maxpool_version == "2":
                scores = self._profile_maxpool_v2(block_keys, block_values)
            elif profile_maxpool_version == "3":
                scores = self._profile_maxpool_v3(block_keys, block_values)
            else:
                raise ValueError(f"Unknown PROFILE_VER: {profile_maxpool_version}")
            return scores

    def prune_prompt(self, key_tensor, value_tensor):
        # remove tokens for a single request
        # start = time.perf_counter()
        q_len, num_heads, head_dim = key_tensor.shape
        if q_len <= self.cache_budget:
            return key_tensor, value_tensor
        
        if self.evict_method in ["streamingLLM", "streamingLLM-1"]:
            # for budget case
            remainder_size = q_len % self.orig_block_size
            # Only the middle slice will be pruned, reserve enough blocks
            end_idx_first_slice = self.initial_blocks * self.orig_block_size
            first_slice = slice(0, end_idx_first_slice)
            
            end_idx_middle_slice = q_len - self.orig_block_size - remainder_size
            middle_slice = slice(end_idx_first_slice, end_idx_middle_slice)
            
            last_slice = slice(end_idx_middle_slice, q_len)
            # print(f"**********1===remainder_size = {remainder_size}, q_len = {q_len}, first_slice = {first_slice}, middle_slice = {middle_slice}, last_slice = {last_slice}") 
            
            # Extract middle tokens and its size
            middle_key = key_tensor[middle_slice, :, :]
            middle_value = value_tensor[middle_slice, :, :]
            middle_tokens = middle_key.shape[0]
            
            head_len = end_idx_first_slice
            tail_len = self.orig_block_size
            middle_allowance = self.cache_budget - head_len - tail_len
            total_prune_tokens = middle_tokens - middle_allowance
            total_prune_tokens = 0 if total_prune_tokens < 0 else total_prune_tokens
    
            # Middle slice after prunning
            middle_slice = slice(end_idx_first_slice + total_prune_tokens, end_idx_middle_slice) 
        
            # Rejoin the tokens from the first slice, prunned_middle_key, and last slice
            rejoined_key = torch.cat([
                key_tensor[first_slice, :, :],
                key_tensor[middle_slice, :, :],
                key_tensor[last_slice, :, :]
            ], dim=0)
            rejoined_value = torch.cat([
                value_tensor[first_slice, :, :],
                value_tensor[middle_slice, :, :],
                value_tensor[last_slice, :, :]
            ], dim=0)
            # print(f"**********6===rejoined_key = {rejoined_key.shape}, rejoined_value = {rejoined_value.shape}")
            # end = time.perf_counter()
            # print(f"Time taken to prune prompt blocks take {end - start:.6f} using streamingLLM on stream {torch.cuda.current_stream()}")
            return rejoined_key, rejoined_value 
        elif self.evict_method in ["local", "global", "inverse_key_l2"]:
            # always keep the first block and the last block
            remainder_size = q_len % self.orig_block_size
            end_idx_first_slice = self.orig_block_size * self.initial_blocks
            first_slice = slice(0, end_idx_first_slice)
            
            end_idx_middle_slice = q_len - self.orig_block_size - remainder_size
            middle_slice = slice(end_idx_first_slice, end_idx_middle_slice)

            last_slice = slice(end_idx_middle_slice, q_len)
            # print(f"**********remainder_size = {remainder_size}, q_len = {q_len}, first_slice = {first_slice}, middle_slice = {middle_slice}, last_slice = {last_slice}")
            
            # Extract middle tokens and its keys
            middle_key = key_tensor[middle_slice, :, :]
            middle_value = value_tensor[middle_slice, :, :]
            middle_tokens = middle_key.shape[0]
            # 修改点：将硬编码的 orig_block_size 替换为受 initial_blocks 控制的 end_idx_first_slice
            middle_pruned_tokens = self.cache_budget - end_idx_first_slice - self.orig_block_size
            total_prune_tokens = middle_tokens - middle_pruned_tokens 
            total_prune_tokens = 0 if total_prune_tokens < 0 else total_prune_tokens
                
            # Get scores for middle tokens
            scores = self.get_token_score(middle_key, middle_value, self.evict_method)

            # Evict least N elements, select the indices along dim=0
            _, least_indices = torch.topk(scores, k=total_prune_tokens, largest=False, dim=0)

            # Create a mask for pruning
            mask = torch.ones_like(scores, dtype=torch.bool)
            mask.scatter_(0, least_indices, False)

            # Prune middle tokens, middle_key[mask] will drop the data based on the mask, only keep the data where mask is True
            pruned_middle_key = middle_key[mask].view(-1, num_heads, head_dim)
            pruned_middle_value = middle_value[mask].view(-1, num_heads, head_dim)

            # Rejoin the tokens from the first slice, prunned_middle_key, and last slice
            rejoined_key = torch.cat([
                key_tensor[first_slice, :, :],
                pruned_middle_key,
                key_tensor[last_slice, :, :]
            ], dim=0)
            
            rejoined_value = torch.cat([
                value_tensor[first_slice, :, :],
                pruned_middle_value,
                value_tensor[last_slice, :, :]
            ], dim=0)
            # end = time.perf_counter()
            # print(f"Time taken to prune prompt blocks take {end - start:.6f} using streamingLLM on stream {torch.cuda.current_stream()}")   
            return rejoined_key, rejoined_value
    
    def prune_prompt_inplace(self, key_tensor, value_tensor, output_key_tensor, output_value_tensor):
        """
        In-place version of prune_prompt that writes directly to pre-allocated output tensors.
        
        Args:
            key_tensor: Input keys to prune
            value_tensor: Input values to prune  
            output_key_tensor: Pre-allocated tensor to write pruned keys
            output_value_tensor: Pre-allocated tensor to write pruned values
        """
        q_len, num_heads, head_dim = key_tensor.shape
        if q_len <= self.cache_budget:
            output_key_tensor.copy_(key_tensor)
            output_value_tensor.copy_(value_tensor)
            return

        if self.evict_method in ["streamingLLM", "streamingLLM-1"]:
            # for budget case
            remainder_size = q_len % self.orig_block_size
            # Only the middle slice will be pruned
            end_idx_first_slice = self.initial_blocks * self.orig_block_size
            first_slice = slice(0, end_idx_first_slice)
            
            end_idx_middle_slice = q_len - self.orig_block_size - remainder_size
            middle_slice = slice(end_idx_first_slice, end_idx_middle_slice)
            
            last_slice = slice(end_idx_middle_slice, q_len)
            
            # Extract middle tokens and its size
            middle_tokens = end_idx_middle_slice - end_idx_first_slice
            
            head_len = end_idx_first_slice
            tail_len = self.orig_block_size
            middle_allowance = self.cache_budget - head_len - tail_len
            total_prune_tokens = middle_tokens - middle_allowance
            total_prune_tokens = 0 if total_prune_tokens < 0 else total_prune_tokens
    
            # Middle slice after pruning
            middle_slice = slice(end_idx_first_slice + total_prune_tokens, end_idx_middle_slice) 
    
            # Write directly to output tensors
            combined_key = torch.cat([
                key_tensor[first_slice, :, :],
                key_tensor[middle_slice, :, :],
                key_tensor[last_slice, :, :]
            ], dim=0)
            combined_value = torch.cat([
                value_tensor[first_slice, :, :],
                value_tensor[middle_slice, :, :],
                value_tensor[last_slice, :, :]
            ], dim=0)
            
            output_key_tensor.copy_(combined_key)
            output_value_tensor.copy_(combined_value)

        elif self.evict_method in ["local", "global", "inverse_key_l2", "paged_ratio", "profile_richness"]:
            # for budget case
            remainder_size = q_len % self.orig_block_size
            # Only the middle slice will be pruned
            end_idx_first_slice = self.initial_blocks * self.orig_block_size
            first_slice = slice(0, end_idx_first_slice)
            
            end_idx_middle_slice = q_len - self.orig_block_size - remainder_size
            middle_slice = slice(end_idx_first_slice, end_idx_middle_slice)
            
            last_slice = slice(end_idx_middle_slice, q_len)
            
            # Extract middle tokens and its size
            middle_tokens = end_idx_middle_slice - end_idx_first_slice
            
            head_len = end_idx_first_slice
            tail_len = self.orig_block_size
            middle_allowance = self.cache_budget - head_len - tail_len
            total_prune_tokens = middle_tokens - middle_allowance
            total_prune_tokens = 0 if total_prune_tokens < 0 else total_prune_tokens
    
            # Middle slice after pruning
            middle_slice = slice(end_idx_first_slice + total_prune_tokens, end_idx_middle_slice) 
    
            # Write directly to output tensors
            combined_key = torch.cat([
                key_tensor[first_slice, :, :],
                key_tensor[middle_slice, :, :],
                key_tensor[last_slice, :, :]
            ], dim=0)
            combined_value = torch.cat([
                value_tensor[first_slice, :, :],
                value_tensor[middle_slice, :, :],
                value_tensor[last_slice, :, :]
            ], dim=0)
            
            output_key_tensor.copy_(combined_key)
            output_value_tensor.copy_(combined_value)
        else:
            raise ValueError(f"Unsupported eviction method: {self.evict_method} for cache_prune_type: {self.cache_prune_type}")
    
     
    def get_pruned_length(self, seq_len):
        """
        Calculate the length after pruning for a given sequence length.
        This is needed for pre-allocating output tensors.
        """
        if seq_len <= self.cache_budget:
            return seq_len
            
        if self.evict_method in ["streamingLLM", "streamingLLM-1", "inverse_key_l2"]: 
            # Need to prune tokens from middle slice
            remainder_size = seq_len % self.orig_block_size
            # Only the middle slice will be pruned
            end_idx_first_slice = self.initial_blocks * self.orig_block_size
            num_first_slice_tokens = end_idx_first_slice

            end_idx_middle_slice = seq_len - self.orig_block_size - remainder_size
            num_middle_tokens = end_idx_middle_slice - end_idx_first_slice
            num_last_slice_tokens = seq_len - end_idx_middle_slice

            middle_unpruned_tokens = self.cache_budget - num_first_slice_tokens - self.orig_block_size

            middle_unpruned_tokens = max(middle_unpruned_tokens, 0)
            total_unpruned_tokens = num_first_slice_tokens + middle_unpruned_tokens + num_last_slice_tokens

            return total_unpruned_tokens
            
        elif self.evict_method in ["local", "global", "paged_ratio", "profile_richness"]:
            # Need to prune tokens from middle slice
            remainder_size = seq_len % self.orig_block_size
            # Only the middle slice will be pruned
            end_idx_first_slice = self.initial_blocks * self.orig_block_size
            num_first_slice_tokens = end_idx_first_slice

            end_idx_middle_slice = seq_len - self.orig_block_size - remainder_size
            num_middle_tokens = end_idx_middle_slice - end_idx_first_slice
            num_last_slice_tokens = seq_len - end_idx_middle_slice

            middle_unpruned_tokens = self.cache_budget - num_first_slice_tokens - self.orig_block_size

            middle_unpruned_tokens = max(middle_unpruned_tokens, 0)
            total_unpruned_tokens = num_first_slice_tokens + middle_unpruned_tokens + num_last_slice_tokens

            return total_unpruned_tokens
        else:
            raise ValueError(f"Unsupported eviction method: {self.evict_method} for cache_prune_type: {self.cache_prune_type}")
    
        
    def get_blocks_to_prune_and_merge_decode(self, seq_kv_len):
        if self.decode_evict_method == "streamingLLM":
            if self.cache_prune_type == "percentage":
                assert self.num_block_merge >= 2
                s_block_id = self.initial_blocks
                e_block_id = s_block_id + self.num_block_merge
                prune_tokens = (self.num_block_merge - 1) * self.orig_block_size
            else:
                # For cache_budget case
                if seq_kv_len <= self.cache_budget:
                    s_block_id, e_block_id = -1, -1 # represent no blocks to prune
                    prune_tokens = 0
                else:
                    s_block_id = self.initial_blocks
                    e_block_id = self.initial_blocks + 1
                    prune_tokens = self.orig_block_size
        else:
            if self.cache_prune_type == "percentage":
                s_block_id = self.initial_blocks
                e_block_id = self.initial_blocks + self.num_block_merge
                prune_tokens = (self.num_block_merge - 1) * self.orig_block_size
            else:
                # For cache_budget case
                if seq_kv_len <= self.cache_budget:
                    s_block_id, e_block_id = -1, -1 # No blocks to prune
                    prune_tokens = 0
                else:
                    s_block_id = self.initial_blocks
                    e_block_id = self.initial_blocks + self.num_block_merge
                    prune_tokens = self.orig_block_size
         
        return s_block_id, e_block_id, prune_tokens
          
    def prune_oldest_block(self, key_tensor, value_tensor, prune_tokens):
        if self.decode_evict_method== "streamingLLM":
            # For vLLM, we just need to remove the block from the block_table, so do nothing here
            return None, None
        else:
            q_len, num_heads, head_dim = key_tensor.shape
            scores = self.get_score(key_tensor, value_tensor, self.decode_evict_method)
            # print(f"********scores.shape = {scores.shape}")
            # print(f"********Number of tokens to prune = {prune_tokens}")
            _, least_indices = torch.topk(scores, k=prune_tokens, largest=False, dim=0)

            mask = torch.ones_like(scores, dtype=torch.bool)
            mask.scatter_(0, least_indices, False)

            # print(f"mask.shape = {mask.shape}, key_tensor[mask].shape = {key_tensor[mask].shape}")
            pruned_key = key_tensor[mask].view(-1, num_heads, head_dim)
            pruned_value = key_tensor[mask].view(-1, num_heads, head_dim)
            
            return pruned_key, pruned_value        
    

    
if __name__ == "__main__":
    num_heads = 4
    head_dim = 64

    q_len = 1024
    block_size = 16
    budget = 512
    initial_blocks = 1
    # prompt_evict_method = "streamingLLM"
    evict_method = "local"
    topk_blocks = 3

    cache_type   = "budget"

    pruner = KVCachePruner(cache_type, evict_method, block_size, budget, initial_blocks)

    keys = torch.randn(q_len, num_heads, head_dim)
    values = torch.randn(q_len, num_heads, head_dim)

    print("before prompt blocks pruning", keys.shape, values.shape)
    keys, values = pruner.prune_prompt(keys, values)
    print("after prompt blocks pruning", keys.shape, values.shape)
    print()

    # s_block_id, e_block_id, prune_tokens = pruner.get_blocks_to_prune_and_merge_decode(keys.shape[0] + block_size * evict_frequency) 
    # print(f"s_block_id = {s_block_id}, e_block_id = {e_block_id}, prune_tokens = {prune_tokens}")
    # keys = torch.randn(bsz, num_heads, q_len, head_dim)
    # values = torch.randn(bsz, num_heads, q_len, head_dim)

    # print("before recent block pruning", keys.shape, values.shape)
    # keys, values = pruner.prune_recent_block(keys, values)
    # print("after recent block pruning", keys.shape, values.shape)

