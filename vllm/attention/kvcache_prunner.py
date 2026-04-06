import torch
import torch.nn.functional as F
import math 
import time

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
                # use inverse key_l2 norm for global and local eviction
                scores = self.compute_hoyer_sparsity(keys)
                # scores = self.inverse_key_l2(keys, values)
                l1_norm = torch.norm(keys, p=1, dim=-1)
                scores = 1.0 / (l1_norm + 1e-8)
        elif evict_method == "inverse_key_l2":
            # use inverse key_l2 norm for inverse_key_l2 eviction
            scores = self.inverse_key_l2(keys, values)
        else:
            raise ValueError(f"Unknown eviction method: {evict_method}")
        return scores

    def _profile_maxpool_onlyVecV(self, block_keys, block_values):
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

    def _profile_maxpool_multSparsity(self, block_keys, block_values):
        """版本 v3: 待调参或修改逻辑的预留位置"""
        # 1. 计算 V 极性 (L2 / L1) -> 越高代表特征越锐利、越不平庸
        v_l2 = torch.linalg.vector_norm(block_values, ord=2, dim=-1)
        v_l1 = torch.linalg.vector_norm(block_values, ord=1, dim=-1)
        v_polarity = v_l2 / (v_l1 + 1e-8) # [seq, heads]

        # 2. 计算 K 极性 (L2 / L1) -> 越高代表作为 Key 的区分度越强 (容易被精确命中)
        k_l2 = torch.linalg.vector_norm(block_keys, ord=2, dim=-1)
        k_l1 = torch.linalg.vector_norm(block_keys, ord=1, dim=-1)
        k_polarity = k_l2 / (k_l1 + 1e-8) # [seq, heads]

        # 3. Token 级联合极性 (相乘代表同时满足)
        token_scores = v_polarity * k_polarity # [seq, heads]

        # 4. Block 级聚合：寻找 Block 内综合极性最锐利的 Token
        block_score = token_scores.max(dim=0).values.mean().item()
        return block_score

    def _profile_maxpool_DevotoEnv(self, block_keys, block_values):
        """
        [Devoto 加权包络] Token 级 K 倒数加权 -> Block Max-Pool
        将 Devoto 定理 (1/L2) 作为该 Token V 向量被检索的概率权重。
        """
        # Step 1: 计算 K 的注意力代理权重 (Addressability)
        k_norm = torch.linalg.vector_norm(block_keys, ord=2, dim=-1, keepdim=True)
        attn_proxy = 1.0 / (k_norm + 1e-8)
        # Step 2: 用注意力权重对 V 的特征幅度进行加权
        weighted_values = torch.abs(block_values) * attn_proxy
        # Step 3: Meso-level Aggregation (Max Pooling on valid features)
        profile_vector = torch.max(weighted_values, dim=0).values
        profile_norm = torch.linalg.vector_norm(profile_vector, ord=2, dim=-1)
        return profile_norm.mean().item()

    def _profile_maxpool_scalerEnvelope(self, block_keys, block_values):
        """
        [双重校准包络] Block 视角独立的 K 与 V 融合 (宏观视角)
        不采用 Token 级加权，而是分别计算 Block 的 V 丰富度和 Block 的 K 极性，然后在宏观相乘。
        """
        # 1. Block V Richness
        v_env = torch.max(torch.abs(block_values), dim=0).values
        v_richness = torch.linalg.vector_norm(v_env, ord=2, dim=-1) # [num_heads]
        # 2. Block K 综合极性 (基于 Devoto)
        k_norm = torch.linalg.vector_norm(block_keys, ord=2, dim=-1)
        k_importance = (1.0 / (k_norm + 1e-8)).mean(dim=0) # [num_heads]
        # 3. Head 级别相乘后求均值
        return (v_richness * k_importance).mean().item()

    def _profile_maxpool_powKV(self, block_keys, block_values):
        v_maxpool = torch.max(torch.abs(block_values), dim=0).values
        v_richness = torch.norm(v_maxpool, p=2, dim=-1) # [num_heads]
        # Hoyer 极性变体 (L2/L1) 幂级数加权
        # K 极性越高 (L2/L1 越大)，表明分布越稀疏/越易被精准检索
        k_l2 = torch.linalg.vector_norm(block_keys, ord=2, dim=-1)
        k_l1 = torch.linalg.vector_norm(block_keys, ord=1, dim=-1)
        k_polarity = (k_l2 / (k_l1 + 1e-8)).mean(dim=0) # [num_heads]
        gamma = float(os.environ.get("PROFILE_GAMMA", "0.5"))
        # V^gamma * K极性^(1-gamma)
        return ((v_richness ** gamma) * (k_polarity ** (1 - gamma))).mean().item()

    def _profile_maxpool_sparsityEnvelope(self, block_keys, block_values):
        """
        [极性/稀疏度加权包络] Token 级 K Hoyer 稀疏度加权 -> Block Max-Pool
        高稀疏度的 K 意味着锋利的专属匹配，将其作为权重。
        """
        import math
        d = block_keys.shape[-1]
        sqrt_d = math.sqrt(d)

        k_l1 = torch.linalg.vector_norm(block_keys, ord=1, dim=-1) # [seq, num_heads]
        k_l2 = torch.linalg.vector_norm(block_keys, ord=2, dim=-1) # [seq, num_heads]

        # 计算 Hoyer Sparsity (0 到 1 之间，越接近 1 越极化/稀疏)
        k_sparsity = (sqrt_d - (k_l1 / (k_l2 + 1e-8))) / (sqrt_d - 1.0) # [seq, num_heads]

        # Token 级加权 & Block 级包络提取
        weighted_values = block_values * k_sparsity.unsqueeze(-1)
        v_env = torch.max(torch.abs(weighted_values), dim=0).values
        return torch.linalg.vector_norm(v_env, ord=2, dim=-1).mean().item()

    def _profile_maxpool_cascadeKmean(self, block_keys, block_values):
        # 方案 3: 级联设计 (Hard Thresholding)
        k_l2 = torch.linalg.vector_norm(block_keys, ord=2, dim=-1)
        k_l1 = torch.linalg.vector_norm(block_keys, ord=1, dim=-1)
        k_polarity = k_l2 / (k_l1 + 1e-8) # [seq_len_in_block, num_heads]

        # 设定阈值：仅保留极性超过均值的 Token 的 V 向量参与包络计算
        threshold = k_polarity.mean(dim=0, keepdim=True) # [1, num_heads]
        valid_mask = k_polarity > threshold # [seq_len_in_block, num_heads]

        # 扩展 Mask 以匹配 head_dim
        valid_mask_expanded = valid_mask.unsqueeze(-1).expand_as(block_values)
        filtered_v = block_values * valid_mask_expanded

        cascade_v_maxpool = torch.max(torch.abs(filtered_v), dim=0).values
        scores = torch.norm(cascade_v_maxpool, p=2, dim=-1).mean().item()
        return scores

    def _profile_maxpool_softmaxKSparse(self, block_keys, block_values):
        # 方案 4 (推荐): 动态 Soft-Gating (K 极性作为 V 包络的注意力权重)
        # 结合了包络的宏观性与 K 向量的微观检索概率
        k_l2 = torch.linalg.vector_norm(block_keys, ord=2, dim=-1)
        k_l1 = torch.linalg.vector_norm(block_keys, ord=1, dim=-1)
        k_sparsity = k_l2 / (k_l1 + 1e-8) # [seq_len_in_block, num_heads]

        # 在 Block 内部对 K 稀疏度进行 Softmax 归一化
        k_weights = torch.nn.functional.softmax(k_sparsity, dim=0).unsqueeze(-1) # [seq_len, num_heads, 1]

        # K 加权后的 V 向量再求 Max-Pool 包络
        weighted_v = block_values * k_weights
        gated_v_maxpool = torch.max(torch.abs(weighted_v), dim=0).values
        scores = torch.norm(gated_v_maxpool, p=2, dim=-1).mean().item()
        return scores

    def _profile_maxpool_powKVmeanFirst(self, block_keys, block_values):
        v_maxpool = torch.max(torch.abs(block_values), dim=0).values
        v_richness = torch.norm(v_maxpool, p=2, dim=-1) # [num_heads]
        # Hoyer 极性变体 (L2/L1) 幂级数加权
        # K 极性越高 (L2/L1 越大)，表明分布越稀疏/越易被精准检索
        k_l2 = torch.linalg.vector_norm(block_keys, ord=2, dim=-1)
        k_l1 = torch.linalg.vector_norm(block_keys, ord=1, dim=-1)
        k_polarity = (k_l2 / (k_l1 + 1e-8)).mean(dim=0) # [num_heads]

        gamma = float(os.environ.get("PROFILE_GAMMA", "0.5"))
        # V^gamma * K极性^(1-gamma)
        return ((v_richness ** gamma) * (k_polarity ** (1 - gamma))).mean().item()

    def _profile_maxpool_GeoWgtFusion0405(self, block_keys, block_values):
        # Hoyer 极性变体 (L2/L1) 幂级数加权
        # K 极性越高 (L2/L1 越大)，表明分布越稀疏/越易被精准检索
        k_l2 = torch.linalg.vector_norm(block_keys, ord=2, dim=-1)
        k_l1 = torch.linalg.vector_norm(block_keys, ord=1, dim=-1)
        k_polarity = (k_l2 / (k_l1 + 1e-8)) # # [S, H]  <-- 保留 Token 维度！

        v_abs = torch.abs(block_values)  # [S, H, D]
        k_polarity = k_polarity.unsqueeze(-1)  # [S, H, 1]
        weighted_v = v_abs * k_polarity        # [S, H, D] <-- 同一个 Token 的 K × V
        gated_v_maxpool = torch.max(weighted_v, dim=0).values  # [H, D]
        v_richness = torch.linalg.vector_norm(gated_v_maxpool, ord=1, dim=-1)  # [H]

        gamma = float(os.environ.get("PROFILE_GAMMA", "0.5"))
        # V^gamma * K极性^(1-gamma)
        return ((v_richness ** gamma) * (k_polarity ** (1 - gamma))).mean().item()

    def _profile_maxpool_alphaL1(self, block_keys, block_values):
        """
        V4 [终极建议版]: 注意力潜能预加权 (Attention Potential Gating)
        逻辑：与其分别求均值再相乘，不如回归 Attention 的物理本质。
              一个 V 向量能否对最终输出产生影响，取决于它的 K 有多“锋利”且模长有多“小”。
              我们将 K_polarity * Devoto 视为该 Token 的“激活潜能”，
              用这个潜能去动态缩放 V 向量，然后再求 Max-pool 包络。
        """
        # 1. 综合 K 潜能 = K极性(L2/L1) * Devoto(1/L2) = 1 / L1
        # 这在数学上异常优雅：L1 范数越小，意味着非零特征越少且绝对值越小，潜能越大。
        k_l1 = torch.linalg.vector_norm(block_keys, ord=1, dim=-1)
        k_potential = 1.0 / (k_l1 + 1e-8) # [seq, num_heads]

        # 为了避免某个极端 Token 霸占数值，对潜能进行 Block 内部的归一化 (0~1 之间)
        k_potential_norm = k_potential / (k_potential.max(dim=0, keepdim=True).values + 1e-8)

        # 2. 用潜能动态缩放 V 向量
        alpha = float(os.environ.get("PROFILE_ALPHA", "0.5"))
        gated_values = block_values * (1+alpha) * k_potential_norm.unsqueeze(-1)

        # 3. 提取预加权后的真实特征包络
        v_envelope = torch.max(torch.abs(gated_values), dim=0).values # [num_heads, head_dim]
        return torch.linalg.vector_norm(v_envelope, ord=2, dim=-1).mean().item()

    def _profile_maxpool_kLinearAmplifier(self, block_keys, block_values):
        """
        [K-Amplified Linear Payload (K极性线性放大器)]
        吸收了 Lab 3 中偏向 K 的结论，并修复了非线性操作破坏 V 空间包络的缺陷。
        """
        # 1. 提取 K 极性标量 (L2/L1) -> 数值通常在 [1/sqrt(D), 1] 之间
        k_l2 = torch.linalg.vector_norm(block_keys, ord=2, dim=-1)
        k_l1 = torch.linalg.vector_norm(block_keys, ord=1, dim=-1)
        k_polarity = k_l2 / (k_l1 + 1e-8)  # [S, H]
        
        # 2. K 极性非线性激励 (kappa > 1.0)
        # 用指数级放大的方式，严厉惩罚平庸的 K，奖励尖锐的 K，但不扭曲 V 本身
        # 建议实验区间: KAPPA = 1.0, 2.0, 3.0, 4.0
        kappa = float(os.environ.get("PROFILE_KAPPA", "2.0"))
        k_weight = k_polarity ** kappa     # [S, H]
        
        # 3. Token 级严格微观绑定
        v_abs = torch.abs(block_values)    # [S, H, D]
        k_weight = k_weight.unsqueeze(-1)  # [S, H, 1]
        
        # 注意：这里是线性乘法，完美保护了 V 的特征包络拓扑
        weighted_v = v_abs * k_weight      # [S, H, D]
        
        # 4. 提取宏观特征包络 (Meso-level Envelope)
        gated_v_maxpool = torch.max(weighted_v, dim=0).values # [H, D]
        
        # 5. 计算包络的综合能量
        v_richness = torch.linalg.vector_norm(gated_v_maxpool, ord=2, dim=-1) # [H]
        return v_richness.mean().item() 

    def _profile_maxpool_maxSoftGate(self, block_keys, block_values):
        # 动态 Soft-Gating (K 极性作为 V 包络的注意力权重)
        # 结合了包络的宏观性与 K 向量的微观检索概率
        k_l2 = torch.linalg.vector_norm(block_keys, ord=2, dim=-1)
        k_l1 = torch.linalg.vector_norm(block_keys, ord=1, dim=-1)
        k_sparsity = k_l2 / (k_l1 + 1e-8) # [seq_len_in_block, num_heads]

        # 在 Block 内部对 K 稀疏度进行 Softmax 归一化
        # k_weights = torch.nn.functional.softmax(k_sparsity, dim=0).unsqueeze(-1) # [seq_len, num_heads, 1]
        k_weights = k_sparsity / (k_sparsity.max(dim=0, keepdim=True).values + 1e-8)
        k_weights = k_weights.unsqueeze(-1)

        # K 加权后的 V 向量再求 Max-Pool 包络
        weighted_v = block_values * k_weights
        gated_v_maxpool = torch.max(torch.abs(weighted_v), dim=0).values
        scores = torch.norm(gated_v_maxpool, p=2, dim=-1).mean().item()
        return scores

    def _profile_maxpool_VabsKpolarEnvelop(self, block_keys, block_values):
        """
        [特征包络幂级数门控] 
        严格的 Token 级绑定 + 几何加权，防止 K 极性重复计算。
        """
        gamma = float(os.environ.get("PROFILE_GAMMA", "0.5"))
        # 1. 提取 K 极性 (Hoyer 极性变体 L2/L1)
        k_l2 = torch.linalg.vector_norm(block_keys, ord=2, dim=-1)
        k_l1 = torch.linalg.vector_norm(block_keys, ord=1, dim=-1)
        k_polarity = k_l2 / (k_l1 + 1e-8)  # [S, H]
        # 2. 在 Token 级别，对 V 特征和 K 极性分别执行几何缩放 必须在融合前幂运算
        v_abs_pow = torch.abs(block_values) ** (1 + gamma)     # [S, H, D]
        k_polarity_pow = k_polarity ** (1.0 - gamma)     # [S, H]
        # 3. Token 级绑定 (Micro-binding)
        k_polarity_pow = k_polarity_pow.unsqueeze(-1)    # [S, H, 1]
        weighted_v = v_abs_pow * k_polarity_pow          # [S, H, D]
        # 4. 提取宏观特征包络 (Meso-level Envelope Extraction)
        gated_v_maxpool = torch.max(weighted_v, dim=0).values # [H, D]
        # 5. 计算包络能量
        v_richness = torch.linalg.vector_norm(gated_v_maxpool, ord=2, dim=-1) # [H]
        return v_richness.mean().item()

    def _profile_maxpool_v1(self, block_keys, block_values):
        """
        [K极性指数级放大 + L1 包络体积]
        严格遵守 Token 微观绑定，保持 V 的线性空间，指数放大 K 极性，使用 L1 衡量信息容量。
        """
        # 1. 提取 K 极性 (L2/L1) -> 衡量检索锐度
        k_l2 = torch.linalg.vector_norm(block_keys, ord=2, dim=-1)
        k_l1 = torch.linalg.vector_norm(block_keys, ord=1, dim=-1)
        k_polarity = k_l2 / (k_l1 + 1e-8)  # [S, H]
        # 2. 指数级放大 K 极性 (建议 KAPPA = 2.0, 3.0, 4.0)
        # 用高次幂将绝佳钩子与平庸钩子拉开巨大鸿沟
        kappa = float(os.environ.get("PROFILE_KAPPA", "3.0"))
        k_weight = k_polarity ** kappa     # [S, H]
        # 3. Token 级微观绑定 (绝对保持 V 的线性！)
        v_abs = torch.abs(block_values)    # [S, H, D]
        weighted_v = v_abs * k_weight.unsqueeze(-1)  # [S, H, D]
        # 4. 提取宏观特征包络 (Meso-level Envelope)
        gated_v_maxpool = torch.max(weighted_v, dim=0).values # [H, D]
        # 5. 使用 L1 范数计算包络体积 (至关重要！取代 L2)
        # L1 完美衡量了该 Block 激活了多少个不同的概念特征
        v_richness_volume = torch.linalg.vector_norm(gated_v_maxpool, ord=1, dim=-1) # [H]
        return v_richness_volume.mean().item()

    def _profile_maxpool_v2(self, block_keys, block_values):
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
        kappa = float(os.environ.get("PROFILE_KAPPA", "1.0"))
        hook_density = (k_polarity ** kappa).mean(dim=0) # [H] 
        # 宏观几何融合 (Geometric Fusion)
        gamma = float(os.environ.get("PROFILE_GAMMA", "0.2"))
        # 两个指标均在 [H] 维度，几何加权后求均值，物理意义绝对严谨
        final_score = ((envelope_volume ** gamma) * (hook_density ** (1.0 - gamma))).mean().item()
        return final_score

    def _profile_maxpool_v4(self, block_keys, block_values):
        pass

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
            key_norms = torch.norm(block_keys, p=2, dim=-1).mean(dim=1).sum(dim=0)
            return 1 / (key_norms + 1e-8)
        elif self.sub_evict_method == "paged_ratio":
            # 避免使用 .mean().sum() 的链式调用，直接使用 linalg.vector_norm
            # value_norms = torch.norm(block_values, p=2, dim=-1).mean(dim=1).sum(dim=0)
            v_norm = torch.linalg.vector_norm(block_values, ord=2, dim=-1).mean(dim=1).sum()
            # key_norms = torch.norm(block_keys, p=2, dim=-1).mean(dim=1).sum(dim=0)
            k_norm = torch.linalg.vector_norm(block_keys, ord=2, dim=-1).mean(dim=1).sum()
            # return value_norms / (key_norms + 1e-8)
            return (v_norm / (k_norm + 1e-8)).item()
        elif self.sub_evict_method == "profile_richness":
            profile_maxpool_version = os.environ.get("PROFILE_MAXPOOL_VER", "1")
            if profile_maxpool_version == "1":
                scores = self._profile_maxpool_v1(block_keys, block_values)
            elif profile_maxpool_version == "2":
                scores = self._profile_maxpool_v2(block_keys, block_values)
            elif profile_maxpool_version == "3":
                scores = self._profile_maxpool_v3(block_keys, block_values)
            elif profile_maxpool_version == "4":
                scores = self._profile_maxpool_v4(block_keys, block_values)
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

