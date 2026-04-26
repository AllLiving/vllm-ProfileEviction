import torch
import torch.nn.functional as F
import math 
import time
import os

# =====================================================================
# [性能优化]: 静态编译 Token 级双轨制打分函数
# =====================================================================
@torch.jit.script
def _jit_profile_sparsity_v1(keys: torch.Tensor, values: torch.Tensor, k_hoyer: torch.Tensor, gamma: float, kappa: float) -> torch.Tensor:
    # --- V 部分 ---
    # 极致优化：L1 范数等价于 abs().sum()，并在 head 维度一并 sum
    # 原逻辑: norm(ord=1, dim=-1).sum(dim=1) -> 优化为: abs().sum(dim=[-1, 1])
    v_score_raw = values.abs().sum(dim=[-1, 1]) # [S]
    v_score = v_score_raw / (v_score_raw.mean() + 1e-8)

    # --- K 部分 ---
    # L2 范数必须用底层 norm
    k_l2 = torch.norm(keys, p=2, dim=-1) # [S, H]
    k_l2_normed = k_l2 / (k_l2.mean(dim=0, keepdim=True) + 1e-8)
    k_kinetic = (1.0 / (k_l2_normed + 1e-2)) * k_hoyer # [S, H]
    k_score_raw = k_kinetic.sum(dim=1) # [S]
    k_score = k_score_raw / (k_score_raw.mean() + 1e-8)

    # 融合：公平对数凸组合
    # 返回形状需增加一维以匹配原逻辑
    return ((v_score ** gamma) * (k_score ** kappa)).unsqueeze(-1)

@torch.jit.script
def _jit_profile_sparsity_v2(keys: torch.Tensor, values: torch.Tensor, gamma: float, kappa: float) -> torch.Tensor:
    # --- V 部分 ---
    # 极致优化：L1 范数等价于 abs().sum()，连带 head 维度一并求和
    v_score_raw = values.abs().sum(dim=[-1, 1]) # [S]
    v_score = v_score_raw / (v_score_raw.mean() + 1e-8)

    # --- K 部分 ---
    # 极致优化：L1 范数等价于 abs().sum()。
    # 这里 k_l1 是 [S, H]，后续要做除法，所以只在 -1 维度求和
    k_l1 = keys.abs().sum(dim=-1) # [S, H]
    k_kinetic = 1.0 / (k_l1 + 1e-8) # [S, H]
    k_score_raw = k_kinetic.sum(dim=1) # [S]
    k_score = k_score_raw / (k_score_raw.mean() + 1e-8)

    # 几何融合
    return ((v_score ** gamma) * (k_score ** kappa)).unsqueeze(-1)

# =====================================================================
# [性能优化]: 静态编译双轨制打分函数，融合碎算子，消除 Python 调度开销
# =====================================================================
@torch.jit.script
def _jit_dual_track_score(block_keys: torch.Tensor, block_values: torch.Tensor, kappa: float, gamma: float) -> float:
    # 1. 基础极性定义
    # 【回退优化】坚决使用 torch.norm，利用底层 C++ ATen 级高度优化的融合算子
    k_l2 = torch.norm(block_keys, p=2, dim=-1)
    k_l1 = torch.norm(block_keys, p=1, dim=-1)
    k_polarity = k_l2 / (k_l1 + 1e-8)  # [S, H]

    # 轨道一：Peak Feature Envelope
    # 【图优化友好】依赖 JIT 编译器的自动算子融合 (Auto-Fusion)，不要用 .mul_ 打断计算图
    weighted_v = torch.abs(block_values) * k_polarity.unsqueeze(-1)  # [S, H, D]
    gated_v_maxpool = torch.max(weighted_v, dim=0).values # [H, D]

    # 【保留胜利】因为 gated_v_maxpool 绝对非负，使用原生的 .sum() 替代 .norm(p=1) 达到极速
    envelope_volume = gated_v_maxpool.sum(dim=-1) # [H]

    # 轨道二：Addressability Density
    hook_density = (k_polarity ** kappa).mean(dim=0) # [H]

    # 宏观几何融合
    final_score = ((envelope_volume ** gamma) * (hook_density ** (1.0 - gamma))).mean()
    return final_score.item()

@torch.jit.script
def _jit_dual_track_score_proto(block_keys: torch.Tensor, block_values: torch.Tensor, kappa: float, gamma: float) -> float:
    # 1. 基础极性定义 (恢复使用轻量级的 torch.norm)
    k_l2 = torch.norm(block_keys, p=2, dim=-1)
    k_l1 = torch.norm(block_keys, p=1, dim=-1)
    k_polarity = k_l2 / (k_l1 + 1e-8)  # [S, H]

    # 轨道一：Peak Feature Envelope
    v_abs = torch.abs(block_values)    # [S, H, D]
    weighted_v = v_abs * k_polarity.unsqueeze(-1)  # [S, H, D]
    gated_v_maxpool = torch.max(weighted_v, dim=0).values # [H, D]
    envelope_volume = torch.norm(gated_v_maxpool, p=1, dim=-1) # [H]

    # 轨道二：Addressability Density
    hook_density = (k_polarity ** kappa).mean(dim=0) # [H]

    # 宏观几何融合
    final_score = ((envelope_volume ** gamma) * (hook_density ** (1.0 - gamma))).mean()
    return final_score.item()

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
        # [性能优化]: 在初始化阶段缓存环境变量，避免热点循环中的系统调用
        self.profile_kappa = float(os.environ.get("PROFILE_KAPPA", "1.1"))
        self.profile_gamma = float(os.environ.get("PROFILE_GAMMA", "0.1"))
        
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

    def _profile_sparsity_v1(self, keys, values):
        """
        [方案：无魔数显式动能门控标准版 (Cobb-Douglas 归一化融合)]
        已通过 @torch.jit.script 静态编译，利用 abs().sum() 数学等价替换极速计算。
        """
        seq_len = keys.shape[0]
        budget = self.cache_budget
        c_ratio = min(1.0, float(budget) / max(seq_len, 1))
        # 将环境变量读取移至外部或使用类缓存 (建议后续将其移至 __init__ 中)
        beta = float(os.environ.get("PROFILE_KAPPA_TOKEN", "1.0"))
        gamma = 1.0 - c_ratio
        kappa = c_ratio * beta
        # Hoyer 稀疏度由于内部涉及较多独立逻辑，保持原函数调用
        k_hoyer = self.compute_hoyer_sparsity(keys) # [S, H]
        # 移交 JIT 引擎执行
        return _jit_profile_sparsity_v1(keys, values, k_hoyer, gamma, kappa)

    def _profile_sparsity_v2(self, keys, values):
        """
        [方案：纯净 L1 正交同构]
        已通过 @torch.jit.script 静态编译，纯净无依赖极速版。
        """
        seq_len = keys.shape[0]
        budget = self.cache_budget
        c_ratio = min(1.0, float(budget) / max(seq_len, 1))
        base_param = float(os.environ.get("PROFILE_KAPPA_TOKEN", "1.0"))
        # 按照您的原始代码中最终生效的赋值逻辑
        gamma = base_param * (1.0 - c_ratio)
        kappa = base_param * c_ratio
        # 移交 JIT 引擎执行
        return _jit_profile_sparsity_v2(keys, values, gamma, kappa)

    def _profile_sparsity_v3(self, keys, values):
        """版本 v3: 待调参或修改逻辑的预留位置"""
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
                else:
                    raise ValueError(f"Unknown PROFILE_VER: {profile_sparsity_version}")
        elif evict_method == "inverse_key_l2":
            # use inverse key_l2 norm for inverse_key_l2 eviction
            scores = self.inverse_key_l2(keys, values)
        else:
            raise ValueError(f"Unknown eviction method: {evict_method}")
        return scores

    def _profile_maxpool_v1(self, block_keys, block_values):
        """
        [提纯版双轨制：峰值特征包络 × 检索连通密度]
        逻辑极其清晰，专为学术论文公式对应而设计。完美复现 Lab 3.2 优势，消灭 broadcasting 隐患。
        已通过 @torch.jit.script 静态编译优化吞吐性能。
        """
        return _jit_dual_track_score(block_keys, block_values, self.profile_kappa, self.profile_gamma)
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
        kappa = float(os.environ.get("PROFILE_KAPPA", "1.1"))
        hook_density = (k_polarity ** kappa).mean(dim=0) # [H]
        # 宏观几何融合 (Geometric Fusion)
        gamma = float(os.environ.get("PROFILE_GAMMA", "0.1"))
        # 两个指标均在 [H] 维度，几何加权后求均值，物理意义绝对严谨
        final_score = ((envelope_volume ** gamma) * (hook_density ** (1.0 - gamma))).mean().item()
        return final_score

    def _batched_profile_maxpool_v1(self, batched_keys, batched_values):
        """
        [Macro Stage 性能优化] 完全消除 for 循环的批量打分算子。
        batched_keys / values shape: [num_blocks, block_size, num_heads, head_dim]
        返回: [num_blocks] 形状的 Tensor (仍在 GPU 上)
        """
        # 1. 基础极性定义 (Batched)
        # dim=-1 依然是 head_dim，求范数后 shape: [num_blocks, block_size, num_heads]
        k_l2 = torch.linalg.vector_norm(batched_keys, ord=2, dim=-1)
        k_l1 = torch.linalg.vector_norm(batched_keys, ord=1, dim=-1)
        k_polarity = k_l2 / (k_l1 + 1e-8)
        # 轨道一：Peak Feature Envelope (峰值特征包络)
        v_abs = torch.abs(batched_values)  # [num_blocks, block_size, num_heads, head_dim]
        # unsqueeze(-1) 让 polarity shape 变为 [..., num_heads, 1] 以便广播
        weighted_v = v_abs * k_polarity.unsqueeze(-1)
        # 沿着 Token 维度 (dim=1，即 block_size 维度) 求 Max
        # gated_v_maxpool shape: [num_blocks, num_heads, head_dim]
        gated_v_maxpool = torch.max(weighted_v, dim=1).values
        # envelope_volume shape: [num_blocks, num_heads]
        envelope_volume = torch.linalg.vector_norm(gated_v_maxpool, ord=1, dim=-1)
        # 轨道二：Addressability Density (检索连通密度)
        kappa = float(os.environ.get("PROFILE_KAPPA", "1.1"))
        # 沿着 Token 维度 (dim=1) 求 Mean
        # hook_density shape: [num_blocks, num_heads]
        hook_density = (k_polarity ** kappa).mean(dim=1)
        # 宏观几何融合 (Geometric Fusion)
        gamma = float(os.environ.get("PROFILE_GAMMA", "0.1"))
        # merged_scores shape: [num_blocks, num_heads]
        merged_scores = (envelope_volume ** gamma) * (hook_density ** (1.0 - gamma))
        # 沿着 Head 维度 (dim=1) 求 Mean
        # 最终 final_batched_scores shape: [num_blocks]
        final_batched_scores = merged_scores.mean(dim=1)
        return final_batched_scores # 注意：这里不加 .item()，返回 GPU Tensor

    def _profile_maxpool_v2(self, block_keys, block_values):
        """版本 v2: 待调参或修改逻辑的预留位置"""
        pass

    def _profile_maxpool_v3(self, block_keys, block_values):
        """版本 v3: 待调参或修改逻辑的预留位置"""
        pass

    def _profile_maxpool_vRich(self, block_keys, block_values):
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
            profile_maxpool_version = os.environ.get("PROFILE_MAXPOOL_VER", "1")
            if profile_maxpool_version == "1":
                scores = self._profile_maxpool_v1(block_keys, block_values)
            elif profile_maxpool_version == "2":
                scores = self._profile_maxpool_v2(block_keys, block_values)
            elif profile_maxpool_version == "3":
                scores = self._profile_maxpool_v3(block_keys, block_values)
            else:
                raise ValueError(f"Unknown PROFILE_VER: {profile_maxpool_version}")
            return scores

    def get_batched_block_score_with_simhash(self, seq_keys, seq_values, evict_method, block_size):
        """
        [Macro Stage - 高性能优化版] 针对 Prefill 阶段，
        一把梭计算所有 Block 的 SimHash，彻底消除 .item() 造成的 CPU-GPU Sync 阻塞。
        """
        num_tokens, num_heads, head_dim = seq_values.shape
        num_blocks = num_tokens // block_size
        if num_blocks == 0:
            return [], []
        # 1. 维度重组：[num_blocks, block_size, num_heads, head_dim]
        # batched_keys = seq_keys.view(num_blocks, block_size, num_heads, head_dim)
        # batched_values = seq_values.view(num_blocks, block_size, num_heads, head_dim)
        valid_tokens = num_blocks * block_size
        batched_keys = seq_keys[:valid_tokens].view(num_blocks, block_size, num_heads, head_dim)
        batched_values = seq_values[:valid_tokens].view(num_blocks, block_size, num_heads, head_dim)
        # 2. 批量获取 Meso Score
        # (由于子打分逻辑包含复杂的切片/倒数运算，安全起见仍在此做 Python 循环提取分数)
        # meso_scores = []
        # for idx in range(num_blocks):
        #     meso_scores.append(self.get_block_score(batched_keys[idx], batched_values[idx], evict_method))
        if self.sub_evict_method == "profile_richness":
            batched_meso_scores = self._batched_profile_maxpool_v1(batched_keys, batched_values)
        else:
            raise NotImplementedError("Please implement batched math for your specific evict_method")
        # 3. [核心优化] 批量计算 V_profile -> [num_blocks, num_heads, head_dim]
        v_profile = torch.max(torch.abs(batched_values), dim=1).values
        # 4. 展平: [num_blocks, num_heads * head_dim]
        flattened_v = v_profile.view(num_blocks, -1)
        # 5. 初始化投影矩阵
        if self.rp_matrix is None:
            feature_dim = flattened_v.shape[1]
            gen = torch.Generator(device=seq_values.device)
            gen.manual_seed(42)
            self.rp_matrix = torch.randn(feature_dim, 63, device=seq_values.device, dtype=seq_values.dtype, generator=gen)
            self.powers_of_two = (2 ** torch.arange(62, -1, -1, device=seq_values.device)).to(torch.int64)
        # 6. 极致并行的 GPU 矩阵乘法
        # projected shape: [num_blocks, 63]
        projected = torch.matmul(flattened_v, self.rp_matrix)
        bits = (projected > 0).to(torch.int64)
        # 7. GPU 并行二进制转 64-bit int: [num_blocks]
        simhash_vals = torch.sum(bits * self.powers_of_two, dim=1)
        # 8. 统一拉回 CPU，整个过程仅触发 1 次 CUDA Sync！
        # return meso_scores, simhash_vals.cpu().tolist()
        return batched_meso_scores.cpu().tolist(), simhash_vals.cpu().tolist()

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
            remainder_size = q_len % self.orig_block_size
            # 物理保护区：由 initial_blocks 决定，而非硬编码的 orig_block_size
            protected_len = self.initial_blocks * self.orig_block_size
            end_idx_first_slice = protected_len
            first_slice = slice(0, end_idx_first_slice)
            # 切片：尾部 (严格与 get_pruned_length 保持一致)
            budget_blocks = self.cache_budget // self.orig_block_size
            protected_tail_blocks = max(1, budget_blocks // 4)
            tail_protect_len = protected_tail_blocks * self.orig_block_size + remainder_size
            # 切片：中间段
            end_idx_middle_slice = max(end_idx_first_slice, q_len - tail_protect_len)
            middle_slice = slice(end_idx_first_slice, end_idx_middle_slice)
            last_slice = slice(end_idx_middle_slice, q_len)
            
            # Extract middle tokens
            middle_key = key_tensor[middle_slice, :, :]
            middle_value = value_tensor[middle_slice, :, :]
            middle_tokens = middle_key.shape[0]
            # 减去实际的保护区长度 Budget = Protected_Head + Middle_Survivor + Protected_Tail 
            middle_allowance = self.cache_budget - end_idx_first_slice - (protected_tail_blocks * self.orig_block_size)
            middle_allowance = max(0, middle_allowance)
            middle_pruned_tokens = self.cache_budget - protected_len - self.orig_block_size
            total_prune_tokens = middle_tokens - middle_pruned_tokens 
            total_prune_tokens = middle_tokens - middle_allowance
            total_prune_tokens = 0 if total_prune_tokens < 0 else total_prune_tokens

            if total_prune_tokens > 0:
                # Get scores for middle tokens
                # raw_scores Shape: [Seq_Len, Num_Heads]
                raw_scores = self.get_token_score(middle_key, middle_value, self.evict_method)
                # 聚合Headscore for Token一致性, when value_l2，求和代表在所有 Head 上的总能量
                scores = raw_scores.sum(dim=1)

                # Evict least N elements, select the indices along dim=0
                _, least_indices = torch.topk(scores, k=total_prune_tokens, largest=False, dim=0)

                # Create a mask for pruning
                # 初始化为 True (保留)，将 least_indices 设为 False (驱逐)
                mask = torch.ones(middle_tokens, dtype=torch.bool, device=middle_key.device)
                mask[least_indices] = False

                # Get pruned middle tensors
                pruned_middle_key = middle_key[mask, :, :]
                pruned_middle_value = middle_value[mask, :, :]
                # print(f"total_prune_tokens={total_prune_tokens} middle_keys={middle_tokens} pruned_middle_key.shape = {pruned_middle_key.shape}, middle_key.shape = {middle_key.shape}")

                # [Assertion Check]
                assert mask.dim() == 1, \
                        f"Head Sync Failed: Mask should be 1D [Seq], but got {mask.shape}"
                # check seq size
                expected_survivors = middle_tokens - total_prune_tokens
                assert pruned_middle_key.shape[0] == expected_survivors, \
                        f"Pruning Count Mismatch: Expected {expected_survivors}, Got {pruned_middle_key.shape[0]}"
                # check num_heads
                assert pruned_middle_key.shape[1] == num_heads, \
                        f"Head Dimension Corrupted: Expected {num_heads}, Got {pruned_middle_key.shape[1]}"
            else:
                # No pruning needed, keep the entire middle slice
                pruned_middle_key = middle_key
                pruned_middle_value = middle_value
            
            # # Calculate sizes
            # first_size = self.orig_block_size
            # pruned_middle_size = pruned_middle_key.shape[0]
            # last_size = q_len - end_idx_middle_slice
            
            # Single concatenation operation (most efficient)
            combined_key = torch.cat([
                key_tensor[first_slice, :, :],
                pruned_middle_key,
                key_tensor[last_slice, :, :]
            ], dim=0)
            
            combined_value = torch.cat([
                value_tensor[first_slice, :, :],
                pruned_middle_value,
                value_tensor[last_slice, :, :]
            ], dim=0)
            
            # print(f"qlen={q_len}, combined_key.shape = {combined_key.shape}, combined_value.shape = {combined_value.shape}, output_key_tensor.shape= {output_key_tensor.shape}, output_value_tensor.shape = {output_value_tensor.shape}") 
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
            protected_len = self.initial_blocks * self.orig_block_size
            end_idx_first_slice = protected_len
            num_first_slice_tokens = end_idx_first_slice
            # 动态保护区：尾部 25% + 当前未满的 Remainder
            budget_blocks = self.cache_budget // self.orig_block_size
            protected_tail_blocks = max(1, budget_blocks // 4)
            tail_protect_len = protected_tail_blocks * self.orig_block_size + remainder_size
            end_idx_middle_slice = seq_len - self.orig_block_size - remainder_size
            # num_middle_tokens = end_idx_middle_slice - end_idx_first_slice
            num_last_slice_tokens = seq_len - end_idx_middle_slice
            middle_allowance = self.cache_budget - num_first_slice_tokens - (protected_tail_blocks * self.orig_block_size)
            middle_allowance = max(0, middle_allowance)
            middle_unpruned_tokens = self.cache_budget - num_first_slice_tokens - self.orig_block_size
            middle_unpruned_tokens = max(middle_unpruned_tokens, 0)
            total_unpruned_tokens = num_first_slice_tokens + middle_unpruned_tokens + num_last_slice_tokens
            total_unpruned_tokens = num_first_slice_tokens + middle_allowance + tail_protect_len

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

