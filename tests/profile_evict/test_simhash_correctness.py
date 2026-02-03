import torch
import torch.nn as nn
import time

class KVCacheProfiler(nn.Module):
    def __init__(self, head_dim: int = 128, hash_bits: int = 64, device: str = "cuda"):
        """
        初始化 Profiler。
        关键点：
        1. 预分配随机投影矩阵 (Random Projection Matrix)，避免推理时 malloc。
        2. 预计算 2 的幂次向量，用于将 bool 向量快速转换为 int64 指纹。
        """
        super().__init__()
        self.head_dim = head_dim
        self.hash_bits = hash_bits
        self.device = device
        
        # 1. 随机投影矩阵：用于 SimHash (LSH)
        # Shape: [Head_Dim, 64]
        # 使用 normal 分布以保证各向同性
        self.register_buffer(
            "proj_matrix", 
            torch.randn(head_dim, hash_bits, device=device, dtype=torch.float32)
        )
        
        # 2. 幂次向量：用于将 64位 bool 映射为 int64
        # Shape: [64] -> [1, 2, 4, ..., 2^63]
        powers = 2 ** torch.arange(hash_bits, device=device, dtype=torch.int64)
        self.register_buffer("powers_of_two", powers)

    def hoyer_sparsity(self, x: torch.Tensor) -> torch.Tensor:
        """
        算子 1: Hoyer Sparsity Calculation
        Input:  [Batch, Block_Size, Head_Dim]
        Output: [Batch, Block_Size] (每个 Token 的稀疏度打分)
        Formula: Hoyer = L1 / L2 (简化版)
        """
        # 加上 1e-6 防止除以零
        l1 = torch.norm(x, p=1, dim=-1)
        l2 = torch.norm(x, p=2, dim=-1) + 1e-6
        return l1 / l2

    def meso_aggregation(self, x: torch.Tensor) -> torch.Tensor:
        """
        算子 2: Meso-level Aggregation (Max Pooling)
        逻辑：沿 Sequence (Block_Size) 维度取最大值，提取该 Block 的轮廓。
        Input:  [Batch, Block_Size, Head_Dim]
        Output: [Batch, Head_Dim] (Block_Profile)
        """
        # dim=1 是 Block_Size 维度
        profile, _ = torch.max(x, dim=1)
        return profile

    def simhash_fingerprint(self, profile: torch.Tensor) -> torch.Tensor:
        """
        算子 3: SimHash Fingerprinting (Vectorized)
        逻辑：投影 -> 二值化 -> 压缩为 Int64
        Input:  [Batch, Head_Dim] (Block_Profile)
        Output: [Batch] (Int64 Hash)
        """
        # 1. Random Projection
        # [Batch, Dim] @ [Dim, 64] -> [Batch, 64]
        projected = torch.matmul(profile, self.proj_matrix)
        
        # 2. Binarization (Sign bit)
        # > 0 为 1，<= 0 为 0。转换为 int64 以便后续计算
        bits = (projected > 0).to(torch.int64) # [Batch, 64]
        
        # 3. Packing Bits into Int64 (No loops, pure vectorization)
        # 利用广播机制: [Batch, 64] * [64] -> [Batch, 64] -> sum -> [Batch]
        # 这是一个极其高效的位压缩操作
        fingerprints = (bits * self.powers_of_two).sum(dim=-1)
        
        return fingerprints

    def forward(self, x: torch.Tensor):
        """
        端到端计算流
        """
        # 1. 计算稀疏度 (用于后续可能的 Eviction 策略，此处仅做计算)
        _ = self.hoyer_sparsity(x)
        
        # 2. 聚合 Block 特征
        profile = self.meso_aggregation(x)
        
        # 3. 生成指纹
        fingerprint = self.simhash_fingerprint(profile)
        
        return fingerprint

# ==========================================
# Unit Test & Benchmark Utilities
# ==========================================

def run_test():
    print(f"{'='*20} Stage 1: Unit Testing {'='*20}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing on Device: {device}")
    
    # 配置
    B, L, D = 16, 16, 128  # Batch=16, Block_Size=16, Head_Dim=128
    profiler = KVCacheProfiler(head_dim=D, device=device)
    
    # --- Case 1: 鲁棒性测试 (Robustness) ---
    print("\n[Test 1] Robustness Check (Block A vs Block A + Noise)")
    
    # Block A: 标准特征块
    block_a = torch.randn(B, L, D, device=device)
    
    # Block A': 叠加微小噪声 (模拟量化误差或浮点漂移)
    # 噪声幅度控制在 1e-3 级别，Max Pooling 应能过滤掉大部分波动，SimHash 保持一致
    # noise = torch.randn_like(block_a) * 1e-3
    noise = torch.randn_like(block_a) * 1e-2
    block_a_prime = block_a + noise
    
    fp_a = profiler(block_a)
    fp_a_prime = profiler(block_a_prime)
    
    # 验证是否全等
    is_identical = torch.equal(fp_a, fp_a_prime)
    if is_identical:
        print(f"✅ PASS: Fingerprints match perfectly despite noise.")
    else:
        # mismatch_count = (fp_a != fp_a_prime).sum().item()
        mismatch_idx = (fp_a != fp_a_prime).nonzero(as_tuple=True)[0]
        count = len(mismatch_idx)
        first_fail_idx = mismatch_idx[0].item()
        print(f"❌ FAIL: {count} blocks have different fingerprints.")
        # Debug info
        print(f"Sample A : {fp_a[first_fail_idx].item():X}")
        print(f"Sample A': {fp_a_prime[first_fail_idx].item():X}")

    # --- Case 2: 区分度测试 (Discrimination) ---
    print("\n[Test 2] Discrimination Check (Block A vs Block B)")
    
    # Block B: 完全不同的数据
    block_b = torch.randn(B, L, D, device=device)
    fp_b = profiler(block_b)
    
    # 理论上两个随机块生成相同 64-bit 指纹的概率极低
    is_different = not torch.equal(fp_a, fp_b)
    if is_different:
        print(f"✅ PASS: Different blocks generated different fingerprints.")
    else:
        print(f"❌ FAIL: Collision detected (Extremely rare for 64-bit).")

def run_benchmark():
    print(f"\n{'='*20} Stage 2: Performance Benchmark {'='*20}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 模拟真实负载
    # 1000 个 Blocks，对应较大的 Batch Size
    # Batch=1000, Block_Size=16, Head_Dim=128
    # 数据类型设为 float16 (vLLM 常用精度)
    BATCH_SIZE = 1000
    L, D = 16, 128
    dtype = torch.float16
    
    profiler = KVCacheProfiler(head_dim=D, device=device).to(dtype)
    # 注意：powers_of_two 是 int64，不能转为 fp16，需要保持原有类型，Module转换时需注意
    # 重新注册一下 int64 的 buffer 以防 .to() 误伤
    profiler.powers_of_two = profiler.powers_of_two.to(torch.int64) 
    # 投影矩阵需要是 fp16
    profiler.proj_matrix = profiler.proj_matrix.to(dtype)

    input_data = torch.randn(BATCH_SIZE, L, D, device=device, dtype=dtype)
    
    # Warmup
    for _ in range(100):
        _ = profiler(input_data)
    torch.cuda.synchronize()
    
    # Timing
    iterations = 1000
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(iterations):
        _ = profiler(input_data)
    end_event.record()
    
    torch.cuda.synchronize()
    
    total_time_ms = start_event.elapsed_time(end_event) # milliseconds
    avg_time_per_iter_us = (total_time_ms / iterations) * 1000
    avg_time_per_block_us = avg_time_per_iter_us / BATCH_SIZE
    
    print(f"Input Shape: [{BATCH_SIZE}, {L}, {D}]")
    print(f"Precision: {dtype}")
    print(f"Total Latency (1000 blocks): {avg_time_per_iter_us:.2f} us")
    print(f"Per-Block Latency: {avg_time_per_block_us:.4f} us (Target: < 10 us)")
    
    if avg_time_per_block_us < 10:
        print("✅ PERFORMANCE TARGET MET (FAST)")
    else:
        print("⚠️ PERFORMANCE TARGET MISSED (OPTIMIZE NEEDED)")

if __name__ == "__main__":
    # 设置随机种子保证可复现性
    torch.manual_seed(42)
    
    run_test()
    run_benchmark()
