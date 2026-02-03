import hashlib
from vllm.config import PagedEvictConfig, CacheConfig

def test_hash_sensitivity():
    print("------ 开始验证 Hash 敏感性 ------")

    # 1. 创建两个只有 cache_budget 不同的 PagedEvictConfig
    evict_config_1 = PagedEvictConfig(
        disable_evict_prefill=False,
        cache_prune_type="budget",
        evict_method="inverse_key_l2",
        cache_budget=1024,  # <--- 差异点 A
        topk_blocks=3
    )
    
    evict_config_2 = PagedEvictConfig(
        disable_evict_prefill=False,
        cache_prune_type="budget",
        evict_method="inverse_key_l2",
        cache_budget=2048,  # <--- 差异点 B
        topk_blocks=3
    )

    # 2. 计算 PagedEvictConfig 的 Hash
    hash_1 = evict_config_1.compute_hash()
    hash_2 = evict_config_2.compute_hash()

    print(f"PagedEvict Config 1 (Budget 1024) Hash: {hash_1}")
    print(f"PagedEvict Config 2 (Budget 2048) Hash: {hash_2}")

    if hash_1 == hash_2:
        print("\n❌ [失败] PagedEvictConfig Hash 未发生变化！")
        print("原因：compute_hash 方法可能未正确包含 cache_budget 参数。")
        return
    else:
        print("✅ [成功] PagedEvictConfig 能够感知参数变化。")

    # 3. 验证 CacheConfig 是否能感知到变化 (集成测试)
    print("\n------ 验证 CacheConfig 集成 ------")
    
    # 创建两个 CacheConfig，分别嵌入上面的 evict_config
    cache_config_1 = CacheConfig(
        block_size=16, gpu_memory_utilization=0.9, swap_space=4, cache_dtype="auto",
        paged_evict_config=evict_config_1  # 嵌入配置 A
    )
    
    cache_config_2 = CacheConfig(
        block_size=16, gpu_memory_utilization=0.9, swap_space=4, cache_dtype="auto",
        paged_evict_config=evict_config_2  # 嵌入配置 B
    )

    cache_hash_1 = cache_config_1.compute_hash()
    cache_hash_2 = cache_config_2.compute_hash()

    print(f"CacheConfig 1 Hash: {cache_hash_1}")
    print(f"CacheConfig 2 Hash: {cache_hash_2}")

    if cache_hash_1 == cache_hash_2:
        print("\n❌ [失败] CacheConfig Hash 未发生变化！")
        print("原因：CacheConfig.compute_hash 可能忘了调用 self.paged_evict_config.compute_hash()。")
        print("后果：vLLM 可能会错误地复用旧的计算图，导致新参数不生效。")
    else:
        print("\n✅ [完美] CacheConfig Hash 已更新。")
        print("结论：vLLM 将会把这视为两个完全不同的运行环境，强制重新初始化。修改有效！")

if __name__ == "__main__":
    test_hash_sensitivity()
