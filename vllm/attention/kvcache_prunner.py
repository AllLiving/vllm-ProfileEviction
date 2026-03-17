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
            # Input: block_values [num_tokens, num_heads, head_dim]
            # 这里的 num_tokens 被视为属于同一个 Block (Scope)

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
            
            end_idx_middle_slice = q_len - self.orig_block_size - remainder_size
            middle_slice = slice(end_idx_first_slice, end_idx_middle_slice)

            last_slice = slice(end_idx_middle_slice, q_len)
            
            # Extract middle tokens
            middle_key = key_tensor[middle_slice, :, :]
            middle_value = value_tensor[middle_slice, :, :]
            middle_tokens = middle_key.shape[0]
            # 减去实际的保护区长度 Budget = Protected_Head + Middle_Survivor + Protected_Tail 
            middle_pruned_tokens = self.cache_budget - protected_len - self.orig_block_size
            total_prune_tokens = middle_tokens - middle_pruned_tokens 
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
            end_idx_middle_slice = seq_len - self.orig_block_size - remainder_size
            # num_middle_tokens = end_idx_middle_slice - end_idx_first_slice
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

