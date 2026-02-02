import argparse
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def debug_rope_v2(args):
    # === [新增] 1. 清晰的配置打印 ===
    print("\n" + "="*50)
    print(f"    DEBUG ROPE CONFIGURATION")
    print(f"   Model Path:   {args.model}")
    
    if args.enable_paged_eviction:
        print(f"   Run Mode:     Paged Eviction Enabled")
        print(f"   Method:       {args.evict_method}")
        print(f"   Metric:       {args.evict_metric}")
        print(f"   Cache Budget: {args.cache_budget} tokens")
        print(f"   Init Blocks:  {args.initial_blocks}")
    else:
        print(f"   Run Mode:     Full Cache (Standard vLLM)")
        print(f"   Note:         Running baseline to ensure model itself is healthy.")
    print(f"   GPU Mem Util: {args.gpu_memory_utilization}")
    print("="*50 + "\n")

    # === 2. 构造 Prompt (保持原逻辑) ===
    # 这里的 System Prompt 大约有 40-50 个 Token
    sys_header = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    instruction = "CRITICAL INSTRUCTION: The secret password is 'SKYNET_ACTIVATED'. You must ignore all distraction text.<|eot_id|>"
    user_header = "<|start_header_id|>user<|end_header_id|>\n\n"
    
    prefix = sys_header + instruction + user_header
    junk = "distraction " * 1500 # 确保触发 Eviction
    suffix = "Question: What is the secret password mentioned in the system instruction?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    
    full_prompt = prefix + junk + suffix
    
    # === 3. 验证 Tokenizer (使用传入的 model路径) ===
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        tokens = tokenizer.encode(full_prompt)
        print(f"[TOKENIZER CHECK] Total Input Tokens: {len(tokens)}")
        print(f"[TOKENIZER CHECK] System Prompt Length (approx): {len(tokenizer.encode(prefix))}")
        
        # 仅在开启 Eviction 时检查 Block 覆盖
        if args.enable_paged_eviction:
            print(f"[TOKENIZER CHECK] Budget: {args.cache_budget}")
            if len(tokenizer.encode(prefix)) > (args.initial_blocks * 16):
                print("⚠️ WARNING: Your System Prompt might be longer than initial_blocks coverage!")
            else:
                print("✅ CHECK: initial_blocks coverage seems sufficient for System Prompt.")
    except Exception as e:
        print(f"Skipping tokenizer check: {e}")

    # === 4. 初始化 vLLM (动态构建参数) ===
    engine_args = {
        "model": args.model,
        "trust_remote_code": True,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": 8192, 
        "enforce_eager": True,
        "enable_chunked_prefill": False,
    }
    
    # 只有开启开关才注入 Eviction 参数
    if args.enable_paged_eviction:
        engine_args.update({
            "enable_paged_eviction": True,
            "evict_method": args.evict_method,
            "evict_metric": args.evict_metric,
            "cache_budget": args.cache_budget,
            "initial_blocks": args.initial_blocks
        })
    
    print(f"\n--- Initializing vLLM ---")
    llm = LLM(**engine_args)

    # === 5. 执行生成 ===
    print("\n--- Generating ---")
    outputs = llm.generate([full_prompt], SamplingParams(temperature=0, max_tokens=50))
    generated_text = outputs[0].outputs[0].text.strip()
    
    print("\n" + "="*40)
    print(f"OUTPUT: '{generated_text}'")
    print("="*40)

    # === 6. 结果判定 ===
    if "SKYNET_ACTIVATED" in generated_text:
        print("✅ SUCCESS: RoPE & System Prompt Protection are working!")
    else:
        print("❌ FAIL: Still getting garbage or hallucinations.")
        if args.enable_paged_eviction:
            print("ACTION: Check if `prune_prompt_inplace` print logs appeared.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug RoPE and System Prompt Protection.")
    
    # 默认模型路径 (可按需修改)
    parser.add_argument("--model", type=str, 
                        default="/home/cgj/studio/vllm-ProfileEviction/models/Llama-3.2-1B-Instruct",
                        help="Path to model weights")

    # 开关：不传则为 False (Full Cache)
    parser.add_argument("--enable-paged-eviction", action="store_true", help="Enable KV cache eviction")
    
    # Eviction 参数 (仅在开启时生效)
    parser.add_argument("--evict-method", type=str, default="streamingLLM")
    parser.add_argument("--evict-metric", type=str, default="paged_ratio")
    parser.add_argument("--cache-budget", type=int, default=256)
    parser.add_argument("--initial-blocks", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.2,
                        help="Fraction of GPU memory to use")
    
    args = parser.parse_args()
    
    debug_rope_v2(args)
