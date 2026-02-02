import argparse
import os
import torch
from vllm import LLM, SamplingParams

# 一段约 600 token 的长文本 (用于触发 512 的 budget)
LONG_PROMPT = """
The history of artificial intelligence (AI) began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of modern AI were planted by classical philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols. This work culminated in the invention of the programmable digital computer in the 1940s, a machine based on the abstract essence of mathematical reasoning. This device and the ideas behind it inspired a handful of scientists to begin seriously discussing the possibility of building an electronic brain. 

The field of AI research was founded at a workshop held on the campus of Dartmouth College in summer 1956. Those who attended would become the leaders of AI research for decades. Many of them predicted that a machine as intelligent as a human being would exist in no more than a generation, and they were given millions of dollars to make this vision come true. Eventually, it became obvious that they had grossly underestimated the difficulty of the project. In 1973, in response to the criticism from Sir James Lighthill and ongoing pressure from the US Congress, the U.S. and British governments stopped funding undirected research into artificial intelligence, and the difficult years that followed would later be known as an "AI winter". Seven years later, a visionary initiative by the Japanese Government inspired governments and industry to provide AI with billions of dollars, but by the late 80s the investors became disillusioned and withdrew funding again.

Investment and interest in AI boomed in the first decades of the 21st century when machine learning was successfully applied to many problems in academia and industry due to new methods, the application of powerful computer hardware, and the collection of immense data sets.

Summary of the above text:
"""

def run_test(args):
    print("\n" + "="*50)
    print(f"VALIDATION TEST CONFIGURATION")
    print(f"Model Path:   {args.model}")

    if args.enable_paged_eviction:
        print(f"   Run Mode:     Paged Eviction Enabled")
        print(f"   Method:       {args.evict_method}")
        print(f"   Metric:       {args.evict_metric}")
        print(f"   Cache Budget: {args.cache_budget} tokens")
        print(f"   Init Blocks:  {args.initial_blocks}")
    else:
        print(f"   Run Mode:     Full Cache (Standard vLLM)")
        print(f"   Note:         Eviction is DISABLED. Running with default memory management.")
    print("="*50 + "\n")

    # 1. 初始化 vLLM
    engine_args = {
        "model": args.model,
        "trust_remote_code": True,
        "gpu_memory_utilization": 0.9,
        # "max_model_len": 4096, # 确保模型能吃下长文本
        "max_model_len": 8192, # 确保模型能吃下长文本
        "max_num_batched_tokens": 8192,
        "enforce_eager": True,
        "dtype": "auto",
        "enable_chunked_prefill": False,
    }
    
    if args.enable_paged_eviction:
        engine_args.update({
            "enable_paged_eviction": True,
            "evict_method": args.evict_method,
            "evict_metric": args.evict_metric,
            "cache_budget": args.cache_budget,
            "initial_blocks": args.initial_blocks
        })

    llm = LLM(**engine_args)

    # 2. 生成参数
    sampling_params = SamplingParams(temperature=0.7, max_tokens=100)

    # 3. 运行推理
    # 我们故意让 Prompt 长度 > Cache Budget，强制触发 Prefill 阶段的驱逐
    prompt = LONG_PROMPT * 2 # 复制两次确保足够长 (~1000 tokens)
    # 如果 budget 是 512，这里一定会触发 eviction
    
    outputs = llm.generate([prompt], sampling_params)

    # 4. 打印结果
    for output in outputs:
        generated_text = output.outputs[0].text
        print("\n" + "="*20 + " GENERATED OUTPUT " + "="*20)
        print(generated_text)
        print("="*60 + "\n")
        
        # 简单判定
        if len(generated_text.strip()) == 0 or "game game" in generated_text or "-------" in generated_text:
             print("❌ FAIL: Output looks like garbage/collapse.")
        else:
             print("✅ PASS: Output looks coherent (human check required).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="/home/cgj/studio/vllm-ProfileEviction/models/Llama-3.2-1B-Instruct",
                        help="Path to the model weights")
    parser.add_argument("--enable-paged-eviction", action="store_true", help="Enable KV cache eviction")
    parser.add_argument("--evict-method", type=str, default="global")
    parser.add_argument("--evict-metric", type=str, default="value_l2")
    parser.add_argument("--cache-budget", type=int, default=512) # 设小一点，方便测试
    parser.add_argument("--initial-blocks", type=int, default=4)
    args = parser.parse_args()
    
    run_test(args)
