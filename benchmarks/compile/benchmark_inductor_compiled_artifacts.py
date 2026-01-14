#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark script to measure inductor cache performance across multiple models.

This script benchmarks the vLLM inductor cache feature by running models twice:
1. Cold Start: First run compiles from scratch and saves artifacts to cache
2. Warm Start: Second run loads precompiled artifacts from cache

The script measures load time and inference performance for both runs and computes
the speedup achieved by using cached artifacts.

Usage:
    # Run all models with default settings
    python benchmark_inductor_compiled_artifacts.py

    # Run specific models by index
    python benchmark_inductor_compiled_artifacts.py --models 0 3 4

    # Run specific models by name (partial match)
    python benchmark_inductor_compiled_artifacts.py --models Qwen Llama

    # List available models
    python benchmark_inductor_compiled_artifacts.py --list-models

    # Customize cache directory and output
    python benchmark_inductor_compiled_artifacts.py \
        --cache-dir /path/to/cache --output results.json

    # Adjust generation parameters
    python benchmark_inductor_compiled_artifacts.py --max-tokens 256 --batch-size 4

Requirements:
    - PyTorch 2.10.0+ (for standalone_compile with serialization support)
    - vLLM with VLLM_USE_BACKEND_WITH_INDUCTOR_COMPILED_ARTIFACTS feature enabled
    - Sufficient GPU memory (8x H100 recommended for largest models)
"""

import argparse
import json
import logging
import os
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a model to benchmark."""

    name: str
    tensor_parallel_size: int

    def __str__(self):
        return f"{self.name} (TP={self.tensor_parallel_size})"


@dataclass
class BenchmarkRunConfig:
    """Configuration for a single benchmark run."""

    key: str  # Key to store results under (e.g., "cold_start", "warm_start")
    label: str  # Display label for logging
    run_number: int  # Run number for logging (1, 2, 3, 4)
    description: str  # Description for logging
    clean_cache: bool  # Whether to clean caches before this run
    setup_env: Callable[[str], None]  # Function to set up environment (takes cache_dir)
    note: str | None = None  # Optional note to log before the run


@dataclass
class RunGroup:
    """A group of related benchmark runs (cold and warm start)."""

    name: str  # Display name for the group (e.g., "Inductor Cache", "Baseline")
    cold_key: str  # Key for cold start results (e.g., "cold_start")
    warm_key: str  # Key for warm start results (e.g., "warm_start")
    runs: list[BenchmarkRunConfig] = field(default_factory=list)


def setup_inductor_cache_env(cache_dir: str) -> None:
    """Set up environment for inductor cache backend (new approach)."""
    os.environ["VLLM_CACHE_ROOT"] = cache_dir
    os.environ["VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE"] = str(512 * 1024 * 1024)
    os.environ["VLLM_USE_BACKEND_WITH_INDUCTOR_COMPILED_ARTIFACTS"] = "1"
    os.environ["VLLM_USE_AOT_COMPILE"] = "1"
    os.environ["VLLM_USE_STANDALONE_COMPILE"] = "1"
    logger.info("Environment: Using inductor cache backend")


def setup_baseline_env(cache_dir: str) -> None:
    """Set up environment for baseline (standalone compile WITHOUT inductor cache)."""
    os.environ["VLLM_CACHE_ROOT"] = cache_dir
    os.environ["VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE"] = str(512 * 1024 * 1024)
    os.environ.pop("VLLM_USE_BACKEND_WITH_INDUCTOR_COMPILED_ARTIFACTS", None)
    os.environ["VLLM_USE_AOT_COMPILE"] = "1"
    os.environ["VLLM_USE_STANDALONE_COMPILE"] = "1"
    logger.info(
        "Environment: Using baseline (standalone compile WITHOUT inductor cache)"
    )


DEFAULT_MODELS = [
    ModelConfig("Qwen/Qwen3-32B", 1),
    ModelConfig("deepseek-ai/DeepSeek-R1-0528", 8),
    ModelConfig("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", 1),
    ModelConfig("meta-llama/Llama-3.3-70B-Instruct", 2),
    ModelConfig("nvidia/Llama-3.3-70B-Instruct-FP8", 2),
    ModelConfig("mistralai/Mistral-Large-Instruct-2411", 2),
]


def run_vllm_inference(
    model: str,
    tensor_parallel_size: int,
    prompt: str,
    max_tokens: int,
    batch_size: int,
    compile_sizes: list[int | str] | None,
    compile_ranges_split_points: list[int] | None,
    enable_compile: bool,
) -> dict[str, float]:
    """Run vLLM inference and measure timing.

    Args:
        model: Model name or path
        tensor_parallel_size: Number of GPUs for tensor parallelism
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        batch_size: Batch size
        compile_sizes: Which batch sizes to compile for
        compile_ranges_split_points: Split points for compile ranges
        enable_compile: Whether to enable torch compile

    Returns:
        Dictionary with timing information

    Note:
        Environment variables (VLLM_CACHE_ROOT, VLLM_USE_AOT_COMPILE, etc.)
        must be set before calling this function.
    """
    os.environ.pop("VLLM_FORCE_AOT_LOAD", None)

    if not enable_compile:
        os.environ.pop("VLLM_USE_BACKEND_WITH_INDUCTOR_COMPILED_ARTIFACTS", None)
        os.environ.pop("VLLM_USE_AOT_COMPILE", None)
        os.environ.pop("VLLM_USE_STANDALONE_COMPILE", None)

    import torch

    from vllm.entrypoints.llm import LLM
    from vllm.sampling_params import SamplingParams

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
    )

    prompts = [prompt] * batch_size

    load_start = time.perf_counter()

    if enable_compile:
        llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=False,
            compilation_config={
                "level": 3,
                "backend": "inductor",
                "compile_sizes": compile_sizes,
                "compile_ranges_split_points": compile_ranges_split_points,
            },
            gpu_memory_utilization=0.95,
            max_model_len=2048,
        )
    else:
        llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=True,
            gpu_memory_utilization=0.95,
            max_model_len=2048,
        )
    load_time = time.perf_counter() - load_start

    logger.info("Running warmup...")
    _ = llm.generate(prompts[:1], sampling_params)

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    logger.info("Running timed inference...")
    inference_start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    torch.cuda.synchronize()
    inference_time = time.perf_counter() - inference_start

    total_tokens = sum(
        len(output.outputs[0].token_ids) if output.outputs else 0 for output in outputs
    )
    tokens_per_second = total_tokens / inference_time if inference_time != 0 else 0.0

    return {
        "load_time": load_time,
        "inference_time": inference_time,
        "total_tokens": total_tokens,
        "tokens_per_second": tokens_per_second,
    }


def format_time(seconds: float) -> str:
    """Format time in seconds to a human-readable string."""
    if seconds < 1:
        return f"{seconds * 1000:.2f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"


def print_model_results(model_name: str, results: dict[str, float], run_type: str = ""):
    """Print results for a single model."""
    prefix = f"  [{run_type}] " if run_type else "  "
    logger.info("%sLoad Time:        %s", prefix, format_time(results["load_time"]))
    logger.info(
        "%sInference Time:   %s", prefix, format_time(results["inference_time"])
    )
    logger.info("%sTotal Tokens:     %.0f", prefix, results["total_tokens"])
    logger.info("%sTokens/Second:    %.2f", prefix, results["tokens_per_second"])
    total_time = format_time(results["load_time"] + results["inference_time"])
    logger.info("%sTotal Time:       %s", prefix, total_time)


def print_summary(
    all_results: dict[str, dict[str, any]],
    run_groups: list[RunGroup],
) -> None:
    """Print summary table of all results."""
    logger.info("")
    logger.info("=" * 120)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 120)

    for model_name in sorted(all_results.keys()):
        model_results = all_results[model_name]
        logger.info("")
        logger.info("Model: %s", model_name)
        logger.info("-" * 80)

        completed_groups = [
            g
            for g in run_groups
            if g.cold_key in model_results and g.warm_key in model_results
        ]

        for group in completed_groups:
            cold = model_results[group.cold_key]
            warm = model_results[group.warm_key]
            load_speedup = (
                cold["load_time"] / warm["load_time"]
                if warm["load_time"] != 0
                else float("inf")
            )
            logger.info(
                "  %s: Cold=%s, Warm=%s (%.2fx speedup)",
                group.name,
                format_time(cold["load_time"]),
                format_time(warm["load_time"]),
                load_speedup,
            )

        for i, group1 in enumerate(completed_groups):
            for group2 in completed_groups[i + 1 :]:
                cold1 = model_results[group1.cold_key]
                cold2 = model_results[group2.cold_key]
                warm1 = model_results[group1.warm_key]
                warm2 = model_results[group2.warm_key]

                cold_speedup = (
                    cold2["load_time"] / cold1["load_time"]
                    if cold1["load_time"] != 0
                    else float("inf")
                )
                warm_speedup = (
                    warm2["load_time"] / warm1["load_time"]
                    if warm1["load_time"] != 0
                    else float("inf")
                )

                cold_time_saved = cold2["load_time"] - cold1["load_time"]
                cold_time_saved_str = (
                    f"+{format_time(cold_time_saved)}"
                    if cold_time_saved >= 0
                    else f"-{format_time(abs(cold_time_saved))}"
                )
                warm_time_saved = warm2["load_time"] - warm1["load_time"]
                warm_time_saved_str = (
                    f"+{format_time(warm_time_saved)}"
                    if warm_time_saved >= 0
                    else f"-{format_time(abs(warm_time_saved))}"
                )
                logger.info(
                    "  %s vs %s: Cold=%.2fx (%s), Warm=%.2fx (%s)",
                    group1.name,
                    group2.name,
                    cold_speedup,
                    cold_time_saved_str,
                    warm_speedup,
                    warm_time_saved_str,
                )

    logger.info("")
    logger.info("=" * 120)


def cleanup_gpu_memory(message: str = "Cleaned up GPU memory"):
    """Clean up GPU memory between runs."""
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    logger.info(message)


def clean_all_caches(cache_dir: str):
    """Clean all compilation caches before running benchmarks.

    Removes:
    1. cache_dir: The entire vLLM cache directory
    2. torchinductor cache: TORCHINDUCTOR_CACHE_DIR or /tmp/torchinductor_{USER}

    Also resets vLLM's environment variable cache to ensure any new
    environment variable settings take effect.
    """
    import getpass

    logger.info("")
    logger.info("=" * 80)
    logger.info("CLEANING ALL CACHES")
    logger.info("=" * 80)

    import vllm.envs as envs

    envs.disable_envs_cache()

    if os.path.exists(cache_dir):
        logger.info("Removing: %s", cache_dir)
        shutil.rmtree(cache_dir)
    else:
        logger.info("Not found: %s", cache_dir)

    torchinductor_cache = os.environ.get(
        "TORCHINDUCTOR_CACHE_DIR",
        f"/tmp/torchinductor_{getpass.getuser()}",
    )
    if os.path.exists(torchinductor_cache):
        logger.info("Removing: %s", torchinductor_cache)
        shutil.rmtree(torchinductor_cache)
    else:
        logger.info("Not found: %s", torchinductor_cache)

    logger.info("=" * 80)
    logger.info("CACHE CLEANING COMPLETE")
    logger.info("=" * 80)


def execute_benchmark_run(
    run_config: BenchmarkRunConfig,
    model_config: ModelConfig,
    cache_dir: str,
    prompt: str,
    max_tokens: int,
    batch_size: int,
    compile_sizes: list[int] | None,
    compile_ranges_split_points: list[int] | None,
) -> dict[str, float]:
    """Execute a single benchmark run with the given configuration.

    Args:
        run_config: Configuration for this benchmark run
        model_config: Model configuration
        cache_dir: Cache directory path
        prompt: Input prompt for generation
        max_tokens: Maximum tokens to generate
        batch_size: Batch size for inference
        compile_sizes: Compile sizes for compilation config
        compile_ranges_split_points: Compile ranges split points

    Returns:
        Dictionary with benchmark results (load_time, inference_time, etc.)
    """
    if run_config.clean_cache:
        clean_all_caches(cache_dir)
        cleanup_gpu_memory("Cleaned up GPU memory after cache cleaning")

    logger.info(
        "--- RUN %d: %s (%s) ---",
        run_config.run_number,
        run_config.label,
        run_config.description,
    )

    if run_config.note:
        logger.info(run_config.note)

    run_config.setup_env(cache_dir)

    results = run_vllm_inference(
        model=model_config.name,
        tensor_parallel_size=model_config.tensor_parallel_size,
        prompt=prompt,
        max_tokens=max_tokens,
        batch_size=batch_size,
        compile_sizes=compile_sizes,
        compile_ranges_split_points=compile_ranges_split_points,
        enable_compile=True,
    )

    print_model_results(model_config.name, results, run_config.label)

    cleanup_gpu_memory("Cleaned up GPU memory after run")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark inductor precompile across multiple models"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help=(
            "Specific models to benchmark (by name or index). "
            "If not specified, runs all models."
        ),
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Tell me a story about a robot learning to paint.",
        help="Input prompt for generation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate (default: 128)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference (default: 1)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=os.path.join(os.path.expanduser("~"), ".cache", "vllm"),
        help="Cache directory for compiled artifacts",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--include-baseline",
        action="store_true",
        help=(
            "Include baseline run with standalone compile "
            "(without inductor cache) for comparison"
        ),
    )
    parser.add_argument(
        "--compile-sizes",
        type=int,
        nargs="+",
        default=None,
        help=("Batch sizes to compile ahead of time."),
    )
    parser.add_argument(
        "--compile-ranges-split-points",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Split points for compile ranges. Creates ranges like "
            "[1, split1), [split1, split2), ..., [splitN, max]. "
            "Example: --compile-ranges-split-points 256 512 1024"
        ),
    )
    args = parser.parse_args()
    if args.list_models:
        logger.info("Available models:")
        for i, model_config in enumerate(DEFAULT_MODELS):
            logger.info("  %d: %s", i, model_config)
        return

    models_to_run = []
    if args.models:
        for model_spec in args.models:
            try:
                idx = int(model_spec)
                if 0 <= idx < len(DEFAULT_MODELS):
                    models_to_run.append(DEFAULT_MODELS[idx])
                else:
                    max_idx = len(DEFAULT_MODELS) - 1
                    logger.warning(
                        "Model index %d out of range (0-%d), skipping", idx, max_idx
                    )
                    continue
            except ValueError:
                found = False
                for config in DEFAULT_MODELS:
                    if model_spec in config.name:
                        models_to_run.append(config)
                        found = True
                        break
                if not found:
                    models_to_run.append(ModelConfig(model_spec, 1))
    else:
        models_to_run = DEFAULT_MODELS

    cache_dir = args.cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    logger.info("Using cache directory: %s", cache_dir)

    all_results = {}

    inductor_group = RunGroup(
        name="Inductor",
        cold_key="cold_start",
        warm_key="warm_start",
        runs=[
            BenchmarkRunConfig(
                key="cold_start",
                label="COLD START",
                run_number=1,
                description="compiling from scratch",
                clean_cache=True,
                setup_env=setup_inductor_cache_env,
            ),
            BenchmarkRunConfig(
                key="warm_start",
                label="WARM START",
                run_number=2,
                description="loading from cache",
                clean_cache=False,
                setup_env=setup_inductor_cache_env,
            ),
        ],
    )

    baseline_group = RunGroup(
        name="Baseline",
        cold_key="baseline_miss",
        warm_key="baseline_hit",
        runs=[
            BenchmarkRunConfig(
                key="baseline_miss",
                label="BASELINE COLD START",
                run_number=3,
                description="first AOT compile",
                clean_cache=True,
                setup_env=setup_baseline_env,
                note=(
                    "Note: This uses AOT compile but WITHOUT the inductor "
                    "cache backend feature"
                ),
            ),
            BenchmarkRunConfig(
                key="baseline_hit",
                label="BASELINE WARM START",
                run_number=4,
                description="loading from AOT cache",
                clean_cache=False,
                setup_env=setup_baseline_env,
            ),
        ],
    )

    run_groups = [inductor_group]
    if args.include_baseline:
        run_groups.append(baseline_group)

    for model_config in models_to_run:
        logger.info("")
        logger.info("=" * 100)
        logger.info("Benchmarking: %s", model_config)
        logger.info("=" * 100)

        model_results = {}

        try:
            for group in run_groups:
                try:
                    for run_config in group.runs:
                        results = execute_benchmark_run(
                            run_config=run_config,
                            model_config=model_config,
                            cache_dir=cache_dir,
                            prompt=args.prompt,
                            max_tokens=args.max_tokens,
                            batch_size=args.batch_size,
                            compile_sizes=args.compile_sizes,
                            compile_ranges_split_points=args.compile_ranges_split_points,
                        )
                        model_results[run_config.key] = results
                except Exception:
                    logger.warning(
                        "%s runs failed for %s, continuing with partial results",
                        group.name,
                        model_config.name,
                        exc_info=True,
                    )

            all_results[model_config.name] = model_results

        except Exception:
            logger.exception(
                "Failed to benchmark %s",
                model_config.name,
            )
            continue
        finally:
            cleanup_gpu_memory("Cleaned up GPU memory")

    if all_results:
        print_summary(all_results, run_groups)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(all_results, f, indent=2)
            logger.info("Results saved to %s", args.output)
    else:
        logger.info("No successful benchmarks to report.")


if __name__ == "__main__":
    main()
