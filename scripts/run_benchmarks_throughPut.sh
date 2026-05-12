#!/bin/bash

# ==============================================================================
# vLLM Benchmark 自动化测试脚本
# ==============================================================================

# 1. 变量与参数配置
MODELS=(
    "./models/Llama-3.2-3B-Instruct"
    "./models/Meta-Llama-3.1-8B-Instruct"
)
BUDGETS=(256 512 1024 2048 4096)
STANDARD_STRATEGIES=("paged_ratio") # 不包含 profile_richness 和 full-cache

# profile_richness 专属配置：循环测试次数
PROFILE_RUNS=50
PROFILE_RUNS=15

# 输出日志与结果文件
SUMMARY_CSV="benchmark_summary.csv"
LOG_DIR="./benchmark_logs"

mkdir -p "$LOG_DIR"
echo "Model,Strategy,Budget,Total_Tokens_Per_Second" > "$SUMMARY_CSV"

echo "========================================================================"
echo "🚀 启动 vLLM 自动化压测"
echo "涵盖模型: ${#MODELS[@]} 个 | Budget挡位: ${#BUDGETS[@]} 个 | Profile循环: $PROFILE_RUNS 次"
echo "所有报错将被拦截记录，不影响整体进度。放宽心，去忙别的吧！"
echo "========================================================================"

# ==============================================================================
# 2. 核心执行与解析函数
# ==============================================================================
run_benchmark() {
    local model=$1
    local strategy=$2
    local budget=$3
    
    # 临时文件用于精准捕获当前执行的控制台输出
    local temp_log=$(mktemp "$LOG_DIR/temp_run_XXXXXX.log")
    
    echo -e "\n[$(date +'%H:%M:%S')] ➡️  正在运行 | 模型: $(basename "$model") | 策略: $strategy | Budget: $budget"

    # 根据策略动态构建命令
    if [ "$strategy" == "full-cache" ]; then
        CMD="python benchmarks/benchmark_throughput.py \
            --backend vllm \
            --model $model \
            --input-len 1024 \
            --output-len 8192 \
            --num-prompts 64 \
            --gpu-memory-utilization 0.9 \
            --enforce-eager \
            --max-model-len 9300"
    elif [ "$strategy" == "profile_richness" ] || [ "$strategy" == "paged_ratio" ]; then
        CMD="python benchmarks/benchmark_throughput.py \
            --backend vllm \
            --model $model \
            --input-len 1024 \
            --output-len 8192 \
            --num-prompts 64 \
            --gpu-memory-utilization 0.9 \
            --enforce-eager \
            --enable-paged-eviction \
            --disable-evict-prefill \
            --cache-prune-type budget \
            --evict-method global \
            --evict-metric $strategy \
            --cache-budget $budget"
    elif [ "$strategy" == "streamingLLM" ] || [ "$strategy" == "inverse_key_l2" ]; then
        CMD="python benchmarks/benchmark_throughput.py \
            --backend vllm \
            --model $model \
            --input-len 1024 \
            --output-len 8192 \
            --num-prompts 64 \
            --gpu-memory-utilization 0.9 \
            --enforce-eager \
            --enable-paged-eviction \
            --disable-evict-prefill \
            --cache-prune-type budget \
            --evict-method $strategy \
            --cache-budget $budget"
    fi

    # 执行命令：同时输出到控制台(供你偶尔查看)和临时日志(用于解析)，并捕获可能崩溃的错误码(|| true)
    eval "$CMD" 2>&1 | tee "$temp_log" || true

    # 解析 Throughput: 提取 total tokens/s 对应的数值
    local throughput=$(grep "Throughput:" "$temp_log" | awk -F', ' '{print $2}' | awk '{print $1}')

    if [ -z "$throughput" ]; then
        throughput="FAILED_OR_OOM"
        echo "⚠️  [警告] 运行失败或未捕获到吞吐量，已标记为 FAILED_OR_OOM"
    else
        echo "✅  [成功] 解析吞吐量: $throughput total tokens/s"
    fi

    # 保存关键日志，清理临时文件
    cat "$temp_log" >> "$LOG_DIR/all_runs_archive.log"
    rm -f "$temp_log"
    
    # 返回结果给调用者
    echo "$throughput"
}

# ==============================================================================
# 3. 优先执行：常规策略 (Paged Evict) & Full-Cache
# ==============================================================================
for model in "${MODELS[@]}"; do
    echo "========================================================================"
    echo "🟢 开始测试模型 (常规策略段): $(basename "$model")"
    echo "========================================================================"

    # 1. 运行 Full-Cache (被注释掉，按原样保留)
    # echo "full-cache checking..."
    # tp=$(run_benchmark "$model" "full-cache" "N/A")
    # echo "$(basename "$model"),full-cache,N/A,$tp" >> "$SUMMARY_CSV"

    # 2. 运行常规策略 (paged_ratio 等)
    for strategy in "${STANDARD_STRATEGIES[@]}"; do
        echo "[$strategy] checking..."
        for budget in "${BUDGETS[@]}"; do
            echo "at ${budget}..."
            tp=$(run_benchmark "$model" "$strategy" "$budget")
            echo "$(basename "$model"),$strategy,$budget,$tp" >> "$SUMMARY_CSV"
        done
    done
done

# ==============================================================================
# 4. 压轴执行：profile_richness 专属循环矩阵分析 (按次循环交叉测试多模型)
# ==============================================================================
echo "========================================================================"
echo "🎯 开始执行压轴任务: profile_richness 专属稳定性/吞吐量分析"
echo "========================================================================"

# 先为每个模型初始化其专属的 Profile 矩阵 CSV 表头
for model in "${MODELS[@]}"; do
    MODEL_NAME=$(basename "$model")
    PROFILE_MATRIX_CSV="profile_matrix_${MODEL_NAME}.csv"
    
    header="Iteration"
    for b in "${BUDGETS[@]}"; do header="${header},Budget_${b}"; done
    echo "$header" > "$PROFILE_MATRIX_CSV"
done

# 最外层按次数循环 (1 到 50 次)
for ((i=1; i<=PROFILE_RUNS; i++)); do
    echo "========================================================================"
    echo "📊 --- 第 $i/$PROFILE_RUNS 次 Profile 循环 ---"
    echo "========================================================================"
    
    # 每次循环内，测试所有模型
    for model in "${MODELS[@]}"; do
        MODEL_NAME=$(basename "$model")
        PROFILE_MATRIX_CSV="profile_matrix_${MODEL_NAME}.csv"
        
        echo "🔄 当前执行模型: $MODEL_NAME"
        row_data="$i"
        
        # 遍历所有 Budget 挡位
        for budget in "${BUDGETS[@]}"; do
            echo "profile is at ${budget}..."
            tp=$(run_benchmark "$model" "profile_richness" "$budget")

            # 记录到当前行数据
            row_data="${row_data},${tp}"

            # 同样存一份到总表中，方便全局对比
            echo "${MODEL_NAME},profile_richness_${i},$budget,$tp" >> "$SUMMARY_CSV"
        done
        
        # 跑完当前模型的所有 Budget 后，将这 1 行数据写入对应的矩阵表中
        echo "$row_data" >> "$PROFILE_MATRIX_CSV"
    done
done

echo "========================================================================"
echo "🎉 所有压测任务已圆满完成！"
echo "总览数据请查看: $SUMMARY_CSV"
echo "Profile专属矩阵请查看: profile_matrix_*.csv"
echo "========================================================================"
