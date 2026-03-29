import re
import sys
import argparse

def parse_and_check_logs(file_path, block_size=16, initial_blocks=4):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ 无法读取文件 {file_path}: {e}")
        return

    # 正则匹配探针日志块
    # 考虑到了终端输出可能的格式错乱和换行
    pattern = re.compile(
        r'========== Step Trigger ==========\n'
        r'ReqID:\s*(?P<req>\d+),\s*SeqLen:\s*(?P<seq>\d+),\s*Budget:\s*(?P<budget>\d+)\n'
        r'Evict Metric:\s*(?P<metric>\S+)\n'
        r'Action:\s*Evicted Logical Index \[(?P<idx>\d+)\]\n'
        r'Block Table BEFORE:\s*\[(?P<before>[^\]]+)\]\n'
        r'Block Table AFTER\s*:\s*\[(?P<after>[^\]]+)\]\n'
        r'-+',
        re.MULTILINE
    )

    matches = list(pattern.finditer(content))
    if not matches:
        print("⚠️ 未在文件中找到符合格式的探针日志。")
        return

    print(f"🔍 发现 {len(matches)} 条探针记录，开始校验 (BlockSize={block_size}, InitialBlocks={initial_blocks})...\n")

    passed_count = 0
    failed_count = 0

    for i, match in enumerate(matches, 1):
        req_id = int(match.group('req'))
        seq_len = int(match.group('seq'))
        budget = int(match.group('budget'))
        rmv_idx = int(match.group('idx'))

        # 清洗可能因为终端折行产生的多余空格 (如 "2    19" -> "219")
        before_str = re.sub(r'\s+', '', match.group('before'))
        after_str = re.sub(r'\s+', '', match.group('after'))

        before_list = [int(x) for x in before_str.split(',') if x]
        after_list = [int(x) for x in after_str.split(',') if x]

        # 获取非零的有效块
        # before_valid = [x for x in before_list if x != 0]
        # after_valid = [x for x in after_list if x != 0]
        # 获取有效物理块：仅剔除末尾连续的 Padding 0，保留可能作为合法 ID 的首部或中部 0
        def get_valid_blocks(block_list):
            idx = len(block_list) - 1
            while idx >= 0 and block_list[idx] == 0:
                idx -= 1
            return block_list[:idx + 1]

        before_valid = get_valid_blocks(before_list)
        after_valid = get_valid_blocks(after_list)

        errors = []

        # ==========================================
        # 1. 触发条件校验
        # ==========================================
        if seq_len <= budget:
            errors.append(f"触发条件违规: SeqLen ({seq_len}) 未超过 Budget ({budget})")
        if seq_len % block_size != 0:
            errors.append(f"触发条件违规: SeqLen ({seq_len}) 不是 BlockSize ({block_size}) 的整数倍")

        # ==========================================
        # 2. 左移阵型 & 物理容量校验
        # ==========================================
        # 按照左移逻辑，BEFORE 去掉 rmv_idx 后应该完全等于 AFTER 的有效块部分
        if rmv_idx >= len(before_valid):
            errors.append(f"索引越界: 尝试驱逐的 Logical Index [{rmv_idx}] 超出当前物理块总数 ({len(before_valid)})")
        else:
            expected_shifted_valid = before_valid[:rmv_idx] + before_valid[rmv_idx+1:]
            
            if len(after_valid) != len(before_valid) - 1:
                errors.append(f"容量守恒异常: BEFORE有效块数({len(before_valid)})，驱逐后AFTER有效块数({len(after_valid)})不等于预期({len(before_valid)-1})")
            
            if after_valid != expected_shifted_valid:
                errors.append("左移阵型异常: AFTER 的物理块排列与 BEFORE 移除指定块后的平移结果不一致")

        # 检查 AFTER 末尾是否正确填充了 0 (Padding ID)
        if len(after_valid) < len(after_list):
            if after_list[len(after_valid)] != 0:
                errors.append("左移阵型异常: AFTER 数组最末尾有效位之后没有正确置 0 (Padding)")

        # ==========================================
        # 3. 保护区免死金牌校验
        # ==========================================
        # 头部 Sink 保护区
        if rmv_idx < initial_blocks:
            errors.append(f"免死金牌违规: 驱逐索引 [{rmv_idx}] 侵入了头部 initial_blocks ({initial_blocks}) 保护区")

        # 尾部 Recent 保护区
        total_budget_blocks = budget // block_size
        protected_tail_blocks = max(1, total_budget_blocks // 4)
        
        # 物理安全区终点: 物理总数 - 1(活跃尾块) - 尾部保护块
        max_physical_safe_end = len(before_valid) - 1 - protected_tail_blocks
        
        if rmv_idx > max_physical_safe_end:
            errors.append(f"免死金牌违规: 驱逐索引 [{rmv_idx}] 侵入了尾部 Recent 保护区 (物理安全区截止至索引 [{max_physical_safe_end}])")

        # ==========================================
        # 打印单条结果
        # ==========================================
        if not errors:
            passed_count += 1
            # print(f"✅ [Pass] 记录 #{i} | ReqID: {req_id}, SeqLen: {seq_len}, 驱逐位置: [{rmv_idx}]")
        else:
            failed_count += 1
            print(f"❌ [Fail] 记录 #{i} | ReqID: {req_id}, SeqLen: {seq_len}, 驱逐位置: [{rmv_idx}]")
            for err in errors:
                print(f"    - {err}")

    # 汇总
    print("\n" + "="*50)
    print(f"📊 校验完成! 总记录数: {len(matches)}")
    print(f"   🟢 通过: {passed_count}")
    print(f"   🔴 失败: {failed_count}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM Paged Eviction 日志检查脚本")
    parser.add_argument("log_file", type=str, help="包含探针输出的日志文件路径")
    parser.add_argument("--block-size", type=int, default=16, help="KV Cache 的 Block Size (默认: 16)")
    parser.add_argument("--initial-blocks", type=int, default=4, help="受保护的 initial-blocks 数量 (默认: 4)")
    
    args = parser.parse_args()
    parse_and_check_logs(args.log_file, args.block_size, args.initial_blocks)
