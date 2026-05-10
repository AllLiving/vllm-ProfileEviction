'''
A class to manage the l2 norm of all the blocks for the running requests.
This class is used when local/global eviction is enabled.
'''
from typing import Dict, List
import collections
import copy

class BlockL2NormManager:
    def __init__(self):
        self.reqid_mapping_blocktables: Dict[str, Dict[int, List[int]]] = None  # request_id -> block_tables mapping
        self.reqid_mapping_seqs: Dict[str, List[int]] = None  # request_id -> seq_ids mapping
        self.flat_sequence_map = [] # used to fetch the request_id and seq_id based on the seq_idx quickly
        self.reqid_mapping_block_l2norms: Dict[str, Dict[int, List[float]]] = None  # request_id -> block_l2norm mapping

        # [ProfileEvict] Macro Stage: SimHash 状态挂载点
        # 记录全局每个 Hash 对应的物理 Block 数量，O(1) 查找
        self.global_hash_freq: Dict[int, int] = {}
        # 记录每个 Request 对应的 Block Hash 账本，与 block_l2norms 结构严格对齐
        self.reqid_mapping_block_hashes: Dict[str, Dict[int, List[int]]] = None
        # [ProfileEvict GC] TTL 垃圾回收状态
        self.step_counter = 0
        self.req_status: Dict[str, str] = {} # 'active' or 'suspended'
        self.reqid_last_active: Dict[str, int] = {}
        # 基于req_status，Suspended 不再污染全局，可设非常大，绝对保护 Swap Out
        self.MAX_IDLE_STEPS = 8192

    def _decrement_req_hashes(self, req_id: str):
        """[GC 钩子] 请求挂起或结束时，瞬间撤销其对全局 Hash 的污染"""
        if req_id in self.reqid_mapping_block_hashes:
            for seq_id, block_hashes in self.reqid_mapping_block_hashes[req_id].items():
                for h in block_hashes:
                    if h != -1:
                        # [核心修复 2]: 使用 .get() 绝对安全，不会污染字典
                        freq = self.global_hash_freq.get(h, 0)
                        if freq > 1:
                            self.global_hash_freq[h] = freq - 1
                        elif freq == 1:
                            del self.global_hash_freq[h]

    def _increment_req_hashes(self, req_id: str):
        """[GC 钩子] 请求从 Swap Out 唤醒时，将其原有 Hash 重新注入全局池"""
        if req_id in self.reqid_mapping_block_hashes:
            for seq_id, block_hashes in self.reqid_mapping_block_hashes[req_id].items():
                for h in block_hashes:
                    if h != -1:
                        self.global_hash_freq[h] = self.global_hash_freq.get(h, 0) + 1

    def _hard_delete_req(self, req_id: str):
        """[GC 钩子] 超出 TTL，执行真正的本地内存清理"""
        if req_id in self.reqid_mapping_block_hashes:
            del self.reqid_mapping_block_hashes[req_id]
        if req_id in self.reqid_mapping_block_l2norms:
            del self.reqid_mapping_block_l2norms[req_id]
        if req_id in self.reqid_mapping_seqs:
            del self.reqid_mapping_seqs[req_id]
        if req_id in self.req_status:
            del self.req_status[req_id]
        if req_id in self.reqid_last_active:
            del self.reqid_last_active[req_id]

    def _init_new_req(self, req_id: str, block_tables: Dict[int, List[int]]):
        """[Helper] 初始化全新请求的账本"""
        self.reqid_mapping_seqs[req_id] = list(block_tables.keys())
        self.reqid_mapping_block_l2norms[req_id] = {}
        self.reqid_mapping_block_hashes[req_id] = {}
        self.req_status[req_id] = 'active'
        self.reqid_last_active[req_id] = self.step_counter
        for seq_id, block_table in block_tables.items():
            # 赋予“免死金牌”和 Hash 占位符
            self.reqid_mapping_block_l2norms[req_id][seq_id] = [float('inf')] * len(block_table)
            self.reqid_mapping_block_hashes[req_id][seq_id] = [-1] * len(block_table)

    def update_reqs(self, reqid_mapping_blocktables: Dict[str, Dict[int, List[int]]]):
        is_reqs_changed = False
        self.step_counter += 1
        current_reqs = set(reqid_mapping_blocktables.keys())
        if self.reqid_mapping_block_l2norms is None:
            # 1. 引擎冷启动：首次分配
            self.reqid_mapping_blocktables = reqid_mapping_blocktables
            self.reqid_mapping_seqs = {}
            self.reqid_mapping_block_l2norms = {}
            self.reqid_mapping_block_hashes = {}
            for req_id, block_tables in reqid_mapping_blocktables.items():
                self._init_new_req(req_id, block_tables)
            is_reqs_changed = True
        else:
            if current_reqs != set(self.reqid_mapping_blocktables.keys()):
                is_reqs_changed = True
            self.reqid_mapping_blocktables = reqid_mapping_blocktables
            all_tracked_reqs = set(self.reqid_mapping_block_hashes.keys())
            # 2. 挂起 (Suspend): 刚刚从当前批次中消失的请求 (Swap Out 或 Finished)
            for req_id in all_tracked_reqs - current_reqs:
                if self.req_status.get(req_id) == 'active':
                    self.req_status[req_id] = 'suspended'
                    self._decrement_req_hashes(req_id) # 瞬间阻断全局污染！
            # 3. 唤醒 (Resume): 之前被挂起，现在又回到批次中的请求 (Swap In)
            for req_id in current_reqs.intersection(all_tracked_reqs):
                if self.req_status.get(req_id) == 'suspended':
                    self.req_status[req_id] = 'active'
                    self._increment_req_hashes(req_id) # 账本复活，重新加入角逐
                # 只要还在跑，就持续刷新心跳
                self.reqid_last_active[req_id] = self.step_counter
            # 4. 新生 (New): 全新的请求到来
            for req_id in current_reqs - all_tracked_reqs:
                self._init_new_req(req_id, reqid_mapping_blocktables[req_id])
        # 5. 死亡 (TTL GC): 处理长期挂起的僵尸请求
        removed_reqs = []
        for req_id, status in self.req_status.items():
            if status == 'suspended':
                if self.step_counter - self.reqid_last_active[req_id] > self.MAX_IDLE_STEPS:
                    removed_reqs.append(req_id)
        for req_id in removed_reqs:
            # 因为挂起时已经从全局池减过了，这里直接释放字典内存即可
            self._hard_delete_req(req_id)
        # 6. 更新物理映射底表 (flat_sequence_map)
        if is_reqs_changed:
            self.flat_sequence_map = []
            for req_id, block_tables in reqid_mapping_blocktables.items():
                for seq_id in block_tables.keys():
                    self.flat_sequence_map.append((req_id, seq_id))

    # 旧版 hash实现
    def update_reqs_defectHash(self, reqid_mapping_blocktables: Dict[str, Dict[int, List[int]]]):
        ### Based on the blocktables of each request to init and update the their block_l2 norms
        ### Goal: add the new request and delete old request
        is_reqs_changed = False
        if self.reqid_mapping_block_l2norms is None:
            self.reqid_mapping_blocktables = reqid_mapping_blocktables
            self.reqid_mapping_seqs = {}
            self.reqid_mapping_block_l2norms = {} 
            self.reqid_mapping_block_hashes = {}
            for req_id, block_tables in reqid_mapping_blocktables.items():
                self.reqid_mapping_seqs[req_id] = list(block_tables.keys())  # Store seq_ids for the request
                self.reqid_mapping_block_l2norms[req_id] = {}
                self.reqid_mapping_block_hashes[req_id] = {}
                for seq_id, block_table in block_tables.items():
                    self.reqid_mapping_block_l2norms[req_id][seq_id] = [0.0 * len(block_table)]  # Initialize with zeros
                    self.reqid_mapping_block_hashes[req_id][seq_id] = [0 * len(block_table)] # [ProfileEvict] 初始化 0
            is_reqs_changed = True
        else:
            # Update existing requests
            # print(f"update_reqs - pass to the function: reqid_mapping_blocktables={reqid_mapping_blocktables.keys()}")
            # compare the reqid_mapping_blocktables with the existing self.reqid_mapping_blocktables to if requests are changed
            if set(reqid_mapping_blocktables.keys()) != set(self.reqid_mapping_blocktables.keys()):
                is_reqs_changed = True
                
            self.reqid_mapping_blocktables = reqid_mapping_blocktables
            for req_id, block_tables in reqid_mapping_blocktables.items():
                if req_id not in self.reqid_mapping_block_l2norms:
                    self.reqid_mapping_seqs[req_id] = list(block_tables.keys()) 
                    self.reqid_mapping_block_l2norms[req_id] = {}
                    self.reqid_mapping_block_hashes[req_id] = {}
                for seq_id, block_table in block_tables.items():
                    if seq_id not in self.reqid_mapping_block_l2norms[req_id]:
                        self.reqid_mapping_block_l2norms[req_id][seq_id] = [float('inf')] * len(block_table)
                        self.reqid_mapping_block_hashes[req_id][seq_id] = [-1] * len(block_table) # -1 表未计算占位
            
            # print(f"update_reqs: reqid_mapping_blocktables={reqid_mapping_blocktables.keys()}, self.reqid_mapping_blocktables={self.reqid_mapping_blocktables.keys()}")
            # print(f"update_reqs: reqid_mapping_block_l2norms={self.reqid_mapping_block_l2norms.keys()}")
            # Remove old requests
            # removed_reqs = [req_id for req_id in self.reqid_mapping_block_l2norms.keys() if req_id not in reqid_mapping_blocktables]
            # print(f"to be removed reqs: {removed_reqs}")
            # for req_id in removed_reqs:
            #     del self.reqid_mapping_block_l2norms[req_id]
            #     del self.reqid_mapping_seqs[req_id]
            # print(f"update_reqs - after: reqid_mapping_seqs={self.reqid_mapping_seqs.keys()}")

            # [ProfileEvict - 阶段四] Teardown Hook: 唤醒并强化旧请求清理逻辑
            self.step_counter += 1
            for req_id in reqid_mapping_blocktables.keys():
                self.reqid_last_active[req_id] = self.step_counter
            if not is_reqs_changed:
                # 即使请求集合没变，也需要检查是否有超时未活跃的僵尸请求
                pass
            # [ProfileEvict GC] TTL 惰性回收逻辑
            removed_reqs = []
            for req_id, last_active in list(self.reqid_last_active.items()):
                if self.step_counter - last_active > self.MAX_IDLE_STEPS:
                    removed_reqs.append(req_id)

            for req_id in removed_reqs:
                # 1. 彻底清退该死去请求造成的全局 Hash 频次污染
                if req_id in self.reqid_mapping_block_hashes:
                    for seq_id, block_hashes in self.reqid_mapping_block_hashes[req_id].items():
                        for h in block_hashes:
                            # 排除占位符 -1，且确保频次大于 0 防止越界
                            if h != -1 and self.global_hash_freq[h] > 0:
                                self.global_hash_freq[h] -= 1
                                # 字典瘦身：如果某个 Hash 频次归零，直接从字典中硬删除，防止长期运行内存爆炸
                                if self.global_hash_freq[h] == 0:
                                    del self.global_hash_freq[h]
                    # 删掉该请求的哈希隔离账本
                    del self.reqid_mapping_block_hashes[req_id]
                if req_id in self.reqid_mapping_block_l2norms:
                    del self.reqid_mapping_block_l2norms[req_id]
                if req_id in self.reqid_mapping_seqs:
                    del self.reqid_mapping_seqs[req_id]
                del self.reqid_last_active[req_id]

        if is_reqs_changed:
            ### reset the flat_sequence_map based on the updated reqid_mapping_seqs
            self.flat_sequence_map = []
            for req_id, block_tables in reqid_mapping_blocktables.items():
                for seq_id in block_tables.keys():
                    self.flat_sequence_map.append((req_id, seq_id))
            # print(f"update_reqs - flat_sequence_map: {self.flat_sequence_map}")

    def update_block_l2norms(self, seq_idx: int, l2_norms: List[float]):
        # calculate the req_id based on the seq_idx
        target_req_id, target_seq_id = self.get_reqid_and_seqid(seq_idx)
        assert target_req_id is not None and target_seq_id is not None, "Invalid seq_idx for block l2 norm update"
        ## update the block l2 norms for the target request and seq
        self.reqid_mapping_block_l2norms[target_req_id][target_seq_id] = l2_norms

    def update_block_hashes(self, seq_idx: int, hashes: List[int]):
        target_req_id, target_seq_id = self.get_reqid_and_seqid(seq_idx)
        self.reqid_mapping_block_hashes[target_req_id][target_seq_id] = hashes
        for h in hashes:
            if h != -1:
                self.global_hash_freq[h] = self.global_hash_freq.get(h, 0) + 1

    def update_last_block_l2norm(self, seq_idx: int, l2_norm: float):
        target_req_id, target_seq_id = self.get_reqid_and_seqid(seq_idx)
        assert target_req_id is not None and target_seq_id is not None, "Invalid seq_idx for block l2 norm update"
        self.reqid_mapping_block_l2norms[target_req_id][target_seq_id].append(l2_norm)
        return self.reqid_mapping_block_l2norms[target_req_id][target_seq_id]
        
    def get_reqid_and_seqid(self, seq_idx: int):
        # if seq_idx >= len(self.flat_sequence_map):
        #     for req_id, seq_ids in self.reqid_mapping_seqs.items():
        #         print(f"raw reqid_mapping_seqs: req_id={req_id}, seq_ids={seq_ids}")
            
        #     for item in self.flat_sequence_map:
        #         print(f"flat_sequence_map: item={item}")
            
        assert seq_idx < len(self.flat_sequence_map), f"seq_idx {seq_idx} is out of range length {len(self.flat_sequence_map)}"
        target_req_id, target_seq_id = self.flat_sequence_map[seq_idx]
        
        return target_req_id, target_seq_id
    
    def get_req_block_tables(self, seq_idx: int):
        req_id, seq_id = self.get_reqid_and_seqid(seq_idx)
        assert req_id is not None and seq_id is not None, "Invalid seq_idx for block tables retrieval"
        return self.reqid_mapping_blocktables[req_id][seq_id]
    
    def get_updated_block_tables(self, seq_idx: int, rmv_idx: int, is_rmv_l2norm: bool = False): 
        req_id, seq_id = self.get_reqid_and_seqid(seq_idx)
        assert req_id is not None and seq_id is not None, f"Invalid seq_idx {seq_idx} for block tables retrieval"
        # block_tables = self.reqid_mapping_blocktables[req_id][seq_id]
        block_tables = copy.deepcopy(self.reqid_mapping_blocktables[req_id][seq_id])
        # print(f"rmv_idx={rmv_idx}, block_tables={block_tables}")
        del block_tables[rmv_idx]  # Remove the block at rmv_idx
        if is_rmv_l2norm:
            # remove the l2-norm of the block at rmv_idx when is_rmv_l2norm is True
            del self.reqid_mapping_block_l2norms[req_id][seq_id][rmv_idx]
        # return the updated block tables
        return block_tables
