import torch
import numpy as np
from typing import List, Dict, Optional
from copy import deepcopy
from torch.nn.utils.rnn import pad_sequence
import re
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_response_mask, compute_advantage, Role
from verl.utils.metric import (
    reduce_metrics,
)
import uuid
from tqdm import tqdm
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.utils.debug import marked_timer
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import Tracking
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.reward_score.math_dapo import is_correct_minerva
from omegaconf import OmegaConf, open_dict
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls

def left_to_right_padding(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    将 Left Padded 的 Tensor 转换为 Right Padded。
    tensor: [B, L]
    mask: [B, L] (1 for valid, 0 for pad)
    """
    B, L = tensor.shape
    device = tensor.device
    
    # 计算每个样本的有效长度
    lens = mask.sum(dim=-1).long() # [B]
    
    # 创建新的 Right Padded Tensor
    new_tensor = torch.zeros_like(tensor)
    
    for i in range(B):
        l = lens[i]
        if l > 0:
            # 取出有效部分 (Left Padded 的有效部分在末尾)
            valid_part = tensor[i, -l:]
            # 放到开头 (Right Padded)
            new_tensor[i, :l] = valid_part
            
    return new_tensor

def extract_answer(text: str) -> str:
    """从文本中提取 #### 后的答案，用于简单的正确性匹配"""
    if "####" in text:
        return text.split("####")[-1].strip()
    if 'Answer:' in text[-20:]:
        return text[-20:].split('Answer:')[-1].strip()
    return "[No Answer]"

class TeacherStudentReflectiveTrainer(RayPPOTrainer):
    """
    Teacher-Student Reflective Trainer.
    
    Architecture:
    - Student (Actor): Trainable. Generates Response.
    - Teacher (RefPolicy): Frozen. Generates Summary AND Computes LogProb.
    
    Resource Strategy:
    - Splits available GPUs into two pools (Student Pool & Teacher Pool) to allow 
      two simultaneous vLLM instances without conflict.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.use_reference_policy, "TeacherStudentReflectiveTrainer requires a Reference Policy (Teacher)!"
        self.use_critic = False
        self.use_rm = False
        
        # === NEW: Load Teacher Tokenizer ===
        # 如果配置了 teacher.model.path，则加载对应的 tokenizer
        # 否则默认 teacher 和 student 使用相同的 tokenizer
        if OmegaConf.select(self.config, "teacher.model.path"):
            print(f">>> Loading specific Teacher Tokenizer from: {self.config.teacher.model.path}")
            self.teacher_tokenizer = hf_tokenizer(
                self.config.teacher.model.path, 
                trust_remote_code=self.config.data.get("trust_remote_code", False)
            )
            self.different_teacher_model = True
        else:
            print(">>> Using Student Tokenizer for Teacher (Shared Model)")
            self.teacher_tokenizer = self.tokenizer
            self.different_teacher_model = False
            
        # === NEW: Standard Distillation Flag ===
        self.standard_distillation = OmegaConf.select(self.config, "teacher.standard_distillation")
        if self.standard_distillation is None:
            self.standard_distillation = False
            
        if self.standard_distillation:
            print(">>> [Mode] Running STANDARD On-Policy Distillation (No Reflection/Hints)")
        else:
            print(">>> [Mode] Running REFLECTIVE On-Policy Distillation (With Summaries & Hints)")
            
        print(">>> TeacherStudentReflectiveTrainer Initialized.")

    def init_workers(self):
        """
        初始化 Worker，支持多机多卡 (Multi-Node) 自动分配与检查。
        """
        # 1. 获取资源配置
        n_gpus_per_node = self.config.trainer.n_gpus_per_node
        n_nodes = self.config.trainer.nnodes
        if n_nodes > 1:
            # === Determine Teacher TP ===
            teacher_tp = 1
            if OmegaConf.select(self.config, "teacher.rollout.tensor_model_parallel_size"):
                teacher_tp = self.config.teacher.rollout.tensor_model_parallel_size
            
            teacher_gpus_per_node = teacher_tp
            student_gpus_per_node = n_gpus_per_node - teacher_gpus_per_node
            
            if student_gpus_per_node < 1:
                raise ValueError(f"Not enough GPUs per node! Got {n_gpus_per_node}, need at least 2.")

            # 计算总的 World Size
            student_world_size = student_gpus_per_node * n_nodes / self.config.teacher.rollout.tensor_model_parallel_size
            teacher_world_size = teacher_gpus_per_node * n_nodes / teacher_tp
            
            print(f"\n{'='*20} Multi-Node Topology Setup {'='*20}")
            print(f">>> Nodes: {n_nodes} | GPUs per Node: {n_gpus_per_node}")
            print(f">>> Student (Actor): {student_gpus_per_node} GPUs/node * {n_nodes} nodes = {student_world_size} Total GPUs")
            print(f">>> Teacher (Ref):   {teacher_gpus_per_node} GPUs/node * {n_nodes} nodes = {teacher_world_size} Total GPUs")

            # === 关键检查: Batch Size 整除性 (防止启动后报错) ===
            # 规则: ppo_mini_batch_size 必须能被 (Student_World_Size * Micro_Batch) 整除
            # 或者至少: (ppo_mini_batch_size / Student_World_Size) 必须是整数，且能被 Micro_Batch 整除
            
            mini_batch = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            micro_batch = self.config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
            
            # 1. 检查每张卡分到的 Batch 是否为整数
            if mini_batch % student_world_size != 0:
                suggested = (mini_batch // student_world_size + 1) * student_world_size
                raise ValueError(
                    f"\n[Config Error] ppo_mini_batch_size ({mini_batch}) cannot be divided evenly by Student World Size ({student_world_size}).\n"
                    f"Student GPUs = {student_world_size} (Nodes: {n_nodes} * GPUs: {student_gpus_per_node})\n"
                    f"Suggested ppo_mini_batch_size: {suggested} or {suggested + student_world_size}"
                )
                
            local_batch = mini_batch // student_world_size
            
            # 2. 检查每张卡的 Local Batch 是否能被 Micro Batch 整除
            if local_batch % micro_batch != 0:
                raise ValueError(
                    f"\n[Config Error] Local Batch ({local_batch}) cannot be divided by Micro Batch ({micro_batch}).\n"
                    f"Global Mini Batch: {mini_batch}, Student World Size: {student_world_size}\n"
                    f"Please adjust ppo_mini_batch_size or ppo_micro_batch_size_per_gpu."
                )
                
            print(f">>> Batch Check Passed: Global={mini_batch} -> PerGPU={local_batch} -> Micro={micro_batch} (Accum={local_batch//micro_batch})")
            print(f"{'='*60}\n")

            # 2. 创建资源池 (RayResourcePool 会自动处理跨节点调度)
            # process_on_nodes=[N, N] 表示 Node 0 需要 N 张卡，Node 1 需要 N 张卡...
            student_pool = RayResourcePool(
                process_on_nodes=[student_gpus_per_node] * n_nodes,
                use_gpu=True,
                max_colocate_count=1,
                name_prefix="student_pool"
            )
            
            teacher_pool = RayResourcePool(
                process_on_nodes=[teacher_gpus_per_node] * n_nodes,
                use_gpu=True,
                max_colocate_count=1,
                name_prefix="teacher_pool"
            )
            
            self.resource_pool_to_cls = {student_pool: {}, teacher_pool: {}}

            # 3. 初始化 Config
            student_config = deepcopy(self.config.actor_rollout_ref)
            raw_max_len = self.config.data.max_prompt_length + self.config.data.max_response_length
            safe_max_model_len = raw_max_len + 512 
            student_config.rollout.max_model_len = safe_max_model_len # <--- 修复 vLLM 长度限制
            student_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=student_config,
                role="actor_rollout",
                profile_option=self.config.trainer.npu_profile.options,
            )
            self.resource_pool_to_cls[student_pool]["actor_rollout"] = student_cls

            teacher_config = deepcopy(self.config.actor_rollout_ref)
            # === NEW: Override Teacher Config ===
            if self.different_teacher_model:
                teacher_config.model.path = self.config.teacher.model.path
                print(f">>> [Config] Overriding Teacher Model Path: {teacher_config.model.path}")
                
                # 只有 Megatron 需要 TP，FSDP 通常不需要手动设置 TP，除非你用的是 Megatron 后端
                # 但如果你的 Teacher 很大 (比如 72B)，可能需要 FSDP Sharding
                # 注意：标准的 RefWorker (FSDP) 会自动处理 Sharding
                
                # 移除 vLLM 特有的配置，防止报错或误导
                if 'rollout' in teacher_config:
                    # 纯 Ref 不需要 gpu_memory_utilization，因为它不用 vLLM
                    # 但保留它也无妨，只要 role="ref"，这些参数通常会被忽略
                    pass
            
            teacher_config.model.torch_dtype = "bfloat16"
            
            # === KEY CHANGE: 这里的 role 改回 "ref" ===
            # 这告诉 verl 使用 ActorRolloutRefWorker 的 ref_model 逻辑，而不是 actor 逻辑
            # 并且不会初始化 vLLM
            # teacher_cls = RayClassWithInitArgs(
            #     cls=self.role_worker_mapping[Role.RefPolicy],
            #     config=teacher_config,
            #     role="ref",  # <--- 改回 ref
            #     profile_option=self.config.trainer.npu_profile.options,
            # )
                        # === 开启 Teacher 的 vLLM 以支持 Debug 续写 ===
            # 必须设置 gpu_memory_utilization，否则 vLLM 会报错
            if 'rollout' not in teacher_config:
                teacher_config.rollout = OmegaConf.create()
            teacher_config.rollout.gpu_memory_utilization = self.config.actor_rollout_ref.rollout.gpu_memory_utilization
            
            teacher_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RefPolicy],
                config=teacher_config,
                role="actor_rollout_ref",  # <--- 改为 actor_rollout_ref，强制初始化 vLLM
                profile_option=self.config.trainer.npu_profile.options,
            )
            self.resource_pool_to_cls[teacher_pool]["ref"] = teacher_cls

            # 4. 启动 Workers
            all_wg = {}
            wg_kwargs = {
                "ray_wait_register_center_timeout": self.config.trainer.ray_wait_register_center_timeout,
                "device_name": self.device_name
            }
            
            for resource_pool, class_dict in self.resource_pool_to_cls.items():
                worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
                wg_dict = self.ray_worker_group_cls(
                    resource_pool=resource_pool,
                    ray_cls_with_init=worker_dict_cls,
                    **wg_kwargs,
                )
                spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
                all_wg.update(spawn_wg)

            self.actor_rollout_wg = all_wg["actor_rollout"]
            self.actor_rollout_wg.init_model()
            
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()
                
            self.async_rollout_mode = False
            if self.config.actor_rollout_ref.rollout.mode == 'async':
                from verl.experimental.agent_loop import AgentLoopManager
                self.async_rollout_mode = True
                self.async_rollout_manager = AgentLoopManager(
                    config=self.config,
                    worker_group=self.actor_rollout_wg
                )
        else:
            # === FIX: 调整分配比例 ===
            # 之前的逻辑: student_gpus = n_gpus_per_node // 2 (4:4)
            # 新的逻辑: 尽可能多给 Student，Teacher 保留 1-2 张即可
            
            # === Determine Teacher TP ===
            teacher_tp = 1
            if OmegaConf.select(self.config, "teacher.rollout.tensor_model_parallel_size"):
                teacher_tp = self.config.teacher.rollout.tensor_model_parallel_size
            
            teacher_gpus_per_node = teacher_tp
            student_gpus_per_node = n_gpus_per_node - teacher_gpus_per_node
            
            if teacher_gpus_per_node < 1:
                raise ValueError(f"Not enough GPUs! Need at least 2 GPUs per node.")
                
            print(f">>> Resource Splitting: Student={student_gpus_per_node} GPUs, Teacher={teacher_gpus_per_node} GPUs (per node)")
            # =========================

            # 2. Create Independent Resource Pools
            student_pool = RayResourcePool(
                process_on_nodes=[student_gpus_per_node] * n_nodes,
                use_gpu=True,
                max_colocate_count=1,
                name_prefix="student_pool"
            )
            
            teacher_pool = RayResourcePool(
                process_on_nodes=[teacher_gpus_per_node] * n_nodes,
                use_gpu=True,
                max_colocate_count=1,
                name_prefix="teacher_pool"
            )
            
            self.resource_pool_to_cls = {student_pool: {}, teacher_pool: {}}

            # 3. Initialize Student (Actor) Config
            # Deepcopy to prevent config pollution
            student_config = deepcopy(self.config.actor_rollout_ref)
            # Adjust micro_batch_size if necessary (optional, but good practice since world_size changed)
            # student_config.actor.ppo_mini_batch_size //= 2 # Logic handled by worker usually
            raw_max_len = self.config.data.max_prompt_length + self.config.data.max_response_length
            safe_max_model_len = raw_max_len + 512 
            student_config.rollout.max_model_len = safe_max_model_len # <--- 修复 vLLM 长度限制
            student_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=student_config,
                role="actor_rollout",
                profile_option=self.config.trainer.npu_profile.options,
            )
            self.resource_pool_to_cls[student_pool]["actor_rollout"] = student_cls

            # 4. Initialize Teacher (Ref) Config
            # We define it as 'actor_rollout' role to FORCE vLLM initialization.
            teacher_config = deepcopy(self.config.actor_rollout_ref)
            # === NEW: Override Teacher Config ===
            if self.different_teacher_model:
                teacher_config.model.path = self.config.teacher.model.path
                print(f">>> [Config] Overriding Teacher Model Path: {teacher_config.model.path}")
                
                # 只有 Megatron 需要 TP，FSDP 通常不需要手动设置 TP，除非你用的是 Megatron 后端
                # 但如果你的 Teacher 很大 (比如 72B)，可能需要 FSDP Sharding
                # 注意：标准的 RefWorker (FSDP) 会自动处理 Sharding
                
                # 移除 vLLM 特有的配置，防止报错或误导
                if 'rollout' in teacher_config:
                    # 纯 Ref 不需要 gpu_memory_utilization，因为它不用 vLLM
                    # 但保留它也无妨，只要 role="ref"，这些参数通常会被忽略
                    pass
            
            teacher_config.model.torch_dtype = "bfloat16"
            
            # === KEY CHANGE: 这里的 role 改回 "ref" ===
            # 这告诉 verl 使用 ActorRolloutRefWorker 的 ref_model 逻辑，而不是 actor 逻辑
            # 并且不会初始化 vLLM
            # teacher_cls = RayClassWithInitArgs(
            #     cls=self.role_worker_mapping[Role.RefPolicy],
            #     config=teacher_config,
            #     role="ref",  # <--- 改回 ref
            #     profile_option=self.config.trainer.npu_profile.options,
            # )
            # === 开启 Teacher 的 vLLM 以支持 Debug 续写 ===
            # 必须设置 gpu_memory_utilization，否则 vLLM 会报错
            if 'rollout' not in teacher_config:
                teacher_config.rollout = OmegaConf.create()
            # [核心修复]：Teacher 独占 GPU 且不训练，直接给 90% 的显存！
            # 如果你在 bash 脚本里传了 +teacher.rollout.gpu_memory_utilization，就用传入的，否则默认 0.9
            if OmegaConf.select(self.config, "teacher.rollout.gpu_memory_utilization"):
                teacher_config.rollout.gpu_memory_utilization = self.config.teacher.rollout.gpu_memory_utilization
            else:
                teacher_config.rollout.gpu_memory_utilization = 0.9
            
            # 同时保证 max_num_batched_tokens 和 max_model_len 也同步过来，防止 Teacher 爆显存
            teacher_config.rollout.max_num_batched_tokens = self.config.actor_rollout_ref.rollout.max_num_batched_tokens
            if OmegaConf.select(self.config, "actor_rollout_ref.rollout.max_model_len"):
                teacher_config.rollout.max_model_len = self.config.actor_rollout_ref.rollout.max_model_len

            teacher_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RefPolicy],
                config=teacher_config,
                role="actor_rollout_ref",  # <--- 改为 actor_rollout_ref，强制初始化 vLLM
                profile_option=self.config.trainer.npu_profile.options,
            )
            self.resource_pool_to_cls[teacher_pool]["ref"] = teacher_cls

            # 5. Spawn Workers
            all_wg = {}
            wg_kwargs = {
                "ray_wait_register_center_timeout": self.config.trainer.ray_wait_register_center_timeout,
                "device_name": self.device_name
            }
            
            for resource_pool, class_dict in self.resource_pool_to_cls.items():
                worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
                wg_dict = self.ray_worker_group_cls(
                    resource_pool=resource_pool,
                    ray_cls_with_init=worker_dict_cls,
                    **wg_kwargs,
                )
                spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
                all_wg.update(spawn_wg)

            # 6. Bind to Trainer
            self.actor_rollout_wg = all_wg["actor_rollout"]
            self.actor_rollout_wg.init_model()
            
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()
                
            # Async Rollout Manager (Only for Student)
            self.async_rollout_mode = False
            if self.config.actor_rollout_ref.rollout.mode == 'async':
                from verl.experimental.agent_loop import AgentLoopManager
                self.async_rollout_mode = True
                self.async_rollout_manager = AgentLoopManager(
                    config=self.config,
                    worker_group=self.actor_rollout_wg
                )

    def _prepare_summary_generation_batch(self, batch: DataProto, raw_prompts: List[str], ground_truths: List[str]) -> DataProto:
        """
        构造用于生成总结 (Reflection) 的 Batch。
        包含: 强力正则清洗，去除 System Prompt 和嵌套标记。
        """
        responses = batch.batch['responses']
        
        assert len(raw_prompts) == len(responses)
        assert len(ground_truths) == len(responses)

        input_ids_list = []
        attention_mask_list = []

        for i, (r_ids, gt_text) in enumerate(zip(responses, ground_truths)):
            p_text_dirty = raw_prompts[i]
            
            # --- 正则清洗逻辑 (提取纯 User Query) ---
            p_text_clean = p_text_dirty
            
            # 1. 尝试截取最后一个 User 标记之后的内容
            split_patterns = [
                r"<\|im_start\|>user\s*",  # Qwen/ChatML
                r"user\s*\n",              # Qwen decode
                r"User:\s*",               # Common
                r"\[INST\]\s*",            # Llama
                r"Human:\s*"               # Anthropic
            ]
            
            found_user = False
            for pattern in split_patterns:
                matches = list(re.finditer(pattern, p_text_dirty, re.IGNORECASE))
                if matches:
                    last_match = matches[-1]
                    p_text_clean = p_text_dirty[last_match.end():]
                    found_user = True
                    break
            
            # 2. 移除可能残留的 Assistant 标记 (Prompt 结尾)
            stop_patterns = [
                r"<\|im_start\|>assistant",
                r"assistant\s*\n",
                r"Assistant:",
                r"\[/INST\]"
            ]
            for pattern in stop_patterns:
                match = re.search(pattern, p_text_clean, re.IGNORECASE) # 使用 escape 防止正则错误
                if match:
                    p_text_clean = p_text_clean[:match.start()]
            
            # 3. 移除可能残留的 User 结束标记 (如 <|im_end|>)
            if "<|im_end|>" in p_text_clean:
                p_text_clean = p_text_clean.replace("<|im_end|>", "")
            p_text_clean = p_text_clean.strip()
            # ----------------------------------------

            r_text = self.tokenizer.decode(r_ids, skip_special_tokens=True)
            r_text = r_text.replace("<think>", "\n\n")
            r_text = r_text.replace("</think>", "\n\n")
            # Prompt for the Teacher (Summary Generation)
            content = (
                "I will give you a math question and it's ground_truth. I will also give you a solution. The solution may be incorrect."
                f"question: {p_text_clean}\n\n"
                f"ground_truth: {gt_text}\n\n"
                f"solution: {r_text}\n\n"
                f"Task: You need to analyze how to solve this problem based on the ground_truth and the solution. If the solution is correct, extract the key points that led to solving it correctly. If the solution is wrong, identify the traps in the question and what to watch out for.\n"
                f"Reply directly with your extracted findings—no extra explanation needed."
            )

            messages = [
                {"role": "system", "content": "You are a helpful math assistant."},
                {"role": "user", "content": content}
            ]

            # 4. 编码 (使用 Teacher Tokenizer)
            enc_ids = self.teacher_tokenizer.apply_chat_template(
                messages, 
                tokenize=True, 
                add_generation_prompt=True,
                return_tensors='pt'
            )
            
            if enc_ids.dim() == 2:
                enc_ids = enc_ids[0]

            input_ids_list.append(enc_ids)
            attention_mask_list.append(torch.ones_like(enc_ids))

        # --- Padding (使用 Teacher Tokenizer 的 Pad ID) ---
        max_len = max([len(t) for t in input_ids_list])
        pad_token_id = self.teacher_tokenizer.pad_token_id # Important!
        if pad_token_id is None:
            pad_token_id = self.teacher_tokenizer.eos_token_id

        padded_input_ids = []
        padded_attention_mask = []
        padded_position_ids = []
        
        for ids, mask in zip(input_ids_list, attention_mask_list):
            pad_len = max_len - len(ids)
            pad_ids = torch.full((pad_len,), pad_token_id, dtype=ids.dtype, device=ids.device)
            pad_mask = torch.full((pad_len,), 0, dtype=mask.dtype, device=mask.device)
            
            final_ids = torch.cat([pad_ids, ids])
            final_mask = torch.cat([pad_mask, mask])
            
            padded_input_ids.append(final_ids)
            padded_attention_mask.append(final_mask)
            
            seq_len = len(ids)
            pos_content = torch.arange(seq_len, dtype=torch.long, device=ids.device)
            pos_pad = torch.zeros(pad_len, dtype=torch.long, device=ids.device)
            final_pos = torch.cat([pos_pad, pos_content])
            padded_position_ids.append(final_pos)
            
        input_ids = torch.stack(padded_input_ids)
        attention_mask = torch.stack(padded_attention_mask)
        position_ids = torch.stack(padded_position_ids)

        summary_batch = DataProto.from_dict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids
        })
        # === FIX: 增强 Summary 生成的鲁棒性 ===
        
        # 1. 尝试读取配置，如果读不到，直接给 4096
        # 注意：OmegaConf 有时需要用 .key 访问，有时用 get
        try:
            max_tokens = self.config.data.summary_max_new_tokens
        except:
            max_tokens = 4096
            
        # 强制保底，防止配置里只有 256 之类的
        if max_tokens < 1024:
            max_tokens = 4096
            
        # 2. 打印调试信息 (只打印一次)
        if self.global_steps == 1:
            print(f">>> [Summary Debug] Prompt Count: {len(input_ids_list)}")
            print(f">>> [Summary Debug] Avg Prompt Len: {np.mean([len(t) for t in input_ids_list]):.1f}")
            print(f">>> [Summary Debug] Setting max_new_tokens = {max_tokens}")

        summary_batch.meta_info = {
            "do_sample": True,
            "temperature": 0.1,
            "max_new_tokens": max_tokens,
            "eos_token_id": self.teacher_tokenizer.eos_token_id, # Teacher EOS
            "pad_token_id": self.teacher_tokenizer.pad_token_id, # Teacher Pad
            "ignore_eos": False,
        }
        
        return summary_batch

    def _prepare_teacher_forward_batch(self, batch: DataProto, summaries: torch.Tensor, ground_truths: List[str]) -> DataProto:
        """
        构造 Teacher Forward 的 Batch。
        修正: 全面采用 Left Padding (左填充) 策略。
        这是配合 DP_Actor 切片逻辑 (取最后 N 个 Token) 的唯一正确解法。
        """
        prompts = batch.batch['prompts']
        responses = batch.batch['responses']
        
        new_input_ids = []
        new_prompts_list = []
        new_responses_list = [] # 纯净的 Response Token

        for i in range(len(prompts)):
            # 1. 解码 Prompt
            p_ids = prompts[i]
            p_ids = p_ids[p_ids != self.tokenizer.pad_token_id]
            p_text_dirty = self.tokenizer.decode(p_ids, skip_special_tokens=False)
            
            # 清洗 User Query
            p_text_clean = p_text_dirty
            split_patterns = [r"<\|im_start\|>user\s*", r"user\s*\n", r"User:\s*", r"\[INST\]\s*", r"Human:\s*"]
            for pattern in split_patterns:
                matches = list(re.finditer(pattern, p_text_dirty, re.IGNORECASE))
                if matches:
                    p_text_clean = p_text_dirty[matches[-1].end():]
                    break
            stop_patterns = [r"<\|im_start\|>assistant", r"assistant\s*\n", r"Assistant:", r"\[/INST\]"]
            for pattern in stop_patterns:
                match = re.search(pattern, p_text_clean, re.IGNORECASE)
                if match:
                    p_text_clean = p_text_clean[:match.start()]
            if "<|im_end|>" in p_text_clean:
                p_text_clean = p_text_clean.replace("<|im_end|>", "")
            p_text_clean = p_text_clean.strip()

            # ----------------------------------------------------
            # 核心修复：必须提前获取 r_ids 和 gt_text，无论哪种模式都需要用到
            # ----------------------------------------------------
            # 获取 Student Response
            r_ids = responses[i]
            r_ids = r_ids[r_ids != self.tokenizer.pad_token_id]
            # 获取 GT
            gt_text = ground_truths[i]

            if getattr(self, "standard_distillation", False):
                # === 标准蒸馏模式 ===
                # 老师只看原问题，不看 Ground Truth 和 Summary
                if self.different_teacher_model:
                    messages = [{"role": "user", "content": p_text_clean}]
                    prefix_ids = self.teacher_tokenizer.apply_chat_template(
                        messages, 
                        tokenize=True, 
                        add_generation_prompt=True,
                        return_tensors='pt'
                    )[0].to(r_ids.device)
                else:
                    # 如果 Tokenizer 一样，直接用原版 Prompt IDs 即可，保证 Teacher 看到的和 Student 完全一致
                    prefix_ids = p_ids
            else:
                # === 反思蒸馏模式 (Reflective) ===
                # 获取 Summary (Teacher Tokenizer -> Text)
                s_ids = summaries[i]
                s_ids = s_ids[s_ids != self.teacher_tokenizer.pad_token_id]
                s_text = self.teacher_tokenizer.decode(s_ids, skip_special_tokens=True)
                if s_text.find('</think>') != -1:
                    s_text[s_text.find('</think>')+len('</think>'):]
                # System Prompt 注入 Hint
                system_content = (
                    "You are a helpful math assistant."
                )
                user_content = (
                    "I will give you a math problem along with key hints for solving it.\n\n"
                    f"Hints: {s_text}\n\n"
                    "Please use these hints to help you solve the math problem."
                )
                
                # 编码 Prompt (Teacher Tokenizer)
                messages = [
                    {"role": "system", "content": system_content}, 
                    {"role": "user", "content": user_content},
                    {"role": "user", "content": p_text_clean} 
                ]
                
                prefix_ids = self.teacher_tokenizer.apply_chat_template(
                    messages, 
                    tokenize=True, 
                    add_generation_prompt=True,
                    return_tensors='pt'
                )[0].to(r_ids.device)
            
            # 6. 拼接 (此时不加 Mask，放到后面统一处理)
            teacher_full = torch.cat([prefix_ids, r_ids])
            
            new_input_ids.append(teacher_full)
            new_prompts_list.append(prefix_ids)
            new_responses_list.append(r_ids)

        # === 核心修正: 手动 Left Padding (Input 和 Labels 双向对齐) ===
        # === Padding (Teacher Tokenizer) ===
        device = batch.batch['input_ids'].device
        pad_token_id = self.teacher_tokenizer.pad_token_id
        if pad_token_id is None: pad_token_id = self.teacher_tokenizer.eos_token_id
        
        # 1. 计算最大长度
        max_len_input = max([x.size(0) for x in new_input_ids])
        max_len_resp = max([x.size(0) for x in new_responses_list])
        
        padded_input_ids = []
        padded_attention_mask = []
        padded_response_mask = []
        padded_position_ids = []
        padded_responses = [] 
        
        for i in range(len(new_input_ids)):
            # --- A. Input Padding (Left) ---
            # 目标结构: [Pad, Pad, Prefix, Response]
            seq_len = new_input_ids[i].size(0)
            pad_len_input = max_len_input - seq_len
            
            id_pad = torch.full((pad_len_input,), pad_token_id, dtype=torch.long, device=device)
            mask_pad = torch.zeros((pad_len_input,), dtype=torch.long, device=device)
            
            # 构造 Input
            cur_input = torch.cat([id_pad, new_input_ids[i]])
            # 构造 Attention Mask (Pad部分为0, 内容部分为1)
            cur_att_mask = torch.cat([mask_pad, torch.ones_like(new_input_ids[i], device=device)])
            
            padded_input_ids.append(cur_input)
            padded_attention_mask.append(cur_att_mask)
            
            # --- B. Response Mask (Left Padded to match Input) ---
            # 目标: 只有 Response 部分为 1，其余(Pad + Prefix) 为 0
            r_len = new_responses_list[i].size(0)
            total_zeros = max_len_input - r_len # 前面所有的 0 (Input Pad + Prefix)
            
            cur_resp_mask = torch.cat([
                torch.zeros((total_zeros,), dtype=torch.long, device=device),
                torch.ones((r_len,), dtype=torch.long, device=device)
            ])
            padded_response_mask.append(cur_resp_mask)
            
            # --- C. Labels/Response Padding (Left) ---
            # 目标结构: [Pad, Pad, Response]
            # 长度必须统一为 max_len_resp，以便 stack
            r_pad_len = max_len_resp - r_len
            r_pad = torch.full((r_pad_len,), pad_token_id, dtype=torch.long, device=device)
            
            cur_labels = torch.cat([r_pad, new_responses_list[i]])
            padded_responses.append(cur_labels)
            
            # --- D. Position IDs ---
            pos_ids = torch.cumsum(cur_att_mask, dim=-1) - 1
            pos_ids.masked_fill_(cur_att_mask == 0, 0)
            padded_position_ids.append(pos_ids)

        # Stack
        input_ids = torch.stack(padded_input_ids)
        attention_mask = torch.stack(padded_attention_mask)
        response_mask = torch.stack(padded_response_mask)
        position_ids = torch.stack(padded_position_ids)
        responses = torch.stack(padded_responses) 
        
        # Prompts 仅用于 logging，使用标准 Pad 即可
        prompts_padded = pad_sequence(new_prompts_list, batch_first=True, padding_value=pad_token_id)

        teacher_batch = DataProto.from_dict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "response_mask": response_mask,
            "prompts": prompts_padded,   
            "responses": responses 
        })
        
        teacher_batch.meta_info = {
            "temperature": 1.0,
            "micro_batch_size": batch.meta_info.get("micro_batch_size", 1),
            "use_dynamic_bsz": batch.meta_info.get("use_dynamic_bsz", False),
            "max_token_len": batch.meta_info.get("max_token_len", 2048)
        }
        
        return teacher_batch

    def fit(self):
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()
        # === 新增：官方 val_only 逻辑修复 ===
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            print(f">>> [Evaluation Mode] Starting initial validation at step {self.global_steps}...")
            val_metrics = self._validate() 
            print(f">>> Initial Validation Metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            
            # 如果设置了 val_only，直接退出，不进入下面的训练循环
            if self.config.trainer.get("val_only", False):
                print(">>> [Evaluation Mode] val_only=True detected. Exiting now.")
                return 
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        self.global_steps += 1

        # === Sanity Check: Is Teacher Brain-Dead? ===
        print("Running Teacher Sanity Check...")
        test_prompts = ["1+1="]
        test_responses = ["2"]
        
        # 使用 Teacher Tokenizer 编码
        p_ids = self.teacher_tokenizer.encode(test_prompts[0], return_tensors='pt')[0]
        r_ids = self.teacher_tokenizer.encode(test_responses[0], return_tensors='pt')[0]
        
        input_ids = torch.cat([p_ids, r_ids])
        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(len(input_ids))
        response_mask = torch.zeros_like(input_ids)
        response_mask[len(p_ids):] = 1
        
        # 2. 获取 Worker 数量 (World Size)
        # 我们需要让 batch size 能够被 world size 整除
        world_size = self.ref_policy_wg.world_size
        
        # 3. 复制数据以匹配 World Size
        input_ids = input_ids.unsqueeze(0).repeat(world_size, 1)
        attention_mask = attention_mask.unsqueeze(0).repeat(world_size, 1)
        position_ids = position_ids.unsqueeze(0).repeat(world_size, 1)
        response_mask = response_mask.unsqueeze(0).repeat(world_size, 1)
        
        # 重新构造 responses (tensor)
        responses_tensor = r_ids.unsqueeze(0).repeat(world_size, 1)

        sanity_batch = DataProto.from_dict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "response_mask": response_mask,
            "responses": responses_tensor # 必须有，用于 compute_log_prob 内部切片
        })
        
        # 补充 meta_info
        sanity_batch.meta_info = {
            "micro_batch_size": world_size, # 一次性跑完
            "temperature": 1.0,
            "use_dynamic_bsz": False
        }
        
        # Teacher Forward
        out = self.ref_policy_wg.compute_ref_log_prob(sanity_batch)
        
        # 取第一个样本的结果
        log_prob = out.batch['ref_log_prob'][0, 0].item()
        
        print(f"Teacher Sanity Check: LogProb('2' | '1+1=') = {log_prob:.4f}")
        
        if log_prob < -5.0:
            print("!!! CRITICAL WARNING: Teacher LogProb is extremely low. Model might be randomly initialized!")
        else:
            print("Teacher seems healthy.")
        # ============================================

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}

                timing_raw = {}

                is_last_step = self.global_steps >= self.total_training_steps

                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0 or self.global_steps == 1)
                ):
                    print(f"Start Testing at step {self.global_steps}...")
                    with marked_timer("testing", timing_raw, color="green"):
                        # 调用父类 RayPPOTrainer 的 _validate 方法
                        val_metrics: dict = self._validate() 
                    
                    # 将测试指标添加到 metrics 中，以便 logger 记录
                    metrics.update(val_metrics)
                    print(f"Testing finished. Metrics: {val_metrics}")
                # =================================================================

                metrics.update({
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch
                })
                

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                
                # --- Step 1: Data Alignment ---
                B = batch.batch.batch_size[0]
                N = self.config.actor_rollout_ref.rollout.n
                
                raw_prompts_list = []
                for p_ids in batch.batch['input_ids']: 
                    raw_prompts_list.append(self.tokenizer.decode(p_ids, skip_special_tokens=True))
                
                ground_truths_list = []
                possible_gt_keys = ["ground_truth", "solution", "reward_model"]
                found_gt = False
                for key in possible_gt_keys:
                    if key in batch.non_tensor_batch:
                        data = batch.non_tensor_batch[key]
                        if key == "reward_model":
                            ground_truths_list = [item.get('ground_truth', '') if isinstance(item, dict) else '' for item in data]
                        else:
                            ground_truths_list = data.tolist() if isinstance(data, np.ndarray) else data
                        found_gt = True
                        break
                if not found_gt:
                    ground_truths_list = [""] * B

                expanded_raw_prompts = [p for p in raw_prompts_list for _ in range(N)]
                expanded_ground_truths = [g for g in ground_truths_list for _ in range(N)]
                
                batch_keys_to_pop = ['input_ids', 'attention_mask', 'position_ids']
                possible_non_tensor_keys = ['raw_prompt_ids', 'multi_modal_data', 'raw_prompt', 'tools_kwargs', 'interaction_kwargs', 'index', 'agent_name']
                non_tensor_batch_keys_to_pop = [k for k in possible_non_tensor_keys if k in batch.non_tensor_batch]
                
                gen_batch = batch.pop(batch_keys=batch_keys_to_pop, non_tensor_batch_keys=non_tensor_batch_keys_to_pop)
                gen_batch.meta_info['global_steps'] = self.global_steps
                gen_batch = gen_batch.repeat(repeat_times=N, interleave=True)

                # --- Step 2: Student Generation ---
                with marked_timer("gen_student", timing_raw):
                    if not self.async_rollout_mode:
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                    else:
                        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                    
                    batch = batch.repeat(repeat_times=N, interleave=True)
                    batch = batch.union(gen_batch_output)

                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )
                if "response_mask" not in batch.batch.keys():
                    batch.batch["response_mask"] = compute_response_mask(batch)
                batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                # --- Step 3: Reflection & Reward ---
                with marked_timer("reward_reflection", timing_raw, color="yellow"):
                    # 3.1 Extract Prompts
                    tensor_prompts = batch.batch['prompts']
                    decoded_prompts = []
                    for p_ids in tensor_prompts:
                        decoded_prompts.append(self.tokenizer.decode(p_ids, skip_special_tokens=False))
                    
                    # 3.2 Extract GT
                    current_ground_truths = []
                    if "reward_model" in batch.non_tensor_batch:
                        rm_data = batch.non_tensor_batch["reward_model"]
                        current_ground_truths = [item.get('ground_truth', '') if isinstance(item, dict) else '' for item in rm_data]
                    elif "solution" in batch.non_tensor_batch:
                        current_ground_truths = batch.non_tensor_batch["solution"].tolist()
                    elif "ground_truth" in batch.non_tensor_batch:
                        current_ground_truths = batch.non_tensor_batch["ground_truth"].tolist()
                    else:
                        current_ground_truths = [""] * len(tensor_prompts)

                    # 3.3 Student Generates Summary
                    summary_input_batch = self._prepare_summary_generation_batch(
                        batch, 
                        decoded_prompts, 
                        current_ground_truths
                    )
                    
                    summary_output = self.actor_rollout_wg.generate_sequences(summary_input_batch)
                    summaries = summary_output.batch['responses']
                    
                    # 3.4 Teacher Computes LogProb
                    teacher_batch = self._prepare_teacher_forward_batch(batch, summaries, current_ground_truths)
                    
                    teacher_log_prob_output = self.ref_policy_wg.compute_ref_log_prob(teacher_batch)
                    teacher_full_log_probs = teacher_log_prob_output.batch['ref_log_prob']
                    
                    # === 新增: 获取 Teacher 的 Argmax Token ID ===
                    # 注意：如果 dp_actor 没改好，这里会报错 KeyError
                    teacher_full_argmax = teacher_log_prob_output.batch['ref_argmax'] 

                    # === FIX: 将 Teacher 的 LogProbs 从 Left Pad 转为 Right Pad ===
                    t_resp_mask = teacher_batch.batch['response_mask']
                    teacher_full_log_probs_aligned = left_to_right_padding(teacher_full_log_probs, t_resp_mask)
                    teacher_full_argmax_aligned = left_to_right_padding(teacher_full_argmax, t_resp_mask) # <--- 对齐 Argmax

                    # === FIX: 形状对齐 (Padding 修复) ===
                    # Teacher 的 LogProb 长度可能只是当前 Batch 的最大长度 (例如 5824)
                    # 而 Student 的 Responses 可能被 Pad 到了全局最大长度 (例如 14336)
                    # 我们需要将 Teacher LogProb Pad 到与 Student 一致
                    
                    student_resp_len = batch.batch['responses'].shape[1]
                    teacher_logp_len = teacher_full_log_probs_aligned.shape[1]
                    
                    if teacher_logp_len < student_resp_len:
                        # 需要 Padding
                        pad_len = student_resp_len - teacher_logp_len
                        # 构造全 0 的 Padding (LogProb 为 0 意味着概率为 1，这不太对，但在 Mask 之外无所谓)
                        # 或者用极小值，但在 MaskedMean 中会被忽略，所以 0 也可以
                        padding = torch.zeros(
                            (teacher_full_log_probs_aligned.shape[0], pad_len),
                            dtype=teacher_full_log_probs_aligned.dtype,
                            device=teacher_full_log_probs_aligned.device
                        )
                        # 拼接 (假设是 Right Padding)
                        teacher_full_log_probs_aligned = torch.cat([teacher_full_log_probs_aligned, padding], dim=1)
                    
                    elif teacher_logp_len > student_resp_len:
                        # 理论上不应该发生，但防一手：截断
                        teacher_full_log_probs_aligned = teacher_full_log_probs_aligned[:, :student_resp_len]

                    # 赋值回 batch
                    batch.batch['ref_log_prob'] = teacher_full_log_probs_aligned
                    
                    # === FIX: 强制 Student LogProb 计算使用 Temp=1.0 ===
                    batch.meta_info['temperature'] = 1.0 
                    student_log_prob_output = self.actor_rollout_wg.compute_log_prob(batch)
                    student_full_log_probs = student_log_prob_output.batch['old_log_probs']
                    
                    # === Log Part 1: Generation Content (Debug Info) ===
                    if True:
                        print(f"\n{'='*20} Teacher-Student Reflection Debug (Step {self.global_steps}) {'='*20}")
                        try:
                            idx = 0 
                            # Summary Input
                            sum_in_ids = summary_input_batch.batch['input_ids'][idx]
                            sum_in_ids = sum_in_ids[sum_in_ids != self.tokenizer.pad_token_id]
                            sum_in_text = self.tokenizer.decode(sum_in_ids, skip_special_tokens=False)
                            
                            # Summary Output
                            s_ids = summaries[idx]
                            s_ids = s_ids[s_ids != self.tokenizer.pad_token_id]
                            s_text = self.tokenizer.decode(s_ids, skip_special_tokens=True)
                            
                            # Teacher Input
                            t_full_ids = teacher_batch.batch['input_ids'][idx]
                            t_full_ids = t_full_ids[t_full_ids != self.tokenizer.pad_token_id]
                            t_full_text = self.tokenizer.decode(t_full_ids, skip_special_tokens=False)

                            # === FIX: 获取 Student Response 和 Ground Truth 进行对比 ===
                            # 1. 获取 GT
                            gt_raw = current_ground_truths[idx]
                            gt_ans = extract_answer(gt_raw)
                            if gt_ans == "[No Answer]" and len(gt_raw) < 20 and type(gt_raw) == str: 
                                gt_ans = gt_raw.strip().replace(',', '')
                            # 2. 获取 Student Response
                            s_ids = batch.batch['responses'][idx]
                            s_ids = s_ids[s_ids != self.tokenizer.pad_token_id]
                            s_resp_text = self.tokenizer.decode(s_ids, skip_special_tokens=True)
                            s_ans = extract_answer(s_resp_text)
                            
                            # 3. 判定
                            # 简单的字符串匹配 (对于数学题通常足够)
                            # 移除逗号以防 "1,000" vs "1000"
                            if type(gt_raw) == list:
                                gt_ans = str(gt_raw)
                                is_correct = (s_ans.split(',') in gt_raw)
                            else:
                                is_correct, s_ans = is_correct_minerva(s_resp_text, gt_ans)
                            status_icon = "✅" if is_correct else "❌"
                            
                            print(f"--- [Answer Check] ---")
                            print(f"Ground Truth Raw: {str(gt_raw)[-50:].strip()}...") # 只打印最后一点
                            print(f"Student Answer:   {s_ans}")
                            print(f"Target Answer:    {gt_ans}")
                            print(f"Result:           {status_icon} (Match: {is_correct})")
                            print(f"----------------------\n")
                            # ==========================================================

                            print(f"--- [0] Summary Generation Input ---\n{sum_in_text.strip()}\n")
                            print(f"--- [1] Summary Output ---\n{s_text.strip()}\n")
                            print(f"--- [2] Teacher LogProb Input (Full Context) ---\n{t_full_text.strip()}\n")

                            # =================================================================
                            # === NEW DEBUG: 让 Teacher 接着 Student 的 response 继续生成 ===
                            # =================================================================
                            print(f"--- [3.5] Teacher Continuation Probe ---")
                            try:
                                # 1. 提取当前样本 (Prompt + Student Response)
                                # teacher_batch.batch['input_ids'] 已经是 Left-Padded，且去除了右侧的 Pad
                                # 结构为: [Pad, Pad, ..., Prompt, Student_Response]
                                cont_input_ids = teacher_batch.batch['input_ids'][idx].unsqueeze(0)
                                cont_att_mask = teacher_batch.batch['attention_mask'][idx].unsqueeze(0)
                                cont_pos_ids = teacher_batch.batch['position_ids'][idx].unsqueeze(0)
                                
                                # 2. 构造单样本 Batch
                                cont_batch = DataProto.from_dict({
                                    "input_ids": cont_input_ids,
                                    "attention_mask": cont_att_mask,
                                    "position_ids": cont_pos_ids
                                })
                                
                                # 3. 获取 World Size 并复制样本 (必须填满所有 GPU，否则无法分发)
                                world_size = self.ref_policy_wg.world_size
                                cont_batch = cont_batch.repeat(repeat_times=world_size, interleave=True)
                                
                                # 4. 设置生成参数 (Greedy Decoding)
                                cont_batch.meta_info = {
                                    "do_sample": False,
                                    "max_new_tokens": 128, # 让 Teacher 续写最多 128 个 Token
                                    "temperature": 1.0,
                                    "micro_batch_size": world_size, 
                                    "use_dynamic_bsz": False,
                                    "eos_token_id": self.teacher_tokenizer.eos_token_id,
                                    "pad_token_id": self.teacher_tokenizer.pad_token_id,
                                }
                                
                                # 5. 调用 Teacher 生成 (续写)
                                cont_output = self.ref_policy_wg.generate_sequences(cont_batch)
                                
                                # 6. 解码 Teacher 续写的内容
                                cont_resp_ids = cont_output.batch['responses'][0]
                                # 过滤掉 padding
                                cont_resp_ids = cont_resp_ids[cont_resp_ids != self.teacher_tokenizer.pad_token_id]
                                cont_resp_text = self.teacher_tokenizer.decode(cont_resp_ids, skip_special_tokens=True)
                                
                                print(f"Student ended at:  ... {s_resp_text[-50:].strip()}")
                                print(f"Teacher continues: \n>> {cont_resp_text.strip()} <<\n")
                                
                            except Exception as e:
                                print(f"[Teacher Continuation Failed] {e}")
                                print(">>> 提示: 如果报错 'NoneType' object has no attribute 'generate_sequences'，")
                                print(">>> 是因为在 init_workers 中 Teacher 的 role 设置为了 'ref'，导致未初始化 vLLM。")
                                print(">>> 如需运行此 Debug，需将 Teacher 的 role 改为 'actor_rollout_ref' 或 'rollout'。")
                            # =================================================================
                            
                            print(f"--- [4] Deep Inspection of Teacher Batch ---")
                            # 打印前 10 个 Input IDs 和 Attention Mask
                            print(f"Input IDs (first 20): {teacher_batch.batch['input_ids'][idx][:20].tolist()}")
                            print(f"Attn Mask (first 20): {teacher_batch.batch['attention_mask'][idx][:20].tolist()}")
                            print(f"Input IDs (last 20): {teacher_batch.batch['input_ids'][idx][-20:].tolist()}")
                            
                            # 检查 Response Mask 对应的 Input IDs 是否真的是 Response
                            rmask = teacher_batch.batch['response_mask'][idx].bool()
                            target_ids = teacher_batch.batch['input_ids'][idx][rmask]
                            print(f"Target IDs from Mask: {target_ids[:10].tolist()}")
                            print(f"Student Response IDs: {batch.batch['responses'][idx][:10].tolist()}")

                        except Exception as e:
                            print(f"Log Error 1: {e}")
                            import traceback
                            traceback.print_exc()
                        print(f"{'='*60}\n")
                    # ======================================

                    # 3.6 Reward Calculation (仅用于 Logging, K2 Loss 不需要 Reward)
                    token_level_rewards = torch.zeros_like(student_full_log_probs)
                    s_probs = student_full_log_probs
                    t_probs = teacher_full_log_probs_aligned 

                    min_len = min(s_probs.shape[1], t_probs.shape[1])
                    s_part = s_probs[:, :min_len]
                    t_part = t_probs[:, :min_len]
                    
                    # 计算原始差异 (用于观察)
                    kl_diff = t_part - s_part
                    distill_reward = kl_diff

                    # =================================================================
                    # === FIX: Add Debug Metrics (Logits Statistics) ===
                    # =================================================================
                    with torch.no_grad():
                        # 1. 获取有效 Mask (切片以匹配 min_len)
                        # 必须过滤掉 Padding，否则 Min LogP 会显示 -14.82 (Padding 的 LogP)
                        valid_mask = batch.batch['response_mask'][:, :min_len].bool()
                        
                        if valid_mask.any():
                            # 2. 提取有效数据 (Flatten)
                            s_valid = s_part[valid_mask]
                            t_valid = t_part[valid_mask]
                            diff_valid = kl_diff[valid_mask]
                            
                            # 3. Student Stats
                            metrics['debug/student_logp_max'] = s_valid.max().item()
                            metrics['debug/student_logp_min'] = s_valid.min().item()
                            metrics['debug/student_logp_mean'] = s_valid.mean().item()
                            
                            # 4. Teacher Stats
                            metrics['debug/teacher_logp_max'] = t_valid.max().item()
                            metrics['debug/teacher_logp_min'] = t_valid.min().item()
                            metrics['debug/teacher_logp_mean'] = t_valid.mean().item()
                            
                            # 5. Diff Stats (Reward Raw)
                            metrics['debug/diff_logp_max'] = diff_valid.max().item()
                            metrics['debug/diff_logp_min'] = diff_valid.min().item()
                            metrics['debug/diff_logp_mean'] = diff_valid.mean().item()
                            
                        # === NEW: Sequence Length Metrics ===
                        # 安全获取 Pad Token ID (防止没有 pad_token_id 的情况)
                        s_pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
                        t_pad_id = self.teacher_tokenizer.pad_token_id if self.teacher_tokenizer.pad_token_id is not None else self.teacher_tokenizer.eos_token_id

                        # 1. Prompt Length (原始问题长度)
                        prompt_lens = (batch.batch['prompts'] != s_pad_id).sum(dim=-1).float()
                        metrics['debug/prompt_len_mean'] = prompt_lens.mean().item()
                        metrics['debug/prompt_len_max'] = prompt_lens.max().item()
                        metrics['debug/prompt_len_min'] = prompt_lens.min().item()
                        
                        # 2. Student Response Length (学生回答长度)
                        student_resp_lens = (batch.batch['responses'] != s_pad_id).sum(dim=-1).float()
                        metrics['debug/student_resp_len_mean'] = student_resp_lens.mean().item()
                        metrics['debug/student_resp_len_max'] = student_resp_lens.max().item()
                        metrics['debug/student_resp_len_min'] = student_resp_lens.min().item()
                        
                        # 3. Teacher Summary Length (老师总结的长度)
                        summary_lens = (summaries != t_pad_id).sum(dim=-1).float()
                        metrics['debug/teacher_summary_len_mean'] = summary_lens.mean().item()
                        metrics['debug/teacher_summary_len_max'] = summary_lens.max().item()
                        metrics['debug/teacher_summary_len_min'] = summary_lens.min().item()
                        
                        # 4. Teacher Forward Length (老师重新计算概率时的总上下文长度)
                        teacher_forward_lens = (teacher_batch.batch['input_ids'] != t_pad_id).sum(dim=-1).float()
                        metrics['debug/teacher_forward_len_mean'] = teacher_forward_lens.mean().item()
                        metrics['debug/teacher_forward_len_max'] = teacher_forward_lens.max().item()
                        metrics['debug/teacher_forward_len_min'] = teacher_forward_lens.min().item()

                    # =================================================================

                    # === Log Part 2: Token-Level KL Analysis ===
                    if True:
                        print(f"\n{'='*20} Token-Level KL Analysis (Step {self.global_steps}) {'='*20}")
                        try:
                            idx = 0 
                            # 1. Student Response (Remove Pad)
                            s_ids = batch.batch['responses'][idx]
                            s_ids = s_ids[s_ids != self.tokenizer.pad_token_id]
                            
                            # 2. Teacher Response (Extract using Mask)
                            t_full_ids = teacher_batch.batch['input_ids'][idx]
                            t_mask = teacher_batch.batch['response_mask'][idx]
                            t_ids = t_full_ids[t_mask.bool()]
                            t_ids = t_ids[t_ids != self.tokenizer.pad_token_id]
                            
                            # 3. Decode
                            s_text_check = self.tokenizer.decode(s_ids, skip_special_tokens=False)
                            t_text_check = self.tokenizer.decode(t_ids, skip_special_tokens=False)
                            
                            print(f"--- Sequence Alignment Check ---")

                            
                            if len(s_ids) != len(t_ids) or not torch.equal(s_ids, t_ids):
                                print("[WARNING] ID Mismatch! Teacher sees different tokens than Student generated!")
                                print(f"S IDs: {s_ids.tolist()}")
                                print(f"T IDs: {t_ids.tolist()}")
                                print(f"Student Seq (Len={len(s_ids)}): {s_text_check}...")
                                print(f"Teacher Seq (Len={len(t_ids)}): {t_text_check}...")
                            else:
                                print("[OK] Token IDs match perfectly.")

                            print(f"\n--- Token-wise KL Breakdown ---")
                            print(f"{'Token':<15} | {'ID':<6} | {'S_LogP':<8} | {'T_LogP':<8} | {'Diff':<8} | {'Teacher Preferred'}")
                            print("-" * 95)
                            s_vals = s_part[idx].tolist()
                            t_vals = t_part[idx].tolist() # 这里引用的 t_part 已经是 Right Padded 的了
                            diff_vals = kl_diff[idx].tolist()
                            # 获取当前样本的 Teacher Preferred Tokens
                            t_preferred_ids = teacher_full_argmax_aligned[idx].tolist()
                            valid_count = 0
                            for i in range(len(s_ids)):
                                # if valid_count >= 50: break
                                if i >= len(s_vals): break
                                
                                tid = s_ids[i].item()
                                token_str = self.tokenizer.decode([tid]).replace('\n', '\\n')
                                                                # 获取 Teacher 想要什么
                                t_pref_id = t_preferred_ids[i]
                                # 如果是 Padding (0)，显示为 Pad
                                if i >= len(t_ids): # 简单的越界保护
                                    t_pref_str = "[PAD]"
                                else:
                                    t_pref_str = self.tokenizer.decode([t_pref_id]).replace('\n', '\\n').strip()
                                diff_val = diff_vals[i]
                                
                                # 格式化输出
                                row_str = f"{token_str:<15} | {tid:<6} | {s_vals[i]:.2f}   | {t_vals[i]:.2f}   | {diff_val:.2f}   "
                                
                                if np.abs(diff_val) < 0.1:
                                    print(f"{row_str} | {t_pref_str}")
                                else:
                                    # 差异大时，高亮显示 Teacher 想要的 Token
                                    print(f"{row_str} | >> {t_pref_str} <<")
                                valid_count += 1
                                
                        except Exception as e:
                            print(f"Log Error 2: {e}")
                        print(f"{'='*60}\n")
                    # ==========================================
                    
                    # 记录 diff 到 reward 以便 tensorboard 观察，但不截断
                    token_level_rewards[:, :min_len] = kl_diff
                    batch.batch['token_level_rewards'] = token_level_rewards
                    
                    valid_mask = batch.batch['response_mask']
                    with torch.no_grad():
                        mean_kl = (token_level_rewards * valid_mask).sum() / (valid_mask.sum() + 1e-6)
                        metrics['reward/reflection_kl'] = mean_kl.item()

                # --- Step 4: Update Flow (K2 Mode) ---
                entropys = student_log_prob_output.batch['entropys']
                entropy_agg = torch.mean(entropys)
                metrics['actor/entropy'] = entropy_agg.item()
                
                student_log_prob_output.batch.pop('entropys', None)
                batch = batch.union(student_log_prob_output)

                if "values" not in batch.batch.keys():
                    batch.batch["values"] = torch.zeros_like(batch.batch["token_level_rewards"])

                # === 核心修改: 禁用 GAE (compute_advantage) ===
                # 专家建议使用 K2 Loss (MSE)，这意味着不需要 GAE 的时间平滑。
                # 直接将 advantages 设为 0 (或者 diff)，因为 dp_actor 会忽略它，直接用 ref_log_prob 算 MSE。
                # 为了防止 verl 内部报错，我们保留数据结构。
                batch.batch['advantages'] = torch.zeros_like(batch.batch['token_level_rewards'])
                batch.batch['returns'] = torch.zeros_like(batch.batch['token_level_rewards'])
                
                # 原有的 compute_advantage 被注释掉，避免 GAE 引入偏差
                # with marked_timer("adv", timing_raw):
                #     batch = compute_advantage(...)

                with marked_timer("update_actor", timing_raw):
                    actor_output = self.actor_rollout_wg.update_actor(batch)
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)
                
                logger.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)
                self.global_steps += 1
                
                if self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0:
                    self._save_checkpoint()

                if self.global_steps >= self.total_training_steps:
                    progress_bar.close()
                    return