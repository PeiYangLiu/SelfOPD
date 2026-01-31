import torch
import numpy as np
from typing import List, Dict, Optional
from copy import deepcopy
from torch.nn.utils.rnn import pad_sequence
import re
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_response_mask, compute_advantage
from verl.utils.metric import (
    reduce_metrics,
)
import uuid
from tqdm import tqdm
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.utils.debug import marked_timer
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import Tracking
from omegaconf import OmegaConf, open_dict


class SelfReflectiveGRPOTrainer(RayPPOTrainer):
    """
    Self-Reflective GRPO Trainer.
    Mechanism:
    1. Generate Response (Student): [Prompt] -> [Response]
    2. Generate Summary (Reflection): [Prompt, Response, Ground Truth] -> [Summary]
    3. Teacher Forward: Compute logits for [Prompt, Summary, Response]
    4. Student Forward: Compute logits for [Prompt, Response]
    5. Reward: KL(Teacher || Student) on the Response tokens.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_critic = False
        self.use_rm = False
        
        if self.config.algorithm.adv_estimator not in ['grpo', 'grpo_outcome', 'grpo_passk']:
            print(f"Warning: You are using SelfReflectiveTrainer but adv_estimator is {self.config.algorithm.adv_estimator}.")

        print(">>> SelfReflectiveGRPOTrainer Initialized.")

    def _prepare_summary_generation_batch(self, batch: DataProto, raw_prompts: List[str], ground_truths: List[str]) -> DataProto:
        """
        构造用于生成总结 (Reflection) 的 Batch。
        修复：手动实现 Left Padding，并构造正确的 position_ids。
        """
        responses = batch.batch['responses']
        
        assert len(raw_prompts) == len(responses)
        assert len(ground_truths) == len(responses)

        input_ids_list = []
        attention_mask_list = []

        for i, (r_ids, gt_text) in enumerate(zip(responses, ground_truths)):
            p_text_dirty = raw_prompts[i]
            
            # ... (正则清洗逻辑保持不变) ...
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
            p_text_clean = p_text_clean.strip()
            # ... (清洗结束) ...

            r_text = self.tokenizer.decode(r_ids, skip_special_tokens=True)
            
            content = (
                f"Question: {p_text_clean}\n\n"
                f"Standard Answer: {gt_text}\n\n"
                f"Student Answer: {r_text}\n\n"
                f"Task: Verify the Student Answer step-by-step. Is it correct?\n"
                f"Answer concisely."
            )

            messages = [
                {"role": "system", "content": "You are a helpful math assistant."},
                {"role": "user", "content": content}
            ]

            enc_ids = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=True, 
                add_generation_prompt=True,
                return_tensors='pt'
            )
            
            if enc_ids.dim() == 2:
                enc_ids = enc_ids[0]

            input_ids_list.append(enc_ids)
            attention_mask_list.append(torch.ones_like(enc_ids))

        # ==================== 修复: 手动 Left Padding & Position IDs ====================
        max_len = max([len(t) for t in input_ids_list])
        pad_token_id = self.tokenizer.pad_token_id
        
        padded_input_ids = []
        padded_attention_mask = []
        padded_position_ids = []
        
        for ids, mask in zip(input_ids_list, attention_mask_list):
            pad_len = max_len - len(ids)
            
            # 构造 Padding
            pad_ids = torch.full((pad_len,), pad_token_id, dtype=ids.dtype, device=ids.device)
            pad_mask = torch.full((pad_len,), 0, dtype=mask.dtype, device=mask.device)
            
            # 拼接: [Pad, Content]
            final_ids = torch.cat([pad_ids, ids])
            final_mask = torch.cat([pad_mask, mask])
            
            padded_input_ids.append(final_ids)
            padded_attention_mask.append(final_mask)
            
            # 构造 Position IDs (Left Padding)
            # 对于 Left Padding:
            # Mask: [0, 0, 1, 1, 1]
            # Pos:  [0, 0, 0, 1, 2] (或者直接 cumsum - 1, 只要非 Pad 部分是从 0 开始递增即可)
            
            # 方法 1: 简单的 cumsum - 1 (vLLM 通常能处理，只要 mask 对了)
            # pos = torch.cumsum(final_mask, dim=-1) - 1
            # pos.masked_fill_(final_mask == 0, 0)
            
            # 方法 2: 严格对齐 (Pad 部分为 0, Content 部分从 0 到 L-1)
            # 这种方式对于 vLLM + Left Padding 最稳妥
            seq_len = len(ids)
            pos_content = torch.arange(seq_len, dtype=torch.long, device=ids.device)
            pos_pad = torch.zeros(pad_len, dtype=torch.long, device=ids.device)
            final_pos = torch.cat([pos_pad, pos_content])
            
            padded_position_ids.append(final_pos)
            
        input_ids = torch.stack(padded_input_ids)
        attention_mask = torch.stack(padded_attention_mask)
        position_ids = torch.stack(padded_position_ids)
        # ==============================================================================

        summary_batch = DataProto.from_dict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids # 必须包含
        })
        
        summary_batch.meta_info = {
            "do_sample": True,
            "temperature": 0.1, 
            "max_new_tokens": 512,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id
        }
        
        return summary_batch

    def _prepare_teacher_forward_batch(self, batch: DataProto, summaries: torch.Tensor) -> DataProto:
        """
        构造 Teacher Forward 的 Batch: [Prompt, Summary, Response]
        修复：移除 Summary 的 EOS Token，并插入更明确的分隔符。
        """
        prompts = batch.batch['prompts']
        responses = batch.batch['responses']
        
        new_input_ids = []
        new_attention_mask = []
        new_response_masks = []
        new_prompts_list = []
        new_responses_list = []

        # 构造更明确的分隔符
        sep_ids = self.tokenizer.encode("\n\nOriginal Response:\n", add_special_tokens=False)
        separator = torch.tensor(sep_ids, device=prompts.device)

        for p, s, r in zip(prompts, summaries, responses):
            p_real = p[p != self.tokenizer.pad_token_id]
            s_real = s[s != self.tokenizer.pad_token_id]
            r_real = r[r != self.tokenizer.pad_token_id]
            
            # 1. 处理 Summary: 移除末尾的 EOS Token
            if s_real.numel() > 0 and s_real[-1] == self.tokenizer.eos_token_id:
                s_real = s_real[:-1] 
            
            # 2. 构造 Teacher Prompt: [Original Prompt] + [Summary (No EOS)] + [Separator]
            teacher_prompt = torch.cat([p_real, s_real, separator])
            
            # 3. Teacher Response: [Original Response]
            teacher_response = r_real
            
            # 4. Full Input
            combined_ids = torch.cat([teacher_prompt, teacher_response])
            
            att_mask = torch.ones_like(combined_ids)
            resp_mask = torch.zeros_like(combined_ids)
            resp_mask[len(teacher_prompt):] = 1 
            
            new_input_ids.append(combined_ids)
            new_attention_mask.append(att_mask)
            new_response_masks.append(resp_mask)
            
            new_prompts_list.append(teacher_prompt)
            new_responses_list.append(teacher_response)

        # Padding
        input_ids = pad_sequence(new_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence(new_attention_mask, batch_first=True, padding_value=0)
        response_mask = pad_sequence(new_response_masks, batch_first=True, padding_value=0)
        
        prompts_padded = pad_sequence(new_prompts_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        responses_padded = pad_sequence(new_responses_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        
        position_ids = torch.cumsum(attention_mask, dim=-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)

        teacher_batch = DataProto.from_dict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "response_mask": response_mask,
            "prompts": prompts_padded,   
            "responses": responses_padded 
        })
        
        return teacher_batch

    def fit(self):
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf
        from tqdm import tqdm
        import uuid
        import numpy as np

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        self.global_steps += 1

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                
                # Pop keys
                batch_keys_to_pop = ['input_ids', 'attention_mask', 'position_ids']
                possible_non_tensor_keys = ['raw_prompt_ids', 'multi_modal_data', 'raw_prompt', 'tools_kwargs', 'interaction_kwargs', 'index', 'agent_name']
                non_tensor_batch_keys_to_pop = [k for k in possible_non_tensor_keys if k in batch.non_tensor_batch]
                
                gen_batch = batch.pop(batch_keys=batch_keys_to_pop, non_tensor_batch_keys=non_tensor_batch_keys_to_pop)
                gen_batch.meta_info['global_steps'] = self.global_steps
                
                # Tensor Repeat
                N = self.config.actor_rollout_ref.rollout.n
                gen_batch = gen_batch.repeat(repeat_times=N, interleave=True)

                # Step 2: Student Generation
                with marked_timer("gen_student", timing_raw):
                    if not self.async_rollout_mode:
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                    else:
                        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                    
                    batch = batch.repeat(repeat_times=N, interleave=True)
                    batch = batch.union(gen_batch_output)

                # Step 3: Meta Info Injection
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )
                if "response_mask" not in batch.batch.keys():
                    batch.batch["response_mask"] = compute_response_mask(batch)
                
                batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                # Step 4: Reflection & Reward
                with marked_timer("reward_reflection", timing_raw, color="yellow"):
                    # ==================== 修复: 从 Tensor 逆向提取 Prompt ====================
                    # 我们不再信任手动扩展的 list，直接从 batch.batch['prompts'] 解码
                    # batch.batch['prompts'] 是生成过程中使用的 Prompt Token IDs，绝对与 Response 对齐
                    
                    tensor_prompts = batch.batch['prompts'] # (B*N, Len)
                    
                    # 提取 Ground Truth (这个只能从 non_tensor_batch 拿)
                    # 假设 batch.repeat 对 non_tensor_batch 的处理是正确的 (通常是 numpy repeat)
                    # 如果这步也错位，那 verl 的 repeat 逻辑就有大问题，但我们先假设它是对的
                    current_ground_truths = []
                    
                    # 尝试获取 GT
                    if "reward_model" in batch.non_tensor_batch:
                        rm_data = batch.non_tensor_batch["reward_model"]
                        # rm_data 应该已经是扩展后的长度 (B*N)
                        current_ground_truths = [item.get('ground_truth', '') if isinstance(item, dict) else '' for item in rm_data]
                    elif "solution" in batch.non_tensor_batch:
                        current_ground_truths = batch.non_tensor_batch["solution"].tolist()
                    elif "ground_truth" in batch.non_tensor_batch:
                        current_ground_truths = batch.non_tensor_batch["ground_truth"].tolist()
                    else:
                        current_ground_truths = [""] * len(tensor_prompts)

                    # 调用 prepare_summary，传入 tensor_prompts (需要先解码)
                    # 为了避免在主进程做大量解码卡顿，我们可以只解码 raw_prompts 需要的部分
                    # 但为了对齐，我们必须解码
                    
                    decoded_prompts = []
                    for p_ids in tensor_prompts:
                        decoded_prompts.append(self.tokenizer.decode(p_ids, skip_special_tokens=True))
                    
                    # A. Summaries
                    summary_input_batch = self._prepare_summary_generation_batch(
                        batch, 
                        decoded_prompts, # 使用从 Tensor 解码的 Prompt
                        current_ground_truths
                    )
                    summary_output = self.actor_rollout_wg.generate_sequences(summary_input_batch)
                    summaries = summary_output.batch['responses']
                    # ========================================================================
                    
                    # B. Student LogProbs
                    student_log_prob_output = self.actor_rollout_wg.compute_log_prob(batch)
                    student_full_log_probs = student_log_prob_output.batch['old_log_probs']
                    
                    # C. Teacher LogProbs
                    teacher_batch = self._prepare_teacher_forward_batch(batch, summaries)
                    
                    # === Debug Logging ===
                    if self.global_steps % 10 == 1:
                        print(f"\n{'='*20} Self-Reflection Debug (Step {self.global_steps}) {'='*20}")
                        try:
                            idx = 0 
                            # 1. 验证 Prompt (从 Tensor 解码)
                            p_text_tensor = decoded_prompts[idx]
                            gt_text = current_ground_truths[idx]
                            
                            # 2. Student Response
                            r_ids = batch.batch['responses'][idx]
                            r_text = self.tokenizer.decode(r_ids, skip_special_tokens=True)
                            
                            # 3. Summary Model Input (ACTUAL)
                            sum_in_ids = summary_input_batch.batch['input_ids'][idx]
                            sum_in_text = self.tokenizer.decode(sum_in_ids, skip_special_tokens=False)

                            # 4. Summary Output
                            s_ids = summaries[idx]
                            s_text = self.tokenizer.decode(s_ids, skip_special_tokens=True)

                            print(f"--- [0] Summary Model Input (ACTUAL) ---\n{sum_in_text.strip()}\n")
                            print(f"--- [1] Prompt (From Tensor) ---\n{p_text_tensor.strip()}...\n")
                            print(f"--- [2] Ground Truth ---\n{gt_text.strip()}\n")
                            print(f"--- [3] Student Response ---\n{r_text.strip()}...\n")
                            print(f"--- [4] Summary Output ---\n{s_text.strip()}...\n")
                            
                            # 再次检查一致性 (Prompt vs Summary Input)
                            # 清洗后的 Prompt 应该出现在 Summary Input 中
                            if p_text_tensor[:20] not in sum_in_text and "Question:" not in sum_in_text:
                                print("\n[WARNING] Potential mismatch or cleaning issue!")

                        except Exception as e:
                            print(f"Log Error: {e}")
                        print(f"{'='*60}\n")
                    # =====================

                    teacher_log_prob_output = self.actor_rollout_wg.compute_log_prob(teacher_batch)
                    teacher_full_log_probs = teacher_log_prob_output.batch['old_log_probs']

                    # D. KL Reward
                    token_level_rewards = torch.zeros_like(student_full_log_probs)
                    s_probs = student_full_log_probs
                    t_probs = teacher_full_log_probs
                    min_len = min(s_probs.shape[1], t_probs.shape[1])
                    s_part = s_probs[:, :min_len]
                    t_part = t_probs[:, :min_len]
                    kl_diff = t_part - s_part
                    token_level_rewards[:, :min_len] = kl_diff
                    batch.batch['token_level_rewards'] = token_level_rewards
                    
                    valid_mask = batch.batch['response_mask']
                    with torch.no_grad():
                        mean_kl = (token_level_rewards * valid_mask).sum() / (valid_mask.sum() + 1e-6)
                        metrics['reward/reflection_kl'] = mean_kl.item()

                # Step 5: PPO Flow
                entropys = student_log_prob_output.batch['entropys']
                entropy_agg = torch.mean(entropys)
                metrics['actor/entropy'] = entropy_agg.item()
                
                student_log_prob_output.batch.pop('entropys', None)
                batch = batch.union(student_log_prob_output)

                if "values" not in batch.batch.keys():
                    batch.batch["values"] = torch.zeros_like(batch.batch["token_level_rewards"])

                with marked_timer("adv", timing_raw):
                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                        num_repeat=self.config.actor_rollout_ref.rollout.n,
                        norm_adv_by_std_in_grpo=self.config.algorithm.get('norm_adv_by_std_in_grpo', True),
                        config=self.config.algorithm
                    )

                with marked_timer("update_actor", timing_raw):
                    actor_output = self.actor_rollout_wg.update_actor(batch)
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)

                metrics.update({
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch
                })
                
                logger.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)
                self.global_steps += 1
                
                if self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0:
                    self._save_checkpoint()

                if self.global_steps >= self.total_training_steps:
                    progress_bar.close()
                    return