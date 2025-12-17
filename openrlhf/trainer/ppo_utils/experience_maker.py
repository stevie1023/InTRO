import time
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import ray
import torch
import torch.distributed as dist
import torch.nn as nn
from tqdm import tqdm

from openrlhf.models.actor import Actor
from openrlhf.models.ring_attn_utils import pad_sequences, unpad_sequences
from openrlhf.models.utils import compute_approx_kl, compute_reward, masked_mean, unpacking_samples
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.remote_rm_utils import remote_rm_fn, remote_rm_fn_ray
import torch.nn.functional as F


logger = init_logger(__name__)

def zero_pad_sequences(sequences, side: str = "left", value=0):
    assert side in ("left", "right")
    max_len = max(seq.size(-1) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(-1)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=value))
    return torch.stack(padded_sequences, dim=0)


def to(tensor: Union[torch.Tensor, list[torch.Tensor]], device):
    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    return tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor


def pin_memory(tensor: Union[torch.Tensor, list[torch.Tensor]]):
    if isinstance(tensor, list):
        return [pin_memory(t) for t in tensor]
    return tensor.pin_memory() if isinstance(tensor, torch.Tensor) else tensor

import re 
def preprocess_box_response_for_qwen_prompt(sequence, answer):
    # breakpoint()
    model_output= re.sub(r'^.*?<\|im_start\|>assistant', '<|im_start|>assistant', sequence, flags=re.DOTALL,count = 1)
    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"] 
    for stop_word in stop_words:
        if stop_word in model_output:
            model_output = model_output.split(stop_word)[0].strip()
    extract_answer = qwen_extract_answer(model_output, data_name="math") #TODO: check the data_name, hard code here for now
    
    
    # temp_query = ""
    # temp_response = ""

    # temp_query = sequence.split("ASSISTANT:\n")[0]

    # temp_response = sequence.split("**Final Answer**")[-1]
    
    # pattern = r'\\boxed\{(.*?)\}\s*\\\]'

    # # 使用 re.DOTALL 确保能匹配跨行文本
    # match = re.search(pattern, temp_response)
    # #match = re.search(pattern, temp_response)
    # if match:
    #     temp_answer = match.group(1)
    # else:
    #     temp_answer = "none"

    # #temp_answer = temp_response.split("\n\n# Answer\n\n")[-1]
    # #temp_response = sequence.split("<|reserved_special_token_0|>The final answer is:")[0]

    # #response_list = temp_response.split("<|reserved_special_token_0|>")

    # processed_solution = temp_response + "\n\n# Answer\n\n" + temp_answer + "<|reserved_special_token_0|>"

    # processed_solution = re.sub(r"<\|end_of_text\|>", "", processed_solution)
    
    if qwen_math_equal_subprocess(prediction=extract_answer, reference=answer):
        box_match = 1.0
    else:
        box_match = -0.5
        
    if "boxed" not in model_output:
        box_match = -1.0
        

    return "", box_match

# from openrlhf.trainer.ppo_utils.qwen_math_eval_toolkit.grader import math_equal as qwen_math_equal
def qwen_math_equal_with_timeout_ray(prediction, reference, include_percentage=True, is_close=True, timeout_duration=3):
    """
    使用Ray的超时机制对math_equal函数进行控制
    """
    @ray.remote
    def _remote_qwen_math_equal(prediction, reference, include_percentage, is_close):
        return qwen_math_equal(prediction, reference, include_percentage, is_close, timeout=False)
    
    try:
        # 启动远程任务并等待结果
        future = _remote_qwen_math_equal.remote(prediction, reference, include_percentage, is_close)
        result = ray.get(future, timeout=timeout_duration)
        return result
    except (GetTimeoutError, Exception) as e:
        # 如果超时或发生其他错误，返回False
        ray.logger.info("Math Eq eval timeout.")
        return False
    
from multiprocessing import Process, Queue
def qwen_math_equal_subprocess(prediction, reference,  timeout_seconds=10):
    def worker(q, prediction, reference):
        result = qwen_math_equal(prediction=prediction, reference=reference, timeout=False)
        q.put(result)

    q = Queue()
    p = Process(target=worker, args=(q, prediction, reference))
    p.start()
    
    # 添加超时处理
    p.join(timeout=timeout_seconds)  # 等待进程完成，最多等待 timeout_seconds 秒
    
    # 如果进程还在运行，则终止它并返回 False
    if p.is_alive():
        p.terminate()
        p.join()  # 确保进程被完全清理
        return False
        
    # 如果进程正常完成，获取结果
    try:
        return q.get_nowait()
    except:
        return False   
    
def preprocess_orm_reward(queries, tokenizer, **generate_kwargs):
    input_ids = []
    input_masks = []
    for input_for_prm in queries:
        input_token = tokenizer(
            input_for_prm,
            max_length=generate_kwargs["max_new_tokens"],
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        for input_id, input_mask in zip(input_token["input_ids"], input_token["attention_mask"]):
            input_ids.append(input_id)
            input_masks.append(input_mask)

    padding_side = "right"
    input_ids = zero_pad_sequences(input_ids, side=padding_side, value=tokenizer.pad_token_id)
    input_masks = zero_pad_sequences(input_masks, side=padding_side)

    return input_ids, input_masks

def preprocess_orm800k_box_responsev1(sequence, answer):
    temp_query = ""
    temp_response = ""

    temp_query = sequence.split("ASSISTANT:\n")[0]
    #temp_query = temp_query.split("# Question\n\n")[-1]

    temp_response = sequence.split("ASSISTANT:\n")[-1]
    
    #print("temp_response", temp_response)
    # 使用正则表达式匹配
    # pattern = r"The final answer is: \\\\boxed\{(.*?)\}"
    # # 正则表达式，处理换行与特殊字符
    # pattern = r"The final answer is: \\\\boxed\{(.*?)\}"
    #pattern = r"The final answer is: \\\\boxed\{(.*?)\}"
    pattern = r"The final answer is: \\boxed\{(.*?)\}"
    # 使用 re.DOTALL 确保能匹配跨行文本
    match = re.search(pattern, temp_response, re.DOTALL)
    #match = re.search(pattern, temp_response)
    if match:
        temp_answer = match.group(1)
    else:
        temp_answer = "none"

    #temp_answer = temp_response.split("\n\n# Answer\n\n")[-1]
    temp_response = sequence.split("<|reserved_special_token_0|>The final answer is:")[0]

    #response_list = temp_response.split("<|reserved_special_token_0|>")

    processed_solution = temp_response + "\n\n# Answer\n\n" + temp_answer + "<|reserved_special_token_0|>"

    # for i in range(len(response_list)-1):
    #     processed_solution = processed_solution + "Step " + str(i+1) + ": " + response_list[i] + " <|reserved_special_token_0|>\n"

    # processed_solution = processed_solution + response_list[-1] + " The answer is: " + temp_answer + " <|reserved_special_token_0|>\n"
    processed_solution = re.sub(r"<\|end_of_text\|>", "", processed_solution)
    
    if temp_answer == answer:
        box_match = 0.5
    else:
        box_match = -0.5
    return processed_solution, box_match


def preprocess_prm_reward(queries, tokenizer, prompt_max_len, **generate_kwargs):
    input_ids = []
    input_masks = []
    for input_for_prm in queries:
        input_token = tokenizer(
            input_for_prm,
            max_length=generate_kwargs["max_new_tokens"] + prompt_max_len,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        for input_id, input_mask in zip(input_token["input_ids"], input_token["attention_mask"]):
            input_ids.append(input_id)
            input_masks.append(input_mask)

    padding_side = "right"
    input_ids = zero_pad_sequences(input_ids, side=padding_side, value=tokenizer.pad_token_id)
    input_masks = zero_pad_sequences(input_masks, side=padding_side)

    return input_ids, input_masks


@dataclass
class Experience:
    """Experience is a batch of data.
    These data should have the the sequence length and number of actions.
    Left padding for sequences is applied.

    Shapes of each tensor:
    sequences: (B, S)
    action_log_probs: (B, A)
    base_action_log_probs: (B, A)
    values: (B, A)
    returns: (B, A)
    advantages: (B, A)
    attention_mask: (B, S)
    action_mask: (B, A)
    kl: (B, A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    reverse_sequences: torch.Tensor
    action_log_probs: torch.Tensor
    base_action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    info: Optional[dict]
    kl: Optional[torch.Tensor] = None
    rule_reward: Optional[torch.Tensor] = None

    @torch.no_grad()
    def to_device(self, device: torch.device):
        self.sequences = to(self.sequences, device)
        self.reverse_sequences = to(self.reverse_sequences, device)
        self.action_log_probs = to(self.action_log_probs, device)
        self.base_action_log_probs = to(self.base_action_log_probs, device)
        self.returns = to(self.returns, device)
        self.advantages = to(self.advantages, device)
        self.values = to(self.values, device)
        self.attention_mask = to(self.attention_mask, device)
        self.action_mask = to(self.action_mask, device)
        self.kl = to(self.kl, device)
        self.rule_reward = to(self.rule_reward, device)
        self.info = {key: to(value, device) for key, value in self.info.items()}
        return self

    def pin_memory(self):
        self.sequences = pin_memory(self.sequences)
        self.reverse_sequences = pin_memory(self.reverse_sequences)
        self.action_log_probs = pin_memory(self.action_log_probs)
        self.base_action_log_probs = pin_memory(self.base_action_log_probs)
        self.returns = pin_memory(self.returns)
        self.advantages = pin_memory(self.advantages)
        self.values = pin_memory(self.values)
        self.attention_mask = pin_memory(self.attention_mask)
        self.action_mask = pin_memory(self.action_mask)
        self.kl = pin_memory(self.kl)
        self.rule_reward = pin_memory(self.rule_reward)
        self.info = {key: pin_memory(value) for key, value in self.info.items()}
        return self


@dataclass
class Samples:
    """Samples is a batch of data.
    There can be 2 formats to store the samples, batched or packed.
    The batched format means padding is applied to the sequences, while the packed format
    will concatenate the prompt and response without padding.

    Shapes of each tensor, when 2 shapes are shown, the first one is for batched format
        and the second one is for packed format:
    sequences: (B, S) or (1, total_length), the tokens of both prompt and response.
    attention_mask: (B, S) or (1, total_length), the attention mask for sequences.
    action_mask: (B, A) or None, the action (response) mask to show which part of the
        sequence is the response. When the samples are packed, this is None.
    num_actions: int or (B,), the number of actions (tokens) in the response.
        When the samples are not packed, we will use action_mask, so this is an int to
        show the size of action_mask. Otherwise, this is a tensor to show the number of
        actions for each sample.
    packed_seq_lens: None or (B,), the length of each sample in the packed samples.
    response_length: (B,), the number of tokens in the response.
    total_length: (B,), the total number of tokens in the sequences.
    prompts: the prompts used to generate responses
    """

    sequences: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    packed_seq_lens: Optional[torch.Tensor]
    response_length: torch.Tensor
    total_length: torch.Tensor
    prompts: list[str]
    labels: list[str]
    pad_len: Optional[int]

@dataclass
class SamplesBOX:
    """Samples is a batch of data.
    There can be 2 formats to store the samples, batched or packed.
    The batched format means padding is applied to the sequences, while the packed format
    will concatenate the prompt and response without padding.

    Shapes of each tensor, when 2 shapes are shown, the first one is for batched format
        and the second one is for packed format:
    sequences: (B, S) or (1, total_length), the tokens of both prompt and response.
    attention_mask: (B, S) or (1, total_length), the attention mask for sequences.
    action_mask: (B, A) or None, the action (response) mask to show which part of the
        sequence is the response. When the samples are packed, this is None.
    num_actions: int or (B,), the number of actions (tokens) in the response.
        When the samples are not packed, we will use action_mask, so this is an int to
        show the size of action_mask. Otherwise, this is a tensor to show the number of
        actions for each sample.
    packed_seq_lens: None or (B,), the length of each sample in the packed samples.
    response_length: (B,), the number of tokens in the response.
    total_length: (B,), the total number of tokens in the sequences.
    """

    sequences: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    packed_seq_lens: Optional[torch.Tensor]
    response_length: torch.Tensor
    total_length: torch.Tensor
    answers: List[str]
    
class NaiveExperienceMaker(ABC):
    """
    Naive experience maker.
    """

    def __init__(
        self,
        actor: Actor,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: Actor,
        tokenizer,
        prompt_max_len: int,
        kl_controller,
        strategy=None,
        remote_rm_url: Union[list[str], str] = None,
        reward_fn=None,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.reward_fn = reward_fn
        self.perf_stats = None
        self.advantage_estimator = strategy.args.advantage_estimator

        # custom reward func for reinforced finetuning
        self.custom_reward_func = None
        remote_rm_url = [remote_rm_url] if isinstance(remote_rm_url, str) else remote_rm_url
        if remote_rm_url and remote_rm_url[0].endswith(".py"):
            print(f"Loading custom `reward_func(queries, prompts, labels)` from {remote_rm_url[0]}")
            import importlib.util

            spec = importlib.util.spec_from_file_location("reward_func", remote_rm_url[0])
            reward_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(reward_module)
            self.custom_reward_func = reward_module.reward_func

    # tokenizer
    def tokenize_fn(self, texts, max_length, padding=True, device=None):
        if not padding:
            # when padding is False, return tokenized texts as list
            return self.tokenizer(
                texts,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    @torch.no_grad()
    def make_experience_list(
        self, all_prompts: Union[str, List[str]], all_labels, **generate_kwargs
    ) -> List[Experience]:
        """
        Make a list of experience with the micro_rollout_batch_size.

        This method will first calculate the response sequences and rewards for the given prompts.
        Then, if we need certain processing for the rewards or do certain filtering, we can process the rollout as a whole.
        After that, we will calculate the advantages and returns for each experience.
        """
        args = self.strategy.args

        # vLLM wakeup when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

            batch_vllm_engine_call(self.vllm_engines, "wake_up")
            torch.distributed.barrier()
            torch.cuda.synchronize()
     
        # generate responses
        if self.strategy.ring_attn_group is not None:
            # Only rank 0 in the ring attention group executes the generation function, and then broadcasts it to all other ranks.
            if self.strategy.ring_attn_rank == 0:
                samples_list = self.generate_samples(all_prompts, all_labels, **generate_kwargs)
                dist.broadcast_object_list(samples_list, src=dist.get_rank(), group=self.strategy.ring_attn_group)
            else:
                world_size = torch.distributed.get_world_size() // args.ring_attn_size
                samples_list = [None] * (
                    args.rollout_batch_size * args.n_samples_per_prompt // world_size // args.micro_rollout_batch_size
                )
                dist.broadcast_object_list(
                    samples_list, src=self.strategy.ring_attn_ranks[0], group=self.strategy.ring_attn_group
                )
        else:
            
            samples_list = self.generate_samples(all_prompts, all_labels, **generate_kwargs)

        # # vLLM offload when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")

        torch.cuda.empty_cache()
        torch.distributed.barrier()
        torch.cuda.synchronize()

        experiences = []
        for samples in tqdm(
            samples_list,
            desc="make_experience",
            disable=not self.strategy.is_rank_0(),
        ):
    
            experiences.append(self.make_experience(samples).to_device("cpu"))

        experiences, rewards = self.process_experiences(experiences)
        # calculate return and advantages
        for experience, reward in zip(experiences, rewards):
            experience = experience.to_device("cuda")
            reward = reward.to(device="cuda")
            num_actions = experience.info["num_actions"]
            reward= compute_reward(
                reward,
                self.kl_ctl.value,
                experience.kl,
                action_mask=experience.action_mask,
                num_actions=num_actions,
                reward_clip_range=args.reward_clip_range,
            )
            # experience.info["rule_reward"] = torch.tensor(norm_rule_reward, device=torch.cuda.current_device()).squeeze()
            if self.advantage_estimator == "gae":
                experience.advantages, experience.returns = self.get_advantages_and_returns(
                    experience.values,
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                    generate_kwargs["lambd"],
                )
            elif self.advantage_estimator in ["reinforce", "rloo", "reinforce_baseline", "group_norm"]:
                experience.returns = self.get_cumulative_returns(
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                )
                experience.advantages = deepcopy(experience.returns)
            else:
                raise Exception(f"Unkown advantage_estimator {self.advantage_estimator}")
            
            # calculate the return info.
            if not getattr(self, "packing_samples", False): ###not used
                return_sums = reward.sum(dim=-1)

            else:
                return_sums = torch.tensor(
                    [each_reward.sum() for each_reward in reward], device=torch.cuda.current_device()
                )
            experience.info["return"] = return_sums    ####tensor[0.1,...   ]
            

            # remove unnecessary info
            experience.kl = None
            del experience.info["num_actions"]
            experience.to_device("cpu")
        return experiences

    @torch.no_grad()
    def generate_samples(self, all_prompts: List[str], all_labels, **generate_kwargs) -> List[Samples]:
        """
        Generate samples and return in batches.
        """
        assert not getattr(self, "packing_samples", False)
        args = self.strategy.args
        self.actor.eval()
        # sample multiple response
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        all_labels = sum([[label] * args.n_samples_per_prompt for label in all_labels], [])
        samples_list = []
        for i in range(0, len(all_prompts), args.micro_rollout_batch_size):
            prompts = all_prompts[i : i + args.micro_rollout_batch_size]
            labels = all_labels[i : i + args.micro_rollout_batch_size]
            inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
        
            sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)
            samples = Samples(
                sequences=sequences,
                attention_mask=attention_mask,
                action_mask=action_mask,
                num_actions=action_mask.size(1),
                packed_seq_lens=None,
                response_length=action_mask.float().sum(dim=-1),
                total_length=attention_mask.float().sum(dim=-1),
                prompts=prompts,
                labels=labels,
                pad_len=None,
            )
            samples_list.append(samples)
        return samples_list

    @torch.no_grad()
    def make_experience(self, samples: Samples) -> Experience:
        raise NotImplementedError("This method should be implemented by the subclass.")

    @torch.no_grad()
    def process_experiences(self, experiences: List[Experience]) -> Tuple[List[Experience], List[torch.Tensor]]:
        """
        Process experiences, this can be used to filter out some experiences or do some processing on the rewards.

        Output:
        - experiences: List of Experience
        - rewards: List of rewards
        """
        args = self.strategy.args

        # reward shaping for rloo and reinforce_baseline
        if args.advantage_estimator == "rloo":
            rewards = torch.cat([experience.info["reward"] for experience in experiences])
            rewards = rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            baseline = (rewards.sum(-1, keepdim=True) - rewards) / (args.n_samples_per_prompt - 1)
            rewards = rewards - baseline
            rewards = rewards.flatten().to(device="cpu").chunk(len(experiences))
            return experiences, rewards
        elif args.advantage_estimator == "reinforce_baseline":
            # REINFORCE++-baseline removed the / std and K3 kl loss in GRPO.
            # `/ std` is not needed in RL variance reduction theory, and `k3 KL` has a larger variance than `k1 KL` under a categorical distribution.
            rewards = torch.cat([experience.info["reward"] for experience in experiences])
            rewards = rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            rewards = rewards - rewards.mean(-1, keepdim=True)
            rewards = rewards.reshape(-1).to(device="cpu").chunk(len(experiences))
            return experiences, rewards
        elif args.advantage_estimator == "group_norm":
            rewards = torch.cat([experience.info["reward"] for experience in experiences])
            rewards = rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            rewards = (rewards - rewards.mean(-1, keepdim=True)) / (rewards.std(-1, keepdim=True) + 1e-9)
            rewards = rewards.reshape(-1).to(device="cpu").chunk(len(experiences))
            return experiences, rewards
        # default rewards
        return experiences, [experience.info["reward"] for experience in experiences]

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        if isinstance(values, list):
            # packing samples
            # TODO: this is slow...
            advantages = []
            returns = []
            for v, r in zip(values, rewards):
                adv, ret = self.get_advantages_and_returns(v.unsqueeze(0), r.unsqueeze(0), action_mask, gamma, lambd)
                advantages.append(adv.squeeze(0))
                returns.append(ret.squeeze(0))
            return advantages, returns

        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        if action_mask is not None:
            values = action_mask * values
            rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns

    @torch.no_grad()
    def get_cumulative_returns(
        self,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function that computes advantages and returns from rewards using REINFORCE.
        REINFORCE uses cumulative returns without the GAE (Generalized Advantage Estimation).

        Input:
        - rewards: Tensor of shape (batch_size, response_size)
        - action_mask: Tensor of shape (batch_size, response_size), binary mask
        - gamma: discount factor

        Output:
        - returns: Tensor of shape (batch_size, response_size)
        """

        if isinstance(rewards, list):
            # packing samples
            # TODO: this is slow...
            returns = []
            for r in rewards:
                ret = self.get_cumulative_returns(r.unsqueeze(0), action_mask, gamma)
                returns.append(ret.squeeze(0))
            return returns

        response_length = rewards.size(1)
        returns = torch.zeros_like(rewards)
        cumulative_return = torch.zeros(rewards.size(0), device=rewards.device)

        # Mask invalid responses if action_mask is provided
        if action_mask is not None:
            rewards = action_mask * rewards

        # Calculate returns by accumulating discounted rewards
        for t in reversed(range(response_length)):
            cumulative_return = rewards[:, t] + gamma * cumulative_return
            returns[:, t] = cumulative_return

        return returns


class RemoteExperienceMaker(NaiveExperienceMaker):
    def __init__(self, *args, vllm_engines: List = None, packing_samples=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.vllm_engines = vllm_engines
        self.packing_samples = packing_samples

        if self.custom_reward_func:
            self.custom_reward_func = ray.remote(self.custom_reward_func)

    @torch.no_grad()
    def make_experience_list(
        self, all_prompts: Union[str, List[str]], all_labels, **generate_kwargs
    ) -> List[Experience]:
        if self.strategy.args.perf:
            self.perf_stats = {
                "generate_time": 0,
                "actor_value_rm_time": 0,
                "wait_time": 0,
            }
        experiences = super().make_experience_list(all_prompts, all_labels, **generate_kwargs)
        if self.critic is not None:
            for experience in experiences:
                # send experience to critic
                experience_cpu = deepcopy(experience)
                experience_cpu.to_device("cpu")
                self._ref = self.critic.append.remote(experience_cpu)
        return experiences

    @torch.no_grad()
    def generate_samples(self, all_prompts: List[str], all_labels, **generate_kwargs) -> List[Samples]:
        """
        Generate samples and return in batches.

        When not using vllm, we will fallback to the default implementation,
        in which actor will be used to generate samples.
        """
        if self.vllm_engines is None:
            return super().generate_samples(all_prompts, all_labels, **generate_kwargs)

        # vLLM generation
        samples = self._generate_vllm(all_prompts, all_labels, **generate_kwargs)
        return samples

    @torch.no_grad()
    def make_experience(self, samples: Samples) -> Experience:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        args = self.strategy.args
        self.actor.eval()
        device = torch.cuda.current_device()

        # extract values from samples
        sequences = samples.sequences
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        num_actions = samples.num_actions
        packed_seq_lens = samples.packed_seq_lens

        
        start = time.time()
        sequences_cpu, attention_mask_cpu = (
            sequences.to("cpu"),
            attention_mask.to("cpu"),
        )

        # init log probs
        if self.initial_model is not None:
            base_action_log_probs_ref = self.initial_model.forward.remote(
                sequences_cpu, num_actions, attention_mask_cpu, logps_allgather=True, packed_seq_lens=packed_seq_lens
            )

            if args.colocate_actor_ref or args.colocate_all_models:
                ray.get([base_action_log_probs_ref])
                ray.get([self.initial_model.empty_cache.remote()])
        else:
            base_action_log_probs_ref = ray.put(None)

        # values
        if self.critic is not None:
            value_ref = self.critic.forward.remote(
                sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens
            )
            # avoid CUDA OOM when colocate models
            if args.colocate_critic_reward or args.colocate_all_models:
                ray.get([value_ref])
                ray.get([self.critic.empty_cache.remote()])
        else:
            value_ref = ray.put(None)

        # rewards
        r_refs = []

        # support remote RM API with ray
        if not self.remote_rm_url:
            for rm in self.reward_model:
                r_refs.append(
                    rm.forward.remote(
                        sequences_cpu, attention_mask_cpu, packed_seq_lens=packed_seq_lens, pad_sequence=True
                    )
                )
        else:
            # remote RM
            if self.strategy.ring_attn_group is None or self.strategy.ring_attn_rank == 0:
                if not self.packing_samples:
                    queries = self.tokenizer.batch_decode(sequences_cpu, skip_special_tokens=False)
                else:
                    sequences_list = []
                    offset = 0
                    tokens_list = sequences_cpu.tolist()[0]
                    for length in packed_seq_lens:
                        sequences_list.append(tokens_list[offset : offset + length])
                        offset += length
                    queries = self.tokenizer.batch_decode(sequences_list, skip_special_tokens=False)

                if self.custom_reward_func:
                    r = self.custom_reward_func.remote(queries, samples.prompts, samples.labels)
                    r_refs.append(r)

                else:
                    for rm in self.remote_rm_url:
                        r = remote_rm_fn_ray.remote(
                            rm, queries=queries, prompts=samples.prompts, labels=samples.labels
                        )
                        r_refs.append(r)

            else:
                r_refs.append(ray.put(None))

        if args.colocate_all_models and not self.remote_rm_url:
            ray.get(r_refs)

            ray.get([self.reward_model[0].empty_cache.remote()])



        ##### 使用新的处理方法
        # 1. 将原始序列解码为文本
        queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=False)
        answers =samples.labels
        
         
        reverse_sequences = []

        # 在重组序列的循环中计算新的长度
        new_packed_seq_lens = []

        print("sequences",sequences.shape)
        
        # 修改循环部分
        sequences_list = []
        attention_mask_list = []
        offset = 0
        tokens_list = sequences.tolist()[0]
        attention_list = attention_mask.tolist()[0]
        for length in packed_seq_lens:
            sequences_list.append(tokens_list[offset : offset + length])
            attention_mask_list.append(attention_list[offset : offset + length])
            offset += length
        for i, (seq, attn_mask) in enumerate(zip(sequences_list, attention_mask_list)):
            # 使用 packed_seq_lens 或 response_length 来确定位置
            # 使用 num_actions 来确定位置
                if isinstance(samples.num_actions, int):
                    # 如果 num_actions 是整数，表示所有样本的响应长度相同
                    response_len = samples.num_actions
                else:
                    # 如果 num_actions 是张量，表示每个样本的响应长度可能不同
                    response_len = samples.num_actions[i]
                # 计算 z 的起始和结束位置
                z_start = len(seq) - response_len
                z_end = len(seq) - 1
                # 提取x部分
                x_part = seq[:z_start]

                #####if use template
                # if self.strategy.args.input_template is not None:
                #     # Decode token IDs to text first
                #     x_part=x_part[:-15]###math template
                #     answer_part = "The answer to this question is "+answers[i]+". Next, please solve the question step by step."
                #     # x_part=x_part[:-3]###r1 template
                #     # answer_part=" <answer> "+answers[i]+" </answer>\\nAssistant:"
                # else:
                x_part = x_part
                answer_part=" ###Answer:"+answers[i]+" ###Solution:"
                    # answer_part="The answer to this question is "+answers[i]+". Next, please solve the question step by step."
                # if self.strategy.args.apply_chat_template:
                #     x_part=x_part[:-5]
                #     answer_part=answer_part+self.tokenizer.decode(x_part[-5:])
                    
                # breakpoint()
                # x_attn_mask = attn_mask[:z_start]
                
                # 提取z部分z
                z_part = seq[z_start:z_end+1]
                # z_attn_mask = attn_mask[z_start:z_end+1]
                
                # 计算各部分长度
                x_len = len(x_part)
                
                # 2对答案进行编码
                answer_inputs = self.tokenizer(
                    answer_part,
                    add_special_tokens=False,
                    max_length=self.prompt_max_len,
                    padding=False,
                    truncation=True,
                    return_tensors="pt"
                ).to(sequences.device)

                y_len = answer_inputs["input_ids"].shape[1]
                z_len = z_end - z_start + 1
                # 新的总长度
                new_len = x_len + y_len + z_len
                new_packed_seq_lens.append(new_len)
                # breakpoint()
                # assert new_len == y_len+packed_seq_lens[i]
                
                # # 获取y部分的attention mask
                # y_attn_mask = torch.ones_like(answer_inputs["input_ids"][i])
                
                # 组合x+y+z
                # breakpoint()
                x_part_tensor = torch.tensor(x_part, device=answer_inputs["input_ids"].device).unsqueeze(0)
                z_part_tensor = torch.tensor(z_part, device=answer_inputs["input_ids"].device).unsqueeze(0)
                
                new_seq = torch.cat([x_part_tensor, answer_inputs["input_ids"], z_part_tensor],dim=1)
                # breakpoint()
                reverse_sequences.append(new_seq)
                
                # # 组合attention mask
                # z_attn_mask_tensor = torch.tensor(z_attn_mask, device=answer_inputs["input_ids"][i].device)
                # x_attn_mask_tensor = torch.tensor(x_attn_mask, device=answer_inputs["input_ids"][i].device)
                # new_attn_mask = torch.cat([x_attn_mask_tensor, y_attn_mask.unsqueeze(0), z_attn_mask_tensor])
                # reverse_attention_mask.append(new_attn_mask)
                
                # # 更新z的位置
                # new_z_start = len(x_part) + len(answer_inputs["input_ids"][i])
                # new_z_end = new_z_start + len(z_part) - 1
                # z_start_idx.append(new_z_start)
                # z_end_idx.append(new_z_end)
                
                # # 创建新的action mask
                # new_act_mask = torch.zeros_like(new_seq, dtype=torch.bool)
                # new_act_mask[new_z_start:new_z_end+1] = True
                # reverse_action_mask.append(new_act_mask)
                
        # 4. 将列表转换为张量
        # breakpoint()
        # reverse_sequences = torch.stack(reverse_sequences)
        # reverse_attention_mask = torch.stack(reverse_attention_mask)
        # reverse_action_mask = torch.stack(reverse_action_mask)
        # new_packed_seq_lens =new_packed_seq_lens, device=sequences.device)

        reverse_sequences = torch.cat(reverse_sequences, dim=1)
        # reverse_attention_mask = torch.cat(reverse_attention_mask, dim=0)
        # reverse_action_mask = torch.cat(reverse_action_mask, dim=0)
        # breakpoint()
        # log probs
        action_log_probs = self.actor(
            sequences,
            num_actions,
            attention_mask,
            ring_attn_group=self.strategy.ring_attn_group,
            logps_allgather=True,
            packed_seq_lens=packed_seq_lens,
        )
        actor_value_rm_time = time.time() - start

        # wait initial/critic/reward model done
        start = time.time()
        ref_values = ray.get([base_action_log_probs_ref, value_ref] + r_refs)
        wait_time = time.time() - start


        base_action_log_probs, value, rewards = ref_values[0], ref_values[1], ref_values[2:]

        if base_action_log_probs is not None:
            base_action_log_probs = base_action_log_probs.to(device)
        if value is not None:
            value = value.to(device)

        # broadcast rewards to all ring attention ranks when using remote RM  ####not used
        if self.remote_rm_url and self.strategy.ring_attn_group is not None:
            if self.strategy.ring_attn_rank == 0:
                dist.broadcast_object_list(rewards, src=dist.get_rank(), group=self.strategy.ring_attn_group)
            else:
                dist.broadcast_object_list(
                    rewards, src=self.strategy.ring_attn_ranks[0], group=self.strategy.ring_attn_group
                )

        rewards = [r.to(device) for r in rewards]  ###rule-based rewards
        r = self.reward_fn(rewards) if len(rewards) > 0 else rewards[0]

        # Handle dictionary return value from reward function (for extra_logs support)


        # avoid CUDA OOM when colocate models
        if args.colocate_critic_reward and not self.remote_rm_url:
            ray.get([self.reward_model[0].empty_cache.remote()])

        if args.colocate_actor_ref or args.colocate_all_models:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        if (self.initial_model is not None) and (not args.use_kl_loss):
            kl = compute_approx_kl(
                action_log_probs,
                base_action_log_probs,
                action_mask=action_mask,
                kl_estimator=self.strategy.args.kl_estimator,
            )
        else:
            kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=device)



        if not self.packing_samples:
            kl_mean = masked_mean(kl, action_mask, dim=-1)
        else:
            # breakpoint()
            if self.strategy.ring_attn_group is not None:
                assert samples.pad_len is not None
                sequences, attention_mask, num_actions, packed_seq_lens, _, _, kl = unpad_sequences(
                    pad_len=samples.pad_len,
                    sequences=sequences,
                    attention_mask=attention_mask,
                    num_actions=num_actions,
                    packed_seq_lens=packed_seq_lens,
                    ring_attn_group=self.strategy.ring_attn_group,
                    action_log_probs=action_log_probs,
                    values=value,
                    kl=kl,
                )
            # convert tensor into list of tensors so that it's easier to manipulate
            # within dataset.
            sequences = unpacking_samples(sequences, packed_seq_lens)
            reverse_sequences = unpacking_samples(reverse_sequences, new_packed_seq_lens)
            attention_mask = None
            action_log_probs = unpacking_samples(action_log_probs, num_actions)
            if value is not None:
                value = unpacking_samples(value, num_actions)
            if base_action_log_probs is not None:
                base_action_log_probs = unpacking_samples(base_action_log_probs, num_actions)

            kl = unpacking_samples(kl, num_actions)
            kl_mean = torch.tensor([each_kl.mean() for each_kl in kl], device=device)

        if not args.use_kl_loss:
            base_action_log_probs = None

                


        info = {
            "kl": kl_mean,
            "reward": r,
            "response_length": samples.response_length,
            "total_length": samples.total_length,
            "num_actions": num_actions,
            # "entropy": mean_entropy,
        }
        
        # # Add extra_logs to info for TensorBoard logging
        # if extra_logs:
        #     info["extra_logs"] = extra_logs

        if self.strategy.args.perf:
            self.perf_stats["actor_value_rm_time"] += actor_value_rm_time
            self.perf_stats["wait_time"] += wait_time

        experience = Experience(
            sequences,
            reverse_sequences,
            action_log_probs,
            base_action_log_probs,
            value,
            None,
            None,
            attention_mask,
            action_mask,
            info,
            kl,
        )

        self.actor.train()  # reset model state
        return experience

    def _generate_vllm(self, all_prompts: List[str], all_labels, **kwargs) -> List[Samples]:
        from vllm import SamplingParams

        # round-robin load balance
        rank = torch.distributed.get_rank() // self.strategy.ring_attn_size
        world_size = torch.distributed.get_world_size() // self.strategy.ring_attn_size

        # Select LLM engines: assign each rank an engine, or cycle through engines if world_size < engine_count
        if len(self.vllm_engines) <= world_size:
            llms = [self.vllm_engines[rank % len(self.vllm_engines)]]
        else:
            llms = self.vllm_engines[rank::world_size]

        args = self.strategy.args

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
            include_stop_str_in_output=True,
        )

        # Expand prompt list based on the number of samples per prompt
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        all_labels = sum([[label] * args.n_samples_per_prompt for label in all_labels], [])
        all_prompt_token_ids = self.tokenize_fn(all_prompts, self.prompt_max_len, padding=False)["input_ids"]

        # Distribute requests to engines and collect responses to outputs
        refs = []
        batch_size = (len(all_prompt_token_ids) + len(llms) - 1) // len(llms)
        for i, llm in enumerate(llms):
            prompt_token_ids = all_prompt_token_ids[i * batch_size : (i + 1) * batch_size]
            refs.append(
                llm.add_requests.remote(rank, sampling_params=sampling_params, prompt_token_ids=prompt_token_ids)
            )
        ray.get(refs)

        # Make sure all requests are sent.
        if self.strategy.ring_attn_group is None:
            torch.distributed.barrier()
        else:
            time.sleep(3)

        # Retrieve and combine results from all outputs
        all_output_refs = []
        for i, llm in enumerate(llms):
            all_output_refs.append(llm.get_responses.remote(rank))
        all_outputs = sum(ray.get(all_output_refs), [])

        samples_list = []
        for i in range(0, len(all_outputs), args.micro_rollout_batch_size):
            outputs = all_outputs[i : i + self.strategy.args.micro_rollout_batch_size]
            prompts = all_prompts[i : i + self.strategy.args.micro_rollout_batch_size]
            labels = all_labels[i : i + self.strategy.args.micro_rollout_batch_size]
            if not self.packing_samples:
                # NOTE: concat all outputs to following format:
                #
                # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
                # | token token token token token | token token [EOS] [PAD] |
                # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
                # |<---------- prompt ----------->|<-------- answer ------->|
                max_input_len, max_output_len = 0, 0
                for output in outputs:
                    max_input_len = max(max_input_len, len(output.prompt_token_ids))
                    max_output_len = max(max_output_len, len(output.outputs[0].token_ids))

                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                for output in outputs:
                    # left padding input
                    input_len = len(output.prompt_token_ids)
                    input_ids = [pad_token_id] * (max_input_len - input_len) + list(output.prompt_token_ids)

                    # right padding output
                    output_len = len(output.outputs[0].token_ids)
                    output_ids = list(output.outputs[0].token_ids) + [pad_token_id] * (max_output_len - output_len)

                    # concat input and output
                    sequences.append(input_ids + output_ids)

                sequences = torch.tensor(sequences)
                sequences, attention_mask, action_mask = self.actor.process_sequences(
                    sequences, max_input_len, eos_token_id, pad_token_id
                )
                sequences = sequences.to("cuda")
                attention_mask = attention_mask.to("cuda")
                action_mask = action_mask.to("cuda")
                samples_list.append(
                    Samples(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=action_mask,
                        num_actions=action_mask.size(1),
                        packed_seq_lens=None,
                        response_length=action_mask.float().sum(dim=-1),
                        total_length=attention_mask.float().sum(dim=-1),
                        prompts=prompts,
                        labels=labels,
                        pad_len=None,
                    )
                )
            else:
                # NOTE: concat all outputs to following format:
                #
                # | token token token | token token [EOS] | token token token token token | token token [EOS] | token token | token token token [EOS] |
                # |<---  prompt ----->|<---- answer ----->|<---------- prompt ----------->|<----- answer ---->|<- prompt -->|<-------- answer ------->|
                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                packed_seq_lens = []
                attention_mask = []
                num_actions = []
                for i, output in enumerate(outputs):
                    input_len = len(output.prompt_token_ids)
                    output_len = len(output.outputs[0].token_ids)
                    packed_seq_lens.append(input_len + output_len)
                    sequences.extend(output.prompt_token_ids + list(output.outputs[0].token_ids))
                    attention_mask.extend([i + 1] * (input_len + output_len))

                    # current_action_mask = [0] * (input_len - 1) + [1] * output_len + [0]
                    # num_actions.append(max(1, sum(current_action_mask)))
                    num_actions.append(max(1, output_len))

                # pad seq makes the sequence a multiple of ring_attention_size.
                pad_len = None
                if self.strategy.ring_attn_group is not None:
                    pad_len, sequences, attention_mask, num_actions, packed_seq_lens = pad_sequences(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        num_actions=num_actions,
                        packed_seq_lens=packed_seq_lens,
                        ring_attn_group=self.strategy.ring_attn_group,
                        pad_token_id=pad_token_id,
                    )

                sequences = torch.tensor(sequences, device="cuda").unsqueeze(0)
                attention_mask = torch.tensor(attention_mask, device="cuda").unsqueeze(0)
                action_mask = None
                response_length = torch.tensor(num_actions, device="cuda", dtype=torch.float)
                total_length = torch.tensor(packed_seq_lens, device="cuda", dtype=torch.float)
                samples_list.append(
                    Samples(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=None,
                        num_actions=num_actions,
                        packed_seq_lens=packed_seq_lens,
                        response_length=response_length,
                        total_length=total_length,
                        prompts=prompts,
                        labels=labels,
                        pad_len=pad_len,
                    )
                )
        return samples_list

    def flush(self):
        "Ensure all experience has been send to critic"
        if self.critic is not None:
            ray.get(self._ref)
            self._ref = None

class NaiveExperienceMakerBOX(ABC):
    """
    Naive experience maker.
    """

    def __init__(
        self,
        actor: Actor,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: Actor,
        tokenizer,
        prompt_max_len: int,
        kl_controller,
        strategy=None,
        remote_rm_url: str = None,
        reward_fn=None,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.reward_fn = reward_fn
        self.perf_stats = None
        self.advantage_estimator = strategy.args.advantage_estimator

    # tokenizer
    def tokenize_fn(self, texts, max_length, padding=True, device=None):
        if not padding:
            # when padding is False, return tokenized texts as list
            return self.tokenizer(
                texts,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    @torch.no_grad()
    def make_experience_list(self, all_prompts: Union[str, List[str]], all_answers:  Union[str, List[str]], **generate_kwargs) -> List[Experience]:
        """
        Make a list of experience with the micro_rollout_batch_size.

        This method will first calculate the response sequences and rewards for the given prompts.
        Then, if we need certain processing for the rewards or do certain filtering, we can process the rollout as a whole.
        After that, we will calculate the advantages and returns for each experience.
        """
        args = self.strategy.args
        experiences = []
        
        for idx, samples in enumerate(tqdm(
            self.generate_samples(all_prompts, all_answers, **generate_kwargs),
            desc=f"make_experience",
            disable=not self.strategy.is_rank_0(),
            )):
            # 获取对应的 answer
            #answers = all_answers[idx]
            #print("samples", samples.)
            experience = self.make_experience(samples, **generate_kwargs)
            experiences.append(experience)
            #reward_indexs_list.append(reward_indexs)
            #step_rewards_list.append(step_rewards)
            
        # for samples in tqdm(
        #     self.generate_samples(all_prompts, **generate_kwargs),
        #     desc="make_experience",
        #     disable=not self.strategy.is_rank_0(),
        # ):
        #     experiences.append(self.make_experience(samples))

        experiences = self.process_experiences(experiences)

        # calculate return and advantages
        for experience in experiences:
            num_actions = experience.info["num_actions"]
            reward = compute_reward(
                experience.info["reward"],
                self.kl_ctl.value,
                experience.kl,
                action_mask=experience.action_mask,
                num_actions=num_actions,
                reward_clip_range=args.reward_clip_range,
            )

            if self.advantage_estimator == "gae":
                experience.advantages, experience.returns = self.get_advantages_and_returns(
                    experience.values,
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                    generate_kwargs["lambd"],
                )
            elif self.advantage_estimator == "reinforce":
                experience.returns = self.get_cumulative_returns(
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                )
                experience.advantages = deepcopy(experience.returns)
            else:
                raise Exception(f"Unkown advantage_estimator {self.advantage_estimator}")

            # calculate the return info.
            if not getattr(self, "packing_samples", False):
                return_sums = reward.sum(dim=-1)
            else:
                return_sums = torch.tensor(
                    [each_reward.sum() for each_reward in reward], device=torch.cuda.current_device()
                )
            experience.info["return"] = return_sums
            # remove unnecessary info
            experience.kl = None
            del experience.info["num_actions"]
        return experiences

    @torch.no_grad()
    def generate_samples(self, all_prompts: List[str], all_answers: List[str], **generate_kwargs) -> List[Samples]:
        """
        Generate samples and return in batches.
        """
        assert not getattr(self, "packing_samples", False)
        args = self.strategy.args
        self.actor.eval()
        # sample multiple response
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        all_answers = sum([[answer] * args.n_samples_per_prompt for answer in all_answers], [])
        samples_list = []
        #self.strategy.print(f"generate samples!!!")
        for i in range(0, len(all_prompts), args.micro_rollout_batch_size):
            prompts = all_prompts[i : i + args.micro_rollout_batch_size]
            answers = all_answers[i : i + args.micro_rollout_batch_size]
            inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
            #self.strategy.print(f"generating!!!")
            sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)
            #self.strategy.print(f"generating!!!", sequences)
            samples = SamplesBOX(
                sequences=sequences,
                attention_mask=attention_mask,
                action_mask=action_mask,
                num_actions=action_mask.size(1),
                packed_seq_lens=None,
                response_length=action_mask.float().sum(dim=-1),
                total_length=attention_mask.float().sum(dim=-1),
                answers=answers,
            )
            samples_list.append(samples)

            #print("sequences", samples.sequences)
            
            #print("attention_mask", samples.attention_mask)


        return samples_list

    
    @torch.no_grad()
    def make_experience(self, samples: Samples, **generate_kwargs) -> Experience:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        self.actor.eval()
        self.initial_model.eval()
        if self.reward_model is not None:
            self.reward_model.eval()
        if self.critic is not None:
            self.critic.eval()

        # extract values from samples
        sequences = samples.sequences
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        num_actions = samples.num_actions
        answers = samples.answers

    

        # log probs
        #print("sequences.shape", sequences.shape)
        #print("sequences", sequences[0])
        #print("num_actions", num_actions)
        #print("attention_mask", attention_mask.shape)
        #print("action_mask", action_mask.shape)
        action_log_probs = self.actor(sequences, num_actions, attention_mask)
        
        #print("action_log_probs", action_log_probs.shape)

        # init log probs
        base_action_log_probs = self.initial_model(sequences, num_actions, attention_mask)

        # values
        if self.critic is not None:
            value = self.critic(sequences, num_actions, attention_mask)
        else:
            value = None

        # rewards
        if self.remote_rm_url is not None:
            # remote RM
            queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=False)
            r = remote_rm_fn(self.remote_rm_url, queries=queries).to(device=action_log_probs.device)
        else:
            # local RM
            
            #print("sequences", sequences[0])

            queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=False)
            
            
            #print("queries", queries[0])

            # 监测不包含 \n\n# Answer\n\n 的 queries
            
            #print("queries", queries[0])
            # 定义正则模式
            #pattern = r"The final answer is: \\\\boxed\{(.*?)\}"
            pattern = r"The final answer is: \\boxed\{(.*?)\}"
            # 找到没有匹配到内容的索引
            missing_answer_indices = [
            i for i, query in enumerate(queries) if not re.search(pattern, query, re.DOTALL)
            ]
            

            processed_queries = []
            box_match_list = []
            for query, answer in zip(queries, answers):
                query, box_match = preprocess_orm800k_box_responsev1(query, answer)
                processed_queries.append(query)
                box_match_list.append(box_match)
                
            queries = processed_queries
            
            
            reward_sequences, reward_attention_mask = preprocess_prm_reward(queries, self.tokenizer, self.prompt_max_len, **generate_kwargs)
            

            candidate_tokens = [128003, 128004]
            logits = self.reward_model(reward_sequences.to(device=action_log_probs.device), attention_mask=reward_attention_mask.to(device=action_log_probs.device), return_output=True).logits[:,:,candidate_tokens]
            scores = logits.softmax(dim=-1)[:,:,0]
            step_scores = []
            step_tag = '<|reserved_special_token_0|>'
            step_tag_id = self.tokenizer.encode(f"{step_tag}")[-1]

            for index, score in enumerate(scores):
                if index in missing_answer_indices:
                # 如果 query 不包含 \n\n# Answer\n\n，设置 step_score 为 0.0
                    step_scores.append(torch.tensor(0.0, device=action_log_probs.device))
                else:
                    # 否则正常计算 step_score
                    step_scores.append(score[reward_sequences[index] == step_tag_id])
                    
            reward_indexs = []
            
            for seq in sequences:
                indices = (seq == step_tag_id).nonzero(as_tuple=True)[0]
                if len(indices) > 0:
                    #print("neg num actions", -num_actions)
                    reverse_indices = [- (len(seq) - idx.item()) for idx in indices]
                    #print("reverse_indices", reverse_indices)
                    
                else:
                    reverse_indices = [-1]
                    
                reward_indexs.append(torch.tensor(reverse_indices, device=action_log_probs.device))
                
                
            # 使用负索引提取128002的位置
            extracted_elements = []
            for seq, negative_indices in zip(sequences, reward_indexs):
                elements = [seq[idx].item() for idx in negative_indices]
                extracted_elements.append(elements)

            # 提取每个张量的最小值
            #min_values = [step_score.min() for step_score in step_scores]

            # 提取每个张量的最小值，若为空则设置为0.0
            min_values = [step_score.min()-0.5 if step_score.numel() > 0 else torch.tensor(-0.5, device=action_log_probs.device) for step_score in step_scores]
            # 将最小值重新合并为一个新的张量
            min_values_tensor = torch.tensor(min_values, device=action_log_probs.device)

            r = min_values_tensor.to(device=action_log_probs.device)
            
            step_scores = [step_score - 0.5 if step_score.numel() > 0 else torch.tensor(-0.5, device=action_log_probs.device) for step_score in step_scores]
            
            # 将 math_box_list 转换为张量，并移动到与 step_scores 相同的设备
            math_box_tensor = torch.tensor(box_match_list, device=r.device)
            # print("r", r)
            
            # print("step_scores", step_scores)
            if len(math_box_tensor) == len(step_scores):
                
                r = r + math_box_tensor
                for i in range(len(math_box_tensor)):
                    step_score = step_scores[i]
                    add_box = math_box_tensor[i]
                    
                    if step_score.dim() == 0:
                        step_scores[i] = step_score + add_box
                    else:
                        step_scores[i][-1] += add_box
                        
        kl = compute_approx_kl(
            action_log_probs,
            base_action_log_probs,
            action_mask=action_mask,
            use_kl_estimator_k3=self.strategy.args.use_kl_estimator_k3,
        )
        #print("kl.shape", kl.shape)
        
        for i, tensor in enumerate(reward_indexs):
            if not torch.equal(tensor, torch.tensor([-1], device=action_log_probs.device)):
                # Remove the second last element
                reward_indexs[i] = torch.cat((tensor[:-2], tensor[-1:]))
            
        #print(masked_mean(kl, action_mask, dim=-1).shape)
        info = {
            "kl": masked_mean(kl, action_mask, dim=-1),
            "reward": r,
            "response_length": samples.response_length,
            "total_length": samples.total_length,
            "num_actions": num_actions,

            #"reward_indexs": reward_indexs,
            #"step_rewards": step_scores,
        }
        # reset model state
        self.actor.train()
        if self.critic is not None:
            self.critic.train()
            
        #print("pre_pre_action_mask", action_mask.size(1))

        return Experience(
            sequences,
            action_log_probs,
            value,
            None,
            None,
            attention_mask,
            action_mask,
            info,
            kl,
        ), reward_indexs, step_scores

    @torch.no_grad()
    def process_experiences(self, experiences: List[Experience]) -> List[Experience]:
        # TODO: add more methods to process experiences
        return experiences

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        if isinstance(values, list):
            # packing samples
            # TODO: this is slow...
            advantages = []
            returns = []
            for v, r in zip(values, rewards):
                adv, ret = self.get_advantages_and_returns(v.unsqueeze(0), r.unsqueeze(0), action_mask, gamma, lambd)
                advantages.append(adv.squeeze(0))
                returns.append(ret.squeeze(0))
            return advantages, returns

        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        if action_mask is not None:
            values = action_mask * values
            rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns

    @torch.no_grad()
    def get_cumulative_returns(
        self,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function that computes advantages and returns from rewards using REINFORCE.
        REINFORCE uses cumulative returns without the GAE (Generalized Advantage Estimation).

        Input:
        - rewards: Tensor of shape (batch_size, response_size)
        - action_mask: Tensor of shape (batch_size, response_size), binary mask
        - gamma: discount factor

        Output:
        - returns: Tensor of shape (batch_size, response_size)
        """

        if isinstance(rewards, list):
            # packing samples
            # TODO: this is slow...
            returns = []
            for r in rewards:
                ret = self.get_cumulative_returns(r.unsqueeze(0), action_mask, gamma)
                returns.append(ret.squeeze(0))
            return returns

        response_length = rewards.size(1)
        returns = torch.zeros_like(rewards)
        cumulative_return = torch.zeros(rewards.size(0), device=rewards.device)

        # Mask invalid responses if action_mask is provided
        if action_mask is not None:
            rewards = action_mask * rewards

        # Calculate returns by accumulating discounted rewards
        for t in reversed(range(response_length)):
            cumulative_return = rewards[:, t] + gamma * cumulative_return
            returns[:, t] = cumulative_return

        return returns

class RemoteExperienceMakerBOX(NaiveExperienceMakerBOX):
    def __init__(self, *args, vllm_engines: List = None, packing_samples=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.vllm_engines = vllm_engines
        self.packing_samples = packing_samples

    @torch.no_grad()
    def make_experience_list(self, all_prompts: Union[str, List[str]], all_answers:  Union[str, List[str]], **generate_kwargs) -> List[Experience]:
        if self.strategy.args.perf:
            self.perf_stats = {
                "generate_time": 0,
                "actor_value_rm_time": 0,
                "wait_time": 0,
            }
        experiences = super().make_experience_list(all_prompts, all_answers, **generate_kwargs)
        if self.critic is not None:
            for experience in experiences:
                # send experience to critic
                experience_cpu = deepcopy(experience)
                experience_cpu.to_device("cpu")
                self._ref = self.critic.append.remote(experience_cpu)
        return experiences

    @torch.no_grad()
    def generate_samples(self, all_prompts: List[str], all_answers: List[str], **generate_kwargs) -> List[Samples]:
        """
        Generate samples and return in batches.

        When not using vllm, we will fallback to the default implementation,
        in which actor will be used to generate samples.
        """
        if self.vllm_engines is None:
            return super().generate_samples(all_prompts, all_answers, **generate_kwargs)

        return self._generate_vllm(all_prompts, all_answers, **generate_kwargs)

    @torch.no_grad()
    def make_experience(self, samples: Samples, **generate_kwargs) -> Experience:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        self.actor.eval()
        device = torch.cuda.current_device()

        # extract values from samples
        sequences = samples.sequences
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        num_actions = samples.num_actions
        packed_seq_lens = samples.packed_seq_lens
        answers = samples.answers

        start = time.time()
        sequences_cpu, attention_mask_cpu = (
            sequences.to("cpu"),
            attention_mask.to("cpu"),
        )
        # ray.logger.info(f"sequences_len: {sequences_cpu.shape}")
        # init log probs
        base_action_log_probs_ref = self.initial_model.forward.remote(
            sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens
        )
        # values
        # ray.logger.info("sent initial model forward request")
        if self.critic is not None:
            value_ref = self.critic.forward.remote(
                sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens
            )
            # avoid CUDA OOM when colocate models
            if self.strategy.args.colocate_critic_reward:
                ray.get([value_ref])
                ray.get([self.critic.empty_cache.remote()])
        else:
            value_ref = ray.put(None)
        
        # ray.logger.info("sent value model forward request")
        if self.strategy.args.colocate_actor_ref:
            ray.get([base_action_log_probs_ref])
            ray.get([self.initial_model.empty_cache.remote()])

        # rewards
        r_refs = []
        # print("answers", len(answers))
        # print("sequences_cpu", sequences_cpu.shape)
        # support remote RM API with ray
        if not self.remote_rm_url:
            queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=False)
            # pattern = r"The final answer is: \\boxed\{(.*?)\}"
            # missing_answer_indices = [
            # i for i, query in enumerate(queries) if not re.search(pattern, query, re.DOTALL)
            # ]
            #print("answers", answers[:5])
            #print("attention_mask_cpu", attention_mask_cpu[:1])
            
            processed_queries = []
            box_match_list = []
            #math_equal_list = []
            for query, answer in zip(queries, answers):
                #temp_query = query
                #query, box_match = preprocess_box_responsev1(query, answer)
                
                ## I will change the response rule-match reward. For my own experiment. By weihao, 12.25 2024
                # query, box_match = preprocess_box_responsev4(query, answer) # original processing func 
                query, box_match = preprocess_box_response_for_qwen_prompt(query, answer)
                #query_v1, equal_match = preprocess_box_responsev1(temp_query, answer)
                processed_queries.append(query)
                box_match_list.append(box_match)
                #math_equal_list.append(equal_match)
                
            queries = processed_queries
            
            final_answer_reward = torch.tensor(box_match_list, device=attention_mask.device)
            
            #queries = [preprocess_orm800k_response(query) for query in queries]
            # print("pro_queries", queries[:5])
            # print("answers", answers[:10])
            
            # print("box_match_list", box_match_list[:10])
            #print("math_equal_list", math_equal_list[:5])
            
            r_refs.append(final_answer_reward)
            # for rm in self.reward_model:
            #     r_refs.append(rm.forward.remote(sequences_cpu, attention_mask_cpu, packed_seq_lens=packed_seq_lens))
                
            # print()
        else:
            # remote RM
            for rm in self.remote_rm_url:
                if not self.packing_samples:
                    queries = self.tokenizer.batch_decode(sequences_cpu, skip_special_tokens=False)
                    r = remote_rm_fn_ray.remote(rm, queries=queries)
                    r_refs.append(r)
                else:
                    sequences_list = []
                    offset = 0
                    tokens_list = sequences_cpu.tolist()[0]
                    for length in packed_seq_lens:
                        sequences_list.append(tokens_list[offset : offset + length])
                        offset += length
                    queries = self.tokenizer.batch_decode(sequences_list, skip_special_tokens=False)
                    r = remote_rm_fn_ray.remote(rm, queries=queries)
                    r_refs.append(r)
        # ray.logger.info("finish cal reward")
        # log probs
        action_log_probs = self.actor(sequences, num_actions, attention_mask, packed_seq_lens=packed_seq_lens)
        actor_value_rm_time = time.time() - start
        # ray.logger.info("finish cal actor log probs")
        # wait initial/critic/reward model done
        start = time.time()
        #print("base_action_log_probs_ref", base_action_log_probs_ref)
        #print("base_action_log_probs_ref", base_action_log_probs_ref.shape)
        ref_values = ray.get([base_action_log_probs_ref, value_ref])
        # ray.logger.info("finish cal init log probs and value")
        wait_time = time.time() - start

        base_action_log_probs, value = ref_values[0], ref_values[1]
        rewards = r_refs
        base_action_log_probs = base_action_log_probs.to(device)
        if value is not None:
            value = value.to(device)
        rewards = [r.to(device) for r in rewards]
        r = self.reward_fn(rewards) if len(rewards) > 0 else rewards[0]

        # Handle dictionary return value from reward function (for extra_logs support)
        # extra_logs = {}
        # if isinstance(r, dict):
        #     if "rewards" in r:
        #         r = r["rewards"]
        #     if "extra_logs" in r:
        #         extra_logs = r["extra_logs"]

        # avoid CUDA OOM when colocate models
        if self.strategy.args.colocate_critic_reward and not self.remote_rm_url:
            ray.get([self.reward_model[0].empty_cache.remote()])

        if self.strategy.args.colocate_actor_ref:
            torch.cuda.empty_cache()

        kl = compute_approx_kl(
            action_log_probs,
            base_action_log_probs,
            action_mask=action_mask,
            use_kl_estimator_k3=self.strategy.args.use_kl_estimator_k3,
        )
        # ray.logger.info("finish cal kl")
        if not self.packing_samples:
            kl_mean = masked_mean(kl, action_mask, dim=-1)
        else:
            # convert tensor into list of tensors so that it's easier to manipulate
            # within dataset.
            sequences = unpacking_samples(sequences, packed_seq_lens)
            attention_mask = None
            action_log_probs = unpacking_samples(action_log_probs, num_actions)
            if value is not None:
                value = unpacking_samples(value, num_actions)

            kl = unpacking_samples(kl, num_actions)
            kl_mean = torch.tensor([each_kl.mean() for each_kl in kl], device=device)

        info = {
            "kl": kl_mean,
            "reward": r,
            "response_length": samples.response_length,
            "total_length": samples.total_length,
            "num_actions": num_actions,
        }
        
        # Add extra_logs to info for TensorBoard logging
        # if extra_logs:
        #     info["extra_logs"] = extra_logs

        if self.strategy.args.perf:
            self.perf_stats["actor_value_rm_time"] += actor_value_rm_time
            self.perf_stats["wait_time"] += wait_time

        experience = Experience(
            sequences,
            action_log_probs,
            value,
            None,
            None,
            attention_mask,
            action_mask,
            info,
            kl,
        )

        self.actor.train()  # reset model state
        return experience

    def _generate_vllm(self, all_prompts: List[str], all_answers: List[str], **kwargs) -> List[Samples]:
        from vllm import SamplingParams

        # round-robin load balance
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        # Select LLM engines: assign each rank an engine, or cycle through engines if world_size < engine_count
        if len(self.vllm_engines) <= world_size:
            llms = [self.vllm_engines[rank % len(self.vllm_engines)]]
        else:
            llms = self.vllm_engines[rank::world_size]

        args = self.strategy.args
        # print("For debugging:", "qwen" in self.strategy.args.pretrain.lower())
        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
            stop=["</s>", "<|im_end|>", "<|endoftext|>"] if "qwen" in self.strategy.args.pretrain.lower() else [],
            stop_token_ids=[151645, 151643] if "qwen" in self.strategy.args.pretrain.lower() else [],
        )

        # Expand prompt list based on the number of samples per prompt
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        all_answers = sum([[answer] * args.n_samples_per_prompt for answer in all_answers], [])
        all_prompt_token_ids = self.tokenize_fn(all_prompts, self.prompt_max_len, padding=False)["input_ids"]

        # Distribute requests to engines and collect responses to outputs
        all_output_refs = []
        batch_size = (len(all_prompt_token_ids) + len(llms) - 1) // len(llms)
        for i, llm in enumerate(llms):
            prompt_token_ids = all_prompt_token_ids[i * batch_size : (i + 1) * batch_size]
            if prompt_token_ids:
                all_output_refs.append(
                    llm.generate.remote(sampling_params=sampling_params, prompt_token_ids=prompt_token_ids)
                )

        # Retrieve and combine results from all outputs
        all_outputs = sum(ray.get(all_output_refs), [])

        samples_list = []
        for i in range(0, len(all_outputs), args.micro_rollout_batch_size):
            outputs = all_outputs[i : i + self.strategy.args.micro_rollout_batch_size]
            answers = all_answers[i : i + self.strategy.args.micro_rollout_batch_size]
            if not self.packing_samples:
                # NOTE: concat all outputs to following format:
                #
                # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
                # | token token token token token | token token [EOS] [PAD] |
                # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
                # |<---------- prompt ----------->|<-------- answer ------->|
                max_input_len, max_output_len = 0, 0
                for output in outputs:
                    max_input_len = max(max_input_len, len(output.prompt_token_ids))
                    max_output_len = max(max_output_len, len(output.outputs[0].token_ids))

                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                for output in outputs:
                    # left padding input
                    input_len = len(output.prompt_token_ids)
                    input_ids = [pad_token_id] * (max_input_len - input_len) + list(output.prompt_token_ids)

                    # right padding output
                    output_len = len(output.outputs[0].token_ids)
                    output_ids = list(output.outputs[0].token_ids) + [pad_token_id] * (max_output_len - output_len)

                    if output_ids[output_len - 1] != eos_token_id:
                        output_ids[min(output_len, len(output_ids) - 1)] = eos_token_id

                    # concat input and output
                    sequences.append(input_ids + output_ids)

                sequences = torch.tensor(sequences)
                sequences, attention_mask, action_mask = self.actor.process_sequences(
                    sequences, max_input_len, eos_token_id, pad_token_id
                )
                sequences = sequences.to("cuda")
                attention_mask = attention_mask.to("cuda")
                action_mask = action_mask.to("cuda")
                samples_list.append(
                    SamplesBOX(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=action_mask,
                        num_actions=action_mask.size(1),
                        packed_seq_lens=None,
                        response_length=action_mask.float().sum(dim=-1),
                        total_length=attention_mask.float().sum(dim=-1),
                        answers=answers,
                    )
                )
            else:
                # NOTE: concat all outputs to following format:
                #
                # | token token token | token token [EOS] | token token token token token | token token [EOS] | token token | token token token [EOS] |
                # |<---  prompt ----->|<---- answer ----->|<---------- prompt ----------->|<----- answer ---->|<- prompt -->|<-------- answer ------->|
                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                packed_seq_lens = []
                attention_mask = []
                num_actions = []
                for i, output in enumerate(outputs):
                    input_len = len(output.prompt_token_ids)
                    output_len = len(output.outputs[0].token_ids)
                    packed_seq_lens.append(input_len + output_len)
                    sequences.extend(output.prompt_token_ids + list(output.outputs[0].token_ids))
                    attention_mask.extend([i + 1] * (input_len + output_len))

                    # current_action_mask = [0] * (input_len - 1) + [1] * output_len + [0]
                    # num_actions.append(max(1, sum(current_action_mask)))
                    num_actions.append(max(1, output_len))

                sequences = torch.tensor(sequences, device="cuda").unsqueeze(0)
                attention_mask = torch.tensor(attention_mask, device="cuda").unsqueeze(0)
                action_mask = None
                response_length = torch.tensor(num_actions, device="cuda", dtype=torch.float)
                total_length = torch.tensor(packed_seq_lens, device="cuda", dtype=torch.float)
                samples_list.append(
                    SamplesBOX(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=None,
                        num_actions=num_actions,
                        packed_seq_lens=packed_seq_lens,
                        response_length=response_length,
                        total_length=total_length,
                        answers=answers,
                        
                    )
                )
        return samples_list

    def flush(self):
        "Ensure all experience has been send to critic"
        if self.critic is not None:
            ray.get(self._ref)
            self._ref = None

