set -x 
HDFS_HOME=/private/home/zhumingye/code/aaai-code-submitted/
RUN_NAME=InTRO-math-ray-qwen2.5-1.5B
FOLDER_NAME=results
export TRITON_CACHE_DIR=$HDFS_HOME/.cache

# launch the master node of ray in container
CUDA_VISIBLE_DEVICES=4,5,6 ray start --head --node-ip-address 0.0.0.0 --num-gpus 3

# # if you want to launch ray on more nodes, use
# ray start --address {MASTER-NODE-ADDRESS}:6379  --num-gpus 8
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"pip": ["timeout_decorator"]}' \
   -- python3 -m openrlhf.cli.train_ppo_ray_intro \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 2 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 0 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 0 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 2 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --colocate_critic_reward \
   --colocate_actor_ref \
   --pretrain /private/model/Qwen/Qwen2.5-1.5B \
   --remote_rm_url examples/grading/math_grade.py \
   --save_path $HDFS_HOME/$FOLDER_NAME/$RUN_NAME \
   --micro_train_batch_size 4 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 1024 \
   --max_samples 10000 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --generate_max_len 2048 \
   --zero_stage 3 \
   --temperature 0.8 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --prompt_data data/math_train_hard_all_ans.json \
   --input_key problem \
   --label_key answer \
   --normalize_reward \
   --packing_samples \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --load_checkpoint \
   --n_samples_per_prompt 4 \
   --advantage_estimator intro \
   --use_tensorboard $HDFS_HOME/tensorboard/$FOLDER_NAME/$RUN_NAME \
   --save_steps 10 \
   --ckpt_path $HDFS_HOME/$FOLDER_NAME/$RUN_NAME \
   --init_kl_coef 1e-3 \
   --gamma 1.0 \
   --use_kl_loss \
   --kl_estimator k3 \
   --input_template None \
