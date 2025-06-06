export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
export TORCHDYNAMO_VERBOSE=1
export WANDB_MODE="offline"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

# Add NCCL debug and optimization settings
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_TIMEOUT=1800
export NCCL_SOCKET_TIMEOUT=1800
export TORCH_NCCL_BLOCKING_WAIT=1

NUM_GPUS=$(nvidia-smi -L | wc -l)
PORT=$(shuf -i 20000-60000 -n 1)
MASTER_ADDR=$HOSTNAME
MASTER_PORT=$SLURM_JOB_ID
NNODES=$SLURM_JOB_NUM_NODES
NODE_RANK=$SLURM_NODEID

# Training Configurations
# Experiment with as many hyperparameters as you want!
LEARNING_RATES=("5e-5")
LR_SCHEDULES=("constant_with_warmup")
OPTIMIZERS=("adamw")
MAX_TRAIN_STEPS=("200000")
frame_length=49

VALIDATION_PROMPT="\"your instruction here Trossen WidowX 250 robot arm:::"\
"pick apple google robot"\
"\""

VALIDATION_IMAGES="data/bridge/processed/1/video/rgb.mp4:::"\
"asset/images/fruit_vangogh.png"

DATA_ROOT="data"
MODEL_PATH="THUDM/CogVideoX-5b-I2V"

for learning_rate in "${LEARNING_RATES[@]}"; do
  for lr_schedule in "${LR_SCHEDULES[@]}"; do
    for optimizer in "${OPTIMIZERS[@]}"; do
      for steps in "${MAX_TRAIN_STEPS[@]}"; do
        output_dir="./runs/models/tesseract-depth-normal-sft-full__framelength_${frame_length}__optimizer_${optimizer}__lr-schedule_${lr_schedule}__learning-rate_${learning_rate}/"

        launcher_cmd="torchrun --master_addr=$MASTER_ADDR --node_rank=$NODE_RANK \
          --rdzv_backend static --nnodes $NNODES --nproc_per_node=$NUM_GPUS --rdzv_id=$NODE_RANK"
        echo "Using launcher command: $launcher_cmd"

        cmd="$launcher_cmd \
          tesseract/i2v_depth_normal_sft.py \
          --pretrained_model_name_or_path $MODEL_PATH \
          --dataset_file cache/samples_depth_normal.json \
          --data_root $DATA_ROOT \
          --height_buckets 240 256 480 512 720 \
          --width_buckets 320 512 640 854 1280 \
          --frame_buckets 9 17 25 33 49 \
          --dataloader_num_workers 2 \
          --pin_memory \
          --validation_prompt $VALIDATION_PROMPT \
          --validation_images $VALIDATION_IMAGES \
          --validation_prompt_separator ::: \
          --num_validation_videos 1 \
          --validation_steps 250 \
          --seed 42 \
          --mixed_precision bf16 \
          --output_dir $output_dir \
          --height 480 \
          --width 640 \
          --guidance_scale 7.5 \
          --max_num_frames $frame_length \
          --train_batch_size 1 \
          --max_train_steps $steps \
          --checkpointing_steps 250 \
          --checkpoints_total_limit 15 \
          --resume_from_checkpoint latest \
          --gradient_accumulation_steps 1 \
          --learning_rate $learning_rate \
          --lr_scheduler $lr_schedule \
          --lr_warmup_steps 200 \
          --lr_num_cycles 1 \
          --noised_image_dropout 0.05 \
          --gradient_checkpointing \
          --enable_slicing \
          --enable_tiling \
          --optimizer $optimizer \
          --beta1 0.9 \
          --beta2 0.95 \
          --weight_decay 0.001 \
          --max_grad_norm 1.0 \
          --allow_tf32 \
          --report_to wandb \
          --ignore_learned_positional_embeddings \
          --nccl_timeout 1800"

        echo "Running command: $cmd"
        eval $cmd
        echo -ne "-------------------- Finished executing script --------------------\n\n"
      done
    done
  done
done
