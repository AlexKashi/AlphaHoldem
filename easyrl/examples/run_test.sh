#!/usr/bin/env bash
trap "exit" INT TERM ERR
trap "kill 0" EXIT

# CUDA_VISIBLE_DEVICES=0 python sac.py --env_name=Walker2d-v3 --max_steps=3000000 &
# CUDA_VISIBLE_DEVICES=0 python sac.py --env_name=Walker2d-v3 --alpha=0.1  --max_steps=3000000&
# CUDA_VISIBLE_DEVICES=0 python sac.py --env_name=Walker2d-v3 --alpha=0.2 --max_steps=3000000 &
# CUDA_VISIBLE_DEVICES=0 python sac.py --env_name=Hopper-v3 --max_steps=3000000 &
# CUDA_VISIBLE_DEVICES=0 python sac.py --env_name=Hopper-v3 --alpha=0.1 --max_steps=3000000 &
# CUDA_VISIBLE_DEVICES=0 python sac.py --env_name=Hopper-v3 --alpha=0.2 --max_steps=3000000 &
# CUDA_VISIBLE_DEVICES=1 python sac.py --env_name=Humanoid-v3 --max_steps=3000000 &
# CUDA_VISIBLE_DEVICES=1 python sac.py --env_name=Humanoid-v3 --alpha=0.1 --max_steps=3000000 &
# CUDA_VISIBLE_DEVICES=1 python sac.py --env_name=Humanoid-v3 --alpha=0.2 --max_steps=3000000 &
CUDA_VISIBLE_DEVICES=0 python sac.py --env_name=Walker2d-v3 --seed=1 --max_steps=3000000 &
CUDA_VISIBLE_DEVICES=0 python sac.py --env_name=Walker2d-v3 --seed=1 --alpha=0.2 --max_steps=3000000 &
CUDA_VISIBLE_DEVICES=0 python sac.py --env_name=Hopper-v3 --seed=1 --max_steps=3000000 &
CUDA_VISIBLE_DEVICES=1 python sac.py --env_name=Hopper-v3 --seed=1 --alpha=0.2 --max_steps=3000000 &
CUDA_VISIBLE_DEVICES=1 python sac.py --env_name=Humanoid-v3 --seed=1 --max_steps=3000000 &
CUDA_VISIBLE_DEVICES=1 python sac.py --env_name=Humanoid-v3 --seed=1 --alpha=0.2 --max_steps=3000000 &



CUDA_VISIBLE_DEVICES=2 python sac.py --env_name=Walker2d-v3 --seed=2 --max_steps=3000000 --save_dir_root=/data/pulkitag/results/taochen/sac &
CUDA_VISIBLE_DEVICES=2 python sac.py --env_name=Walker2d-v3 --seed=2 --alpha=0.2 --max_steps=3000000 --save_dir_root=/data/pulkitag/results/taochen/sac &
CUDA_VISIBLE_DEVICES=2 python sac.py --env_name=Hopper-v3 --seed=2 --max_steps=3000000 /data/pulkitag/results/taochen/sac &
CUDA_VISIBLE_DEVICES=1 python sac.py --env_name=Hopper-v3 --seed=2 --alpha=0.2 --max_steps=3000000 --save_dir_root=/data/pulkitag/results/taochen/sac &
CUDA_VISIBLE_DEVICES=1 python sac.py --env_name=Humanoid-v3 --seed=2 --max_steps=3000000 --save_dir_root=/data/pulkitag/results/taochen/sac &
CUDA_VISIBLE_DEVICES=1 python sac.py --env_name=Humanoid-v3 --seed=2 --alpha=0.2 --max_steps=3000000  --save_dir_root=/data/pulkitag/results/taochen/sac &


CUDA_VISIBLE_DEVICES=0 python sac.py --env_name=Walker2d-v3 --seed=1 --max_steps=3000000 --tgt_entropy=0. --save_dir_root=/data/pulkitag/results/taochen/sac &
CUDA_VISIBLE_DEVICES=0 python sac.py --env_name=Hopper-v3 --seed=1 --max_steps=3000000  --tgt_entropy=0. --save_dir_root=/data/pulkitag/results/taochen/sac &
CUDA_VISIBLE_DEVICES=4 python sac.py --env_name=Humanoid-v3 --seed=1 --max_steps=3000000  --tgt_entropy=0. --save_dir_root=/data/pulkitag/results/taochen/sac &
CUDA_VISIBLE_DEVICES=4 python sac.py --env_name=Walker2d-v3 --seed=0 --max_steps=3000000 --tgt_entropy=0. --save_dir_root=/data/pulkitag/results/taochen/sac &
CUDA_VISIBLE_DEVICES=5 python sac.py --env_name=Hopper-v3 --seed=0 --max_steps=3000000  --tgt_entropy=0. --save_dir_root=/data/pulkitag/results/taochen/sac &
CUDA_VISIBLE_DEVICES=5 python sac.py --env_name=Humanoid-v3 --seed=0 --max_steps=3000000  --tgt_entropy=0. --save_dir_root=/data/pulkitag/results/taochen/sac &
CUDA_VISIBLE_DEVICES=0 python sac.py --env_name=Walker2d-v3 --seed=2 --max_steps=3000000 --tgt_entropy=0. --save_dir_root=/data/pulkitag/results/taochen/sac &
CUDA_VISIBLE_DEVICES=4 python sac.py --env_name=Hopper-v3 --seed=2 --max_steps=3000000  --tgt_entropy=0. --save_dir_root=/data/pulkitag/results/taochen/sac &
CUDA_VISIBLE_DEVICES=5 python sac.py --env_name=Humanoid-v3 --seed=2 --max_steps=3000000  --tgt_entropy=0. --save_dir_root=/data/pulkitag/results/taochen/sac &
wait
