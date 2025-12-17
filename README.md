
python train_experts.py --expert all


python train_experts.py --expert des


python train_experts.py --expert post

python train_experts.py --expert graph

python train_experts.py --expert graph --num_epochs 20

python train_experts.py --expert cat --num_epochs 10

python train_experts.py --expert num --num_epochs 10

python gate_rl.py --epochs 20 --diversity_coef 0.9 --entropy_coef 0.9 --lr 1e-6

### 自定义参数
python train_experts.py --expert des,tweets --num_epochs 15 --batch_size 64

### 查看所有参数
python train_experts.py --help
