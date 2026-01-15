python train_experts.py --expert all

python train_experts.py --expert des

python train_experts.py --expert post

python train_experts.py --expert graph

python train_experts.py --expert graph --num_epochs 20

python train_experts.py --expert cat --num_epochs 10

python train_experts.py --expert num --num_epochs 10

python train_cat_num_experts.py --expert both --epochs 10

自定义参数
python train_experts.py --expert des,tweets --num_epochs 15 --batch_size 64

查看所有参数
python train_experts.py --help

chmod +x test_thresholds.sh
./test_thresholds.sh

Mixing_Expert_bot/
├── configs/
│   └── expert_configs.py
├── scripts/
│   ├── gate.py
│   ├── iso_and_nonisol.py
│   ├── feature_fusion.py
│   ├── train_experts.py
│   └── expert_trainer.py
├── src/
│   ├── dataset.py
│   ├── model.py
│   └── metrics.py
