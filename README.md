
python train_experts.py --expert all


python train_experts.py --expert des


python train_experts.py --expert post

python train_experts.py --expert graph


### 自定义参数
python train_experts.py --expert des,tweets --num_epochs 15 --batch_size 64

### 查看所有参数
python train_experts.py --help
