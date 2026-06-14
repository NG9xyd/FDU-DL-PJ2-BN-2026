# 任务一：CIFAR-10 图像分类

所有命令均在本目录执行：

```powershell
cd codes/task1_cifar10
```

## 快速检查

使用假数据检查训练和输出流程：

```powershell
python train.py --experiment-name smoke --fake-data --epochs 1 --num-workers 0
```

## 复现实验

基线实验：

```powershell
python train.py --experiment-name baseline --epochs 30
```

不同通道数、激活函数、损失函数和优化器的对比实验：

```powershell
python train.py --experiment-name small --channels 32,64,128 --epochs 30
python train.py --experiment-name gelu --activation gelu --epochs 30
python train.py --experiment-name focal --loss focal --epochs 30
python train.py --experiment-name label_smoothing --label-smoothing 0.1 --epochs 30
python train.py --experiment-name no_regularization --dropout 0 --weight-decay 0 --epochs 30
python train.py --experiment-name sgd --optimizer sgd --lr 0.1 --epochs 50
```

实验完成后生成汇总表：

```powershell
python summarize_results.py
```

## 输出文件

每次实验会在 `results/task1/<experiment-name>/` 中生成：

- `best.pt`：验证集准确率最高的模型权重
- `config.json`：完整实验配置
- `history.csv`：每个 epoch 的训练和验证指标
- `summary.json`：最终验证集和测试集结果
- `training_curves.png`：训练曲线
- `confusion_matrix.png`：混淆矩阵
- `per_class_accuracy.csv`：逐类别准确率
- `first_layer_filters.png`：第一层卷积核
- `misclassified.png`：错误分类样例

首次正式运行时，程序会自动将 CIFAR-10 下载到 `dataset/`。
