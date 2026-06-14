# 神经网络与深度学习 Project 2

本项目使用 PyTorch 完成以下任务：

1. 在 CIFAR-10 上训练可配置卷积神经网络，并进行消融实验。
2. 使用 VGG-A 比较有无 Batch Normalization 时的训练表现，并分析优化过程。

## 主要结果

| 实验 | 测试准确率 |
|---|---:|
| 任务一最佳 CNN | 92.06% |
| VGG-A（无 BN） | 75.88% |
| VGG-A + BN | 82.50% |

完整中文实验报告位于 `report/report.tex`。

## 目录结构

```text
codes/
  task1_cifar10/       任务一模型、训练与可视化代码
  task2_batchnorm/     任务二 VGG-A 与 BN 实验代码
report/
  report.tex           LaTeX 实验报告
  figures/             报告使用的实验图片
MODEL_WEIGHTS.md       模型权重信息与下载链接
```

数据集和模型权重未直接提交到 Git 仓库。

## 环境配置

原实验环境为 Python 3.12、PyTorch 2.9.1 和 CUDA 12.6。一般的
PyTorch 2.x 环境也可以运行。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 下载数据集

任务一首次运行时会自动下载 CIFAR-10：

```powershell
cd codes\task1_cifar10
python train.py --experiment-name data_check --epochs 1 --train-items 128 --test-items 128
cd ..\..
```

数据集将保存在 `codes/task1_cifar10/dataset/cifar-10-batches-py`。
任务二复用同一数据目录。

## 快速检查

以下命令使用生成的假数据，仅用于检查代码、GPU 和结果输出流程是否正常：

```powershell
cd codes\task1_cifar10
python train.py --experiment-name smoke --fake-data --epochs 1 --num-workers 0

cd ..\task2_batchnorm
python run_experiments.py --mode all --fake-data --epochs 1 `
  --landscape-epochs 1 --learning-rates 0.001,0.002 `
  --max-steps 2 --num-workers 0 --output-dir smoke_results
cd ..\..
```

## 复现任务一

在 `codes/task1_cifar10` 目录中执行以下命令。

复现最佳模型：

```powershell
python train.py --experiment-name sgd --optimizer sgd --lr 0.1 `
  --epochs 50 --no-download
```

复现消融实验：

```powershell
python train.py --experiment-name baseline --epochs 30 --no-download
python train.py --experiment-name small --channels 32,64,128 --epochs 30 --no-download
python train.py --experiment-name gelu --activation gelu --epochs 30 --no-download
python train.py --experiment-name focal --loss focal --epochs 30 --no-download
python train.py --experiment-name label_smoothing --label-smoothing 0.1 --epochs 30 --no-download
python train.py --experiment-name no_regularization --dropout 0 --weight-decay 0 --epochs 30 --no-download
python summarize_results.py
```

每次实验的配置、指标和图片保存在
`results/task1/<experiment-name>/`。

## 复现任务二

在 `codes/task2_batchnorm` 目录中执行以下命令。实验从 CIFAR-10 标准训练集
划分 5,000 张作为验证集，使用其余 45,000 张训练图像。

复现 VGG-A 有无 BN 的对比实验：

```powershell
python run_experiments.py --mode compare `
  --data-dir ..\task1_cifar10\dataset `
  --train-items -1 --epochs 20 --num-workers 4 `
  --output-dir results_full
```

复现 loss landscape 和梯度分析：

```powershell
python run_experiments.py --mode landscape `
  --data-dir ..\task1_cifar10\dataset `
  --landscape-items -1 --landscape-epochs 5 `
  --learning-rates 0.0001,0.0005,0.001,0.002 `
  --num-workers 4 --output-dir results_full
```

实验结果保存在 `results_full/`。

## 模型权重

项目报告要求提供训练后的模型权重。权重文件单独发布，不直接写入 Git 历史。
文件名、校验值和下载链接见 `MODEL_WEIGHTS.md`。

## 编译报告

使用 XeLaTeX 编译中文报告：

```powershell
cd report
xelatex report.tex
xelatex report.tex
```
