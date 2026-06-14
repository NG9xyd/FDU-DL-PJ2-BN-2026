# 任务二：Batch Normalization

所有命令均在本目录执行：

```powershell
cd codes/task2_batchnorm
```

## VGG-A 有无 BN 的对比

```powershell
python run_experiments.py --mode compare `
  --data-dir ..\task1_cifar10\dataset `
  --train-items -1 --epochs 20 --num-workers 4 `
  --output-dir results_full
```

该命令使用相同的数据划分、随机种子、优化器和学习率分别训练 VGG-A 与
VGG-A + BN。结果保存在 `results_full/comparison/`。

## Loss landscape 与梯度分析

```powershell
python run_experiments.py --mode landscape `
  --data-dir ..\task1_cifar10\dataset `
  --landscape-items -1 --landscape-epochs 5 `
  --learning-rates 0.0001,0.0005,0.001,0.002 `
  --num-workers 4 --output-dir results_full
```

实验使用完整的 45,000 张训练图像和四个学习率。梯度指标使用 FP32 记录，
避免混合精度缩放影响统计结果。输出保存在 `results_full/landscape/`。

主要结果图片：

- `results_full/comparison/bn_training_comparison.png`
- `results_full/landscape/loss_landscape_comparison.png`
- `results_full/landscape/loss_landscape_comparison_zoomed.png`
- `results_full/landscape/gradient_smoothness_comparison.png`

## 快速检查

```powershell
python run_experiments.py --mode all --fake-data --epochs 1 `
  --landscape-epochs 1 --max-steps 2 --num-workers 0 `
  --output-dir smoke_results
```
