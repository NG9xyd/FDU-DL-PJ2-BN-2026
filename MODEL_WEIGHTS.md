# 模型权重

模型权重未直接提交到 Git 历史中。请将以下三个文件上传到 GitHub Release
或网盘，并将表格中的“待填写”替换为实际下载链接。

| 模型 | 本地路径 | 大小 | SHA-256 | 下载链接 |
|---|---|---:|---|---|
| 任务一最佳 CNN | `codes/task1_cifar10/results/task1/sgd/best.pt` | 4.40 MB | `552F77D228503BB637E1B2E42B8903153A325BADD92707F6891BCA99BA898EFD` | 待填写 |
| VGG-A + BN | `codes/task2_batchnorm/results_full/comparison/bn/best.pt` | 37.24 MB | `0AA27102E9BAA132634AEAFFC99D20B5484F153FC79749AA55739DCF0C81F05F` | 待填写 |
| VGG-A（无 BN） | `codes/task2_batchnorm/results_full/comparison/no_bn/best.pt` | 37.20 MB | `88AB24086CD273813C8DC2C5E4B8E9ABF20E5E13B17494F492047E86C8CB8B44` | 待填写 |

推荐将权重作为 GitHub Release 附件发布。这样既能提供稳定下载链接，也不会让
每次克隆仓库时都下载约 79 MB 的二进制文件。
