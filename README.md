# 二维 VAE-GAN 手写数字生成

本项目继承二维 VAE，增加具有谱归一化的卷积判别器，使用 L1 重构、KL 正则与最小二乘对抗目标。包含相同生成模型更新轮次的纯 VAE 对照、实例噪声与对抗权重消融，以及独立分类器评价。

完整成稿为 `reports/vaegan_mnist_report.md`，正文包含方法、全部指定生成结果、实际实验数据与局限，附录包含不超过 800 行的完整可运行实现。报告可独立阅读，复现无需随附预训练权重。未生成 PDF。

## 项目结构

```text
.
├── src/vaegan.py                 # 完整训练、评价与绘图实现
├── tests/test_vaegan.py          # 数学、梯度、流程及成果一致性检查
├── data/MNIST/raw/               # 原始数据，自动下载，忽略提交
├── artifacts/                   # 预训练与各实验权重，忽略提交
├── results/                     # 图像、数组、指标、训练记录及摘要
│   └── ablation_no_noise/        # 初步实验本地备份，忽略提交
├── reports/vaegan_mnist_report.md
└── requirements.txt
```

## 运行

在安装依赖的 Python 环境中，从项目根目录执行：

```powershell
python -m pip install -r requirements.txt
python src/vaegan.py --mode all
python -m unittest discover -s tests -v
```

程序自动下载 MNIST，优先使用 CUDA，未检测到 CUDA 时使用 CPU。首次运行在缺少 `artifacts/vae_base.pt` 时完成 160 轮预训练；随后运行 80 轮主模型训练、各 80 轮的两组对抗消融、80 轮纯 VAE 对照和 6 轮分类器训练。已有兼容的 VAE 检查点时可保存在 `artifacts/vae_base.pt`，其必须包含 `model`、`config`、`history` 字段。

训练完成后，单独重建指标与图像：

```powershell
python src/vaegan.py --mode generate
```

单独运行各阶段可用 `--mode pretrain`、`--mode gan`、`--mode ablation`、`--mode control` 或 `--mode classifier`。重新训练会覆盖对应阶段的权重及训练记录；`generate` 只读取模型并更新结果。程序只在当前进程设置运行库兼容选项，不修改系统配置。

## 结果与报告

| 文件 | 内容 |
| --- | --- |
| `results/latent_grid.png` | 指定范围与间隔的全部 21×21 个输出 |
| `results/five_samples.png` | 固定训练集首个数字 5 的输入、均值参考及十次后验输出 |
| `results/five_samples.csv` | 十组随机噪声及对应隐变量 |
| `results/prior_comparison.png` | 相同先验隐变量下的三组模型对比 |
| `results/ablation_comparison.png` | 相同先验隐变量下的三组对抗配置对比 |
| `results/reconstructions.png` | 各类别首个测试样本的重构对比 |
| `results/training_curves.png` | 主模型和纯 VAE 对照的训练曲线 |
| `results/latent_scatter.png` | 全测试集后验均值分布 |
| `results/grid_comparison.png` | 完整生成网格的并排对照 |
| `results/metrics.json` | 五组模型的指标、采样参数与配置 |
| `results/samples.npz` | 绘图使用的原始浮点数组 |
| `results/*_history.csv` | 各阶段逐轮记录 |
| `results/provenance.json` | 原有检查点核验与公开数据的摘要 |
| `results/sha256.json` | 当前实现、模型与结果的完整性摘要 |

报告图像使用相对路径引用。导出 Markdown 时，保留 `reports/` 与 `results/` 的相对位置，并确保导出工具支持 LaTeX 公式、表格与代码换页。全文不依赖外部代码附件；模型、数据、原始浮点数组、初步实验备份和缓存不纳入版本控制。

所有生成图像统一灰度范围，未进行二值化、锐化或样本筛选。识别置信度不等于生成准确率；使用 MNIST 分类器特征的 Fréchet 距离不属于标准 Inception FID。报告同时披露重构误差、分布差异及探索性比较的限制。
