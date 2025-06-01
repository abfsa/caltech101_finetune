# Caltech101 图像分类项目

基于PyTorch的迁移学习框架，实现Caltech101数据集的图像分类任务。

## 环境配置
```bash
pip install -r requirements.txt
```
数据准备
下载Caltech101数据集至data/caltech101目录
目录结构应自动组织为：


```
data/caltech101/
    ├── annotations/
    ├── images/
    └── splits/       # 训练/验证/测试划分
```
快速开始
训练模型

```bash
python codes/train.py \
    --config config.yaml \
    --data_path data/caltech101 \
    --output_dir models/
```
测试模型

```bash
python codes/test.py \
    --model_path models/best_model.pth \
    --data_path data/caltech101
    --model resnet50
```
超参数搜索

```bash
python hparamsearch.py \
    --data_path data/caltech101 \
    --n_trials 50 \
    --study_name caltech_hparam
```
可视化分析
参考notebooks/vis.ipynb进行：

训练过程可视化
混淆矩阵分析
特征可视化
配置说明
参考config.yaml中的参数设置：


yaml
Apply
model: resnet50
batch_size: 32
epochs: 50
learning_rate: 0.001
optimizer: adam
注意事项
首次运行时会自动进行数据集划分
默认使用GPU加速训练，CPU模式会自动降级
模型保存路径为models/目录