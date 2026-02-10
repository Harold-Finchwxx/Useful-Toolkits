ImageNet-1k 数据集转换使用说明
环境准备
确保在 preprocess conda 环境中运行：

conda activate preprocess
如果缺少依赖，请安装：

pip install "fsspec>=2023.1.0,<=2025.10.0" dill filelock "httpx<1.0.0" "huggingface-hub<2.0,>=0.25.0" "multiprocess<0.70.19" "pyyaml>=5.1" "requests>=2.32.2" xxhash datasets pillow tqdm
使用方法
转换训练集
cd /mnt/petrelfs/wangxuanxu/RAE_for_Video/RAE
python3 convert_to_imagenet_official.py \
    --parquet-dir data/imagenet-1k/data \
    --output-dir data/imagenet-1k/imagenet_official \
    --split train
转换验证集
python3 convert_to_imagenet_official.py \
    --parquet-dir data/imagenet-1k/data \
    --output-dir data/imagenet-1k/imagenet_official \
    --split val
转换测试集
python3 convert_to_imagenet_official.py \
    --parquet-dir data/imagenet-1k/data \
    --output-dir data/imagenet-1k/imagenet_official \
    --split test
测试模式（只处理少量样本）
# 只处理前 100 个样本进行测试
python3 convert_to_imagenet_official.py \
    --parquet-dir data/imagenet-1k/data \
    --output-dir data/imagenet-1k/imagenet_official \
    --split train \
    --max-samples 100
输出格式
转换后的目录结构：

imagenet_official/
├── train/
│   ├── n01440764/
│   │   ├── n01440764_00000000.JPEG
│   │   ├── n01440764_00000001.JPEG
│   │   └── ...
│   ├── n01443537/
│   │   └── ...
│   └── ...
├── val/
│   ├── n01440764/
│   │   ├── ILSVRC2012_val_00000000.JPEG
│   │   └── ...
│   └── ...
└── test/
    ├── n01440764/
    │   ├── ILSVRC2012_test_00000000.JPEG
    │   └── ...
    └── ...
注意事项
环境要求: 必须在 preprocess conda 环境中运行
磁盘空间: 确保有足够的磁盘空间存储转换后的图像文件
处理时间: 完整数据集转换可能需要较长时间，建议使用 --max-samples 先测试
文件格式: 输出图像格式为 JPEG，质量设置为 95
验证转换结果
检查输出文件：

# 查看训练集目录结构
ls data/imagenet-1k/imagenet_official/train/ | head -20

# 统计每个类别的图像数量
find data/imagenet-1k/imagenet_official/train -name "*.JPEG" | wc -l

# 查看某个类别的图像
ls data/imagenet-1k/imagenet_official/train/n01440764/ | head -5
