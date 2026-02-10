#!/usr/bin/env python3
"""
将 ImageNet-1k 数据集从 Parquet 格式转换为官方 ImageNet 目录结构

用法:
    python convert_to_imagenet_official.py --parquet-dir data/imagenet-1k/data --output-dir data/imagenet-1k/imagenet_official --split train
    python convert_to_imagenet_official.py --parquet-dir data/imagenet-1k/data --output-dir data/imagenet-1k/imagenet_official --split val
    python convert_to_imagenet_official.py --parquet-dir data/imagenet-1k/data --output-dir data/imagenet-1k/imagenet_official --split test
"""

import argparse
import sys
from pathlib import Path
from tqdm import tqdm
import os

# 添加 classes.py 所在目录到路径
sys.path.insert(0, str(Path(__file__).parent / 'data' / 'imagenet-1k'))
from classes import IMAGENET2012_CLASSES

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("警告: datasets 库未安装，尝试使用 pyarrow...")
    try:
        import pyarrow.parquet as pq
        import pandas as pd
        HAS_PYARROW = True
    except ImportError:
        HAS_PYARROW = False
        print("错误: 需要安装 datasets 或 pyarrow+pandas")
        sys.exit(1)


def create_label_to_wnid_mapping():
    """创建从整数标签到 WordNet ID 的映射"""
    label_to_wnid = {}
    for idx, (wnid, desc) in enumerate(IMAGENET2012_CLASSES.items()):
        label_to_wnid[idx] = wnid
    return label_to_wnid


def convert_parquet_to_imagenet_format(
    parquet_dir: str,
    output_dir: str,
    split: str = 'train',
    max_samples: int = None,
    image_format: str = 'JPEG'
):
    """
    将 Parquet 格式的数据集转换为官方 ImageNet 格式
    
    Args:
        parquet_dir: Parquet 文件所在目录
        output_dir: 输出目录
        split: 数据集分割 ('train' / 'val' / 'test')
        max_samples: 最大处理样本数（用于测试）
        image_format: 图像格式（默认 'JPEG'）
    """
    parquet_dir = Path(parquet_dir)
    output_dir = Path(output_dir)
    
    # 创建输出目录
    split_output_dir = output_dir / split
    split_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建标签到 WordNet ID 的映射
    label_to_wnid = create_label_to_wnid_mapping()
    
    # 查找所有 parquet 文件（val 对应 validation-*.parquet，test 对应 test-*.parquet）
    if split == 'train':
        parquet_files = sorted(parquet_dir.glob('train-*.parquet'))
    elif split == 'val':
        parquet_files = sorted(parquet_dir.glob('validation-*.parquet'))
    elif split == 'test':
        parquet_files = sorted(parquet_dir.glob('test-*.parquet'))
    else:
        raise ValueError(f"未知的 split: {split}，必须是 'train'、'val' 或 'test'")
    
    if not parquet_files:
        raise FileNotFoundError(f"在 {parquet_dir} 中未找到 {split} 的 parquet 文件")
    
    print(f"找到 {len(parquet_files)} 个 {split} parquet 文件")
    
    total_processed = 0
    image_counter = {}  # 用于统计每个类别的图像数量
    
    for parquet_file in tqdm(parquet_files, desc=f"处理 {split} 文件"):
        # 尝试使用 datasets 库加载
        use_datasets = HAS_DATASETS
        
        if use_datasets:
            try:
                dataset = load_dataset('parquet', data_files=str(parquet_file), split='train')
            except Exception as e:
                print(f"警告: 使用 datasets 加载 {parquet_file.name} 失败: {e}")
                use_datasets = False
        
        # 如果 datasets 失败，尝试使用 pyarrow+pandas
        if not use_datasets and HAS_PYARROW:
            try:
                table = pq.read_table(parquet_file)
                df = pd.DataFrame(table.to_pandas())
                dataset = df.to_dict('records')
            except Exception as e:
                print(f"警告: 使用 pyarrow 加载 {parquet_file.name} 失败: {e}")
                continue
        
        # 处理每个样本
        for idx, sample in enumerate(tqdm(dataset, desc=f"处理 {Path(parquet_file).name}", leave=False)):
            if max_samples is not None and total_processed >= max_samples:
                print(f"\n达到最大样本数限制 ({max_samples})，停止处理")
                break
            
            # 获取图像和标签
            if use_datasets:
                image = sample['image']
                label = sample['label']
            else:
                image = sample.get('image')
                label = sample.get('label')
            
            if image is None:
                print(f"警告: 样本 {idx} 没有图像数据，跳过")
                continue
            
            # train/val 必须有标签；test 允许无标签（label 为 -1 或 None）
            if split != 'test' and (label is None or (label == -1)):
                print(f"警告: 样本 {idx} 没有标签数据，跳过")
                continue
            
            # 获取 WordNet ID（test 无标签时为 None，仍会保存到 test/ 根目录）
            wnid = label_to_wnid.get(label) if label is not None and label >= 0 else None
            
            if split != 'test' and wnid is None:
                print(f"警告: 未知的 label {label}，跳过")
                continue
            
            # 确定保存目录：test 无标签时保存到 test/ 根目录，否则按 wnid 分子目录
            if split == 'test' and wnid is None:
                wnid_dir = split_output_dir
            else:
                wnid_dir = split_output_dir / (wnid if wnid else 'unknown')
            wnid_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成文件名
            if split == 'train':
                # train: n01440764_00000000.JPEG
                image_counter[wnid] = image_counter.get(wnid, 0)
                filename = f"{wnid}_{image_counter[wnid]:08d}.JPEG"
                image_counter[wnid] += 1
            elif split == 'val':
                # val: ILSVRC2012_val_00000000.JPEG
                image_counter['val'] = image_counter.get('val', 0)
                filename = f"ILSVRC2012_val_{image_counter['val']:08d}.JPEG"
                image_counter['val'] += 1
            else:
                # test: ILSVRC2012_test_00000000.JPEG
                image_counter['test'] = image_counter.get('test', 0)
                filename = f"ILSVRC2012_test_{image_counter['test']:08d}.JPEG"
                image_counter['test'] += 1
            
            # 保存图像
            output_path = wnid_dir / filename
            
            try:
                # datasets 库返回的 image 已经是 PIL Image，可以直接保存
                # 但为了安全，确保是 RGB 模式
                if hasattr(image, 'mode') and image.mode not in ['RGB', 'RGBA', 'L', 'P']:
                    image = image.convert('RGB')
                elif hasattr(image, 'mode') and image.mode in ['RGBA', 'L', 'P']:
                    image = image.convert('RGB')
                
                # 保存图像
                image.save(str(output_path), format=image_format.upper(), quality=95)
                total_processed += 1
                
            except Exception as e:
                print(f"警告: 无法保存图像到 {output_path}: {e}")
                print(f"  图像类型: {type(image)}")
                if hasattr(image, 'mode'):
                    print(f"  图像模式: {image.mode}")
                if hasattr(image, 'size'):
                    print(f"  图像大小: {image.size}")
                continue
        
        if max_samples is not None and total_processed >= max_samples:
            break
    
    print(f"\n转换完成！共处理 {total_processed} 个样本")
    print(f"输出目录: {split_output_dir}")
    
    # 统计每个类别的图像数量
    print(f"\n各类别图像数量统计:")
    for wnid in sorted(image_counter.keys()):
        if wnid not in ('val', 'test'):
            count = image_counter[wnid]
            class_name = IMAGENET2012_CLASSES.get(wnid, 'Unknown')
            print(f"  {wnid} ({class_name[:30]}): {count} 张")


def main():
    parser = argparse.ArgumentParser(
        description='将 ImageNet-1k 数据集从 Parquet 格式转换为官方 ImageNet 目录结构'
    )
    parser.add_argument(
        '--parquet-dir',
        type=str,
        required=True,
        help='Parquet 文件所在目录'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='输出目录'
    )
    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'val', 'test'],
        default='train',
        help='数据集分割 (train / val / test)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='最大处理样本数（用于测试，默认处理所有样本）'
    )
    parser.add_argument(
        '--image-format',
        type=str,
        default='JPEG',
        help='图像格式（默认 JPEG）'
    )
    
    args = parser.parse_args()
    
    convert_parquet_to_imagenet_format(
        parquet_dir=args.parquet_dir,
        output_dir=args.output_dir,
        split=args.split,
        max_samples=args.max_samples,
        image_format=args.image_format
    )


if __name__ == '__main__':
    main()
