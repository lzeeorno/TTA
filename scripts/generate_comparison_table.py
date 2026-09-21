#!/usr/bin/env python3
"""
生成实验结果对比表格
从results/目录收集所有JSON结果文件，生成Markdown格式的对比表格
"""

import json
import os
from pathlib import Path
import pandas as pd
import argparse
from typing import List, Dict

def collect_results(results_dir: str = './results') -> pd.DataFrame:
    """收集所有实验结果"""
    results = []

    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return pd.DataFrame()

    # 遍历所有JSON文件
    json_files = list(results_path.rglob('*.json'))
    print(f"📊 Found {len(json_files)} result files")

    for json_file in json_files:
        try:
            with open(json_file) as f:
                data = json.load(f)

                # 提取关键信息
                results.append({
                    'method': data.get('method', 'unknown'),
                    'backbone': data.get('backbone', 'unknown'),
                    'dataset': data.get('dataset', 'unknown'),
                    'seed': data.get('seed', 0),
                    'mean_accuracy': data.get('mean_accuracy', 0.0),
                    'timestamp': data.get('timestamp', ''),
                    'file': str(json_file.relative_to(results_path))
                })
        except Exception as e:
            print(f"⚠️  Error reading {json_file}: {e}")

    if not results:
        print("❌ No valid results found")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    print(f"✅ Loaded {len(df)} experiment results")
    return df

def generate_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """生成汇总表格（按dataset, backbone, method分组）"""
    if df.empty:
        return pd.DataFrame()

    # 按dataset, backbone, method分组，计算平均值和标准差
    summary = df.groupby(['dataset', 'backbone', 'method'])['mean_accuracy'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('count', 'count')
    ]).reset_index()

    # 格式化结果字符串
    summary['result'] = summary.apply(
        lambda x: f"{x['mean']:.2f} ± {x['std']:.2f} ({int(x['count'])})"
        if x['count'] > 1 else f"{x['mean']:.2f}",
        axis=1
    )

    # 创建透视表
    pivot = summary.pivot_table(
        index='method',
        columns=['dataset', 'backbone'],
        values='result',
        aggfunc='first'
    )

    return pivot

def generate_detailed_table(df: pd.DataFrame) -> pd.DataFrame:
    """生成详细表格（包含每个种子的结果）"""
    if df.empty:
        return pd.DataFrame()

    # 选择关键列
    detailed = df[['dataset', 'backbone', 'method', 'seed', 'mean_accuracy']].copy()
    detailed['mean_accuracy'] = detailed['mean_accuracy'].apply(lambda x: f"{x:.2f}")

    # 排序
    detailed = detailed.sort_values(['dataset', 'backbone', 'method', 'seed'])

    return detailed

def find_best_methods(df: pd.DataFrame) -> Dict[str, Dict]:
    """找出每个数据集上表现最好的方法"""
    if df.empty:
        return {}

    best_methods = {}

    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]

        # 按方法分组，计算平均准确率
        method_avg = dataset_df.groupby('method')['mean_accuracy'].mean().sort_values(ascending=False)

        best_methods[dataset] = {
            'best_method': method_avg.index[0],
            'best_accuracy': method_avg.iloc[0],
            'ranking': method_avg.to_dict()
        }

    return best_methods

def main():
    parser = argparse.ArgumentParser(description='Generate comparison table from experiment results')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory containing result JSON files')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (default: print to stdout)')
    parser.add_argument('--format', type=str, default='markdown',
                        choices=['markdown', 'latex', 'csv'],
                        help='Output format')
    parser.add_argument('--detailed', action='store_true',
                        help='Show detailed results (all seeds)')

    args = parser.parse_args()

    # 收集结果
    print("=" * 60)
    print("Collecting Results...")
    print("=" * 60)
    df = collect_results(args.results_dir)

    if df.empty:
        print("\n❌ No results to display")
        return

    print(f"\n📈 Dataset Summary:")
    print(df.groupby(['dataset', 'method']).size().unstack(fill_value=0))

    # 生成表格
    print("\n" + "=" * 60)
    print("Generating Tables...")
    print("=" * 60)

    if args.detailed:
        table = generate_detailed_table(df)
        title = "Detailed Results (All Seeds)"
    else:
        table = generate_summary_table(df)
        title = "Summary Results (Mean ± Std)"

    # 找出最佳方法
    best_methods = find_best_methods(df)

    # 输出
    output_lines = []
    output_lines.append(f"\n# {title}\n")

    if args.format == 'markdown':
        output_lines.append(table.to_markdown())
    elif args.format == 'latex':
        output_lines.append(table.to_latex())
    elif args.format == 'csv':
        output_lines.append(table.to_csv())

    # 添加最佳方法总结
    output_lines.append("\n\n# Best Methods by Dataset\n")
    for dataset, info in best_methods.items():
        output_lines.append(f"\n## {dataset}")
        output_lines.append(f"**Best Method**: {info['best_method']} ({info['best_accuracy']:.2f}%)\n")
        output_lines.append("\nRanking:")
        for i, (method, acc) in enumerate(sorted(info['ranking'].items(),
                                                   key=lambda x: x[1], reverse=True), 1):
            output_lines.append(f"  {i}. {method}: {acc:.2f}%")

    output_text = "\n".join(output_lines)

    # 保存或打印
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_text)
        print(f"\n✅ Results saved to: {args.output}")
    else:
        print(output_text)

if __name__ == '__main__':
    main()
