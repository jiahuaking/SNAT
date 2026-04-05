#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础使用示例 - 展示本库的核心功能
"""

import numpy as np
import sys
import os

# 添加src目录到路径，使得可以导入本地模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.snat.Node_Evaluation import (
    Calculate_H_Index, 
    Calculate_C_Index, 
    Calculate_BI_directional_h_index,
    Calculate_H_L_Index,
    Calculate_Iterative_C_Index
)

def demo_h_index():
    """演示基础H-index计算"""
    print("=" * 50)
    print("基础H-index计算示例")
    print("=" * 50)
    
    # 示例数据：论文引用次数
    citations = [10, 8, 5, 4, 3, 2, 1]
    h_index = Calculate_H_Index(citations)
    print(f"论文引用次数: {citations}")
    print(f"H-index: {h_index}")
    print()

def demo_c_index():
    """演示C-index计算"""
    print("=" * 50)
    print("C-index计算示例（加权无向网络）")
    print("=" * 50)
    
    # 示例：一个4节点的加权网络邻接矩阵
    adjacency_matrix = [
        [0, 2, 1, 0],
        [2, 0, 3, 1],
        [1, 3, 0, 2],
        [0, 1, 2, 0]
    ]
    
    print("网络邻接矩阵:")
    for row in adjacency_matrix:
        print(row)
    
    c_indices = Calculate_C_Index(adjacency_matrix)
    print(f"\n各节点C-index: {c_indices}")
    print()

def demo_bi_directional_h_index():
    """演示双向H-index计算"""
    print("=" * 50)
    print("双向H-index计算示例（有向网络）")
    print("=" * 50)
    
    # 示例：有向网络邻接矩阵
    directed_matrix = np.array([
        [0, 3, 1, 0],
        [0, 0, 2, 1],
        [1, 0, 0, 2],
        [0, 1, 0, 0]
    ])
    
    print("有向网络邻接矩阵:")
    print(directed_matrix)
    
    h_in, h_out = Calculate_BI_directional_h_index(directed_matrix)
    print(f"\n各节点入向H-index: {h_in}")
    print(f"各节点出向H-index: {h_out}")
    print()

def demo_hl_index():
    """演示Hl-index计算"""
    print("=" * 50)
    print("Hl-index计算示例（多层次评价）")
    print("=" * 50)
    
    # 示例：每个研究者及其论文的引用数据
    researcher_data = [
        [10, 8, 5, 3],      # 研究者1的论文引用
        [15, 12, 8, 6, 4],  # 研究者2的论文引用
        [20, 15, 10, 7, 5, 3]  # 研究者3的论文引用
    ]
    
    print("研究者论文引用数据（每个子列表代表一个研究者）:")
    for i, data in enumerate(researcher_data):
        print(f"研究者{i+1}: {data}")
    
    h_values, final_h_index = Calculate_H_L_Index(researcher_data)
    print(f"\n各研究者H-index: {h_values}")
    print(f"整体Hl-index: {final_h_index}")
    print()

def demo_iterative_c_index():
    """演示迭代C-index计算"""
    print("=" * 50)
    print("迭代C-index计算示例")
    print("=" * 50)
    
    # 示例：一个5节点的加权网络
    adjacency_matrix = [
        [0, 2, 1, 0, 1],
        [2, 0, 3, 1, 0],
        [1, 3, 0, 2, 1],
        [0, 1, 2, 0, 2],
        [1, 0, 1, 2, 0]
    ]
    
    print("网络邻接矩阵:")
    for row in adjacency_matrix:
        print(row)
    
    iterative_c_indices = Calculate_Iterative_C_Index(adjacency_matrix)
    print(f"\n各节点迭代C-index: {iterative_c_indices}")
    print()

if __name__ == "__main__":
    print("网络节点重要性评价库 - 使用示例")
    print("本示例展示了本库的核心功能\n")
    
    demo_h_index()
    demo_c_index()
    demo_bi_directional_h_index()
    demo_hl_index()
    demo_iterative_c_index()
    
    print("=" * 50)
    print("示例运行完毕！")
    print("更多信息请参考README.md文档。")