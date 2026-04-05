import pandas as pd
import numpy as np
from typing import List, Union, Tuple, Any, Optional
import numpy.typing as npt # 导入 numpy 的类型模块
import numbers

# 定义一个类型别名，表示所有可能的数字类型，包括 Python 原生 int, float 和 NumPy 的各种数字类型 (np.int32, np.float64 等)
Number = Union[int, float, npt.NDArray[np.integer], npt.NDArray[np.floating]]

def Calculate_H_Index(value: Union[List[Number], np.ndarray]): #输入数据
    '''
    输入：value为一个列表，以无序列表形式呈现该结点的一组绩效值

    输出：标量值，涵义为该结点重要性

    定义：
        一名学者有h篇论文分别被引用了至少 h次，且其余论文的引用次数均不超过h次，
        延伸到社交网络领域，其适用于对单一结点的自身重要性评价，无视了其与其他结点的交互。
        其输入应该是自身产出的一组绩效值，输出是标量形式的自身重要性度量。
    异常:
        TypeError: 如果输入不是列表、元组或 numpy 数组。
        ValueError: 如果列表中包含非数字元素或负数。

    时间复杂度：O(NlogN)，主要性能瓶颈在排序阶段。后续可尝试使用计数法排序，将时间复杂度降为O(N)
    '''
    if value is None:
        raise TypeError("输入不能为 None。请提供一个数值列表或 numpy 数组。")

    # 支持 numpy 数组转换
    if isinstance(value, np.ndarray):
        if value.ndim > 1:
            raise ValueError("输入必须是扁平的一维数组，检测到多维数组。")
        data_list = value.flatten().tolist()
    elif isinstance(value, (list, tuple)):
        data_list = list(value)
    else:
        raise TypeError(f"输入类型不支持: {type(value)}。期望 list, tuple 或 np.ndarray。")

    # 边界情况快速返回 ---
    n = len(data_list)
    if n == 0:
        return 0

    # 检查是否全为数字，且无负数
    for i, v in enumerate(data_list):
        if not isinstance(v, numbers.Number):
            raise TypeError(f"列表中包含非数字元素: 索引 {i} 的值为 '{v}' ({type(v)})。")
        if v < 0:
            raise ValueError(f"列表中包含负数: 索引 {i} 的值为 {v}。绩效值/引用数不能为负。")
    data_list.sort(reverse=True)

    # 排序后，如果最大值(第一个)是0，则 h 必为 0
    if data_list[0] == 0:
        return 0
    # 如果最小的值(最后一个) >= 列表长度，则 h = 列表长度
    if data_list[-1] >= n:
        return n

    # 通用遍历查找
    h = 0
    for i in range(n):
        # 当前排名 (从1开始): rank = i + 1
        # 当前值: val = data_list[i]
        # 条件: val >= rank
        current_rank = i + 1
        current_val = data_list[i]
        if current_val >= current_rank:
            # 暂时满足条件，更新 h
            h = current_rank
        else:
            # 一旦遇到 current_val < current_rank，由于列表是降序的，
            # 后面的值只会更小，排名只会更大，不可能再满足条件。
            # 因此可以直接跳出循环，当前的 h 就是最大值。
            break
    return h

def Calculate_H_L_Index( data_matrix: Union[List[List[Number]], np.ndarray]=None):
    """
    计算 Hl-index
    输入:
        data_matrix，二维列表，表示待评价结点的一组绩效项及绩效项的评价依据，可以是以下两种类型:
            1. List[List[int/float]]: Python 原生嵌套列表 。
            2. np.ndarray: NumPy 数组 (dtype=object)，用于存储不等长的行。
    输出:
        list: h_values 待评价结点的一组绩效值
        int: Hl-index 标量值。
    定义:
        引申自学者的引文网络评价学者重要性。该Hl指标相对传统H指标不仅考虑了学者论文的被引数量，还考虑了引文的质量。
        具体来说，使用H指标递归的计算引文的影响力
        """
    # 输入类型处理
    if isinstance(data_matrix, np.ndarray):
        if data_matrix.ndim != 1 and data_matrix.ndim != 2:
            raise ValueError("NumPy 输入必须是一维对象数组或二维数组。")
        # 如果是二维常规数组 (所有行等长)，直接迭代
        # 如果是不等长，通常是 dtype=object 的一维数组，每个元素是一个列表
        if data_matrix.ndim == 1:
            # 对象数组模式: array([list1, list2, ...])
            rows = data_matrix.tolist()
        else:
            # 二维模式: 转换为列表的列表
            rows = data_matrix.tolist()
    elif isinstance(data_matrix, list):
        if not data_matrix:
            return [], 0
        # 检查是否是嵌套列表
        if not all(isinstance(row, (list, tuple, np.ndarray)) for row in data_matrix):
            raise TypeError("输入列表必须包含子列表 (List of Lists)。")
        rows = data_matrix
    else:
        raise TypeError(f"输入必须是 List[List] 或 np.ndarray，得到: {type(data_matrix)}")

    # --- 快速空值处理 ---
    if not rows:
        return [], 0

    # --- 迭代计算第一层 h-index ---
    h_values = []
    for i, row in enumerate(rows):
        try:
            # 对每一行数据计算 h-index
            h_val = Calculate_H_Index(row)
            h_values.append(h_val)
        except Exception as e:
            raise type(e)(f"在处理第 {i} 个绩效项时出错: {e}") from e

    # --- 4. 计算最终 Hl-index ---
    return h_values,Calculate_H_Index(h_values)

def get_hl_index(data: Any) -> int:
    """
    便捷接口：仅返回最终的 Hl-index 标量值。
    参数:
        data: 嵌套列表数据。
        k: 最大递归深度 (整数或 None)。
    """
    _, result = Calculate_H_L_Index(data)
    return result

def Calculate_Recursive_Hl_Index(
        data: Any,
        k: Optional[int] = None,
        _depth: int = 0
) -> Tuple[List[int], int]:
    """
    计算任意层级嵌套列表的递归 Hl-index。

    输入:
        data: 嵌套列表 (List[List[...]]) 或 NumPy 对象数组。最深层应为数字列表。
        k: 最大递归深度 (考虑的最深层数)。若为 None，则递归至最底层数字。
        _depth: 当前递归深度 (内部自动维护，无需手动设置)。

    输出:
        Tuple[List[int], int]: (当前层各子项的 h-index 列表, 当前层聚合后的 h-index 值)

    逻辑说明:
        1. 若当前层全为数字：直接计算 h-index。
        2. 若达到深度限制 k 且仍含列表：将子列表长度作为估算值，计算 h-index (截断策略)。
        3. 否则：递归计算子项 h-index，再对结果列表计算 h-index。
    """
    # 数据标准化
    if isinstance(data, np.ndarray):
        if data.ndim == 0:
            items = [data.item()]
        elif data.ndim == 1:
            items = list(data)
        else:
            items = list(data)  # 处理二维或不规则对象数组
    elif isinstance(data, list):
        items = data
    elif isinstance(data, (int, float)):
        # 单个数字视为长度为1的列表
        val = 1 if data >= 1 else 0
        return [int(data)], val
    else:
        raise TypeError(f"不支持的数据类型: {type(data)}")

    if not items:
        return [], 0

    # 判断是否到达最底层 (全为数字)
    is_numeric_layer = all(isinstance(x, (int, float)) for x in items)

    if is_numeric_layer:
        h_val = Calculate_H_Index(items)
        return [int(x) for x in items], h_val

    # 检查深度限制 (截断策略)
    if k is not None and _depth >= k:
        # 不再深入，用子元素长度 (拓扑规模) 代替具体数值
        estimated_scores = []
        for item in items:
            if isinstance(item, (list, tuple, np.ndarray)):
                estimated_scores.append(len(item))
            elif isinstance(item, (int, float)):
                estimated_scores.append(int(item))
            else:
                estimated_scores.append(0)

        h_val = Calculate_H_Index(estimated_scores)
        return estimated_scores, h_val

    # 递归计算
    child_h_values = []
    for item in items:
        _, child_h = Calculate_Recursive_Hl_Index(item, k=k, _depth=_depth + 1)
        child_h_values.append(child_h)

    if not child_h_values:
        return [], 0

    final_h = Calculate_H_Index(child_h_values)
    return child_h_values, final_h


def get_Recursive_hl_index(data: Any, k: Optional[int] = None) -> int:
    """
    便捷接口：仅返回最终的递归 Hl-index 标量值。

    参数:
        data: 嵌套列表数据。
        k: 最大递归深度 (整数或 None)。
    """
    _, result = Calculate_Recursive_Hl_Index(data, k=k)
    return result

def Calculate_C_Index(
        matrix: Union[List[List[float]], np.ndarray, pd.DataFrame],
        strong: Optional[Union[List[float], np.ndarray]] = None
) -> List[float]:
    '''
    计算加权无向图中节点的 C-index。

    定义:
        C-index 衡量节点在网络中的加权连接强度，并结合 H-index 评价其邻居的质量。

    逻辑：
        1. 若未提供初始强度 strong，则 strong[i] = sum(matrix[i]) (即节点的加权度)。
        2. 对每个节点 i，收集其所有邻居 j 的加权贡献值：w_ij * strong[j]。
        3. 对这些贡献值列表计算 H-index，作为节点 i 的 C-index。

        通过传入自定义的 strong 向量（例如节点的 H-degree），本函数也可用于计算交流中心性

    输入:
        matrix: 邻接矩阵，支持 List[List], np.ndarray, pd.DataFrame。
        strong: 每个节点的初始强度向量 (可选)。若为 None，则自动计算加权度。

    输出:
        List[int]: 包含每个节点的 C-index 值的列表。

    异常:
        ValueError: 如果矩阵为空、非方阵、包含负数或维度不匹配。
        TypeError: 如果输入数据类型不支持。
    '''

    # 输入标准化
    if matrix is None:
        raise ValueError("输入矩阵不能为 None。")

    if isinstance(matrix, pd.DataFrame):
        np_mat = matrix.values
    elif isinstance(matrix, list):
        if not matrix: return []
        np_mat = np.array(matrix, dtype=float)
    elif isinstance(matrix, np.ndarray):
        np_mat = matrix.astype(float)
    else:
        raise TypeError(f"不支持的矩阵类型: {type(matrix)}")

    if np_mat.ndim != 2:
        raise ValueError(f"输入必须是二维矩阵，当前维度: {np_mat.ndim}")

    rows, cols = np_mat.shape
    if rows != cols:
        raise ValueError("邻接矩阵必须是对称方阵 (N x N)。")

    if rows == 0:
        return []

    if np.any(np_mat < 0):
        raise ValueError("邻接矩阵中包含负数权重。")

    # 初始强度计算
    if strong is None:
        current_strong = np_mat.sum(axis=1)
    else:
        if isinstance(strong, list):
            current_strong = np.array(strong, dtype=float)
        elif isinstance(strong, np.ndarray):
            current_strong = strong.flatten().astype(float)
        else:
            raise TypeError(f"strong 参数类型错误: {type(strong)}")

        if len(current_strong) != rows:
            raise ValueError(f"strong 向量的长度 ({len(current_strong)}) 必须与矩阵维度 ({rows}) 一致。")

    # 核心计算逻辑
    c_indices = []

    for i in range(rows):
        weights = np_mat[i]
        # 提取非零邻居索引
        non_zero_indices = np.where(weights > 0)[0]

        if len(non_zero_indices) == 0:
            c_indices.append(0)
            continue

        # 向量化计算贡献值
        neighbor_contributions = weights[non_zero_indices] * current_strong[non_zero_indices]

        # 计算 H-index
        h_val = Calculate_H_Index(neighbor_contributions.tolist())
        c_indices.append(h_val)

    return c_indices

def Calculate_Iterative_C_Index(
        matrix: Union[List[List[Number]], np.ndarray],
        strong: Optional[Union[List[Number], np.ndarray]] = None,
        nums: int = 0,
        tolerance: float = 1e-6,
        max_iter: int = 1000
) -> List[float]:
    '''
    迭代计算 C-index (Iterative C-index)。通过迭代法更新节点强度，直到收敛或达到指定迭代次数。


    算法逻辑：
        Strong(t+1) = Calculate_C_Index(matrix, Strong(t))

        1. 初始化: 基于输入的 strong (或默认节点强度) 计算初始 C-index。
           此步骤不计入 nums 迭代次数。

        2. 迭代阶段: 基于上一轮的 C-index 结果作为权重，继续计算。

    参数:
        matrix: 邻接矩阵 (支持 List, np.ndarray, pd.DataFrame)。
        strong: 初始强度向量 (支持 List, np.ndarray)。
        nums: 迭代次数控制。
              - 若 > 0: 执行固定次数迭代。
              - 若 = 0: 执行自适应迭代，直到收敛 (前后两次差值 < tolerance)。
        tolerance: 收敛阈值 (仅当 nums=0 时生效)，默认为 1e-6。
        max_iter: 最大迭代次数保护 (仅当 nums=0 时生效)，防止死循环。

    返回:
        List[int]: 收敛后的节点强度列表。

    异常:
        ValueError: 如果输入维度不匹配或矩阵无效。
    '''

    # 输入标准化与校验 ---
    # 转换为 numpy 数组以便高效计算差值
    if isinstance(matrix, pd.DataFrame):
        np_mat = matrix.values
    elif isinstance(matrix, list):
        if not matrix: return []
        np_mat = np.array(matrix, dtype=float)
    elif isinstance(matrix, np.ndarray):
        np_mat = matrix.astype(float)
    else:
        raise TypeError(f"不支持的矩阵类型: {type(matrix)}")

    if np_mat.ndim != 2:
        raise ValueError("输入必须是二维矩阵。")
    if np_mat.shape[0] != np_mat.shape[1]:
        raise ValueError("邻接矩阵必须是方阵。")

    N = np_mat.shape[0]
    if N == 0: return []

    # 初始 Strong 向量处理
    if strong is not None:
        if isinstance(strong, list):
            strong = np.array(strong, dtype=float)
        elif isinstance(strong, np.ndarray):
            strong = strong.flatten().astype(float)
        else:
            raise TypeError(f"strong 参数类型错误: {type(strong)}")

        if len(strong) != N:
            raise ValueError(f"strong 向量长度 ({len(strong)}) 与矩阵维度 ({N}) 不匹配。")

    # ---初始化 ---
    # 计算初始 C-index
    # 这一步不算作迭代次数
    try:
        c_current = Calculate_C_Index(np_mat, strong=strong)
    except Exception as e:
        raise RuntimeError(f"初始化 C-index 计算失败: {e}")

    # --- 迭代阶段 ---

    # 情况 A: 固定次数迭代 (nums > 0)
    if nums > 0:
        for step in range(nums):
            # 将上一轮的 C-index 作为下一轮的 strong 输入
            c_current = Calculate_C_Index(np_mat, strong=c_current)
        return c_current

    # 情况 B: 收敛迭代 (nums == 0)
    else:
        iter_count = 0
        while iter_count < max_iter:
            iter_count += 1
            c_next = Calculate_C_Index(np_mat, strong=c_current)

            # 计算损失 (L1 范数)
            loss = np.sum(np.abs(np.array(c_next) - np.array(c_current)))
            c_current = c_next

            if loss < tolerance:
                break

        return c_current

def Calculate_CG_Index(
        matrix: Union[List[List[Number]], np.ndarray],
        strong: Union[List[Number], np.ndarray]= None
) -> List[int]:
    '''
    计算节点的 CG-index (C-g Index)。基于 G-index 逻辑，关注累积贡献值。

    算法逻辑：
        1. 计算序列：对于节点 i，计算其所有邻居的贡献值 P_j = w_ij * strong_j。
        2. 排序：将贡献值 P_j 按降序排列。
        3. G-index 判定：寻找最大整数 g，使得前 g 个贡献值的累加和 >= g²。

    参数:
        matrix: 邻接矩阵 (支持 List, np.ndarray, pd.DataFrame)。
        strong: 节点强度向量 (List 或 np.ndarray)。

    返回:
        List[int]: 每个节点的 CG-index 值列表。

    异常:
        ValueError: 如果输入维度不匹配或矩阵无效。
    '''

    # --- 输入标准化与校验 ---
    if isinstance(matrix, pd.DataFrame):
        np_mat = matrix.values
    elif isinstance(matrix, list):
        if not matrix: return []
        np_mat = np.array(matrix, dtype=float)
    elif isinstance(matrix, np.ndarray):
        np_mat = matrix.astype(float)
    else:
        raise TypeError(f"不支持的矩阵类型: {type(matrix)}")

    if np_mat.ndim != 2:
        raise ValueError("输入必须是二维矩阵。")
    if np_mat.shape[0] != np_mat.shape[1]:
        raise ValueError("邻接矩阵必须是方阵。")

    N = np_mat.shape[0]
    if N == 0: return []

    # 处理 Strong 向量
    if strong is not None:
        if isinstance(strong, list):
            strong = np.array(strong, dtype=float)
        elif isinstance(strong, np.ndarray):
            strong = strong.flatten().astype(float)
        else:
            raise TypeError(f"strong 参数类型错误: {type(strong)}")

        if len(strong) != N:
            raise ValueError(f"strong 长度 ({len(strong)}) 与矩阵维度 ({N}) 不匹配。")

    # 若未提供 strong，计算节点强度
    if strong is None:
        current_strong = np_mat.sum(axis=1)
    else:
        current_strong = strong

    # --- 计算G-index---
    cg_indices = []

    for i in range(N):
        weights = np_mat[i]
        neighbor_idx = np.where(weights > 0)[0]

        if len(neighbor_idx) == 0:
            cg_indices.append(0)
            continue

        # 计算贡献值并降序排列
        contributions = weights[neighbor_idx] * current_strong[neighbor_idx]
        sorted_contributions = np.sort(contributions)[::-1]

        # G-index 判定: 累加和 >= g^2
        cum_sum = 0.0
        g_val = 0

        for rank, val in enumerate(sorted_contributions):
            cum_sum += val
            current_g = rank + 1

            if cum_sum >= (current_g ** 2):
                g_val = current_g
            else:
                break

        cg_indices.append(int(g_val))

    return cg_indices


def Calculate_Communication_Centrality(
        matrix: Union[List[List[Number]], np.ndarray, pd.DataFrame]
) -> List[int]:
    '''
    计算加权无向图中节点的交流中心性 (Communication Centrality)。

    定义:
        衡量节点在网络中的沟通与传播能力。
        引入 H-degree 评价邻居节点的“结构稳固性”，而非单纯的资源总量。

    算法逻辑：
        1. 预计算 H-degree: 计算每个节点 i 的边权重 H-index，记为 h_i。
        2. 渠道能力: 对每条边 (i, j)，计算传播势能 P_ij = w_ij * h_j。
        3. 聚合评价: 对节点 i 的所有 P_ij 列表计算 H-index。

    参数:
        matrix: 邻接矩阵 (支持 List, np.ndarray, pd.DataFrame)。

    返回:
        List[int]: 每个节点的交流中心性值列表。

    异常:
        ValueError: 如果输入维度不匹配或矩阵无效。
        TypeError: 如果输入数据类型不支持。
    '''

    # --- 输入标准化与校验 ---
    if isinstance(matrix, pd.DataFrame):
        np_mat = matrix.values
    elif isinstance(matrix, list):
        if not matrix: return []
        np_mat = np.array(matrix, dtype=float)
    elif isinstance(matrix, np.ndarray):
        np_mat = matrix.astype(float)
    else:
        raise TypeError(f"不支持的矩阵类型: {type(matrix)}")

    if np_mat.ndim != 2:
        raise ValueError("输入必须是二维矩阵。")
    if np_mat.shape[0] != np_mat.shape[1]:
        raise ValueError("邻接矩阵必须是方阵。")

    N = np_mat.shape[0]
    if N == 0: return []

    if np.any(np_mat < 0):
        raise ValueError("邻接矩阵中包含负数权重。")

    # --- 核心计算逻辑 ---

    # 预计算所有节点的 H-degree
    h_degrees = np.zeros(N)
    for i in range(N):
        weights = np_mat[i]
        non_zero_weights = weights[weights > 0]

        if len(non_zero_weights) > 0:
            h_degrees[i] = Calculate_H_Index(non_zero_weights.tolist())

    # 计算交流中心性
    centrality_values = []

    for i in range(N):
        weights = np_mat[i]
        neighbor_idx = np.where(weights > 0)[0]

        if len(neighbor_idx) == 0:
            centrality_values.append(0)
            continue

        # 计算渠道能力: 边权重 * 邻居的 H-degree
        channel_abilities = weights[neighbor_idx] * h_degrees[neighbor_idx]

        # 聚合计算
        c_val = Calculate_H_Index(channel_abilities.tolist())
        centrality_values.append(int(c_val))

    return centrality_values


def Calculate_BI_directional_h_index(
        adjacency_matrix: np.ndarray,
        max_iter: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算加权有向网络中节点的双向H指数中心性。

    输入：
        adjacency_matrix: numpy 二维数组。表示加权有向网络的邻接矩阵。
                          A[i][j] 代表从节点 i 指向节点 j 的边的权重。
        max_iter: 整数，最大迭代次数，防止在非收敛图中死循环。

    输出：
        Tuple[np.ndarray, np.ndarray]: 包含两个一维数组的元组。
            - h_in_final: 各节点的入向 H 指数（权威值）。
            - h_out_final: 各节点的出向 H 指数（枢纽值）。

    计算逻辑：
        1. 初始化 (n=0)：节点的入向/出向指数直接设为其入度/出度。
        2. 迭代更新 (n>0)：采用同步更新机制。
           - 入向更新：收集所有指向该节点的邻居的 [边权重 × 邻居上一轮出向指数] 构成列表，计算其 H-index。
           - 出向更新：收集该节点指向的所有邻居的 [边权重 × 邻居上一轮入向指数] 构成列表，计算其 H-index。
        3. 终止条件：当连续两次迭代结果完全一致或达到最大迭代次数时停止。

    异常:
        ValueError: 如果输入矩阵不是二维方阵。
        TypeError: 如果输入不是 numpy 数组。
    """

    # --- 严格输入校验 ---

    # 检查类型：必须是 numpy 数组
    if not isinstance(adjacency_matrix, np.ndarray):
        raise TypeError(f"输入必须是 numpy 数组，得到: {type(adjacency_matrix)}")

    # 检查维度：必须是 2 维
    if adjacency_matrix.ndim != 2:
        raise ValueError(f"输入必须是二维邻接矩阵，检测到维度: {adjacency_matrix.ndim}")

    N = adjacency_matrix.shape[0]

    # 检查形状：必须是方阵 (N x N)
    if N != adjacency_matrix.shape[1]:
        raise ValueError(f"邻接矩阵必须是方阵 (N x N)，得到形状: {adjacency_matrix.shape}")

    # 1. 初始化
    N = adjacency_matrix.shape[0]
    # 0阶值为节点的入度和出度（计算非零连接数）
    h_in_prev = np.array([np.sum(adjacency_matrix[:, i] > 0) for i in range(N)])
    h_out_prev = np.array([np.sum(adjacency_matrix[i, :] > 0) for i in range(N)])

    # 2. 迭代更新
    for n in range(1, max_iter + 1):
        h_in_new = np.zeros(N)
        h_out_new = np.zeros(N)

        for i in range(N):
            # 2.1 计算入向中心性 h_in[i]
            # 收集指向 i 的节点 j 的加权出向指数
            in_products = [adjacency_matrix[j][i] * h_out_prev[j]
                           for j in range(N) if adjacency_matrix[j][i] > 0]
            h_in_new[i] = Calculate_H_Index(in_products)

            # 2.2 计算出向中心性 h_out[i]
            # 收集 i 指向的节点 k 的加权入向指数
            out_products = [adjacency_matrix[i][k] * h_in_prev[k]
                            for k in range(N) if adjacency_matrix[i][k] > 0]
            h_out_new[i] = Calculate_H_Index(out_products)

        # 3. 收敛性检查
        if np.array_equal(h_in_new, h_in_prev) and np.array_equal(h_out_new, h_out_prev):
            return h_in_new, h_out_new

        h_in_prev = h_in_new.copy()
        h_out_prev = h_out_new.copy()

    return h_in_prev, h_out_prev
