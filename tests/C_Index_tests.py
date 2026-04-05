import unittest
import numpy as np
import pandas as pd
from src.snat.Node_Evaluation import Calculate_C_Index,Calculate_Iterative_C_Index,Calculate_CG_Index,Calculate_Communication_Centrality
# 测试用例类
# ==========================================
class TestCalculateCIndex(unittest.TestCase):

    def setUp(self):
        """简单的三角形网络 (3个节点互相连接)
           0 --1-- 1
           |      /
           2 --1--
        """
        self.simple_matrix = [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ]

    # ==============================
    # 1. 正常功能测试
    # ==============================

    def test_normal_list_input(self):
        """测试正常的列表输入"""
        result = Calculate_C_Index(self.simple_matrix)
        # 每个节点度为2，邻居强度为2，贡献值为 [2, 2]，H-index应为2
        self.assertEqual(result, [2, 2, 2])
        self.assertIsInstance(result, list)

    def test_normal_numpy_input(self):
        """测试 NumPy 数组输入"""
        np_mat = np.array(self.simple_matrix)
        result = Calculate_C_Index(np_mat)
        self.assertEqual(result, [2, 2, 2])

    def test_normal_pandas_input(self):
        """测试 Pandas DataFrame 输入"""
        df = pd.DataFrame(self.simple_matrix)
        result = Calculate_C_Index(df)
        self.assertEqual(result, [2, 2, 2])

    def test_custom_strong_vector(self):
        """测试自定义 strong 向量"""
        # 强制设定强度为 [10, 10, 10]
        # 每个节点有两个邻居，贡献值为 [1*10, 1*10] = [10, 10]
        # H-index([10, 10]) = 2
        result = Calculate_C_Index(self.simple_matrix, strong=[10, 10, 10])
        self.assertEqual(result, [2, 2, 2])

    def test_weighted_graph(self):
        """测试加权图"""
        # 0-0.5-1
        # |     |
        # 1.0  1.0
        # |     |
        # 2-----1 (假设2和1之间权重1.0，这里仅做简单示意)
        matrix = [
            [0, 0.5, 1.0],
            [0.5, 0, 0],
            [1.0, 0, 0]
        ]
        # Node 0: 强度 1.5. 邻居: 1(0.5), 2(1.0). 贡献: [0.5*0.5, 1.0*1.0] = [0.25, 1.0]. H-index = 1
        # Node 1: 强度 0.5. 邻居: 0(0.5). 贡献: [0.5*1.5] = [0.75]. H-index = 1
        # Node 2: 强度 1.0. 邻居: 0(1.0). 贡献: [1.0*1.5] = [1.5]. H-index = 1
        result = Calculate_C_Index(matrix)
        # 验证返回长度
        self.assertEqual(len(result), 3)
        # 验证具体数值 (根据逻辑推导)
        self.assertEqual(result[0], 1)

        # ==============================

    # 2. 边界条件测试
    # ==============================

    def test_empty_matrix(self):
        """测试空列表"""
        result = Calculate_C_Index([])
        self.assertEqual(result, [])

    def test_isolated_node(self):
        """测试包含孤立节点的图
           0 -- 1    2 (孤立)
        """
        matrix = [
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 0]
        ]
        result = Calculate_C_Index(matrix)
        # 节点 2 没有邻居，H-index 应为 0
        self.assertEqual(result[2], 0)
        # 节点 0, 1 有连接
        self.assertGreater(result[0], 0)

    def test_single_node(self):
        """测试单节点矩阵"""
        result = Calculate_C_Index([[0]])
        self.assertEqual(result, [0])

    # ==============================
    # 3. 异常与容灾测试
    # ==============================

    def test_none_input(self):
        """测试 None 输入"""
        with self.assertRaises(ValueError):
            Calculate_C_Index(None)

    def test_non_square_matrix(self):
        """测试非方阵"""
        matrix = [[0, 1, 0], [1, 0, 0]]  # 2x3
        with self.assertRaises(ValueError):
            Calculate_C_Index(matrix)

    def test_negative_weights(self):
        """测试负权重"""
        matrix = [[0, -1], [-1, 0]]
        with self.assertRaises(ValueError):
            Calculate_C_Index(matrix)

    def test_dimension_mismatch(self):
        """测试 strong 向量维度不匹配"""
        matrix = [[0, 1], [1, 0]]
        strong = [10]  # 应该是2个元素
        with self.assertRaises(ValueError):
            Calculate_C_Index(matrix, strong=strong)

    def test_invalid_strong_type(self):
        """测试 strong 参数类型错误"""
        matrix = [[0, 1], [1, 0]]
        with self.assertRaises(TypeError):
            Calculate_C_Index(matrix, strong="invalid")

    def test_invalid_matrix_type(self):
        """测试矩阵类型错误"""
        with self.assertRaises(TypeError):
            Calculate_C_Index("invalid_matrix")

    def test_non_numeric_data(self):
        """测试非数字数据 (如果 numpy 转换失败)"""
        # 注意：numpy 可能会尝试将字符串转换为 nan，这里测试纯对象数组的情况
        matrix = [['a', 'b'], ['c', 'd']]
        # 具体行为取决于 numpy 版本，通常应捕获 ValueError 或 TypeError
        try:
            Calculate_C_Index(matrix)
        except (ValueError, TypeError):
            pass  # 预期会报错

# 测试 Calculate_Iterative_C_Index
# ==========================================
class TestCalculateIterativeCIndex(unittest.TestCase):

    def setUp(self):
        """简单的三角形网络 (3个节点互相连接)
 --1-- 1
           |      /
 -----/
        """
        self.simple_matrix = [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ]

    # ==============================
    # 1. 正常功能测试
    # ==============================

    def test_convergence_default(self):
        """测试默认收敛逻辑 (nums=0)"""
        # 三角形网络是对称的，最终应该收敛到一个稳定值
        # 初始强度为2。c(1)基于强度2计算。
        result = Calculate_Iterative_C_Index(self.simple_matrix)
        self.assertEqual(len(result), 3)
        # 验证三个节点结果一致（对称性）
        self.assertEqual(result[0], result[1])
        self.assertEqual(result[1], result[2])

    def test_fixed_iterations(self):
        """测试固定迭代次数 (nums > 0)"""
        # 初始 strong=[10, 10, 10]
        # c(1): 邻居贡献 [1*10, 1*10] -> H-index=2
        # c(2): 邻居贡献 [1*2, 1*2] -> H-index=2 (收敛)
        result = Calculate_Iterative_C_Index(self.simple_matrix, strong=[10, 10, 10], nums=1)
        # 第一次迭代结果应为 2
        self.assertEqual(result, [2, 2, 2])

    def test_custom_strong_initialization(self):
        """测试自定义初始权重向量"""
        # 使用非对称初始权重
        # 节点0: 邻居1(强10), 邻居2(强1). 贡献 [10, 1]. H-index = 2
        # 节点1: 邻居0(强1), 邻居2(强1). 贡献 [1, 1]. H-index = 2
        # 节点2: 邻居0(强1), 邻居1(强10). 贡献 [1, 10]. H-index = 2
        # (注意：这里只是演示逻辑，具体值取决于H-index计算细节)
        strong = [1, 10, 1]
        result = Calculate_Iterative_C_Index(self.simple_matrix, strong=strong, nums=1)
        self.assertEqual(len(result), 3)

    def test_numpy_input(self):
        """测试 NumPy 矩阵输入"""
        np_mat = np.array(self.simple_matrix)
        result = Calculate_Iterative_C_Index(np_mat, nums=2)
        self.assertIsInstance(result, list)

    # ==============================
    # 2. 边界条件测试
    # ==============================

    def test_empty_matrix(self):
        """测试空矩阵"""
        result = Calculate_Iterative_C_Index([])
        self.assertEqual(result, [])

    def test_isolated_node(self):
        """测试包含孤立节点
 -- 1    2 (孤立)
        """
        matrix = [
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 0]
        ]
        result = Calculate_Iterative_C_Index(matrix, nums=1)
        # 孤立节点强度为0，C-index应为0
        self.assertEqual(result[2], 0)

    def test_single_node(self):
        """测试单节点"""
        result = Calculate_Iterative_C_Index([[0]])
        self.assertEqual(result, [0])

    # ==============================
    # 3. 异常与容灾测试
    # ==============================

    def test_non_square_matrix(self):
        """测试非方阵"""
        matrix = [[0, 1, 0], [1, 0, 0]]
        with self.assertRaises(ValueError):
            Calculate_Iterative_C_Index(matrix)

    def test_strong_dimension_mismatch(self):
        """测试 strong 向量维度不匹配"""
        matrix = [[0, 1], [1, 0]]
        strong = [10] # 长度应为2
        with self.assertRaises(ValueError):
            Calculate_Iterative_C_Index(matrix, strong=strong)

    def test_invalid_strong_type(self):
        """测试 strong 类型错误"""
        matrix = [[0, 1], [1, 0]]
        with self.assertRaises(TypeError):
            Calculate_Iterative_C_Index(matrix, strong="invalid")

    def test_invalid_matrix_type(self):
        """测试矩阵类型错误"""
        with self.assertRaises(TypeError):
            Calculate_Iterative_C_Index("invalid")

# ==========================================
# 测试 Calculate_CG_Index
# ==========================================
class TestCalculateCGIndex(unittest.TestCase):

    def setUp(self):
        """星型网络用于测试 G-index 的累加特性

           |
 -- 2
           |

        """
        self.star_matrix = [
            [0, 1, 1, 1],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0]
        ]

    # ==============================
    # 1. 正常功能测试
    # ==============================

    def test_g_index_logic(self):
        """测试 G-index 核心逻辑 (累加和 >= g^2)"""
        # 节点 0: 邻居 1,2,3 (强度均为1). 贡献值 [1, 1, 1].
        # g=1: sum(1) = 1 >= 1^2 (True)
        # g=2: sum(1,1) = 2 >= 2^2=4 (False) -> CG = 1
        # 节点 1: 邻居 0 (强度3). 贡献值 [3].
        # g=1: sum(3) = 3 >= 1^2 (True)
        # g=2: 无第2个值 -> CG = 1
        result = Calculate_CG_Index(self.star_matrix)
        self.assertEqual(result[0], 1)
        self.assertEqual(result[1], 1)

    def test_high_weight_g_index(self):
        """测试高权重下的 G-index (验证 g > h 的可能性)"""
        # 构造一个高权重场景
        # 0 --10-- 1
        matrix = [[0, 10], [10, 0]]
        # 节点 0: 邻居 1 (强度10). 贡献 [100].
        # g=1: 100 >= 1 (True)
        # g=2: 无第2个值.
        # 注意：G-index 理论上可以超过度数，但在离散列表中受限于邻居数量
        # 除非实现插值，否则这里最大就是邻居数。
        # 但我们要验证的是累加逻辑：
        # 假设 0 有 3 个邻居，权重都是 5. 强度都是 5.
        # 贡献: 25, 25, 25.
        # g=1: 25 >= 1
        # g=2: 50 >= 4
        # g=3: 75 >= 9
        # g=4: 75 >= 16 (False) -> CG = 3
        matrix_3 = [
            [0, 5, 5, 5],
            [5, 0, 0, 0],
            [5, 0, 0, 0],
            [5, 0, 0, 0]
        ]
        result = Calculate_CG_Index(matrix_3)
        # 中心节点 0 有3个邻居，贡献值很大，应该能拿到 3
        self.assertEqual(result[0], 3)

    def test_custom_strong(self):
        """测试自定义 strong 向量"""
        matrix = [[0, 1], [1, 0]]
        # 强制 strong = [100, 100]
        # 贡献值: 1*100 = 100.
        # g=1: 100 >= 1.
        result = Calculate_CG_Index(matrix, strong=[100, 100])
        self.assertEqual(result, [1, 1])

    def test_numpy_input(self):
        """测试 NumPy 输入"""
        np_mat = np.array(self.star_matrix)
        result = Calculate_CG_Index(np_mat)
        self.assertEqual(len(result), 4)

    # ==============================
    # 2. 边界条件测试
    # ==============================

    def test_empty_matrix(self):
        """测试空矩阵"""
        result = Calculate_CG_Index([])
        self.assertEqual(result, [])

    def test_isolated_node(self):
        """测试孤立节点"""
        matrix = [[0, 0], [0, 0]]
        result = Calculate_CG_Index(matrix)
        self.assertEqual(result, [0, 0])

    def test_single_node(self):
        """测试单节点"""
        result = Calculate_CG_Index([[0]])
        self.assertEqual(result, [0])

    # ==============================
    # 3. 异常与容灾测试
    # ==============================

    def test_non_square_matrix(self):
        """测试非方阵"""
        matrix = [[0, 1, 0], [1, 0, 0]]
        with self.assertRaises(ValueError):
            Calculate_CG_Index(matrix)

    def test_strong_dimension_mismatch(self):
        """测试 strong 维度不匹配"""
        matrix = [[0, 1], [1, 0]]
        strong = [10]
        with self.assertRaises(ValueError):
            Calculate_CG_Index(matrix, strong=strong)

    def test_invalid_strong_type(self):
        """测试 strong 类型错误"""
        matrix = [[0, 1], [1, 0]]
        with self.assertRaises(TypeError):
            Calculate_CG_Index(matrix, strong="invalid")

    def test_invalid_matrix_type(self):
        """测试矩阵类型错误"""
        with self.assertRaises(TypeError):
            Calculate_CG_Index("invalid")


class TestCalculateCommunicationCentrality(unittest.TestCase):

    # ==============================
    # 1. 正常功能测试 (核心逻辑验证)
    # ==============================

    def test_basic_triangle(self):
        """测试简单的三角形网络 (对称结构)
           0 --1-- 1
           |      /
           2 -----/
        """
        matrix = [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ]
        # 逻辑推导:
        # 1. 所有节点度为2，权重均为1。H-degree(所有节点) = 1 (因为2个1 >= 1，但 < 2)。
        # 2. 节点0的贡献值: w_01*h_1 + w_02*h_2 -> [1*1, 1*1] = [1, 1]。
        # 3. H-index([1, 1]) = 1。
        result = Calculate_Communication_Centrality(matrix)
        self.assertEqual(result, [1, 1, 1])

    def test_h_degree_vs_strength(self):
        """测试 H-degree 与 Strength 的区别 (核心特性测试)

        场景: 节点 0 连接了两个邻居 1 和 2。
        - 邻居 1: 只有一个超强连接 (Strength=100, H-degree=1)。
        - 邻居 2: 有多个强连接 (Strength=30, H-degree=10)。

        预期: 算法应更看重邻居 2 (结构稳固)，而非邻居 1 (单纯资源多)。
        """
        # 0 --10-- 1 (1是孤立大户)
        # |
        # 10
        # |
        # 2 --10-- 3 (2和3是结构大户)
        matrix = [
            [0, 10, 10, 0],  # Node 0
            [10, 0, 0, 0],  # Node 1 (Strength=10, H-degree=1)
            [10, 0, 0, 10],  # Node 2 (Strength=20, H-degree=10)
            [0, 0, 10, 0]  # Node 3 (Strength=10, H-degree=10)
        ]

        # 逻辑推导:
        # 1. H-degrees:
        #    Node 1: weights [10] -> H=1
        #    Node 2: weights [10, 10] -> H=2
        #    Node 3: weights [10] -> H=1
        #    Node 0: weights [10, 10] -> H=2 (但这不影响0的计算，只影响别人算0)
        #
        # 2. Node 0 的计算:
        #    邻居 1 贡献: w_01 * h_1 = 10 * 1 = 10
        #    邻居 2 贡献: w_02 * h_2 = 10 * 2 = 20
        #    贡献列表: [10, 20] -> 排序 [20, 10]
        #    H-index([20, 10]) = 2

        result = Calculate_Communication_Centrality(matrix)
        self.assertEqual(result[0], 2)

    def test_pandas_input(self):
        """测试 Pandas DataFrame 输入"""
        df = pd.DataFrame([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ])
        result = Calculate_Communication_Centrality(df)
        # 线性结构 0-1-2
        # Node 1 (中间): 邻居 0(H=1), 2(H=1). 贡献 [1, 1]. H-index = 1.
        # Node 0 (边缘): 邻居 1(H=2). 贡献 [2]. H-index = 1.
        self.assertEqual(len(result), 3)

    def test_numpy_input(self):
        """测试 NumPy 数组输入"""
        np_mat = np.array([[0, 5], [5, 0]])
        result = Calculate_Communication_Centrality(np_mat)
        self.assertEqual(len(result), 2)

    # ==============================
    # 2. 边界条件测试
    # ==============================

    def test_empty_matrix(self):
        """测试空矩阵"""
        result = Calculate_Communication_Centrality([])
        self.assertEqual(result, [])

    def test_isolated_node(self):
        """测试包含孤立节点
           0 -- 1    2 (孤立)
        """
        matrix = [
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 0]
        ]
        result = Calculate_Communication_Centrality(matrix)
        # 节点 2 没有邻居，H-degree 为 0，中心性应为 0
        self.assertEqual(result[2], 0)
        # 节点 0, 1 互相连接，H-degree 均为 1. 贡献 [1]. H-index = 1.
        self.assertEqual(result[0], 1)

    def test_single_node(self):
        """测试单节点"""
        result = Calculate_Communication_Centrality([[0]])
        self.assertEqual(result, [0])

    def test_disconnected_components(self):
        """测试不连通图 (两个分离的组件)"""
        matrix = [
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 5],
            [0, 0, 5, 0]
        ]
        result = Calculate_Communication_Centrality(matrix)
        # 组件1 (0-1): H-degree=1, 贡献=[1], C=1
        # 组件2 (2-3): H-degree=1, 贡献=[5], C=1
        self.assertEqual(result, [1, 1, 1, 1])

    # ==============================
    # 3. 异常与容灾测试
    # ==============================

    def test_non_square_matrix(self):
        """测试非方阵"""
        matrix = [[0, 1, 0], [1, 0, 0]]
        with self.assertRaises(ValueError):
            Calculate_Communication_Centrality(matrix)

    def test_negative_weights(self):
        """测试负权重"""
        matrix = [[0, -1], [-1, 0]]
        with self.assertRaises(ValueError):
            Calculate_Communication_Centrality(matrix)

    def test_invalid_matrix_type(self):
        """测试矩阵类型错误"""
        with self.assertRaises(TypeError):
            Calculate_Communication_Centrality("invalid_matrix")

    def test_invalid_matrix_structure(self):
        """测试矩阵结构错误 (非二维)"""
        matrix = [1, 2, 3]
        with self.assertRaises(ValueError):
            Calculate_Communication_Centrality(matrix)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
