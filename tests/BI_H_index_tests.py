import unittest
import numpy as np
from src.snat.Node_Evaluation import Calculate_BI_directional_h_index


class TestBiDirectionalHIndex(unittest.TestCase):

    # ==============================
    # 1. 核心功能测试
    # ==============================

    def test_basic_convergence(self):
        """测试基本加权有向图的收敛与计算逻辑"""
        # 构建一个简单的 4 节点图
        # 0 -> 1 (w=2), 0 -> 2 (w=1)
        # 1 -> 0 (w=1), 1 -> 3 (w=3)
        # 2 -> 1 (w=1), 2 -> 3 (w=1)
        # 3 -> (无出链，悬挂节点)
        A = np.array([
            [0, 2, 1, 0],
            [1, 0, 0, 3],
            [0, 1, 0, 1],
            [0, 0, 0, 0]
        ])

        h_in, h_out = Calculate_BI_directional_h_index(A)

        # 验证返回类型
        self.assertIsInstance(h_in, np.ndarray)
        self.assertIsInstance(h_out, np.ndarray)
        self.assertEqual(len(h_in), 4)

        # 验证节点 3 (悬挂节点)
        # 入度为 2 (来自1和2)，出度为 0
        # h_out[3] 初始为 0，后续迭代中因为没有出链，乘积列表为空，H-index 保持 0
        self.assertEqual(h_out[3], 0)
        # h_in[3] 应该大于 0，因为它接收来自 1 和 2 的连接

    def test_isolated_node(self):
        """测试包含孤立节点（完全无连接）的图"""
        # 节点 0 和 1 互连，节点 2 完全孤立
        A = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 0]
        ])

        h_in, h_out = Calculate_BI_directional_h_index(A)

        # 孤立节点 2 的入向和出向中心性都应为 0
        self.assertEqual(h_in[2], 0)
        self.assertEqual(h_out[2], 0)

        # 节点 0 和 1 应该有非零值
        self.assertGreater(h_in[0], 0)
        self.assertGreater(h_out[0], 0)

    def test_weight_impact(self):
        """测试边权重对结果的影响"""
        # 场景：节点 0 指向 节点 1
        # 情况 A: 权重为 1
        A_low = np.array([[0, 1], [0, 0]])
        # 情况 B: 权重为 10
        A_high = np.array([[0, 10], [0, 0]])

        _, h_out_low = Calculate_BI_directional_h_index(A_low)
        _, h_out_high = Calculate_BI_directional_h_index(A_high)

        # 节点 0 的出向中心性计算：
        # 列表为 [权重 * h_in(节点1)]。
        # 节点 1 是悬挂节点，h_in(1) = 入度 = 1。
        # 列表 A: [1 * 1] = [1] -> H-index = 1
        # 列表 B: [10 * 1] = [10] -> H-index = 1 (因为只有一个元素)

        # 注意：在这个简单例子中 H-index 可能相同（受限于列表长度）。
        # 我们主要测试代码不报错，且能处理不同权重。
        self.assertIsNotNone(h_out_low)
        self.assertIsNotNone(h_out_high)

    # ==============================
    # 2. 边界与压力测试
    # ==============================

    def test_empty_graph(self):
        """测试空矩阵 (0x0)"""
        A = np.array([]).reshape(0, 0)
        h_in, h_out = Calculate_BI_directional_h_index(A)
        self.assertEqual(len(h_in), 0)
        self.assertEqual(len(h_out), 0)

    def test_max_iter_limit(self):
        """测试最大迭代次数限制 (虽然 H-index 迭代通常收敛很快)"""
        # 构造一个稍微大一点的随机图
        np.random.seed(42)
        A = np.random.randint(0, 5, (10, 10))

        # 设置很小的 max_iter 看是否能正常返回而不报错
        h_in, h_out = Calculate_BI_directional_h_index(A, max_iter=2)

        self.assertEqual(len(h_in), 10)
        # 如果迭代次数太少未收敛，函数应返回当前值而不是报错

    # ==============================
    # 3. 异常处理测试
    # ==============================

    def test_invalid_input_type(self):
        """测试非 numpy 数组输入"""
        A_list = [[0, 1], [1, 0]]
        with self.assertRaises(TypeError):
            Calculate_BI_directional_h_index(A_list)

    def test_non_square_matrix(self):
        """测试非方阵输入"""
        A_rect = np.array([[0, 1, 0], [1, 0, 1]])  # 2x3 矩阵
        with self.assertRaises(ValueError):
            Calculate_BI_directional_h_index(A_rect)

    def test_1d_input(self):
        """测试一维数组输入"""
        A_1d = np.array([1, 2, 3])
        with self.assertRaises(ValueError):
            Calculate_BI_directional_h_index(A_1d)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)