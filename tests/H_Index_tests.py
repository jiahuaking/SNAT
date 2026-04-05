import unittest
import numpy as np
from src.snat.Node_Evaluation import Calculate_H_Index

class TestCalculateHIndex(unittest.TestCase):

    # ==============================
    # 1. 核心逻辑测试 (标准案例)
    # ==============================

    def test_basic_h_index(self):
        """测试标准的 H-index 计算逻辑"""
        # 数据: [43, 12, 5, 1, 7, 9, 4, 46, 23, 7]
        # 排序后: [46, 43, 23, 12, 9, 7, 7, 5, 4, 1]
        # 排名:   1   2   3   4   5  6  7  8  9  10
        # 判定: 前7个数都 >= 7 (46...5)，第8个数4 < 8。
        # 预期结果: 7
        data = [43, 12, 5, 1, 7, 9, 4, 46, 23, 7]
        result = Calculate_H_Index(data)
        self.assertEqual(result, 7)

    def test_single_element(self):
        """测试单元素列表"""
        # [24] -> 24 >= 1
        self.assertEqual(Calculate_H_Index([24]), 1)

    def test_all_ones(self):
        """测试全为1的列表"""
        # [1, 1, 1] -> 3个元素 >= 1
        self.assertEqual(Calculate_H_Index([1, 1, 1]), 1)

    def test_exact_cutoff(self):
        """测试精确截断点 (经典案例)"""
        # [10, 8, 5, 4, 3] -> 排序后不变
        # 1: 10>=1, 2: 8>=2, 3: 5>=3, 4: 4>=4, 5: 3<5
        # 预期结果: 4
        self.assertEqual(Calculate_H_Index([10, 8, 5, 4, 3]), 4)

    def test_large_numbers(self):
        """测试数值远大于长度的情况"""
        # [100, 100] -> 2个元素，值都很大
        # 预期结果: 2
        self.assertEqual(Calculate_H_Index([100, 100]), 2)

    # ==============================
    # 2. 边界与特殊输入测试
    # ==============================

    def test_with_zeros(self):
        """测试包含零值的情况"""
        # [5, 0, 0] -> 只有1个非零元素
        # 预期结果: 1 (因为有1个元素 >= 1)
        self.assertEqual(Calculate_H_Index([5, 0, 0]), 1)

    def test_empty_list(self):
        """测试空列表"""
        self.assertEqual(Calculate_H_Index([]), 0)

    def test_numpy_input(self):
        """测试 NumPy 数组输入"""
        # [3, 3, 3] -> 3个元素 >= 3
        result = Calculate_H_Index(np.array([3, 3, 3]))
        self.assertEqual(result, 3)

    def test_unsorted_input(self):
        """测试未排序输入 (验证函数内部排序逻辑)"""
        # 乱序输入，结果应与排序后一致
        # [1, 5, 3, 4, 2] -> 排序后 [5, 4, 3, 2, 1] -> h=3
        self.assertEqual(Calculate_H_Index([1, 5, 3, 4, 2]), 3)

    # ==============================
    # 3. 异常与容灾测试
    # ==============================

    def test_non_numeric_elements(self):
        """测试包含非数字元素"""
        with self.assertRaises(Exception):
            Calculate_H_Index([10, "20", 5])

    def test_negative_weights(self):
        """测试包含负数"""
        with self.assertRaises(Exception):
            Calculate_H_Index([10, -5, 8])

    def test_none_input(self):
        """测试 None 输入"""
        with self.assertRaises(Exception):
            Calculate_H_Index(None)

    def test_invalid_type(self):
        """测试错误的输入类型 (字符串)"""
        with self.assertRaises(Exception):
            Calculate_H_Index("not a list")

    def test_dict_input(self):
        """测试字典输入 (不支持的类型)"""
        with self.assertRaises(Exception):
            Calculate_H_Index({"a": 1, "b": 2})

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)