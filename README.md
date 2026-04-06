# 社交网络节点重要性评价库

一套基于 H-index 及其衍生算法的 Python 工具库，旨在量化社交网络中节点的多维重要性。本库填补了现有工具在多层级递归评价指标上的空白，支持从基础的 H-index 到复杂的双向 H-index 等多种算法。

**作者：** Xiangbin Yan, Zhai Li, Jiahua Jin, Wei Gu

---

## 📚 目录

- [背景](#-背景)
- [安装](#-安装)
- [快速开始](#-快速开始)
- [API 文档](#-api-文档)
- [算法说明](#-算法说明)
- [测试](#-测试)
- [贡献](#-贡献)
- [许可证](#-许可证)
- [引用](#-引用)
- [版本历史](#-版本历史)

---

## 📝 背景

在社交网络分析及复杂网络研究中，准确评估单一节点的重要性至关重要。传统的中心性指标如度中心性、接近中心性、介数中心性等各有侧重，但往往难以同时捕捉节点的多维特征和复杂网络中的非线性关系。
本项目基于闫相斌等学者发表的系列重要论文的研究成果，将这些理论成果进行工程化实现，为研究者和开发者提供了一套即插即用的Python工具库。

### 核心理论基础

**1. C-index算法** (Yan et al., 2013)
- **创新点**：首次将H-index机制引入加权网络分析，提出基于邻居"质量"而非单纯数量的中心性度量
- **应用场景**：科研合作网络、社交网络影响力分析
- **核心思想**：节点的中心性由其邻居的"加权贡献"决定，而非简单的连接数

**2. 交流中心性算法** (Zhai et al., 2013)
- **创新点**：提出基于H-degree的网络交流能力度量，区分"资源型"和"结构型"邻居的不同贡献
- **应用场景**：信息传播网络、通信网络优化
- **核心思想**：使用H-degree而非Strength作为邻居评价指标，更准确地反映节点的沟通传播能力

**3. Hl-index算法** (Zhai et al., 2014)
- **创新点**：改进传统H-index，引入"质量"维度，通过递归思想评价多层次绩效
- **应用场景**：学术评价、多层次网络分析
- **核心思想**：不仅考虑绩效数量，还考虑绩效依据的质量，实现多层次递归评价

**4. 双向H-index算法** (Zhai et al., 2018)
- **创新点**：针对有向网络设计，分别计算节点的入向权威值和出向枢纽值
- **应用场景**：有向社交网络、引文网络、网页排名
- **核心思想**：区分节点的"权威性"(被重要节点指向)和"枢纽性"(指向重要节点)

### 工程化价值

本库作为上述理论成果的工程化实现，填补了现有工具在多层级递归评价指标上的空白，提供：
- **标准化接口**：统一的API设计，便于集成和使用
- **性能优化**：利用NumPy等库进行向量化计算，提高处理大规模网络的效率
- **健壮性保证**：完善的异常处理和输入验证，确保在各种边界条件下的稳定运行
- **多格式支持**：兼容Python列表、NumPy数组、Pandas DataFrame等多种数据格式

通过这些工程化改进，研究者和开发者可以更便捷地将先进的网络分析理论应用于实际问题解决。

> **核心参考文献**：
> 本库核心算法基于以下文献实现，具体原理请参考原论文：
> - [1]Zhai, Li & Yan, Xiangbin. (2026). The impact of mentorship on the citation-based performance of graduate dissertation: A social capital perspective. J. Informetrics, 20, 101781.
> - [2]Zhai, Li & Yan, Xiangbin. (2022). A directed collaboration network for exploring the order of scientific collaboration. Journal of Informetrics. 16. 101345. 10.1016/j.joi.2022.101345. 
> - [3]Zhai, Li & Yan, Xiangbin & Zhang, Guojing. (2018). Bi-directional h-index: A new measure of node centrality in weighted and directed networks. Journal of Informetrics. 12. 299-314. 10.1016/j.joi.2018.01.004. 
> - [4]Zhai, Li & Yan, Xiangbin & Zhu, Bin. (2014). The Hl-index: Improvement of H-index based on quality of citing papers. Scientometrics. 98. 10.1007/s11192-013-1039-z. 
> - [5]Zhai, Li & Yan, Xiangbin & Zhang, Guojing. (2013). A centrality measure for communication ability in weighted network. Physica A Statistical Mechanics and its Applications. 392. 6107-6117. 10.1016/j.physa.2013.07.056.
> - [6]Yan, Xiangbin & Zhai, Li & Fan, Weiguo. (2013). C-index: A weighted network node centrality measure for collaboration competence. Journal of Informetrics. 7. 223-239. 10.1016/j.joi.2012.11.004.
---

## 🔧 安装

### 从pypi安装

```bash
pip install snat
```

### 从github克隆

```bash
# 克隆仓库
git clone https://github.com/jiahuaking/SNAT.git

# 安装依赖
pip install -r requirements.txt

# 安装包
pip install -e .
```

### 依赖要求

- Python >= 3.7
- numpy >= 1.19.0
- pandas >= 1.3.0

---

## 🚀 快速开始

```python
#从github使用
#from src.snat import Calculate_H_Index, Calculate_C_Index, Calculate_BI_directional_h_index
#从pypi安装
#!pip install snat
from snat import (
    Calculate_H_Index,
    Calculate_C_Index,
    Calculate_BI_directional_h_index,
    Calculate_H_L_Index,
    Calculate_Iterative_C_Index
)
import numpy as np

# 计算基础 H-index
citations = [10, 8, 5, 4, 3]
h_index = Calculate_H_Index(citations)
print(f"H-index: {h_index}")  # 输出: 4

# 计算加权网络的 C-index
adjacency_matrix = [
  [0, 2, 1, 0],
  [2, 0, 3, 1],
  [1, 3, 0, 2],
  [0, 1, 2, 0]
]
c_indices = Calculate_C_Index(adjacency_matrix)
print(f"C-indices: {c_indices}")

# 计算有向网络的双向 H-index
directed_matrix = np.array([
  [0, 3, 1, 0],
  [0, 0, 2, 1],
  [1, 0, 0, 2],
  [0, 1, 0, 0]
])
h_in, h_out = Calculate_BI_directional_h_index(directed_matrix)
print(f"入向 H-index: {h_in}")
print(f"出向 H-index: {h_out}")
```

---

## 📖 API 文档

#### ✅ `Calculate_H_Index(value)`
- **功能**: 计算基础 H-index。
- **逻辑**: 
  - 接收无序列表或 NumPy 数组，进行降序排序。
  - 寻找最大的h，使得至少有**h**个元素的值 **≥h**。
- **输出**: 返回标量统计值h_index，衡量结点重要性
- **优化**: 
  - 增加了边界条件快速返回剪枝。
  - 时间复杂度控制在**O(NlogN)**，主要消耗在排序。后续可向技术排序方式优化，时间复杂度预计能达到**O(N)**
- **健壮性**: 
  - 完善的类型检查（支持 `list`, `tuple`, `np.ndarray`）。
  - 严格的异常处理：拦截非数字元素、负数输入及多维数组错误。

#### ✅ `Calculate_H_L_Index(data_matrix)`
- **功能**: 计算 Hl-index，不仅考虑绩效数量，还通过递归思想考量绩效的“质量”。
- **逻辑**: 
  - 输入为二维结构（绩效项及其依据）。
  - **第一步**: 对每一行（子绩效列表）计算基础的 H-index，生成中间向量 `h_values`。
  - **第二步**: 对 `h_values` 再次计算 H-index，得到最终的标量评价指标。
- **输出**: 返回元组 `(h_values, final_h_index)`，包含子节点重要性列表及当前节点聚合值，便于调试与分析。
- **优化**: 
  - 采用迭代方式处理第一层数据，避免深层递归带来的栈溢出风险。
  - 异常信息增强：捕获单行计算错误并附带索引信息，快速定位脏数据。
- **健壮性**: 
  - 支持多种二维输入格式（嵌套列表、常规二维数组、不等长对象数组）。
  - 严格的空值与类型校验，确保输入必须是合法的“列表的列表”结构。

#### ✅ `Calculate_Recursive_Hl_Index(data, k=None)`
- **功能**: 通用递归计算 Hl-index，支持任意层级深度的嵌套列表计算。
- **逻辑**: 
  - **自动探测**: 自动判断当前层是数值层还是列表层，动态决定继续递归或终止计算。
  - **深度控制 (`k`)**: 支持设置最大递归深度。若达到深度限制仍未到底层，采用“截断策略”（使用子元素长度作为估算值），防止无限递归或过深计算。
  - **聚合计算**: 收集子节点的 H-index 结果形成新向量，再次调用基础 H-index 算法得出当前层重要性。
- **输出**: 返回元组 `(child_h_values, final_h_index)`，包含子节点重要性列表及当前节点聚合值。
- **优化**: 
  - 引入 `_depth` 内部计数器，配合用户定义的 `k` 参数实现灵活的剪枝策略。
  - 针对纯数字层进行快速路径判断，避免不必要的递归调用开销。
- **健壮性**: 
  - 能够处理不规则的嵌套结构（如 `List[List[int]]` 或 `np.ndarray` 对象数组）。
  - 兼容标量输入（单个数字），提供合理的默认返回值，防止程序崩溃。


#### ✅ `Calculate_C_Index(matrix, strong=None)`
- **功能**: 计算加权无向图中节点的 C-index，结合拓扑连接强度与 H-index 机制评价节点重要性。
- **逻辑**: 
  - **强度定义**: 若未提供外部强度向量 `strong`，默认以节点的加权度（邻接矩阵行和）作为初始强度；支持自定义 `strong` 向量以适配不同评价场景。
  - **加权贡献**: 对每个节点，计算其所有邻居的“加权贡献值”（即：边权重 × 邻居节点强度）。
  - **H-index 聚合**: 将节点的邻居贡献值集合输入 H-index 算法，得出的标量值即为该节点的 C-index。
- **输出**: 返回 `List[int]`，按矩阵行顺序包含图中每个节点的 C-index 值。
- **优化**: 
  - **多格式兼容**: 自动识别并标准化输入数据，原生支持 Python 列表、NumPy 数组及 Pandas DataFrame。
  - **向量化加速**: 利用 NumPy 的布尔索引和向量运算替代传统 Python 循环，快速提取非零邻居并计算贡献值，显著提升稀疏矩阵下的计算效率。
- **健壮性**: 
  - **严格校验**: 内置针对空图、非方阵、负权重、维度不匹配等异常情况的检测与报错机制。
  - **边界处理**: 自动处理孤立节点（无邻居）情况，直接返回 0，确保计算流程不中断。

#### ✅ `Calculate_Iterative_C_Index(matrix, strong=None, nums=0, tolerance=1e-6, max_iter=1000)`
- **功能**: 计算迭代 C-index，通过自引用机制动态更新节点强度，直至收敛或达到指定迭代次数。
- **逻辑**: 
  - **第 0 步（初始化）**: 基于输入的 `strong` 向量（若为 `None` 则默认为节点加权度）计算初始 C-index，此步骤不计入迭代次数。
  - **迭代阶段**: 将上一轮计算得到的 C-index 值作为下一轮的 `strong` 输入，循环调用基础 `Calculate_C_Index` 函数。
  - **收敛判定**: 若 `nums=0`，则持续迭代直至前后两次结果的 L1 范数差值小于 `tolerance`。
- **输出**: 返回 `List[float]`，包含图中每个节点收敛后的迭代 C-index 值。
- **优化**: 
  - **模块化复用**: 核心计算完全复用 `Calculate_C_Index` 函数，确保算法逻辑的一致性，避免代码冗余。
  - **冷启动设计**: 明确区分“初始化计算”与“迭代计算”，初始化结果作为迭代的基准（第 0 步），符合论文定义的数学逻辑。
- **健壮性**: 
  - **全面容灾**: 继承并增强了对矩阵类型（List/ndarray/DataFrame）、维度匹配、非方阵及空图的检查机制。
  - **类型安全**: 强制统一内部计算数据类型为 `float`，避免类型混用导致的精度丢失或运行时错误。

#### ✅ `Calculate_CG_Index(matrix, strong=None)`
- **功能**: 计算加权无向图中节点的 CG-index，基于 G-index 逻辑改进，关注累积贡献值而非单纯的排序位置。
- **逻辑**: 
  - **贡献计算**: 计算每个邻居的加权贡献值（即：边权重 × 邻居节点强度）。
  - **G-index 判定**: 将贡献值降序排列，寻找最大整数 $g$，使得前 $g$ 个贡献值的累加和 $\ge g^2$。
  - **差异点**: 相比 C-index 的 H-index 逻辑（关注位置 $h$），CG-index 更侧重于高权重邻居的累积效应。
- **输出**: 返回 `List[int]`，按矩阵行顺序包含图中每个节点的 CG-index 值。
- **优化**: 
  - **风格统一**: 完全沿用 `Calculate_Iterative_C_Index` 的输入标准化与容灾检查代码块，确保库内函数风格高度一致。
  - **向量化排序**: 使用 `np.sort` 进行降序排列，配合累加器快速判定 $g$ 值，计算效率高。
- **健壮性**: 
  - **边界处理**: 自动识别孤立节点（无邻居），直接返回 0，防止索引越界。
  - **输入校验**: 严格检查 `strong` 向量的长度与矩阵维度是否匹配，确保计算逻辑闭环。

#### ✅ `Calculate_Communication_Centrality(matrix)`
- **功能**: 计算加权网络中节点的交流中心性，衡量节点的沟通与传播能力。
- **理论依据**: 与 C-index 的核心区别在于使用 **H-degree** 而非 **Strength** 作为邻居评价指标。
- **逻辑**: 
  - **预计算 H-degree**: 对每个节点计算其所有边权重的 H-index，作为其“结构稳固性”指标。
  - **渠道能力**: 计算传播势能 边权重 × 邻居 H-degree。
  - **聚合评价**: 对节点 i 的所有 P_ij 列表计算 H-index。
- **输出**: 返回 `List[int]`，包含每个节点的交流中心性值。
- **优化**: 
  - **向量化计算**: 使用 NumPy 索引乘法优化渠道能力计算。
  - **代码复用**: 沿用库内统一的输入标准化模块。
- **健壮性**: 
  - 严格处理孤立节点和空图情况。
  - 通过结构敏感性测试，确保算法能区分“资源型”与“结构型”邻居。

#### ✅ `Calculate_BI_directional_h_index(adjacency_matrix, max_iter)`
- **功能**: 计算加权有向网络中节点的双向H指数中心性（权威值与枢纽值）。
- **逻辑**: 
    - **初始化**: 节点的入向/出向指数初始化为入度/出度。
    - **迭代更新**: 采用同步更新机制。入向更新基于邻居的“加权出向指数”集合；出向更新基于邻居的“加权入向指数”集合。
    - **收敛判定**: 当连续两次迭代结果完全一致或达到最大迭代次数时停止。
- **输出**: 返回元组 `(h_in_final, h_out_final)`，分别为各节点的入向和出向 H 指数数组。
- **优化**: 
    - **防御性校验**: 优先执行严格的类型与维度检查（必须是二维方阵 NumPy 数组），防止后续计算出现晦涩的索引错误。
    - **算法复用**: 核心计算步骤直接复用 `Calculate_H_Index` 函数，确保底层评价标准的一致性。
- **健壮性**: 
    - **异常处理**: 明确抛出 `TypeError`（非数组输入）和 `ValueError`（非方阵输入）。
    - **边界兼容**: 能够正确处理悬挂节点（出度为0）和孤立节点（度为0）的情况，计算结果稳定。

---

## 🔬 算法说明

### H-index
经典学术评价指标，表示一个研究者有 h 篇论文每篇至少被引用 h 次。在社交网络中，可用来评价节点的自身重要性。

### Hl-index
Hl-index 在 H-index 基础上考虑绩效质量，通过递归思想对绩效依据进行评价，适用于多层次绩效评估场景。

### C-index
C-index 结合拓扑连接强度与 H-index 机制，考虑节点邻居的质量，适用于加权无向网络的中心性分析。

### CG-index
基于 G-index 逻辑，关注累积贡献值而非单纯的排序位置，更侧重于高权重邻居的累积效应。

### 交流中心性
衡量节点在网络中的沟通与传播能力，使用 H-degree 而非 Strength 作为邻居评价指标。

### 双向 H-index
针对有向网络设计，分别计算节点的入向权威值和出向枢纽值，适用于分析有向网络中的节点重要性。

---

## 🧪 测试

项目包含完整的单元测试，覆盖所有核心功能：

```bash
# 运行所有测试
python -m unittest discover tests/

# 运行特定测试
python -m unittest tests.H_Index_tests
```

### 测试覆盖率

- 基础 H-index 计算逻辑
- 边界条件处理
- 异常输入处理
- 多种输入格式支持
- 网络中心性算法验证

---

## 🤝 贡献

我们热烈欢迎各种形式的贡献！无论您是开发者、研究者还是用户，都可以通过以下方式为本项目做出贡献：

### 代码贡献
如果您想改进核心功能或添加新特性：

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

### 使用案例贡献
我们特别欢迎用户分享实际应用案例！如果您在以下领域使用了本库，请考虑贡献您的案例：

- **学术研究**：社交网络分析、复杂网络研究、文献计量学应用
- **工业应用**：推荐系统、风险控制、社交网络分析
- **教育领域**：教学案例、学生项目、课程作业
- **其他创新应用**：任何创造性的使用场景

**如何贡献使用案例：**
- 在`example/`目录下添加您的案例代码和说明文档
- 或者提交Issue分享您的应用经验
- 优秀的案例将被收录到官方文档中

### 其他贡献方式
- **文档改进**：修正错误、补充说明、翻译文档
- **问题反馈**：提交Bug报告或功能建议
- **性能优化**：提出性能改进方案
- **测试增强**：增加测试覆盖率

### 开发规范

- 遵循 PEP 8 编码规范
- 为新增功能添加单元测试
- 更新文档和示例代码
- 使用类型提示增强代码可读性
- 提交清晰的commit信息

---

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

---

## 📚 引用

如果您在研究中使用本库，请引用相关论文：

### 核心算法参考文献

```bibtex
@article{Zhai2026TheIO,
  title={The impact of mentorship on the citation-based performance of graduate dissertation: A social capital perspective},
  author={Li Zhai and Xiangbin Yan},
  journal={J. Informetrics},
  year={2026},
  volume={20},
  pages={101781},
  url={https://api.semanticscholar.org/CorpusID:285997522}
}

@article{article,
author = {Zhai, Li and Yan, Xiangbin},
year = {2022},
month = {11},
pages = {101345},
title = {A directed collaboration network for exploring the order of scientific collaboration},
volume = {16},
journal = {Journal of Informetrics},
doi = {10.1016/j.joi.2022.101345}
}

@article{article,
author = {Zhai, Li and Yan, Xiangbin and Zhang, Guojing},
year = {2018},
month = {02},
pages = {299-314},
title = {Bi-directional h-index: A new measure of node centrality in weighted and directed networks},
volume = {12},
journal = {Journal of Informetrics},
doi = {10.1016/j.joi.2018.01.004}
}

@article{article,
author = {Zhai, Li and Yan, Xiangbin and Zhu, Bin},
year = {2014},
month = {02},
pages = {},
title = {The Hl-index: Improvement of H-index based on quality of citing papers},
volume = {98},
journal = {Scientometrics},
doi = {10.1007/s11192-013-1039-z}
}

@article{article,
author = {Zhai, Li and Yan, Xiangbin and Zhang, Guojing},
year = {2013},
month = {12},
pages = {6107-6117},
title = {A centrality measure for communication ability in weighted network},
volume = {392},
journal = {Physica A Statistical Mechanics and its Applications},
doi = {10.1016/j.physa.2013.07.056}
}

@article{article,
author = {Yan, Xiangbin and Zhai, Li and Fan, Weiguo},
year = {2013},
month = {01},
pages = {223-239},
title = {C-index: A weighted network node centrality measure for collaboration competence},
volume = {7},
journal = {Journal of Informetrics},
doi = {10.1016/j.joi.2012.11.004}
}
```
请在您的研究中适当引用这些原始论文，以支持学术工作的可追溯性和完整性。

---
