# Python 实现说明

本目录是论文 HFA / SHFA（Li et al., TPAMI 2014）的 **Python 侧代码与结果**。
官方算法在 MATLAB 里；这里有两套 Python 程序，用途不同。

## 目录里有什么

| 文件 / 目录 | 作用 |
|-------------|------|
| `reproduce_official.py` | **正式复现**：按作者 MATLAB release 逐段移植的凸 cutting-plane MKL |
| `hfa_numpy.py` | **教学用**线性交替求解器（类似被 TPAMI 替换掉的 ICML 风格），不是正式算法 |
| `run_demo.py` | 在合成数据上跑 `hfa_numpy.py` |
| `RUN_OFFICIAL.md` | 如何跑正式复现、与作者打印数字对照 |
| `HFA_SHFA_RESULTS.md` | 正式复现的准确率、混淆矩阵、耗时等 |
| `output/` | `official_results.json`、预测、`figures/` |

数据与官方 MATLAB 求解器在：

```text
../matlab_implementation/official_shfa_release/
```

---

## 正式实现：`reproduce_official.py`

### 目标

在作者公开的 **Amazon → DSLR** 固定划分上，复现：

- HFA（`demo.m`）期望打印 **0.567901**
- SHFA（`demo_shfa.m`）期望打印 **0.572842**

本机无 scikit-learn 的 LIBSVM；作者用 LIBSVM 3.20 mex。因此 HFA 可能差 **1 张图**；SHFA 应对齐到 **232/405**。

### 依赖

仓库根目录的 `.venv`（numpy / scipy / scikit-learn / matplotlib / threadpoolctl）。

```text
..\.venv\Scripts\python.exe reproduce_official.py
```

（从本目录执行；从仓库根执行见下方。）

从仓库根：

```text
.venv\Scripts\python.exe python_implementation\reproduce_official.py
```

可选：`--methods SVM_T,HFA,SHFA`，`--max-classes N`（冒烟，不写正式 json）。

### 流水线（对应 MATLAB）

1. **读数据**  
   `categories.mat`、`data_amazon.mat`、`data_dslr_600.mat`、`rand_indices.mat`  
   → 620 源 / 93 有标目标 / 405 测试；列 ℓ₂ 归一化。

2. **高斯核**（`getKernel` / `return_GaussianKernel`）  
   带宽由平方距离均值（含对角 0）得到；源、目标各自一个 γ。

3. **SVM_T**  
   仅在 93 个有标 DSLR 上、同一核、C=1、one-vs-all。

4. **HFA**（对每个类别一个二分类）  
   - `K_root = blkdiag(Ks^{1/2}, Kt^{1/2})`  
   - 初始原子 `h ∝ 1`，带 √λ  
   - 外层 cutting-plane：在当前原子上做 **p=1** MKL（加权 C-SVM）→ 用 `h ∝ K^{1/2}(α∘y)` 加最违原子  
   - 核形式与 MATLAB `sqrtm` 一致：`(K_root @ Hv)(K_root @ Hv)^T + K_root@K_root`  
   - 预测：目标块核 + `H` + ρ（与 `demo.m` 一致）

5. **SHFA**  
   - 405 测试图作为无标目标训练样本（直推）  
   - **p=1.5**，ρ-SVM（平方铰链，one-class SVM ν=1/n）  
   - 无标正类比例设为该类别真实测试率（与论文 T-SVM 协议相同）  
   - `sumkernels`：按标签与 `d` 组装基核并加对角权重  
   - 每轮：∞ 范数启发搜标签 → 加原子 → 再解 p-MKL  
   - 测试决策值除以该二分类的 ρ 再 argmax

6. **写出**  
   `output/official_results.json`、`official_predictions.npz`、`figures/*.png`

### 与 MATLAB 的对应关系

| Python | MATLAB |
|--------|--------|
| `gaussian_kernel` / `symmetric_sqrt` | `getKernel`, `sqrtm` |
| `fit_csvc` | `svmtrain` C-SVM, `-t 4` |
| `fit_onesvm` | one-class SVM |
| HFA 训练循环 | `train_hfa_mkl` + `LpMKL_H_fast` |
| SHFA 训练循环 | `train_shfa_pnorm` + `LpMKL_H_labels_v4` |
| SHFA 核组装 | `sumkernels.cpp` |
| SHFA 预测 | `predict_ifa_semi_kernel` |

### 刻意不做的事

- **不用 GPU**：官方方法是 CPU 上 cutting-plane + LIBSVM。
- **不把 `hfa_numpy.py` 当正式结果**。
- **不重算论文 Table 4 的 20 次随机划分均值**（release 只有这一份 index）。

---

## 教学实现：`hfa_numpy.py` + `run_demo.py`

- 显式维护 `P`、`Q`，在增广特征上交替训线性 hinge 与更新投影。
- 合成高斯数据，维度小，几秒跑完。
- **没有**核化、**没有**凸 MKL、**没有**全局最优保证。  
  论文正式算法请看 `reproduce_official.py` 与 `../matlab_implementation/`。

```text
..\..\.venv\Scripts\python.exe run_demo.py
```
