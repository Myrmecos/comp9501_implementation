# Guide: install full MATLAB and run the official HFA / SHFA demo

This machine currently has only the **MATLAB Runtime**, which cannot execute `.m` scripts. You need a full MATLAB desktop install (R2018b or newer is fine; the release is old and does not need deep-learning toolboxes).

Official code and data are already in:

```text
matlab_implementation/official_shfa_release/
```

Source: [wenli-vision/SHFA_release](https://github.com/wenli-vision/SHFA_release).

Expected prints (authors’ README):

| Script        | Method | Printed accuracy |
|---------------|--------|------------------:|
| `demo.m`      | HFA    | **0.567901** (230/405) |
| `demo_shfa.m` | SHFA   | **0.572842** (~232/405) |

---

## 1. Download and install MATLAB

1. Go to [MathWorks MATLAB](https://www.mathworks.com/products/matlab.html) → **Download** / **Get MATLAB**.
2. Sign in with a MathWorks account (university license, student, home, or trial).
3. Install **MATLAB** (desktop product), not “MATLAB Runtime” alone.
4. Toolboxes required for this demo:
   - Base MATLAB is enough for `sqrtm`, `pinv`, path handling.
   - No Statistics, Parallel, or Deep Learning toolbox is required by the release scripts.
5. Optional but useful on Windows if you must rebuild mex files:
   - A supported C/C++ compiler (Visual Studio Build Tools, or MinGW-w64 via MATLAB’s Add-On Explorer).
   - In MATLAB: `mex -setup C++` and pick that compiler.

After install, launch MATLAB and confirm:

```matlab
version
computer('arch')   % should be win64 on this PC
```

---

## 2. What the release already contains

Under `matlab_implementation/official_shfa_release/`:

| Path | Role |
|------|------|
| `demo.m` | Supervised HFA on Amazon → DSLR |
| `demo_shfa.m` | Semi-supervised SHFA on the same split |
| `data/*.mat` | 31-class Office features and the fixed indices |
| `libs/libsvm-weights-3.20/matlab/svmtrain.mexw64` | Prebuilt weighted LIBSVM (Windows 64-bit) |
| `libs/sumkernels/sumkernels.mexw64` | Prebuilt kernel-sum mex for SHFA |
| `train_*.m`, `LpMKL_*.m`, `predict_*.m`, `commfuns/` | Solver and helpers |

On **64-bit Windows**, the shipped `.mexw64` files often run without recompilation. Rebuild only if MATLAB reports “Invalid MEX-file” or a missing dependency.

---

## 3. One-line fix required before SHFA

`demo.m` loads data from `./data` and is consistent when the current folder is the release root.

`demo_shfa.m` calls `object_prepare_data.m`, which hard-codes:

```matlab
datapath = '../data';
```

The mats live in `matlab_implementation/official_shfa_release/data`, so that path fails when MATLAB’s current directory is the release root. Change it to:

```matlab
datapath = './data';
```

(or `datapath = fullfile(fileparts(mfilename('fullpath')), 'data');`).

Do **not** change anything else if the goal is to match the authors’ printed check.

---

## 4. Run HFA (`demo.m`)

1. In MATLAB: **Home → Set Path** is optional; the script adds its own paths.
2. Set the current folder to the release:

```matlab
cd('C:\Users\etmph\Desktop\machine_learning_gp\matlab_implementation\official_shfa_release')
```

3. Confirm the LIBSVM mex is visible (already on the path set inside `demo.m`):

```matlab
which svmtrain
% expect: ...\libs\libsvm-weights-3.20\matlab\svmtrain.mexw64
```

4. Run:

```matlab
demo
```

5. Wait for 31 one-versus-all models. Wall time is typically on the order of **1–3 minutes** on a modern CPU (cutting-plane MKL around LIBSVM; no GPU).
6. Final line should be close to:

```text
The accuracy = 0.567901
```

If `svmtrain` is missing or the mex fails, see §6.

---

## 5. Run SHFA (`demo_shfa.m`)

1. Apply the `datapath` fix in §3.
2. Stay in the release root:

```matlab
cd('C:\Users\etmph\Desktop\machine_learning_gp\matlab_implementation\official_shfa_release')
which sumkernels
% expect: ...\libs\sumkernels\sumkernels.mexw64
demo_shfa
```

3. This is slower (405 unlabeled points, 50 outer rounds × 31 classes). Expect on the order of **5–15 minutes** on CPU.
4. Final line should be close to:

```text
The accuracy = 0.572842
```

---

## 6. If mex files fail: rebuild

### Weighted LIBSVM

The authors’ README points to LIBSVM and asks for the **weighted** MATLAB interface. This tree only ships `svmtrain.mexw64`, not the full LIBSVM sources.

If you must rebuild:

1. Obtain a weighted LIBSVM 3.x MATLAB package compatible with the demos’ `svmtrain` calling convention (instance weights + precomputed kernel). The release layout expects files under:

```text
matlab_implementation\official_shfa_release\libs\libsvm-weights-3.20\matlab\
```

2. Open MATLAB, `cd` into that `matlab` folder, run `make` (or `make.m`).
3. Point `demo.m` / `demo_shfa.m` at that folder (first `addpath` line).

Standard (unweighted) LIBSVM is **not** a drop-in replacement for this release.

### `sumkernels` (SHFA only)

```matlab
cd('C:\Users\etmph\Desktop\machine_learning_gp\matlab_implementation\official_shfa_release\libs\sumkernels')
mex -setup C++    % once
make              % runs make.m → mex sumkernels.cpp
```

`make.m` uses MSVC-style OpenMP flags. On MinGW you may need to drop `/openmp` or adapt flags if the mex build errors.

---

## 7. Checklist of common failures

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `Undefined function 'svmtrain'` | Path / missing mex | `addpath` to `libs\libsvm-weights-3.20\matlab`; rebuild if needed |
| `Invalid MEX-file` / architecture mismatch | 32-bit mex or wrong MATLAB bitness | Use 64-bit MATLAB; rebuild `.mexw64` |
| `Unable to read file ... data\...` in SHFA | `datapath = '../data'` | Change to `'./data'` (§3) |
| HFA runs, SHFA dies on `sumkernels` | Mex missing or not on path | `addpath('.\libs\sumkernels')`; rebuild (§6) |
| Accuracy far from 0.56 / 0.57 | Wrong cwd, wrong data, or edited params | `cd` to release root; do not change `C`, `λ`, `p`, indices |
| Only Runtime installed | Cannot run `.m` files | Install full MATLAB (§1) |

---

## 8. What you should not change (to match the paper check)

- Data mats and `rand_indices.mat`
- `param.lambda = 100`, `C_s = C_t = 1`
- SHFA: `C_x = 1e-3`, `mkl_degree = 1.5`, `hfa_iter = 50`
- Gaussian kernel construction in `commfuns/`
- Class prior taken from the true test positive rate inside `demo_shfa.m`

The demo is **one fixed split**, not the paper’s 20-split Table 4 average.

---

## 9. Optional: compare to the Python port

Without waiting for MATLAB, the same protocol was already ported and run:

```text
.venv\Scripts\python.exe python_implementation\reproduce_official.py
```

Details: `python_implementation/HFA_SHFA_RESULTS.md`, `python_implementation/RUN_OFFICIAL.md`.

| Method | Official MATLAB (authors) | Python port on this machine |
|--------|---------------------------|-----------------------------|
| HFA    | 0.567901 (230/405)        | 229/405 = 0.565432          |
| SHFA   | 0.572842                  | 232/405 = 0.572840          |

The one-image HFA gap is between LIBSVM 3.20 (release mex) and scikit-learn’s LIBSVM. Running `demo.m` in full MATLAB is the way to recover the exact printed HFA digit.
