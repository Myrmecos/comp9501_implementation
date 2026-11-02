# Running the Python port of the authors’ HFA / SHFA check

MATLAB sources and data live in:

```text
../matlab_implementation/official_shfa_release/
```

That folder is Wen Li’s public release: Amazon (800-d SURF) → DSLR (600-d SURF), 31 categories, 20 labeled source images per category, 3 labeled DSLR images per category, 405 DSLR test images.

## This machine

Full MATLAB may be absent (Runtime alone cannot run `demo.m`). `reproduce_official.py` is the port that was actually run.

From the **repo root**:

```text
.venv\Scripts\python.exe python_implementation\reproduce_official.py
```

From **this directory**:

```text
..\.venv\Scripts\python.exe reproduce_official.py
```

It follows `demo.m` and `demo_shfa.m`: Gaussian bandwidth, \(\lambda=100\), \(C=1\), SHFA’s \(C_u=10^{-3}\) and \(p=1.5\), cutting-plane cap 50, class prior from the test-set positive rate. SVM calls go through scikit-learn’s LIBSVM. The release’s own solver is LIBSVM 3.20 via `svmtrain.mexw64`.

The GPU is not used. The algorithm is not a GPU method.

## What the authors say the demo prints

| Script | Method | Their printed accuracy | This port |
|---|---|---:|---:|
| `demo.m` | HFA | 0.567901 (230/405) | 229/405 = 0.565432 |
| `demo_shfa.m` | SHFA | 0.572842 | 232/405 = 0.572840 |

232/405 is the unique image count consistent with 0.572842. The HFA difference is one test image (LIBSVM 3.20 vs scikit-learn’s LIBSVM). Details and figures: `HFA_SHFA_RESULTS.md`.

## If you want native MATLAB instead

See `../matlab_implementation/RUN_MATLAB.md`.  
`hfa_numpy.py` is not a substitute for either official script.
