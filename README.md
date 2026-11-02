# HFA / SHFA

Reproduction of Wen Li, Lixin Duan, Dong Xu, and Ivor W. Tsang, *Learning with Augmented Features for Supervised and Semi-Supervised Heterogeneous Domain Adaptation*, IEEE TPAMI, 2014.

## Layout

| Path | Contents |
|------|----------|
| `python_implementation/` | Python port, educational NumPy demo, run notes, results & figures |
| `matlab_implementation/` | Authors’ MATLAB release + MATLAB install/run guide |
| `HFA_SHFA_ALGORITHM.md` | Algorithm note (shared) |
| `slides/index.html` | 15-minute talk |
| `.venv/` | Python environment (repo root) |

## Python (official check)

```text
.venv\Scripts\python.exe python_implementation\reproduce_official.py
```

See `python_implementation/README.md` for how the port maps to the MATLAB demos.

Authors’ printed one-round checks: HFA **0.567901**, SHFA **0.572842**.  
This port: HFA **229/405**, SHFA **232/405**. Details in `python_implementation/HFA_SHFA_RESULTS.md`.

`hfa_numpy.py` / `run_demo.py` are a synthetic alternating linear demo, **not** the published solver.

## MATLAB (authors’ scripts)

```text
matlab_implementation/RUN_MATLAB.md
matlab_implementation/official_shfa_release/
```

Full MATLAB (not Runtime) is required to execute `demo.m` / `demo_shfa.m`.
