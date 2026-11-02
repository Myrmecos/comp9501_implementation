"""Reproduce the authors' HFA and SHFA demo on the released Office split.

This is a line-by-line port of official_shfa_release (Wen Li), not the
alternating linear solver in hfa_numpy.py. The numerical target published
in that release is one round of Amazon -> DSLR:

    demo.m      HFA   0.567901
    demo_shfa.m SHFA  0.572842

The solver is the authors' cutting-plane p-norm MKL loop around LIBSVM.
It is a CPU algorithm. A GPU is not used.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import linalg
from sklearn.svm import SVC, OneClassSVM
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA = REPO / "matlab_implementation" / "official_shfa_release" / "data"
OUT = HERE / "output"


def decode_categories(cell) -> list[str]:
    names = []
    for row in np.asarray(cell).ravel():
        item = row
        if isinstance(item, np.ndarray):
            if item.dtype.kind in "SU":
                item = item.item() if item.shape == () else "".join(item.astype(str).ravel())
            elif item.size == 1:
                item = item.reshape(-1)[0]
            else:
                item = "".join(chr(int(c)) for c in item.ravel())
        names.append(str(item).strip())
    return names


def l2_normalize_columns(features: np.ndarray) -> np.ndarray:
    # demo.m: features ./ repmat(sqrt(sum(features.^2)), ...)
    scale = np.sqrt(np.sum(features * features, axis=0, keepdims=True))
    return features / scale


def squared_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """MATLAB L2_distance_2. a, b are d x n."""
    aa = np.sum(a * a, axis=0)
    bb = np.sum(b * b, axis=0)
    dist = aa[:, None] + bb[None, :] - 2.0 * (a.T @ b)
    np.maximum(dist, 0.0, out=dist)
    return dist


def gaussian_kernel(a: np.ndarray, b: np.ndarray | None = None, gamma: float | None = None):
    """return_GaussianKernel.m. If gamma is None, set it from mean squared distance."""
    if b is None:
        b = a
    dist = squared_l2(a, b)
    if gamma is None:
        sigma = np.sqrt(float(np.mean(dist)) * 0.5)
        gamma = 0.0 if sigma == 0 else 1.0 / (2.0 * sigma * sigma)
    return np.exp(-dist * gamma), float(gamma)


def symmetric_sqrt(kernel: np.ndarray) -> np.ndarray:
    # MATLAB: real(sqrtm(K)). Principal square root, imaginary dust dropped.
    root = linalg.sqrtm(kernel)
    root = np.real(np.asarray(root))
    return 0.5 * (root + root.T)


def fit_csvc(kernel: np.ndarray, labels: np.ndarray, sample_weight: np.ndarray):
    """Weighted C-SVC, precomputed kernel. Port of svmtrain(weight, y, K, '-t 4 -c 1')."""
    kernel = np.ascontiguousarray(kernel, dtype=np.float64)
    clf = SVC(C=1.0, kernel="precomputed", tol=1e-3, shrinking=True)
    clf.fit(kernel, labels, sample_weight=sample_weight)
    alpha = np.zeros(len(labels), dtype=np.float64)
    # sklearn dual_coef_ is oriented so that decision_function > 0 predicts +1.
    # The MATLAB release stores |sv_coef| and multiplies by the original ±1 label.
    signed = np.zeros(len(labels), dtype=np.float64)
    signed[clf.support_] = clf.dual_coef_[0]
    alpha[clf.support_] = np.abs(clf.dual_coef_[0])
    y_alpha = labels * alpha
    # Prefer the coefficient vector that reproduces sklearn's decision function.
    # They coincide when the positive class is +1, which it is for ±1 labels.
    if not np.allclose(y_alpha[clf.support_], signed[clf.support_], atol=1e-8):
        y_alpha = signed
        alpha = np.abs(y_alpha)
    rho = float(-clf.intercept_[0])
    return alpha, y_alpha, rho


def fit_onesvm(kernel: np.ndarray, nu: float):
    """One-class SVM, precomputed kernel. Port of svmtrain(..., '-s 2 -t 4 -n 1/n')."""
    kernel = np.ascontiguousarray(kernel, dtype=np.float64)
    n = kernel.shape[0]
    clf = OneClassSVM(kernel="precomputed", nu=nu, tol=1e-3, shrinking=True)
    clf.fit(kernel)
    alpha = np.zeros(n, dtype=np.float64)
    alpha[clf.support_] = np.abs(clf.dual_coef_[0])
    return alpha


def hfa_kernel(k_root: np.ndarray, k_root_sq: np.ndarray, hvs: np.ndarray, coefficients: np.ndarray):
    """LpMKL_H_fast local sumkernels: K_root (H + I) K_root, H = sum d_i h_i h_i'."""
    scaled = hvs * np.sqrt(np.maximum(coefficients, 0.0))[None, :]
    projected = k_root @ scaled
    return projected @ projected.T + k_root_sq


def shfa_kernel(k_mat, k_root, hvs, label_mat, coefficients):
    """libs/sumkernels/sumkernels.cpp, then the caller adds diag(weight)."""
    wlabels = label_mat * np.sqrt(np.maximum(coefficients, 0.0))[None, :]
    projected = (k_root @ hvs) * wlabels
    ones_kernel = k_mat + 1.0
    return projected @ projected.T + ones_kernel * (wlabels @ wlabels.T)


def mkl_update(wn: np.ndarray, degree: float, d_norm: float = 1.0) -> np.ndarray:
    # coefficients = d_norm * wn^(2/(p+1)) / ||.||_p
    power = 2.0 / (degree + 1.0)
    wnp = np.power(np.maximum(wn, 0.0), power)
    eta = np.sum(np.power(wnp, degree)) ** (1.0 / degree)
    if not np.isfinite(eta) or eta <= 0:
        return np.full(len(wn), d_norm / len(wn))
    return d_norm * wnp / eta


def train_hfa_one(source_y, target_y, k_root, k_root_sq, lam: float, degree: float,
                   max_outer: int, tau: float):
    n_s = len(source_y)
    n_l = len(target_y)
    n = k_root.shape[0]
    labels = np.concatenate([source_y, target_y]).astype(np.float64)
    weight = np.ones(n, dtype=np.float64)  # C_s = C_t = 1
    hvs = np.sqrt(lam) * np.ones((n, 1)) / np.sqrt(n)
    objectives = []
    inner_iters = []
    alpha = y_alpha = None
    rho = 0.0
    coefficients = None

    for outer in range(1, max_outer + 1):
        coefficients = np.full(hvs.shape[1], (1.0 / hvs.shape[1]) ** (1.0 / degree))
        alpha, y_alpha, rho, obj, wn = _hfa_svm(labels, k_root, k_root_sq, hvs, coefficients, weight)
        inner = 1
        for inner in range(2, 101):
            coefficients = mkl_update(wn, degree)
            alpha, y_alpha, rho, obj_new, wn = _hfa_svm(
                labels, k_root, k_root_sq, hvs, coefficients, weight
            )
            if abs(obj_new - obj) <= tau * abs(obj_new):
                obj = obj_new
                break
            obj = obj_new
        objectives.append(float(obj))
        inner_iters.append(int(inner))
        if outer > 1 and (
            abs(objectives[-1] - objectives[-2]) <= tau * abs(objectives[-1]) or outer == max_outer
        ):
            break
        beta = k_root @ y_alpha
        norm = np.sqrt(float(beta @ beta))
        if norm <= 0:
            break
        hvs = np.concatenate([hvs, (np.sqrt(lam) * beta / norm)[:, None]], axis=1)

    scaled = hvs * np.sqrt(np.maximum(coefficients, 0.0))[None, :]
    metric = scaled @ scaled.T
    return {
        "y_alpha": y_alpha,
        "rho": rho,
        "H": metric,
        "objectives": objectives,
        "inner_iters": inner_iters,
        "n_kernels": int(hvs.shape[1]),
    }


def _hfa_svm(labels, k_root, k_root_sq, hvs, coefficients, weight):
    kernel = hfa_kernel(k_root, k_root_sq, hvs, coefficients)
    alpha, y_alpha, rho = fit_csvc(kernel, labels, weight)
    kay = k_root @ y_alpha
    hkay = hvs.T @ kay
    q = hkay ** 2
    obj = float(np.sum(alpha) - 0.5 * (np.sum(q * coefficients) + kay @ kay))
    wn = coefficients * np.sqrt(np.maximum(q, 0.0))
    return alpha, y_alpha, rho, obj, wn


def predict_hfa(k_test, l_t_inv, metric, k_root, y_alpha, rho, n_s):
    # demo.m testing block
    tmp = k_test @ l_t_inv.T @ metric @ k_root
    return tmp @ y_alpha + k_test @ y_alpha[n_s:] - rho


def clip_boolean_columns(initial: np.ndarray, rank_key: np.ndarray, lower: float, upper: float):
    """MATLAB ratio clip. initial is boolean (n_u, m), rank_key is the sort key."""
    n_u, m = initial.shape
    out = initial.copy()
    order = np.argsort(-rank_key, axis=0, kind="stable")
    upper_n = upper * n_u
    lower_n = lower * n_u
    keep = int(np.floor(upper_n))
    fill = int(np.ceil(lower_n))
    for j in range(m):
        count = int(np.sum(out[:, j]))
        column = order[:, j]
        if count > upper_n:
            out[column[keep:], j] = False
        elif count < lower_n:
            out[column[:fill], j] = True
    return out


def train_shfa_one(source_y, target_y, k_mat, k_root, aug, weight, lam, degree,
                    ratio, max_outer, tau):
    n_s = len(source_y)
    n_l = len(target_y)
    n = k_mat.shape[0]
    n_u = n - n_s - n_l
    labeled = np.concatenate([source_y, target_y]).astype(np.float64)

    # SVM on labeled target only, used to initialize unlabeled labels.
    block = k_mat[n_s:n_s + n_l, n_s:n_s + n_l]
    q_l = (block + 1.0) * np.outer(target_y, target_y) + np.diag(weight[n_s:n_s + n_l])
    alpha_l = fit_onesvm(q_l, nu=1.0 / n_l)
    y_alpha_l = target_y * alpha_l
    cross = k_mat[n_s + n_l:, n_s:n_s + n_l]
    decs = (cross + 1.0) @ y_alpha_l
    # Same clip as train_shfa_pnorm.m: only the tail is changed, so a negative
    # score inside the top floor(ratio*n_u) stays negative.
    y_u = clip_boolean_columns((decs > 0)[:, None], decs[:, None], ratio, ratio)[:, 0]
    labels = np.concatenate([labeled, np.where(y_u, 1.0, -1.0)])[:, None]

    hvs = np.sqrt(lam) * np.ones((n, 1)) / np.sqrt(n)
    objectives = []
    inner_iters = []
    alpha = None
    coefficients = None

    for outer in range(1, max_outer + 1):
        coefficients = np.full(hvs.shape[1], (1.0 / hvs.shape[1]) ** (1.0 / degree))
        alpha, obj, wn, kernel = _shfa_svm(k_mat, k_root, hvs, labels, coefficients, weight)
        inner = 1
        for inner in range(2, 101):
            coefficients = mkl_update(wn, degree)
            alpha, obj_new, wn, kernel = _shfa_svm(
                k_mat, k_root, hvs, labels, coefficients, weight
            )
            if abs(obj_new - obj) <= tau * abs(obj_new):
                obj = obj_new
                break
            obj = obj_new
        objectives.append(float(obj))
        inner_iters.append(int(inner))
        if (outer > 1 and abs(objectives[-1] - objectives[-2]) <= tau * abs(objectives[-1])) or outer == max_outer:
            break

        dim = aug.shape[0]
        # ax(i, j) = aug'(i, j) * alpha(i) = aug[j, i] * alpha[i]
        ax = aug.T * alpha[:, None]
        axu = ax[n_s + n_l:, :]
        pos_key = axu
        neg_key = -axu
        pos0 = axu > 0
        neg0 = -axu > 0
        pos = clip_boolean_columns(pos0, pos_key, ratio, ratio)
        neg = clip_boolean_columns(neg0, neg_key, ratio, ratio)
        y_u_mat = np.concatenate([pos, neg], axis=1)
        y_u_pm = np.where(y_u_mat, 1.0, -1.0)
        labeled_block = np.repeat(labeled[:, None], y_u_pm.shape[1], axis=1)
        candidates = np.concatenate([labeled_block, y_u_pm], axis=0)
        signed_ax = np.concatenate([ax, -ax], axis=1)
        values = np.abs(np.sum(signed_ax * candidates, axis=0))
        best = int(np.argmax(values))
        y = candidates[:, best]
        y_alpha = y * alpha
        beta = k_root @ y_alpha
        norm = np.sqrt(float(beta @ beta))
        if norm <= 0:
            break
        hvs = np.concatenate([hvs, (np.sqrt(lam) * beta / norm)[:, None]], axis=1)
        labels = np.concatenate([labels, y[:, None]], axis=1)

    rho_scores = kernel @ alpha
    support = alpha > 0
    rho = float(np.mean(rho_scores[support])) if np.any(support) else float(np.mean(rho_scores))
    return {
        "alpha": alpha,
        "Hvs": hvs,
        "labels": labels,
        "coefficients": coefficients,
        "rho": rho,
        "objectives": objectives,
        "inner_iters": inner_iters,
        "n_kernels": int(hvs.shape[1]),
    }


def _shfa_svm(k_mat, k_root, hvs, labels, coefficients, weight):
    kernel = shfa_kernel(k_mat, k_root, hvs, labels, coefficients)
    kernel = kernel + np.diag(weight)
    # Symmetrize tiny asymmetries before LIBSVM.
    kernel = 0.5 * (kernel + kernel.T)
    alpha = fit_onesvm(kernel, nu=1.0 / labels.shape[0])
    # q matches LpMKL_H_labels_v4.m. Entries with alpha=0 do not contribute.
    ay = alpha[:, None] * labels
    su = k_root @ hvs
    q = np.sum(su * ay, axis=0) ** 2
    q = q + np.sum((k_root @ ay) ** 2, axis=0)
    q = q + np.sum(ay, axis=0) ** 2
    obj = float(-0.5 * (np.sum(q * coefficients) + alpha @ (alpha * weight)))
    wn = coefficients * np.sqrt(np.maximum(q, 0.0))
    return alpha, obj, wn, kernel


def predict_shfa(k_test, model, k_root, l_t_inv, n_s):
    alpha = model["alpha"]
    labels = model["labels"]
    hvs = model["Hvs"]
    coefficients = model["coefficients"]
    dec = np.zeros(k_test.shape[0], dtype=np.float64)
    for i in range(labels.shape[1]):
        y_alpha = alpha * labels[:, i]
        # (K_test L_t_inv' h) (h' K_root) y_alpha
        left = k_test @ (l_t_inv.T @ hvs[:, i])
        right = hvs[:, i] @ k_root
        dec = dec + coefficients[i] * (left[:, None] * right[None, :] @ y_alpha
                                        + k_test @ y_alpha[n_s:]
                                        + np.sum(y_alpha))
    return dec / model["rho"]


def train_svmt_one(target_y, k_train, k_test):
    n = len(target_y)
    alpha, y_alpha, rho = fit_csvc(k_train, target_y.astype(np.float64), np.ones(n))
    return k_test @ y_alpha - rho


def load_split():
    import scipy.io as sio

    categories = decode_categories(sio.loadmat(DATA / "categories.mat")["categories"])
    amazon = sio.loadmat(DATA / "data_amazon.mat")
    dslr = sio.loadmat(DATA / "data_dslr_600.mat")
    indices = sio.loadmat(DATA / "rand_indices.mat")

    source_x = l2_normalize_columns(amazon["features"].T.astype(np.float64))
    source_y_all = amazon["labels"].ravel().astype(np.int32)
    target_x = l2_normalize_columns(dslr["features"].T.astype(np.float64))
    target_y_all = dslr["labels"].ravel().astype(np.int32)

    source_index = indices["source_index"].ravel().astype(np.int64) - 1
    train_index = indices["target_training_index"].ravel().astype(np.int64) - 1
    test_index = indices["target_test_index"].ravel().astype(np.int64) - 1

    return {
        "categories": categories,
        "source_x": source_x[:, source_index],
        "source_y": source_y_all[source_index],
        "target_x": target_x[:, train_index],
        "target_y": target_y_all[train_index],
        "test_x": target_x[:, test_index],
        "test_y": target_y_all[test_index],
    }


def prepare_hfa(data):
    k_s, gamma_s = gaussian_kernel(data["source_x"])
    k_t, gamma_t = gaussian_kernel(data["target_x"])
    k_test, _ = gaussian_kernel(data["test_x"], data["target_x"], gamma=gamma_t)
    k_s_root = symmetric_sqrt(k_s)
    k_t_root = symmetric_sqrt(k_t)
    n_s = k_s.shape[0]
    n_t = k_t.shape[0]
    k_root = np.zeros((n_s + n_t, n_s + n_t))
    k_root[:n_s, :n_s] = k_s_root
    k_root[n_s:, n_s:] = k_t_root
    k_t_inv = np.real(linalg.pinv(k_t_root))
    l_t_inv = np.zeros((n_s + n_t, n_t))
    l_t_inv[n_s:, :] = k_t_inv
    return {
        "k_root": k_root,
        "k_root_sq": k_root @ k_root,
        "k_test": k_test,
        "l_t_inv": l_t_inv,
        "n_s": n_s,
        "gamma_s": gamma_s,
        "gamma_t": gamma_t,
        "k_t": k_t,
    }


def prepare_shfa(data):
    target_all = np.concatenate([data["target_x"], data["test_x"]], axis=1)
    k_s, gamma_s = gaussian_kernel(data["source_x"])
    k_t, gamma_t = gaussian_kernel(target_all)
    k_test, _ = gaussian_kernel(data["test_x"], target_all, gamma=gamma_t)
    k_s_root = symmetric_sqrt(k_s)
    k_t_root = symmetric_sqrt(k_t)
    n_s = k_s.shape[0]
    n_t = k_t.shape[0]
    k_mat = np.zeros((n_s + n_t, n_s + n_t))
    k_mat[:n_s, :n_s] = k_s
    k_mat[n_s:, n_s:] = k_t
    k_root = np.zeros_like(k_mat)
    k_root[:n_s, :n_s] = k_s_root
    k_root[n_s:, n_s:] = k_t_root
    k_t_inv = np.real(linalg.pinv(k_t_root))
    l_t_inv = np.zeros((n_s + n_t, n_t))
    l_t_inv[n_s:, :] = k_t_inv
    aug = symmetric_sqrt((1.0 + 100.0) * k_mat + np.ones_like(k_mat))
    return {
        "k_mat": k_mat,
        "k_root": k_root,
        "k_test": k_test,
        "l_t_inv": l_t_inv,
        "aug": aug,
        "n_s": n_s,
        "n_l": data["target_y"].shape[0],
        "gamma_s": gamma_s,
        "gamma_t": gamma_t,
    }


def binary_labels(labels: np.ndarray, category: int) -> np.ndarray:
    return np.where(labels == category, 1.0, -1.0)


def accuracy(pred, truth) -> float:
    return float(np.mean(pred == truth))


def per_class_accuracy(pred, truth, classes):
    rows = []
    for c in classes:
        mask = truth == c
        rows.append(float(np.mean(pred[mask] == truth[mask])) if np.any(mask) else float("nan"))
    return rows


def confusion(pred, truth, classes):
    k = len(classes)
    lookup = {c: i for i, c in enumerate(classes)}
    mat = np.zeros((k, k), dtype=np.int32)
    for p, t in zip(pred, truth):
        mat[lookup[int(t)], lookup[int(p)]] += 1
    return mat


def save_figures(summary, class_names):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    methods = [m for m in ("SVM_T", "HFA", "SHFA") if m in summary["accuracy"]]
    colors = {"SVM_T": "#9a7b4f", "HFA": "#0f6e6b", "SHFA": "#9a3412"}

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    xs = np.arange(len(methods))
    vals = [100 * summary["accuracy"][m] for m in methods]
    bars = ax.bar(xs, vals, color=[colors[m] for m in methods], width=0.62)
    ax.set_xticks(xs, methods)
    ax.set_ylabel("Accuracy on the 405 DSLR test images (%)")
    ax.set_ylim(0, max(vals) * 1.25 if vals else 1)
    ax.set_title("Amazon to DSLR, one official split, 31 categories")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8, f"{val:.2f}", ha="center", va="bottom")
    if "HFA" in summary["accuracy"]:
        ax.axhline(56.7901, color="#0f6e6b", ls="--", lw=0.8, label="Author HFA check 56.7901")
    if "SHFA" in summary["accuracy"]:
        ax.axhline(57.2842, color="#9a3412", ls=":", lw=0.8, label="Author SHFA check 57.2842")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "overall_accuracy.png", dpi=160)
    plt.close(fig)

    if all(m in summary["per_class"] for m in methods):
        fig, ax = plt.subplots(figsize=(12.4, 4.6))
        x = np.arange(len(class_names))
        width = 0.25
        for i, m in enumerate(methods):
            ax.bar(x + (i - (len(methods) - 1) / 2) * width,
                   [100 * v for v in summary["per_class"][m]],
                   width=width, label=m, color=colors[m])
        ax.set_xticks(x, class_names, rotation=70, ha="right", fontsize=8)
        ax.set_ylabel("Per-class accuracy (%)")
        ax.set_title("Per-class accuracy on the held-out DSLR images")
        ax.legend(frameon=False, ncol=len(methods))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(fig_dir / "per_class_accuracy.png", dpi=160)
        plt.close(fig)

    for m in methods:
        mat = np.asarray(summary["confusion"][m])
        fig, ax = plt.subplots(figsize=(8.2, 7.2))
        im = ax.imshow(mat, cmap="YlGnBu")
        ax.set_xticks(range(len(class_names)), class_names, rotation=70, ha="right", fontsize=7)
        ax.set_yticks(range(len(class_names)), class_names, fontsize=7)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{m} confusion matrix")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(fig_dir / f"confusion_{m.lower()}.png", dpi=150)
        plt.close(fig)

    # Outer-loop objective for the class whose trace length is the median.
    traces = summary.get("objectives", {})
    if traces:
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        for m, series in traces.items():
            # series: list over classes of objective lists. Plot the class median final value.
            finals = [s[-1] for s in series if s]
            if not finals:
                continue
            order = np.argsort(finals)
            chosen = series[int(order[len(order) // 2])]
            ax.plot(np.arange(1, len(chosen) + 1), chosen, marker="o", ms=3.5, label=m, color=colors.get(m, "#333"))
        ax.set_xlabel("Cutting-plane iteration")
        ax.set_ylabel("MKL objective of the median class")
        ax.set_title("Outer-loop objective")
        ax.legend(frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(fig_dir / "convergence.png", dpi=160)
        plt.close(fig)

    counts = summary["test_counts"]
    fig, ax = plt.subplots(figsize=(12.2, 4.2))
    ax.bar(np.arange(len(class_names)), counts, color="#315c7a")
    ax.set_xticks(np.arange(len(class_names)), class_names, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("Test images")
    ax.set_title("DSLR test-set size by category (overall accuracy weights these unequally)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "test_support.png", dpi=160)
    plt.close(fig)


def run(methods: list[str], max_classes: int = 0):
    OUT.mkdir(parents=True, exist_ok=True)
    print("Loading official split...", flush=True)
    data = load_split()
    classes = list(range(1, len(data["categories"]) + 1))
    names = data["categories"]
    if max_classes:
        classes = classes[:max_classes]
        names = names[:max_classes]
        print(f"Smoke test: first {max_classes} classes only", flush=True)
    print(
        f"classes={len(names)} source={data['source_x'].shape} "
        f"labeled_target={data['target_x'].shape} test={data['test_x'].shape}",
        flush=True,
    )

    decisions = {}
    objectives = {}
    timing = {}
    kernel_info = {}

    if "SVM_T" in methods:
        print("Training SVM_T (labeled target only)...", flush=True)
        t0 = time.perf_counter()
        k_train, gamma = gaussian_kernel(data["target_x"])
        k_test, _ = gaussian_kernel(data["test_x"], data["target_x"], gamma=gamma)
        scores = np.zeros((data["test_y"].shape[0], len(classes)))
        for i, c in enumerate(classes):
            scores[:, i] = train_svmt_one(binary_labels(data["target_y"], c), k_train, k_test)
            print(f"  SVM_T class {c}/{len(classes)}", flush=True)
        decisions["SVM_T"] = scores
        timing["SVM_T"] = time.perf_counter() - t0
        kernel_info["SVM_T_gamma"] = gamma

    if "HFA" in methods:
        print("Preparing HFA kernels (matrix square roots)...", flush=True)
        t0 = time.perf_counter()
        pack = prepare_hfa(data)
        prep = time.perf_counter() - t0
        print(f"  HFA kernel prep {prep:.1f}s  gamma_t={pack['gamma_t']:.6g}", flush=True)
        scores = np.zeros((data["test_y"].shape[0], len(classes)))
        traces = []
        t1 = time.perf_counter()
        for i, c in enumerate(classes):
            tc = time.perf_counter()
            with threadpool_limits(limits=1):
                model = train_hfa_one(
                    binary_labels(data["source_y"], c),
                    binary_labels(data["target_y"], c),
                    pack["k_root"],
                    pack["k_root_sq"],
                    lam=100.0,
                    degree=1.0,
                    max_outer=50,
                    tau=1e-3,
                )
            scores[:, i] = predict_hfa(
                pack["k_test"], pack["l_t_inv"], model["H"], pack["k_root"],
                model["y_alpha"], model["rho"], pack["n_s"],
            )
            traces.append(model["objectives"])
            objs = model["objectives"]
            print(
                f"  HFA {names[i]:20s} outer={len(objs):2d} "
                f"kernels={model['n_kernels']:2d} "
                f"obj {objs[0]:.6g} -> {objs[-1]:.6g} {time.perf_counter() - tc:.1f}s",
                flush=True,
            )
        decisions["HFA"] = scores
        objectives["HFA"] = traces
        timing["HFA"] = time.perf_counter() - t1
        timing["HFA_prep"] = prep
        kernel_info["HFA_gamma_s"] = pack["gamma_s"]
        kernel_info["HFA_gamma_t"] = pack["gamma_t"]

    if "SHFA" in methods:
        print("Preparing SHFA kernels (matrix square roots)...", flush=True)
        t0 = time.perf_counter()
        pack = prepare_shfa(data)
        prep = time.perf_counter() - t0
        print(f"  SHFA kernel prep {prep:.1f}s", flush=True)
        n = pack["k_mat"].shape[0]
        n_s = pack["n_s"]
        n_l = pack["n_l"]
        weight = np.concatenate([
            np.full(n_s, 1.0),
            np.full(n_l, 1.0),
            np.full(n - n_s - n_l, 1.0 / 1e-3),
        ])
        scores = np.zeros((data["test_y"].shape[0], len(classes)))
        traces = []
        t1 = time.perf_counter()
        for i, c in enumerate(classes):
            tc = time.perf_counter()
            ratio = float(np.mean(data["test_y"] == c))
            with threadpool_limits(limits=1):
                model = train_shfa_one(
                    binary_labels(data["source_y"], c),
                    binary_labels(data["target_y"], c),
                    pack["k_mat"],
                    pack["k_root"],
                    pack["aug"],
                    weight,
                    lam=100.0,
                    degree=1.5,
                    ratio=ratio,
                    max_outer=50,
                    tau=1e-3,
                )
            scores[:, i] = predict_shfa(pack["k_test"], model, pack["k_root"], pack["l_t_inv"], n_s)
            traces.append(model["objectives"])
            print(
                f"  SHFA {names[i]:20s} outer={len(model['objectives']):2d} "
                f"kernels={model['n_kernels']:2d} ratio={ratio:.3f} {time.perf_counter() - tc:.1f}s",
                flush=True,
            )
        decisions["SHFA"] = scores
        objectives["SHFA"] = traces
        timing["SHFA"] = time.perf_counter() - t1
        timing["SHFA_prep"] = prep

    summary = {
        "categories": names,
        "counts": {
            "source": int(data["source_y"].shape[0]),
            "target_labeled": int(data["target_y"].shape[0]),
            "test": int(data["test_y"].shape[0]),
            "source_dim": int(data["source_x"].shape[0]),
            "target_dim": int(data["target_x"].shape[0]),
        },
        "test_counts": [int(np.sum(data["test_y"] == c)) for c in classes],
        "accuracy": {},
        "correct": {},
        "per_class": {},
        "confusion": {},
        "timing_sec": timing,
        "kernel": kernel_info,
        "objectives": objectives,
        "author_check": {"HFA": 0.567901, "SHFA": 0.572842},
    }
    if max_classes:
        print("Smoke test finished; not writing official_results.json", flush=True)
        for method, scores in decisions.items():
            print(method, "score range", float(np.min(scores)), float(np.max(scores)), flush=True)
        return summary
    predictions = {"test_y": data["test_y"].tolist()}
    class_ids = np.asarray(classes)
    for method, scores in decisions.items():
        pred = class_ids[np.argmax(scores, axis=1)]
        summary["accuracy"][method] = accuracy(pred, data["test_y"])
        summary["correct"][method] = int(np.sum(pred == data["test_y"]))
        summary["per_class"][method] = per_class_accuracy(pred, data["test_y"], classes)
        summary["confusion"][method] = confusion(pred, data["test_y"], classes).tolist()
        predictions[method] = pred.tolist()
        print(
            f"{method}: {summary['correct'][method]}/{summary['counts']['test']} "
            f"= {summary['accuracy'][method]:.6f}",
            flush=True,
        )

    if max_classes:
        print("Smoke test finished; not writing official_results.json", flush=True)
        return summary
    serializable = dict(summary)
    (OUT / "official_results.json").write_text(json.dumps(serializable), encoding="utf-8")
    np.savez(
        OUT / "official_predictions.npz",
        test_y=data["test_y"],
        **{m: np.asarray(predictions[m]) for m in decisions},
    )
    save_figures(summary, names)
    print("Wrote", OUT / "official_results.json", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="SVM_T,HFA,SHFA")
    parser.add_argument("--max-classes", type=int, default=0)
    args = parser.parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    run(methods, max_classes=args.max_classes)


if __name__ == "__main__":
    main()
