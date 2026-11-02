"""Educational linear HFA feature augmentation with a NumPy max-margin solver.

This mirrors Eq. (1) in Li et al. (2014):
  phi_s(x) = [P x, x, 0] and phi_t(x) = [Q x, 0, x].
It alternates between one-vs-rest hinge-loss classifiers and P/Q updates. The
published implementation in official_shfa_release/ should be used for the
exact kernelized convex-MKL HFA/SHFA algorithms (see reproduce_official.py
and ../matlab_implementation/official_shfa_release/).
"""
from __future__ import annotations
import numpy as np


class LinearHFA:
    def __init__(self, common_dim=8, epochs=24, svm_epochs=12, lr=0.06,
                 reg=2e-3, random_state=7):
        self.common_dim, self.epochs, self.svm_epochs = common_dim, epochs, svm_epochs
        self.lr, self.reg = lr, reg
        self.rng = np.random.default_rng(random_state)

    @staticmethod
    def _l2_rows(x):
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

    def _features(self, xs, xt):
        ns, nt = len(xs), len(xt)
        z_s = np.zeros((ns, self.dt_))
        z_t = np.zeros((nt, self.ds_))
        return np.vstack((np.hstack((xs @ self.P_.T, xs, z_s)),
                          np.hstack((xt @ self.Q_.T, z_t, xt))))

    def _fit_ovr(self, x, y, classes):
        w = np.zeros((len(classes), x.shape[1] + 1))
        xb = np.c_[x, np.ones(len(x))]
        for ci, label in enumerate(classes):
            yy = np.where(y == label, 1.0, -1.0)
            order = np.arange(len(y))
            for _ in range(self.svm_epochs):
                self.rng.shuffle(order)
                for i in order:
                    margin = yy[i] * (xb[i] @ w[ci])
                    w[ci, :-1] *= (1 - self.lr * self.reg)
                    if margin < 1:
                        w[ci] += self.lr * yy[i] * xb[i]
        return w

    def fit(self, source_x, source_y, target_x, target_y):
        """Fit from labeled source and labeled target samples (rows are samples)."""
        xs, xt = self._l2_rows(np.asarray(source_x, float)), self._l2_rows(np.asarray(target_x, float))
        ys, yt = np.asarray(source_y), np.asarray(target_y)
        self.ds_, self.dt_ = xs.shape[1], xt.shape[1]
        self.classes_ = np.unique(np.r_[ys, yt])
        self.P_ = self.rng.normal(0, .05, (self.common_dim, self.ds_))
        self.Q_ = self.rng.normal(0, .05, (self.common_dim, self.dt_))
        y = np.r_[ys, yt]
        for _ in range(self.epochs):
            x = self._features(xs, xt)
            self.W_ = self._fit_ovr(x, y, self.classes_)
            # Gradient of hinge loss through the shared (P/Q) block.
            grad_p, grad_q = self.reg * self.P_, self.reg * self.Q_
            scores = np.c_[x, np.ones(len(x))] @ self.W_.T
            for ci, label in enumerate(self.classes_):
                sign = np.where(y == label, 1.0, -1.0)
                bad = sign * scores[:, ci] < 1
                wc = self.W_[ci, :self.common_dim]
                for row in np.where(bad[:len(xs)])[0]:
                    grad_p -= sign[row] * np.outer(wc, xs[row])
                for row in np.where(bad[len(xs):])[0]:
                    grad_q -= sign[len(xs) + row] * np.outer(wc, xt[row])
            scale = max(len(y), 1)
            self.P_ -= self.lr * grad_p / scale
            self.Q_ -= self.lr * grad_q / scale
        self.W_ = self._fit_ovr(self._features(xs, xt), y, self.classes_)
        return self

    def predict_target(self, target_x):
        xt = self._l2_rows(np.asarray(target_x, float))
        x = np.hstack((xt @ self.Q_.T, np.zeros((len(xt), self.ds_)), xt))
        scores = np.c_[x, np.ones(len(x))] @ self.W_.T
        return self.classes_[np.argmax(scores, axis=1)]
