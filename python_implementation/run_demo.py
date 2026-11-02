"""Self-contained heterogeneous-domain demonstration for LinearHFA."""
import numpy as np
from hfa_numpy import LinearHFA


def accuracy(a, b):
    return float(np.mean(np.asarray(a) == np.asarray(b)))


def main():
    rng = np.random.default_rng(22)
    # A shared latent 3-class problem observed through feature spaces of
    # different dimensions. Target labels are deliberately scarce.
    means = np.array([[-1.6, 0.0, .8], [1.4, .3, -.5], [.1, 1.7, .2]])
    latent_s = np.vstack([rng.normal(m, .65, (90, 3)) for m in means])
    y_s = np.repeat(np.arange(3), 90)
    latent_t = np.vstack([rng.normal(m, .65, (18, 3)) for m in means])
    y_t = np.repeat(np.arange(3), 18)
    latent_test = np.vstack([rng.normal(m, .65, (120, 3)) for m in means])
    y_test = np.repeat(np.arange(3), 120)
    a_s, a_t = rng.normal(size=(3, 7)), rng.normal(size=(3, 5))
    xs = latent_s @ a_s + rng.normal(0, .12, (len(latent_s), 7))
    xt = latent_t @ a_t + rng.normal(0, .12, (len(latent_t), 5))
    x_test = latent_test @ a_t + rng.normal(0, .12, (len(latent_test), 5))
    model = LinearHFA(common_dim=5, epochs=18, svm_epochs=8, lr=.05).fit(xs, y_s, xt, y_t)
    print(f"Synthetic target accuracy: {accuracy(model.predict_target(x_test), y_test):.3f}")
    print("For the exact paper experiment, see RUN_OFFICIAL.md / reproduce_official.py.")


if __name__ == '__main__':
    main()
