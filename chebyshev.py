import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torchvision import datasets
import skimage.transform


def chebyshev_basis(K, n_points=100):
    x = np.linspace(-1, 1, n_points)
    # Create basis 'T'
    T = np.zeros((K, len(x)))
    T[0, :] = 1
    T[1, :] = x
    for n in range(1, K - 1):
        T[n + 1, :] = 2 * x * T[n, :] - T[n - 1, :]
    return T


def plot_chebyshev():
    fig = plt.figure(figsize=(14, 4))

    def _plot(basis, title, ax):
        x = np.linspace(-1, 1, basis.shape[1])
        for i in range(basis.shape[0]):
            y = np.zeros(len(x)) + i
            ax.plot(y, x, basis[i, :])
        ax.set_title(title, fontsize=16)

    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")

    _plot(chebyshev_basis(5), f"Chebyshev Basis (K=5)", ax1)
    _plot(chebyshev_basis(10), f"Chebyshev Basis (K=10)", ax2)
    _plot(chebyshev_basis(15), f"Chebyshev Basis (K=15)", ax3)

    plt.tight_layout()

    plt.savefig(
        "./figures/chebyshev_basis/chebyshev_basis.png", dpi=300, transparent=True
    )


if __name__ == "__main__":
    plot_chebyshev()
