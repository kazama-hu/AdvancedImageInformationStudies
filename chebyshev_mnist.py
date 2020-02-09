from chebyshev import chebyshev_basis
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform
from scipy.fftpack import fftn, fftshift
from torchvision import datasets
from scipy.special import chebyu, eval_chebyu


def chebyshev_mnist(K):
    dataset = datasets.MNIST("./data", train=False, download=True)

    images = dataset.data.numpy() / 255.0
    X = images[0].astype(np.float32)

    # K次第二種Chebyshev多項式
    var = eval_chebyu(K, X)  # 値を評価

    return X, var


if __name__ == "__main__":
    fig = plt.figure(figsize=(14, 7))

    for K, sub_plot in zip(range(10, 100 + 1, 10), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
        X, var = chebyshev_mnist(K)

        ax = fig.add_subplot(2, 5, sub_plot)
        ax.imshow(var)
        ax.set_title(f"K = {K}")

    plt.rcParams["font.size"] = 34
    plt.tight_layout()

    # Show
    plt.savefig(
        f"./figures/chebyshev_mnist/chebyshev_mnist.png", dpi=300, transparent=True
    )
