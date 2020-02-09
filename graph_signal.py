import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.mplot3d.art3d as art3d
import skimage.transform
from scipy.fftpack import fftn, fftshift
from torchvision import datasets
from scipy.sparse.linalg import eigsh
import numpy.linalg as LA
from scipy.sparse.csgraph import laplacian


def graph_laplacian(algo="chebyshev"):
    dataset = datasets.MNIST("./data", train=False, download=True)

    images = dataset.data.numpy() / 255.0
    A = images[0].astype(np.float32)

    # グラフラプラシアン
    L = laplacian(A, normed=True)

    # 実対称行列の固有値
    Λ, V = eigsh(L, k=L.shape[0], which="SM")

    # 固有値の分布を可視化
    plt.figure(figsize=(16, 8))
    plt.xlabel("λ")
    x = Λ
    y = [max(v) for v in V.T]
    plt.stem(x, y, use_line_collection=True)
    plt.savefig("./figures/signal_eigen/eigen_dist.png", dpi=300, transparent=True)

    # 固有ベクトルの各成分可視化
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    x = np.linspace(-1.0, 1.0, 28)
    ax.plot(x, Λ, zs=0, zdir="x")
    ax.scatter(Λ, V[0], zs=V[1], zdir="x")

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-3, 3)

    plt.tight_layout()
    plt.savefig("./figures/signal_eigen/eigen_vector.png", dpi=300, transparent=True)


if __name__ == "__main__":
    graph_laplacian()
