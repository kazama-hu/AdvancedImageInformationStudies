import matplotlib.pyplot as plt
import numpy as np
import skimage.transform
from scipy.fftpack import fftn, fftshift
from torchvision import datasets


def fourier_mnist():
    dataset = datasets.MNIST("./data", train=False, download=True)

    images = dataset.data.numpy() / 255.0
    X = images[0].astype(np.float32)

    # Define Gabor Filter
    N = X.shape[0]

    x, y = np.meshgrid(np.arange(-float(N), N), np.arange(-float(N), N))

    y = skimage.transform.rotate(y, 35)
    x2 = skimage.transform.rotate(x, -35)

    sigma = 0.75 * np.pi
    lmbda = 1.5 * sigma
    gamma = 1.3

    gabor = np.exp(-(x ** 2 + gamma * y ** 2) / (2 * sigma ** 2)) * np.cos(
        2 * np.pi * x2 / lmbda
    )

    # Create adjacency matrix with gabor filter
    A = np.zeros((N ** 2, N ** 2))
    for i in range(N):
        for j in range(N):
            A[i * N + j, :] = gabor[N - i : N - i + N, N - j : N - j + N].flatten()

    # Calculate FFT
    fft = np.zeros((N, N, 2), dtype=np.float32)

    fft[:, :, 0] = np.abs(fftshift(fftn(X))).squeeze()
    fft[:, :, 1] = np.angle(fftshift(fftn(X))).squeeze()

    # Plot
    plt.figure(figsize=(15, 12))
    plt.subplot(221)
    plt.title("Input Image")
    plt.imshow(X, cmap="gray")

    plt.subplot(222)
    ax = plt.imshow(A)
    plt.title("Adjacency Matrix")
    plt.colorbar(ax, fraction=0.046, pad=0.04)

    plt.subplot(223)
    ax = plt.imshow(fft[:, :, 0])
    plt.title("Frequency Space")
    plt.colorbar(ax, fraction=0.046, pad=0.04)

    plt.subplot(224)
    ax = plt.imshow(fft[:, :, 1])
    plt.title("Transformed Graph Laplacian")
    plt.colorbar(ax, fraction=0.046, pad=0.04)

    plt.tight_layout()

    # Show
    plt.savefig("./figures/fourier_mnist/fourier_mnist.png", dpi=300, transparent=True)


if __name__ == "__main__":
    fourier_mnist()
