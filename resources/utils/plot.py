import matplotlib.pyplot as plt
import numpy as np

from resources.model.core import QMLWrapper

from typing import List


def plot_qml_landscape_binary(
    X: np.ndarray, y: np.ndarray, wrapper: QMLWrapper, cmap="viridis", title=""
):
    """
    Plot the separation boundaries in the 2D input space.

    Args:
        X: N x d matrix of N samples and d features.
        y: Length N vector with labels.
        wrapper: The QMLWrapper we used for learning
        cmap: String with name of matplotlib colormap, see MPL docs
        title: String with title of the figure

    """

    if wrapper.model.bias:
        X = wrapper.add_bias(X)

    class_0, class_1 = np.unique(y)
    plt.rc("font", size=15)
    cmap = plt.cm.get_cmap(cmap)
    blue = cmap(0.0)
    red = cmap(1.0)
    h = 25
    max_grid = 2
    x_min, x_max = X[:, 0].min() - max_grid, X[:, 0].max() + max_grid
    y_min, y_max = X[:, 1].min() - max_grid, X[:, 1].max() + max_grid
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, h), np.linspace(y_min, y_max, h))
    if wrapper.bias:
        z = wrapper.predict(np.c_[xx.ravel(), yy.ravel(), np.ones_like(yy).ravel()])
    else:
        z = wrapper.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z[:, 1] - z[:, 0]
    z = z.reshape(xx.shape)
    fig, ax = plt.subplots()
    ax.contour(xx, yy, z, cmap=cmap)
    # Plot also the training points
    y = y.flatten()
    np.random.seed(123)
    spread = 0.3
    ax.scatter(
        X[(y == class_0), 0]
        + np.random.uniform(-spread, spread, np.sum((y == class_0))),
        X[(y == class_0), 1]
        + np.random.uniform(-spread, spread, np.sum((y == class_0))),
        marker=".",
        c=np.array([blue]),
        label="-1",
        s=25,
    )
    ax.scatter(
        X[(y == class_1), 0]
        + np.random.uniform(-spread, spread, np.sum((y == class_1))),
        X[(y == class_1), 1]
        + np.random.uniform(-spread, spread, np.sum((y == class_1))),
        marker="x",
        c=np.array([red]),
        label="+1",
        s=25,
    )

    ax.set_xlabel("$x_0$")
    ax.set_ylabel("$x_1$")
    ax.set_title(title)
    ax.legend()

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
    m = plt.cm.ScalarMappable(cmap=cmap)
    m.set_array(np.linspace(-1, 1, 11))
    plt.colorbar(m, cax=cbar_ax, boundaries=np.linspace(-1, 1, 11))

    plt.show()


def plot_lh(wrapper: QMLWrapper, cmap="viridis", title=""):
    """

    Args:
        wrapper: The QMLWrapper we used for learning

    """
    cmap = plt.cm.get_cmap(cmap)

    fig, ax = plt.subplots(1, 1)
    ax.plot(wrapper.lh, c=cmap(0.2))
    ax.set_xlabel("number of iterations")
    ax.set_ylabel("Likelihood $\mathcal{L}$")
    ax.set_title(title)
    plt.show()


def plot_qml_landscape_multiclass(
    X: np.ndarray,
    y: np.ndarray,
    wrapper: QMLWrapper,
    subplot_grid: List[int],
    cmap="viridis",
    title="",
):
    """
    Plot the separation boundaries of a multiclass qml model in 2D space.

    Args:
        X: N x d matrix of N samples and d features.
        y: Length N vector with labels.
        wrapper: The QMLWrapper we used for learning
        subplot_grid: List that specifies the grid of the subplots
        cmap: Name of MPL colormap
        title: Title of the figure

    """

    if wrapper.model.bias:
        X = wrapper.add_bias(X)

    assert (
        len(subplot_grid) == 2
    ), "Expected subplot_grid to have length 2, but go iterable with length {}".format(
        len(subplot_grid)
    )
    labels = np.unique(y)
    num_classes = len(np.unique(y))
    assert num_classes > 2, "Only {} classes found, use binary plotter instead".format(
        num_classes
    )
    assert (
        np.product(subplot_grid) == num_classes
    ), "wrong grid size {} for {} classes".format(subplot_grid, num_classes)
    plt.rc("font", size=15)
    cmap = plt.cm.get_cmap(cmap)
    clrs = [cmap(0.0), cmap(0.5), cmap(1.0)]
    h = 25
    max_grid = 2
    spread = 0.2
    y = y.flatten()
    x_min, x_max = X[:, 0].min() - max_grid, X[:, 0].max() + max_grid
    y_min, y_max = X[:, 1].min() - max_grid, X[:, 1].max() + max_grid
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, h), np.linspace(y_min, y_max, h))
    if wrapper.bias:
        z = wrapper.predict(np.c_[xx.ravel(), yy.ravel(), np.ones_like(yy).ravel()])
    else:
        z = wrapper.predict(np.c_[xx.ravel(), yy.ravel()])
    sections = np.zeros_like(z)
    idx = np.argmax(z, axis=1)
    sections[np.arange(len(idx)), np.argmax(z, axis=1)] = 1
    idx = idx.reshape(xx.shape)

    markers = [".", "*", "x", "v", "s", "1"]
    z = [el.reshape(xx.shape) for el in z.T]
    fig, axs = plt.subplots(*subplot_grid)

    if subplot_grid[0] == 1:
        axs = axs.reshape(1, -1)
    if subplot_grid[1] == 1:
        axs = axs.reshape(-1, 1)
    for i, ax in enumerate(axs.flatten()):
        for j, label in enumerate(labels):
            np.random.seed(2342)
            if j != i:
                ax.scatter(
                    X[(y == label), 0]
                    + np.random.uniform(-spread, spread, np.sum((y == label))),
                    X[(y == label), 1]
                    + np.random.uniform(-spread, spread, np.sum((y == label))),
                    c="gray",
                    label=label,
                    marker=markers[label],
                    s=50,
                )
            else:
                ax.scatter(
                    X[(y == label), 0]
                    + np.random.uniform(-spread, spread, np.sum((y == label))),
                    X[(y == label), 1]
                    + np.random.uniform(-spread, spread, np.sum((y == label))),
                    c=np.array([clrs[2]]),
                    marker=markers[label],
                    label=label,
                    s=50,
                )
                polygon = np.zeros_like(idx)
                polygon[idx == i] = 1

                ax.contourf(
                    xx,
                    yy,
                    polygon,
                    alpha=0.1,
                    cmap=cmap,
                    levels=[0.5, 1],
                    vmin=0,
                    vmax=1,
                )

        ax.legend(prop={"size": 6})
        cs = ax.contour(xx, yy, z[i], cmap=cmap, vmin=0, vmax=1)
        ax.clabel(cs, inline=True, inline_spacing=2, fontsize=10)

    ylabels = list((x, 0) for x in range(subplot_grid[0]))
    xlabels = list((subplot_grid[0] - 1, x) for x in range(subplot_grid[1]))
    for ax_id in ylabels:
        axs[ax_id].set_ylabel(r"$x_1$")
    for ax_id in xlabels:
        axs[ax_id].set_xlabel(r"$x_0$")

    for ix, iy in np.ndindex(axs.shape):
        if (ix, iy) not in xlabels:
            axs[ix, iy].set_xticks([])
        if (ix, iy) not in ylabels:
            axs[ix, iy].set_yticks([])
    fig.suptitle(title)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
    m = plt.cm.ScalarMappable(cmap=cmap)
    m.set_array(np.linspace(0, 1, 11))
    plt.colorbar(m, cax=cbar_ax, boundaries=np.linspace(0, 1, 11))

    plt.show()
