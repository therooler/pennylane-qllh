import numpy as np
import matplotlib.pyplot as plt

from resources.model.amplitude import AmplitudeModel
from resources.model.core import QMLWrapper
from resources.utils.plot import plot_qml_landscape_multiclass, plot_lh


def multiclass_amplitude():
    """

    Test the amplitude QML model for a simple data set with three classes.

    """

    model = AmplitudeModel(nclasses=3, device="default.qubit")
    wrapper = QMLWrapper(model)

    noise = 0.25

    number_of_copies = 3
    # PERFECT PROBLEM
    X_1 = np.tile([1, 1], (number_of_copies, 1))
    X_2 = np.tile([-1, -1], (number_of_copies, 1))
    X_3 = np.tile([-1, 1], (number_of_copies, 1))
    X_4 = np.tile([1, -1], (number_of_copies, 1))

    Y_1 = np.tile([0], (number_of_copies, 1))
    Y_2 = np.tile([1], (number_of_copies, 1))
    Y_3 = np.tile([2], (number_of_copies, 1))
    Y_4 = np.tile([2], (number_of_copies, 1))

    # Y_3[: int(noise * number_of_copies)] = np.random.randint(
    #     0, 2, (int(noise * number_of_copies), 1)
    # )

    X = np.vstack((X_1, X_2, X_3, X_4)) + 1
    y = np.vstack((Y_1, Y_2, Y_3, Y_4)).flatten()

    wrapper.train(X, y, maxiter=150, epsilon=0.1, tol=1e-6)
    plot_qml_landscape_multiclass(X, y, wrapper, [1, 3])
    plot_lh(wrapper)


if __name__ == "__main__":

    multiclass_amplitude()
