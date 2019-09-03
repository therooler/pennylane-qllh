import numpy as np

import pennylane as qml
from pennylane.ops.qubit import QubitStateVector
from pennylane.interfaces.tfe import TFEQNode

from rockyraccoon.model.core import RockyModel

import tensorflow.contrib.eager as tfe
import tensorflow as tf

tf.enable_eager_execution()


class WaveFunction(RockyModel):
    """
    QML model which encodes a wave function and learns a parameterized quantum circuit.
    """

    def __init__(self, nclasses, device):
        """
        Initialize the keras model interface.


        Args:
            nclasses: The number of classes in the data, used the determine the required output qubits.
            dev: name of Pennylane Device backend.
        """
        super(WaveFunction, self).__init__(nclasses, device)
        self.req_qub_out = int(np.ceil(np.log2(nclasses)))
        self.req_qub_in = 2 * self.req_qub_out
        self.data_dev = qml.device(device, wires=self.req_qub_out)
        self.model_dev = qml.device(device, wires=self.req_qub_in)
        self.device = device
        self.bias = True

    def __str__(self):
        return "Wave Function Model"

    def initialize(self, nfeatures: int):
        """
        Model initialization.

        Args:
            nfeatures: The number of features in X

        """
        self.init = True
        nparams = 2 ** self.req_qub_in

        # parameters for the linear predictor that encodes the amplitude
        self.w = tfe.Variable(
            0.1 * (np.random.rand(nfeatures, nparams) - 0.5),
            name="weights",
            dtype=tf.float64,
        )
        # quantum circuit
        def circuit(params, obs=None):
            QubitStateVector(params, wires=list(range(self.req_qub_in)))
            return qml.expval.Hermitian(obs, wires=list(range(self.req_qub_out)))

        # create a QNode that can be incorporated in an Eager TF model
        self.circuit = TFEQNode(qml.QNode(circuit, device=self.model_dev, cache=True))
        # add the trainable variables to a list for optimization
        self.trainable_vars.append(self.w)

    def call(self, inputs, observable):
        """
        Given some obsersable, we calculate the output of the model.

        Args:
            inputs: N x d matrix of N samples and d features.
            observable: Hermitian matrix containing an observable

        Returns: N expectation values of the observable

        """
        # encode the features into a normed wavefunctions
        phi = inputs @ self.w
        phi /= tf.reshape(
            tf.sqrt(tf.reduce_sum(tf.math.abs(phi) ** 2, axis=1)), (-1, 1)
        )
        return tf.map_fn(
            lambda x: self.circuit(x, obs=observable), elems=phi, dtype=tf.float64
        )
