import numpy as np

import pennylane as qml
from pennylane.ops.qubit import QubitStateVector
from pennylane.templates.layers import StronglyEntanglingLayers
from pennylane.interfaces.tfe import TFEQNode
from resources.model.core import QMLModel
import tensorflow.contrib.eager as tfe
import tensorflow as tf


class AmplitudeModel(QMLModel):
    """
    QML model that only amplitude encoding.
    """

    def __init__(self, nclasses, device="default.qubit"):
        """
        Initialize the keras model interface.

        Args:
            nclasses: The number of classes in the data, used the determine the required output qubits.
            dev_name: name of Pennylane Device backend.
        """
        super(AmplitudeModel, self).__init__(nclasses, device)
        self.req_qub_out = int(np.ceil(np.log2(nclasses)))
        self.req_qub_in = self.req_qub_out
        self.data_dev = qml.device(device, wires=self.req_qub_out)
        self.device = device
        self.model_dev = None
        self.bias = True

    def __str__(self):
        return "Amplitude Model"

    def initialize(self, nfeatures: int):
        """
        Model initialization.

        Args:
            nfeatures: The number of features in X
            **kwargs: Additional model arguments

        """

        self.req_qub_in = int(np.ceil(np.log2(nfeatures)))
        if self.req_qub_in <= self.req_qub_out:
            self.req_qub_in = 2 * self.req_qub_out

        self.model_dev = qml.device(self.device, wires=self.req_qub_in)

        self.init = True
        nparams = self.req_qub_in
        self.w = tfe.Variable(
            0.1 * (np.random.rand(2, nparams, 3) - 0.5),
            name="weights",
            dtype=tf.float64,
        )

        def circuit(params0, params1, state=None, obs=None):
            QubitStateVector(state, wires=list(range(self.req_qub_out)))
            StronglyEntanglingLayers(
                params0, list(reversed(range(self.req_qub_in))), ranges=[1]
            )
            StronglyEntanglingLayers(params1, list(range(self.req_qub_in)), ranges=[1])
            return qml.expval.Hermitian(obs, wires=list(range(self.req_qub_out)))

        self.circuit = TFEQNode(qml.QNode(circuit, self.model_dev, cache=True))
        self.trainable_vars.append(self.w)

    def call(self, inputs, observable):
        """
        Given some obsersable, we calculate the output of the model.

        Args:
            inputs: N x d matrix of N samples and d features.
            observable: Hermitian matrix containing an observable

        Returns: N expectation values of the observable

        """
        phi = inputs / tf.reshape(
            tf.sqrt(tf.reduce_sum(tf.math.abs(inputs) ** 2, axis=1)), (-1, 1)
        )
        if phi.shape[1] < 2 ** self.req_qub_in:
            phi = tf.concat(
                (
                    phi,
                    tf.zeros(
                        (phi.shape[0], 2 ** self.req_qub_in - phi.shape[1]),
                        dtype=tf.float64,
                    ),
                ),
                axis=1,
            )
        # strange bug with tensors not mapping correctly, unless we do this.
        w = self.w[:, tf.newaxis]
        return tf.map_fn(
            lambda x: self.circuit(w[0], w[1], state=x, obs=observable),
            elems=phi,
            dtype=tf.float64,
        )
