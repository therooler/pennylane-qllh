import numpy as np

import pennylane as qml
from pennylane.templates.layers import StronglyEntanglingLayers
from pennylane.interfaces.tfe import TFEQNode

from resources.model.core import QMLModel

import tensorflow as tf
from tensorflow import keras

tf.enable_eager_execution()


class HybridNN(QMLModel):
    """
    QML model that uses a neural network to learn quantum circuit parameters.
    """

    def __init__(self, nclasses, device):
        """
        Initialize the keras model interface.

        Args:
            nclasses: The number of classes in the data, used the determine the required output qubits.
            device: name of Pennylane Device backend.
        """
        super(HybridNN, self).__init__(nclasses, device)
        self.req_qub_out = int(np.ceil(np.log2(nclasses)))
        self.req_qub_in = self.req_qub_out
        self.data_dev = qml.device(device, wires=self.req_qub_out)
        self.device = device
        self.model_dev = None
        self.nn = None
        self.bias = True

    def __str__(self):
        return "Hybrid Neural Network Model"

    def initialize(self, nfeatures: int):
        """
        Model initialization.

        Args:
            nfeatures: The number of features in X

        """

        self.req_qub_in = int(np.ceil(np.log2(nfeatures)))
        if self.req_qub_in < self.req_qub_out:
            self.req_qub_in = self.req_qub_out

        self.model_dev = qml.device(self.device, wires=self.req_qub_in)

        self.init = True
        nparams = self.req_qub_in
        self.nn = keras.Sequential(
            [
                keras.layers.Flatten(input_shape=(nfeatures,), dtype=tf.float64),
                keras.layers.Dense(10, activation=tf.nn.relu, dtype=tf.float64),
                keras.layers.Dense(
                    (1 * nparams * 3), activation=tf.nn.softmax, dtype=tf.float64
                ),
            ]
        )

        def circuit(params, obs):
            StronglyEntanglingLayers(params, list(range(self.req_qub_in)), ranges=[1])
            return qml.expval.Hermitian(obs, wires=list(range(self.req_qub_out)))

        self.circuit = TFEQNode(qml.QNode(circuit, device=self.model_dev, cache=True))
        self.trainable_vars = self.nn.trainable_weights

    def call(self, inputs, observable):
        """
        Given some obsersable, we calculate the output of the model.

        Args:
            inputs: N x d matrix of N samples and d features.
            observable: Hermitian matrix containing an observable

        Returns: N expectation values of the observable

        """

        nn_out = self.nn(inputs)
        theta = 4 * np.pi * nn_out - 2 * np.pi
        theta = tf.reshape(theta, (-1, 1, self.req_qub_in, 3))
        return tf.map_fn(
            lambda x: self.circuit(x, obs=observable), elems=theta, dtype=tf.float64
        )
