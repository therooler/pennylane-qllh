import numpy as np

import pennylane as qml
from pennylane.templates.layers import StronglyEntanglingLayers
from pennylane.interfaces.tfe import TFEQNode

from rockyraccoon.model.core import RockyModel

import tensorflow as tf
from tensorflow import keras

tf.enable_eager_execution()


class HybridNN(RockyModel):
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
        # we require qubits equal to log2 of the number of features for amplitude encoding
        self.req_qub_in = int(np.ceil(np.log2(nfeatures)))
        # ensure that the number input qubits is twice the number of output qubits
        self.req_qub_in = 2 * self.req_qub_out

        self.model_dev = qml.device(self.device, wires=self.req_qub_in)
        self.init = True
        nparams = self.req_qub_in

        # make a small neural network with 10 hidden neurons
        self.nn = keras.Sequential(
            [
                keras.layers.Flatten(input_shape=(nfeatures,), dtype=tf.float64),
                keras.layers.Dense(10, activation=tf.nn.relu, dtype=tf.float64),
                keras.layers.Dense(
                    (1 * nparams * 3), activation=tf.nn.softmax, dtype=tf.float64
                ),
            ]
        )

        # quantum circuit
        def circuit(params, obs):
            StronglyEntanglingLayers(params, list(range(self.req_qub_in)), ranges=[1])
            return qml.expval.Hermitian(obs, wires=list(range(self.req_qub_out)))

        # create a QNode that can be incorporated in an Eager TF model
        self.circuit = TFEQNode(qml.QNode(circuit, device=self.model_dev, cache=True))
        # add the trainable variables to a list for optimization
        self.trainable_vars = self.nn.trainable_weights

    def call(self, inputs, observable):
        """
        Given some obsersable, we calculate the output of the model.

        Args:
            inputs: N x d matrix of N samples and d features.
            observable: Hermitian matrix containing an observable

        Returns: N expectation values of the observable

        """
        # calculate te neural network output (0,1)
        nn_out = self.nn(inputs)
        # scale to (-2pi, 2pi
        theta = 2 * np.pi * nn_out
        # pass these as parameters to the circuit.
        theta = tf.reshape(theta, (-1, 1, self.req_qub_in, 3))
        return tf.map_fn(
            lambda x: self.circuit(x, obs=observable), elems=theta, dtype=tf.float64
        )
