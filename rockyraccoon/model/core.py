import numpy as np

import pennylane as qml
from pennylane.ops.qubit import QubitStateVector
import itertools as it

import tensorflow as tf

tf.enable_eager_execution()


class RockyModel(tf.keras.Model):
    """
    QML model template.
    """

    def __init__(self, nclasses: int, device="default.qubit"):
        """
        Initialize the keras model interface.

        Args:
            nclasses: The number of classes in the data, used the determine the required output qubits.
            device: name of Pennylane Device backend.
        """
        super(RockyModel, self).__init__()
        self.req_qub_out = None
        self.req_qub_in = None
        self.device = device
        self.data_dev = None
        self.model_dev = None
        self.nclasses = nclasses
        self.init = False
        self.circuit = None
        self.trainable_vars = []

    def __str__(self):
        return "RockyModel"

    def initialize(self, nfeatures: int):
        """
        Model initialization.

        Args:
            nfeatures: The number of features in X

        """
        self.init = True
        raise NotImplementedError

    def call(self, inputs, observable):
        """
        Given some obsersable, we calculate the output of the model.

        Args:
            inputs: N x d matrix of N samples and d features.
            observable: Hermitian matrix containing an observable

        Returns: N expectation values of the observable

        """

        raise NotImplementedError


class RaccoonWrapper:
    """
    QML model training
    """

    # TODO convert everything to complex128?
    measurements = {
        "sx": tf.constant(np.array([[0, 1], [1, 0]]), dtype=tf.float64),
        "sy": tf.constant(
            np.array([[0, complex(0, -1)], [complex(0, 1), 0]]), dtype=tf.float64
        ),
        "sz": tf.constant(np.array([[1, 0], [0, -1]]), dtype=tf.float64),
        "id": tf.constant(np.array([[1, 0], [0, 1]]), dtype=tf.float64),
    }

    def __init__(self, model: RockyModel):
        """
        Wrapper allows one to minimize the quantum log-likelihood for a given RockyModel

        Args:
            model: The QML model we want to use for learning
        """

        self.Q = None
        self.q_y_x = None
        self.q_x = None
        self.req_measurements = None
        self.model = model
        self.bias = True

        def circuit(x, obs=None):
            QubitStateVector(x, wires=list(range(self.model.req_qub_out)))
            return qml.expval.Hermitian(obs, wires=list(range(self.model.req_qub_out)))

        self.data_circuit = qml.QNode(circuit, device=self.model.data_dev, cache=True)

    def _get_discr_statistics(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Calculate the empirical distribution q(y|x) for data X and labels y.

        Args:
            X: N x d matrix of N samples and d features.
            y: Length N vector with labels

        """
        y = y.flatten()
        assert X.shape[0] == y.shape[0], "Data has {} samples with {} labels".format(
            X.shape[0], X.shape[1]
        )

        classes = np.unique(y)
        nsamples, nfeatures = X.shape

        def _q_y_x(c: int, sample: np.ndarray) -> np.ndarray:
            # find copies of our sample
            _idx = np.where((X == tuple(sample)).all(axis=1))[0]
            # return probablity q(y=1|x)
            return np.sum(y[_idx] == c) / len(_idx)

        def _q_x(sample: np.ndarray) -> np.ndarray:
            # find copies of our sample
            _idx = np.where((X == tuple(sample)).all(axis=1))[0]
            # calcualte emperical probability of sample
            return len(_idx) / X.shape[0]

        # Initialize arrays
        self.q_y_x = np.zeros((nsamples, len(classes)))
        self.q_x = np.zeros((nsamples, 1))
        # Get statistics per class
        for i in range(nsamples):
            for j, cl in enumerate(classes):
                self.q_y_x[i, j] = _q_y_x(cl, X[i, :])
            self.q_x[i] = _q_x(X[i, :])

    def _determine_req_measurements(self, complex=False) -> None:
        """
        Determine which measurements are required for constructing the density matrix.

        Args:
            complex: Boolean that determines whether we care about SigmaY observables.

        Returns: list of requirement measurements.

        """

        measurements = []
        if complex:
            for m in it.product(
                ["sx", "sy", "sz", "id"], repeat=self.model.req_qub_out
            ):
                measurements.append(m)
        else:
            for m in it.product(["sx", "sz", "id"], repeat=self.model.req_qub_out):
                measurements.append(m)
        # remove the idenity measurement
        measurements.remove(tuple("id" for _ in range(self.model.req_qub_out)))
        # create constant tensor containing the required measurements.
        self.req_measurements = []
        for m in measurements:
            obs = 1
            for ob in m:
                obs = np.kron(obs, RaccoonWrapper.measurements[ob])
            self.req_measurements.append(obs)
        self.req_measurements = tf.constant(self.req_measurements, dtype=tf.float64)

    def construct_density_matrix(self, phi: np.ndarray) -> np.ndarray:
        """
        Construct the pure density matrix belonging to a certain wavefunction using a quantum circuit.

        Args:
            phi: Amplitudes of wavefunction (should be normed).

        Returns: A density matrix as ndarray.

        """
        # Fill the full wavefunction with the conditional probabilities q(x|y)
        full_phi = np.zeros((2 ** self.model.req_qub_out))
        full_phi[: len(phi)] = phi

        assert np.isclose(
            sum(np.abs(full_phi) ** 2), 1
        ), "Wavefunction must be normed to 1"
        # obtain the required expectation values to construct the density matrix from the data circuit.
        expval = (
            self.req_measurements
            * tf.map_fn(
                lambda x: self.data_circuit(full_phi, obs=x.numpy()),
                self.req_measurements,
                dtype=tf.float64,
            )[:, tf.newaxis, tf.newaxis]
        )
        # combine the tomography measurements
        eta = tf.reduce_sum(expval, axis=0)
        eta += tf.eye(
            2 ** self.model.req_qub_out, 2 ** self.model.req_qub_out, dtype=tf.float64
        )
        # normalization,
        eta /= tf.trace(eta)
        return eta

    def loss(self, inputs: np.ndarray, eta: tf.Tensor) -> tf.float64:
        """
        Determine the quantum log-likelihood of a batch of inputs, eta is the
        corresponding data density matrix

        Args:
            inputs: Mxd matrix of a batch of M samples with d features
            eta: Data density matrices for each sample so a rank 3 tensor.

        Returns: Scalar with quantum log-likelihood

        """
        assert self.model.init, "Initialize the model before calculating the loss"

        # obtain the required expectation values to construct the density matrix from the data circuit.
        expval = tf.map_fn(
            lambda x: self.model(inputs, x), self.req_measurements, dtype=tf.float64
        )
        expval = tf.transpose(expval)
        rho = tf.einsum("no,oij->nij", expval, self.req_measurements)
        rho += tf.eye(
            2 ** self.model.req_qub_out, 2 ** self.model.req_qub_out, dtype=tf.float64
        )
        # normalization
        rho /= tf.trace(rho)[:, tf.newaxis, tf.newaxis]
        log_rho = self._matrix_log(rho)
        # calculate quantum log-likelihood
        lh = -tf.trace(eta @ log_rho)
        # aggregate loss over all samples
        return tf.reduce_mean(lh)

    def train(self, X: np.ndarray, y: np.ndarray, epsilon=0.1, maxiter=100, tol=0.1):
        """
        Train the QML model with gradient descent.

        Args:
            X: N x d matrix of N samples and d features.
            y: Length N vector with labels.
            epsilon: Learning rate of the optimizer.
            maxiter: Maximum number of iterations before convergence.
            tol: Likelihood tolerance threshhold.

        """
        # add bias to inputs
        if self.model.bias:
            X = self.add_bias(X)

        assert self.model.nclasses == len(
            np.unique(y)
        ), "Model expects at {} classes, when y contains {} classes".format(
            self.model.nclasses, len(np.unique(y))
        )
        # determine the required measurements and get empirical statistics
        self._determine_req_measurements(False)
        self._get_discr_statistics(X, y)
        data_states = tf.map_fn(
            self.construct_density_matrix, np.sqrt(self.q_y_x), dtype=tf.float64
        )
        # multiply with qx (prior)
        data_states *= tf.reshape(self.q_x, (-1, 1, 1))
        # intialize the model
        self.model.initialize(X.shape[1])

        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=epsilon)
        self.lh = []
        # print training info
        print("-------TRAINING-------")
        print(self.model)
        print("{} samples, {} features".format(*X.shape))
        print(
            "{} input qubits, {} output qubits".format(
                self.model.req_qub_in, self.model.req_qub_out
            )
        )
        # training loop
        for i in range(maxiter):
            # calculate and apply gradients
            with tf.GradientTape() as tape:
                loss_value = self.loss(X, data_states)
                grads = tape.gradient(loss_value, self.model.trainable_vars)
            optimizer.apply_gradients(
                zip(grads, self.model.trainable_vars),
                global_step=tf.compat.v1.train.get_or_create_global_step(),
            )
            self.lh.append(float(loss_value))
            if i % 20 == 0:
                print("Loss at step {:03d}: {:.6f}".format(i, loss_value))
                if abs(loss_value - self.loss(X, data_states)) < tol:
                    print("L<{} after {:03d} iterations".format(tol, i))

                    break
        print("Final loss: {:.6f}".format(self.loss(X, data_states)))

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """

        Args:
            inputs: Mxd matrix of a batch of M samples with d features

        Returns: array of size M x nclasses with probabilities

        """

        assert self.model.init, "Initialize the model before predicting data"
        # obtain the required expectation values to construct the density matrix from the data circuit.
        expval = tf.map_fn(
            lambda x: self.model(inputs, x), self.req_measurements, dtype=tf.float64
        )
        expval = tf.transpose(expval)
        # combine the tomography measurements
        rho = tf.einsum("no,oij->nij", expval, self.req_measurements)
        rho += tf.eye(
            2 ** self.model.req_qub_out, 2 ** self.model.req_qub_out, dtype=tf.float64
        )
        # normalization
        rho /= tf.trace(rho)[:, tf.newaxis, tf.newaxis]
        # stack probabilties for their respective classes
        return tf.stack(
            [tf.to_float(rho[:, i, i]) for i in range(rho.shape[1])], axis=1
        ).numpy()

    @staticmethod
    def _matrix_log(matrix: tf.Tensor) -> tf.Tensor:
        """
        Calculate matrix log2 through diagonalization of Hermitian matrix  M = U^-1 D U

        Args:
            matrix: N x N matrix

        Returns: N x N matrix

        """
        rx, Ux = tf.linalg.eigh(matrix)
        Ux_inv = tf.linalg.adjoint(Ux)
        rx = tf.cast(
            tf.math.log(tf.clip_by_value(tf.math.real(rx), 1e-13, 1e13)), rx.dtype
        )
        tx = tf.linalg.LinearOperatorDiag(rx).to_dense()
        return tf.matmul(Ux, tf.matmul(tx, Ux_inv))

    @staticmethod
    def add_bias(X: np.ndarray) -> np.ndarray:
        """
        Add a bias to the input data by adding a column of ones.

        Args:
            X: Numpy array of size N x d

        Returns: Numpy array of size N x d+1

        """
        n_samples = X.shape[0]
        return np.hstack([X, np.ones((n_samples, 1))])
