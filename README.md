![](docs/resources/RR_raccoon_wiersema.jpg)

# Penny Lane and the Quantum Log-Likelihood

A quantum machine learning framework for minimizing the quantum log-likelihood. In line with Penny Lane, 
Strawberry Fields and Blackbird, my working title for this framework is Rocky Raccoon.

This project is far from finished, but the most important code is there: The `RockyModel` and `RaccoonWrapper` classes 
are the core of this project and seem to work fine for now.

# Installing

The following works with Conda 4.6.12. Please use the provided environment.yml. 
If you have a CUDA enabled GPU, consider installing `tensorflow-gpu==1.14.0` instead by 
changing the `environment.yml` file accordingly.

 1. Install Conda: [Quickstart](https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart)

 2. Clone the git:
 
    `git clone https://gitlab.com/rooler/pennylane-qllh.git`

 3. Create a Virtual Environment:

    `conda env create -f environment.yml`

    It is good to know that you can use this to update the env:

    `conda env update -f environment.yml`

    And removing it can be done with:

    `conda remove --name pennylane-qllh --all`
 4. Activate the environment and install the `rockyraccoon` package:
    
    `conda activate pennylane-qllh`
    
    `python setup.py install clean`


# Research

The research behind this project can be found in the [whitepaper](https://github.com/therooler/pennylane-qllh/blob/master/docs/pennylane_qllh.pdf) (work in progress). 
The article about the quantum log-likelihood can be found on [arXiv](https://arxiv.org/abs/1905.06728) and is 
published in [Physical Review A](http://doi.org/10.1103/PhysRevA.100.020301).

# Documentation

The code documentation can be found [here]( https://therooler.github.io/pennylane-qllh/). The Docs are generated automatically
with [pdoc3](https://pypi.org/project/pdoc3/). Code is formatted according to PEP8 standards using 
[Black](https://black.readthedocs.io/en/stable/).

# Constructing your own model

Rocky Raccoon consists of two parts. A `RockyModel` class that serves as a template for the hybrid-quantum model 
we wish to train, and a `RaccoonWrapper` class that minimizes the quantum log-likelihood. Both these classes 
can be found in the `model.core` module. 

Writing your own `RockyModel` means defining a class that inherits from `RockyModel` and overloading the 
methods defined there. If this is done correctly, `RaccoonWrapper` will do the heavy lifting for us. 
Below we will discuss the requirements of a `RockyModel` by looking at the individual methods. In the section after that
we will look at some example models that make use of this template. The 
[whitepaper](https://github.com/therooler/pennylane-qllh/blob/master/docs/pennylane_qllh.pdf) is a good reference to 
help understand the design choices made here.

```python
class RockyModel(tf.keras.Model):
    """
    QML model template.
    """
```

 1.) `RockyModel` inherits from keras Models, so that we have a clear template already for what is required
 for tensorflow to work.

```python
def __init__(self, nclasses: int, device="default.qubit"):
    """
    Initialize the keras model interface.

    Args:
        nclasses: The number of classes in the data, used the determine the required output qubits.
        device: name of Pennylane Device backend.
    """
    super(RockyModel, self).__init__()
        self.req_qub_out = int(np.ceil(np.log2(nclasses)))
        self.req_qub_in = None
        self.device = device
        self.data_dev = None
        self.model_dev = None
        self.nclasses = nclasses
        self.init = False
        self.circuit = None
        self.trainable_vars = []
```

 2.) In order to determine the subsystem that we will measure to construct the density matrix, we need
 to determine beforehand how many classes we want to learn, `nclasses`. With regards to the `device` parameter,
 at the moment Rocky Raccoon only supports the `default.qubit` device. In principle we rely only on the 
 `TFEQnode` Penny Lane interface, but some preliminary tests with the Qiskit backend led to buggy behaviour.
 The variable `self.req_qub_out` is determined by the number of classes in your implementation, since we need to 
 construct an appropriately sized density matrix from the circuit. On the other hand, `self.req_qub_in` can be whatever 
 we want, as long as it is equal to or greater as `self.req_qub_out`. The variables `self.model_dev` and `self.data_dev` 
 contain the Penny Lane device objects used for executing the quantum and data circuits, which as mentioned before only 
 supports `default.qubit` for now. `self.circuit` has to be assigned a `TFEQnode` Penny Lane quantum circuit. In order
 for `RaccoonWrapper` to properly update the gradients, we initalize a list of trainable variables in `self.trainable_vars`.

```python
def __str__(self):
    return 'Gradient Ob-la-descent'
```

 3.) Give your model a name.

```python
def initialize(self, nfeatures: int):
    """
    Model initialization.

    Args:
        nfeatures: The number of features in X

    """
    self.init = True
    raise NotImplementedError
```

 4.) `initialize` is called in `RaccoonWrapper` before training begins. In this method we need to make sure that our 
tensorflow variables are initialized and appended to `self.trainable_vars`, that `self.circuit` is assigned a proper `TFEQnode`
and that `self.init` is set to `True` so that training the model can begin. Since some quantum circuits might want to 
amplitude encode features from the data into the wavefunction, we need to be aware of `nfeatures`.

```python
def call(self, inputs: tf.Tensor, observable: tf.Tensor):
    """
    Given some obsersable, we calculate the output of the model.

    Args:
        inputs: N x d tf.Tensor of N samples and d features.
        observable: tf.Tensor with Hermitian matrix containing an observable

    Returns: N expectation values of the observable

    """

    raise NotImplementedError
```

 5.) Finally, we reach the most important method: `call`. This method is called in the
`RaccoonWrapper` loss function to perform tomography of the model density matrix. Given 
a tensor of inputs and an observable, it should return the expectation value of the
given `observable`. To return this value for each sample, it is recommended to use
 `tf.map_fn`, which performs a parallel map over the batch dimension. 

**Note:** Unfotunately, `tf.map_fn` cannot be run in parallel since the Penny Lane 
`TFQENode` casts the tensors to arrays in order work with all device backends. This
means that `tf.map_fn` is simply a fancy wrapper of a for-loop. 

# Examples

Explain examples from whitepaper.

# Future Work

Parallelize everything.
