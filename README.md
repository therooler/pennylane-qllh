# Quantum Machine Learning Library for Quantum log-likelihood minimization (QMLQLLM)

A quantum machine learning framework for minimizing the quantum log-likelihood. In line with PennyLane, Strawberry Fields and Blackbird
my working title for this framework is Rocky Raccoon.

This project is far from finished, but the most important code is there: The `QLModel` and `QMLWrapper` classes are the core of this project and seem to work fine for now.

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

 

# Whitepaper

The research behind this project can be found in the [whitepaper](https://github.com/therooler/pennylane-qllh/blob/master/docs/pennylane_qllh.pdf) (work in progress). 
The article about the quantum log-likelihood can be found on [arXiv](https://arxiv.org/abs/1905.06728) and is 
published in [Physical Review A](http://doi.org/10.1103/PhysRevA.100.020301).

# Documentation

The code documentation can be found [here]( https://therooler.github.io/pennylane-qllh/). The Docs are generated automatically
with [pdoc3](https://pypi.org/project/pdoc3/). Code is formatted according to PEP8 standards using 
[Black](https://black.readthedocs.io/en/stable/).

# Constructing your own model

Explain `QLModel` and `QMLWrapper`.

# Examples

Explain examples from whitepaper.

# Future Work

Parallelize everything.
