from setuptools import setup, Command
import os


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info')


setup(
    name='rockyraccoon',
    version='0.1',
    packages=['rockyraccoon', 'rockyraccoon.examples', 'rockyraccoon.utils', 'rockyraccoon.model'],
    license='MIT License',
    author='Roeland Wiersema',
    package_dir={'pennylane_qllh': 'rockyraccoon'},
    description='A python framework for minimizing the quantum log-likelihood using PennyLane and TensorFlow.',
    cmdclass={
        'clean': CleanCommand,
    }
)