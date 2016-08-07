# Machine Learning Workshop

[![hive entrance](images/classifiers.png)](http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)

## Aim

The aim of this workshop is to introduce the basics of machine learning with Python and scikit-learn.


## Workshop Structure

1. [Workshop dataset](dataset.md)
2. [Introducing machine learning](intro.md)
3. [Image and feature analysis](images_features.md)
4. [Building a machine learning program](training_testing.md)
5. [Putting it all together and validating](pipelines_validation.md)


## Learning Objectives

The aim of this workshop is to provide participants with the following skills:

* Data manipulation to feed your machine learning algorithm
* Choosing features for your particular problem
* Organising your data
* Clustering your data
* Training a support vector machine classification program
* Building a pipeline from preprocessing to final classification
* Choosing the parameters of your classifier
* Validating your approach


## Dataset

The data used for this workshop come in the form of images of tags on the backs of honeybees which were filmed as part of an experiment into bee behaviour. We are interested in training a machine learning algorithm to automatically classify the different types of tag automatically. More information is available in the [workshop dataset](dataset.md) section of the workshop materials.

![hive entrance](images/entrance.jpg)


## Software Installation

We will be demonstrating using scikit-learn 0.17 on Python 3.5. Our demonstrations will be in a Jupyter notebook. Here are two ways to install these packages. Note that older versions may work, but they have not been tested.

Installing a numerical stack for Python is getting easier all the time, but can still be tricky. We give two recipes below, firstly for pip in a virtual environment and secondly using the Conda package manager.


### Option 1. Pip in a virtual environment

    # Setup a virtual env
    /path/to/python3 -m venv ml_tutorial
    source ml_tutorial/bin/activate

    # Upgrade pip to newest version
    python -m pip install --upgrade pip

    # Install the dependencies
    python -m pip install numpy scipy scikit-learn

Note the pip upgrade is necessary so Linux distributions will use precompiled wheels (see the [manylinux](https://github.com/pypa/manylinux) project for details). With a pip version older than 8.1 numpy and scikit-learn will be installed from source*.

* We really don't recommend this.


### Option 2. Conda/Miniconda

The Anaconda distribution of Python [(located here).](https://www.continuum.io/downloads) includes everything we need in the default installation. You can also create a specific environment for this tutorial following the instructions below.

An alternative is [Miniconda](http://conda.pydata.org/miniconda.html), or installing the Conda package manager through pip, but you will need to create an environment following the instructions below.

    # After installing conda or miniconda
    conda create -n ml_tutorial python=3.5 scikit-learn
    source activate ml_tutorial

    # Activate the Conda environment on Windows:
    activate ml_tutorial


### Optional Elements

The workshop demonstrations will use the Jupyter notebook and Matplotlib for a few plots. If you'd like to work in exactly the same environment you can install these libraries with the below instructions. Note that these are big libraries with many dependencies and we don't recommend trying to set them up on the day using conference wifi.

    # Using pip, in the appropriate virtual environment:
    pip install jupyter matplotlib

    # Inside a Conda environment (these are already installed in the base distribution)
    conda install jupyter matplotlib


## Workshop Dataset

The workshop dataset is located [here](https://raw.githubusercontent.com/SamHames/MLD/master/data.npz). For the convenience of this workshop it is distributed as a preprocessed set of numpy arrays that we can load directly.


## Testing Your Installation

If the following works you are ready for this workshop.

    import numpy as np
    import sklearn

    dataset = np.load('path/to/data.npz')
    images = dataset['images']
    labels = dataset['labels']
