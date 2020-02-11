# Experiments With MNIST Using Metaflow

Used Metaflow to run experiments for classification of digits on a MNIST dataset

## Setting it Up 

- ``pip install -r requirements.txt``
- Conda is required to Run this. 
    1. Download Miniconda at https://docs.conda.io/en/latest/miniconda.html
    2. ```conda config --add channels conda-forge```

- export PATH="/Users/valaydave/miniconda3/bin:$PATH" --> Change this to where u install miniconda. U need to run this before executing the Experiments. 


## Experimentation Strategy :

- Run individual Experiments Using ``python Experiments-with-MNIST/hello_mnist.py --environment=conda run --num_training_examples 10000``

## Metaflow Help 

-  ``python Experiments-with-MNIST/hello_mnist.py --environment=conda run --help`` : This will show all the parameters which are currently avaliable to perform experiments on. 

-  ``python Experiments-with-MNIST/hello_mnist.py --environment=conda show`` : This will show the steps of the DAG used for this Experiment. 

## Experiment Analysis

- After running a couple of Experiments successfully running the [experiments_analytics.ipynb](experiments_analytics.ipynb) will help create charts that will help analyse different parameters of the Experiment. 
