import utils
import os
# $ Step 1: Download the Data. 
mnist_folder = 'data/mnist'
utils.safe_mkdir('data')
utils.safe_mkdir(mnist_folder)
utils.download_mnist(mnist_folder)