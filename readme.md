## Analytic insights into structure and rank of neural network Hessian maps


This is the code for the paper „Analytic insights into structure and rank of neural network Hessian maps“. The code is largely based on JAX and the library stax. 
 

### Requirements 

The following packages need to be installed to run all of our code:

	- numpy 1.16.2
	- pandas 1.1.5
	- seaborn 0.11.2
	- jax 0.1.77
	- jaxlib 0.1.56
	- torch 1.8.1
	- torchvision 0.9.1
	

### Juypter Notebook for Hessian Rank

We provide a Jupyter notebook to perform experiments, 

                                hessian_rank.ipynb

comparing the empirical rank and our theoretical predictions. Along the computations, we outline the basic theoretical components of our paper and put them into correspondence with the code.  


### Basic Rank Experiments 

If you prefer not to use Jupyter notebooks, we provide the same functionality in the standard python file 

                                    hessian_rank.py 

You can specify several hyperparameters at the start of the file, including sample size, dataset, loss function, initialization scheme, number of neurons etc. Running the file prints the empirical ranks of the loss Hessian, outer Hessian and functional Hessian along with the corresponding predictions from our paper.

						

### Rank versus Sample Size Plots

In order to obtain the plots visualizing rank as a function of the sample size, run the following commands:


    python3 rank_vs_sample_size.py --units 10,10 --loss mse --dataset MNIST --batch_size 10 --init glorot --dim 64 --K 10


    python3 plotting.py --units 10,10 --loss mse --dataset MNIST --task samplesize --dim 64 --K 10


This will produce a visualization of the rank of the loss, outer and functional Hessian as a function of the sample size. Here we use the dataset MNIST, mean-squared loss, an architecture with hidden units 10x10, with 10 classes and Glorot initialization. The plot  will be saved in a directory, here this would be saved to results/store/samplesize/mse/MNIST/64x10x10x10.

### Rank versus Width Plots 

In order to obtain the plots visualizing rank as a function of the width of the network run the following commands:


    python3 rank_vs_width.py --loss mse --dataset MNIST --batch_size 10 --init glorot --dim 64 --K 10

    python3 plotting.py --loss mse --dataset MNIST --task width --dim 64 --K 10


This will produce a visualization of the rank of the loss, outer and functional Hessian as a function of width. The network architecture has hidden layer sizes width x width. Here we use the dataset MNIST, mean-squared loss with 10 classes and Glorot initialization. The plot  will be saved in a directory, here this would be saved to results/store/width/mse/MNIST.


### Rank versus Depth Plots 

In order to obtain the plots visualizing rank as a function of the depth of the network, run the following commands:


    python3 rank_vs_depth.py --loss mse --dataset MNIST --batch_size 10 --init glorot --dim 16 --K 10 --width 25

    python3 plotting.py --loss mse --dataset MNIST --task depth --dim 16 --K 10 --width 25


This will produce a visualization of the rank of the loss, outer and functional Hessian as a function of the depth. The network architecture has hidden layer sizes width of varying depth. Here we use the dataset MNIST, mean-squared loss with 10 classes and Glorot initialization. The plot  will be saved in a directory, here this would be saved to results/store/width/mse/MNIST/25.

