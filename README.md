# Spike-and-Slab-RNN

Code for paper [Ensemble perspective for understanding temporal credit assignment]([[2102.03740\] Ensemble perspective for understanding temporal credit assignment (arxiv.org)](https://arxiv.org/abs/2102.03740))(arXiv:2102.03740). Here, we propose a recurrent neural network structure that each individual connection in the recurrent computation is modeled by a spike and slab distribution, rather than a precise weight value. We also derive the mean-field algorithm to train the network at the ensemble level. The method is applied to classify handwritten digits when pixels are read in sequence, and to the multisensory integration task that is a fundamental cognitive function of animals. 

# Requirements

Python 3.8.5



# Acknowledgement

- [THE MNIST DATABASE of handwritten digits](http://yann.lecun.com/exdb/mnist/)



# Some Instructions

- The code of the paper is divided into different sections. Among them, our network structure is placed in the model.py file in each section.
- The result data is too large to upload, please contact me if you need it.
- Please contact me if you have any questions about this code. My email: zouwx5@mail2.sysu.edu.cn



# Citation

This code is the product of work carried out by the group of [PMI lab, Sun Yat-sen University](https://www.labxing.com/hphuang2018). If the code helps, consider giving us a shout-out in your publications.