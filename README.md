# FortNN_Forward_Propagation

A flexible Fortran program that reads weights and biases from files (saved from any program, e.g., Python - Tensorflow or Pytorch, etc.) and performs a forward propagation. Inspired by: https://github.com/ketetefid/FortNN

# To compile: 

$ gfortran forward_pass.f90 fortfp.f90 -o forward_pass.exe

# How to save weights and biases from a trained neural network:

Current example demonstrates a neural network with architecture (can be easily modified in the main program):
6 layers with: 9 input neurons, 100-100-100-100 (hidden layers), and 6 output nerons;
activations - leaky relu; no activation in the last layer.

Of course, the architecture should match one that obtained after training:
- weights are saved in wb/w_0 (for each layer weights should be stored in 1 separate line).
- biases are saved in wb/b_0 (for each layer biases should be stored in 1 separate line).

# Contributions

Bug submits, improvements, and suggestions are welcomed.
