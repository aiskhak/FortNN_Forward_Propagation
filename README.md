# FortNN_Forward_Propagation

A flexible Fortran program that reads weights and biases from files (saved from any program, e.g., Python - Tensorflow or Pytorch, etc.) and performs a forward propagation. Inspired by: https://github.com/ketetefid/FortNN

# Please cite our work if useful:

Cheng-Kai Tai, Arsen S. Iskhakov, Nam T. Dinh, and Igor A. Bolotnov
TOWARDS DATA-DRIVEN TURBULENCE MODELING OF MIXED CONVECTION IN ADVANCED REACTORS USING DNS DATA
The 19th International Topical Meeting on Nuclear Reactor Thermal Hydraulics (NURETH-19) Log nr.: 36432
Brussels, Belgium, March 6 - 11, 2022

# To compile: 

```
gfortran forward_pass.f90 fortfp.f90 -o forward_pass.exe
```
replace ```gfortran``` with ```ifort``` or ```mpif77``` if necessary.

# How to save weights and biases from a trained neural network:

Current example demonstrates a neural network with architecture (can be easily modified in the main program):

6 layers with: 9 input neurons, 100-100-100-100 (hidden layers), and 6 output neurons;
activations - leaky relu; no activation in the last layer.

Of course, the architecture should match one that obtained after training:
- weights are saved in wb/w_0 (for each layer weights should be stored in 1 separate line).
- biases are saved in wb/b_0 (for each layer biases should be stored in 1 separate line).

## Example of saving from tensorflow (1.15):

```
vec = []
vec.extend([w_fc1, b_fc1, w_fc2, b_fc2, w_fc3, b_fc3, w_fc4, b_fc4, w_fc5, b_fc5])
def save_w_b(vec):

    # biases
    b1 = tf.compat.v1.layers.flatten(vec[1]).eval()
    b2 = tf.compat.v1.layers.flatten(vec[3]).eval()
    b3 = tf.compat.v1.layers.flatten(vec[5]).eval()
    b4 = tf.compat.v1.layers.flatten(vec[7]).eval()
    b5 = tf.compat.v1.layers.flatten(vec[9]).eval()

    # weights
    w1 = tf.compat.v1.layers.flatten(tf.transpose(vec[0])).eval()   # need to transpose to make it cosistent with pytorch
    w2 = tf.compat.v1.layers.flatten(tf.transpose(vec[2])).eval()
    w3 = tf.compat.v1.layers.flatten(tf.transpose(vec[4])).eval()
    w4 = tf.compat.v1.layers.flatten(tf.transpose(vec[6])).eval()
    w5 = tf.compat.v1.layers.flatten(tf.transpose(vec[8])).eval()

    with open('../wb/b_0', 'wb') as f:
        np.savetxt(f, b1, newline=" ")
        f.write(b'\n')
        np.savetxt(f, b2, newline=" ")
        f.write(b'\n')
        np.savetxt(f, b3, newline=" ")
        f.write(b'\n') 
        np.savetxt(f, b4, newline=" ")
        f.write(b'\n')  
        np.savetxt(f, b5, newline=" ")       

    with open('../wb/w_0', 'wb') as f:
        np.savetxt(f, w1, newline=" ")
        f.write(b'\n')
        np.savetxt(f, w2, newline=" ")
        f.write(b'\n')
        np.savetxt(f, w3, newline=" ")
        f.write(b'\n') 
        np.savetxt(f, w4, newline=" ")
        f.write(b'\n')  
        np.savetxt(f, w5, newline=" ")
```

## Example of saving from pytorch (1.8.1):

```
vec = []
for param in model.parameters():
    vec.append(param.data.view(-1))

    # biases
    b1 = vec[2].numpy()
    b2 = vec[4].numpy()
    b3 = vec[6].numpy()
    b4 = vec[8].numpy()
    b5 = vec[10].numpy()

    # weights
    w1 = vec[1].numpy()
    w2 = vec[3].numpy()
    w3 = vec[5].numpy()
    w4 = vec[7].numpy()
    w5 = vec[9].numpy()

    with open('../wb/b_', 'wb') as f:
        np.savetxt(f, b1, newline=" ")
        f.write(b'\n')
        np.savetxt(f, b2, newline=" ")
        f.write(b'\n')
        np.savetxt(f, b3, newline=" ")
        f.write(b'\n') 
        np.savetxt(f, b4, newline=" ")
        f.write(b'\n')  
        np.savetxt(f, b5, newline=" ")       

    with open('../wb/w_', 'wb') as f:
        np.savetxt(f, w1, newline=" ")
        f.write(b'\n')
        np.savetxt(f, w2, newline=" ")
        f.write(b'\n')
        np.savetxt(f, w3, newline=" ")
        f.write(b'\n') 
        np.savetxt(f, w4, newline=" ")
        f.write(b'\n')  
        np.savetxt(f, w5, newline=" ")
```

# Contributions

Bug submits, improvements, and suggestions are welcomed.
