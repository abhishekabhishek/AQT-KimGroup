# Benchmarking Attention-based Quantum State Tomography

***Abstract*** - Quantum tomography is the process of characterizing a quantum system through a series of measurements. Recent experimental realizations of increasingly large and complex quantum information processing devices have led to a need for resource efficient and accurate tomography techniques. An important problem in quantum tomography is the exponential scaling of resources with system size. In this project, we explore a recent work which applies transformer neural networks to improve resource efficiency of quantum state tomography by modelling correlations in measurement outcomes. We perform empirical evaluation of the framework on different quantum states of interest in quantum information processing, and benchmark its performance against standard techniques in quantum state tomography.

You can read the project report here : [report.pdf](report.pdf) 

The work was conducted using a forked version of the open-source package linked with the original work (https://github.com/KimGroup/AQT). We made several additions and changes to the original codebase to perform the empirical analysis presented above and the modified codebase is attached to this report. 

- The (**notebooks**) directory contains all newly added IPython notebooks to preare the different quantum states, perform the tomography experiments, collect measurement outcome data, perform standard linear inversion and MLE (**ibm\_get\_data.ipynb**), compute the quantum fidelities and visualize the results (**results.ipynb**). 

- The (**circuits**) directory contains newly added state preparation circuit for the W state. 

- The file  was modified to allow setting of hyperparameters, timing various components of the AQT pipeline and saving density matrices before and after post-processing. 

- Missing documentation and docstrings were added through files in the repository including in (**aqt.py**, **fidelity.py**) and most importantly in (**ann.py**) to explain the computation graph of the transformer model. 

- All experiments were conducted using newly generated simulation datasets, and all results were developed using newly added analysis notebooks.

To run,

you need to first generate a dataset using (**ibm\_get\_data.ipynb**) by configuring the system setup, 

```# System setup
n_qubits = 6
n_shots = 100

# Initialize the circuit
circ = QuantumCircuit(n_qubits)

# Choose which type of state to prepare
# One of ghz, w, bisep, random
state = 'ghz'
```

then setup the hyper-parameters in (**aqt.py**), for e.g.

```
# data
POVM = 'pauli6'
STATE = 'ghz'
HARDWARE = ''
N_SHOTS = 100
N_QUBITS = 6
N_MEAS = (3**N_QUBITS)*N_SHOTS if POVM == 'pauli6' else 0

# training
N_EPOCHS = 100
BATCH_SIZE = 100
LR = 1e-3
SAVE_MLE_DM = True

# model
N_LAYERS = 4
D_MODEL = 128
```

and finally, to evaluate the metrics and visualize the results setup the system in (**results.ipynb**)

```
# System setup
n_qubits = 6
n_shots = 100
povm = 'pauli6'
n_meas = (3**n_qubits)*n_shots if povm == 'pauli6' else 0

# Choose which type of state to prepare
# One of ghz, w, bisep, random
state = 'ghz'
hardware = ''

# Training setup
n_epochs = 200
n_layers = 2
d_model = 128
n_heads = 4

# setup directories
data_dir = 'data'
state_dir = data_dir + f'/{state}_{n_qubits}{hardware}'

dm_info = f'{n_qubits}_{n_meas}_'
dm_types = ['ideal', 'linear', 'linear_mle', 'gaussian_mle']
```