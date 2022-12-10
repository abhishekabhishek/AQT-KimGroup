import time
import numpy as np
import torch

import fidelity as F
import povm as P
import ann as A
import sys


# Basic parameters

def AQT(datapath, Nq, Nep, Nl=2, dmodel=64, Nh=4, save_model=True,
        save_loss=True, save_pt=True, save_dm=True):
    """Train Transformer and extract density matrix using inversion
    
    Args:
        datapath (str) : Local path to the measurment dataset
        Nq (int) : Number of qubits
        Nl (int) : Number of attention decoder layers
        dmodel (int) : Embedding dimension
        Nh (int) : ?
        save_model (bool) : Save the trained model
        save_loss (bool) : Save a NumPy array of training loss values
        save_pt (bool) : Save POVM probability table
        save_dm (bool) : Save reconstructed density matrix
    Returns:
        dm_model (np.ndarray) : Reconstructed density matrix using AQT
    Outputs:
        Saves model_filetag{.mod, _loss.npy, _pt.npy, _dm.npy} if corresponding
        bool set to True
    """
    # Tuple of (N_a (int), M (shape = N_a, 2, 2), TinvM (shape = N_a, 2, 2))
    # N_a = 6 for Pauli-6 POVM
    povm = P.POVM('pauli6')
    Na = povm.Na

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # Load data, shape = ?
    data = np.load(f'{datapath}.npy')
    np.random.shuffle(data)

    # No validation 80-20 train-test split
    split = 0.8
    traindata = data[:int(len(data)*split)]
    testdata = data[int(len(data)*split):]

    # Create a unique identified for the run
    model_filetag = f'{datapath}_{Nep}-{Nl}-{dmodel}-{Nh}'

    # Initialize nn.Module model
    model = A.InitializeModel(Nq, Nlayer=Nl, dmodel=dmodel, Nh=Nh, Na=Na).to(
        device)

    # Training loop
    t = time.time()
    model, loss = A.TrainModel(model, traindata, testdata, device,
                               batch_size=50, lr=1e-4, Nep=Nep)
    print(f'Took {round(((time.time()-t)/60), 2)} minutes')
    model.to('cpu')

    if save_model:
        torch.save(model, f'{model_filetag}.mod')
    if save_loss:
        np.save(f'{model_filetag}_loss.npy', loss)

    # Build POVM probability table
    pt_model = F.POVMProbTable(model)

    if save_pt:
        np.save(f'{model_filetag}_pt.npy', pt_model)

    # Reconstruct density matrix
    dm8 = np.zeros((8, 2**Nq, 2**Nq), dtype=complex)

    for xyz in range(8):
        dm8[xyz] = F.GetDMFull(pt_model, Nq, P.POVM('pauli6', xyz))

    _, _, dm_model = F.GetBestDM(dm8)

    if save_dm:
        np.save('{}_dm.npy'.format(model_filetag), dm_model)

    return dm_model

if __name__ == '__main__':

    datapath = 'data/w_3/pauli6_2700'
    Nq = 3
    Nep = 100

    Nl = 2
    dmodel = 64

    AQT(datapath, Nq, Nep, Nl=Nl, dmodel=dmodel)
