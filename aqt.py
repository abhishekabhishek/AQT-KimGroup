import time
import numpy as np
import torch

import fidelity as F
import povm as P
import ann as A
import sys


# Basic parameters

def AQT(datapath, Nq, Nep, Nl=2, dmodel=64, Nh=4, batch_size=100, lr_rate=1e-4,
        save_model=True, save_loss=True, save_pt=True, save_dm=True,
        save_mle_dm=True):
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
    print('using device : ', device)

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
    t1 = time.time()
    model, loss = A.TrainModel(model, traindata, testdata, device,
                               batch_size=batch_size, lr=lr_rate, Nep=Nep)
    t = divmod(time.time() - t1, 60)
    print(f'training : took {t[0]} minutes {t[1]} seconds')
    

    if save_model:
        torch.save(model, f'{model_filetag}.mod')
    if save_loss:
        np.save(f'{model_filetag}_loss.npy', loss)

    # Build POVM probability table
    if Nq <= 10:
        # Original - non-batched method
        model.to('cpu')
        pt_model = F.POVMProbTable(model)
    else:
        # new batched method to generate prob. table
        pt_model = F.GenPOVMProbTable(model, batch_size)
        model.to('cpu')

    if save_pt:
        np.save(f'{model_filetag}_pt.npy', pt_model)

    # Reconstruct density matrix
    dm8 = np.zeros((8, 2**Nq, 2**Nq), dtype=complex)

    t1 = time.time()
    for xyz in range(8):
        dm8[xyz] = F.GetDMFull(pt_model, Nq, P.POVM('pauli6', xyz))
    dm_return = F.WeightedDM(np.full((8), 1/8), dm8)
    if save_dm:
        np.save('{}_pre_dm.npy'.format(model_filetag), dm_return)

    if save_mle_dm:
        t = divmod(time.time() - t1, 60)
        print(f'GetDMFull : took {t[0]} minutes {t[1]} seconds')

        _, _, dm_model = F.GetBestDM(dm8)
        np.save('{}_post_dm.npy'.format(model_filetag), dm_model)
        dm_return = dm_model

    return dm_return


if __name__ == '__main__':
    # data
    POVM = 'pauli6'
    STATE = 'ghz'
    HARDWARE = ''
    N_SHOTS = 100
    N_QUBITS = 6
    N_MEAS = (3**N_QUBITS)*N_SHOTS if POVM == 'pauli6' else 0

    # training
    N_EPOCHS = 200
    BATCH_SIZE = 100
    LR = 1e-3
    SAVE_MLE_DM = True

    # model
    N_LAYERS = 2
    D_MODEL = 128

    t1 = time.time()
    datapath = f'data/{STATE}_{N_QUBITS}{HARDWARE}/{N_QUBITS}_{N_MEAS}'
    AQT(datapath, N_QUBITS, N_EPOCHS, Nl=N_LAYERS, dmodel=D_MODEL,
        batch_size=BATCH_SIZE, lr_rate=LR, save_mle_dm=SAVE_MLE_DM)
    t = divmod(time.time() - t1, 60)
    print(f'main : took {t[0]} minutes {t[1]} seconds')
