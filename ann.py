import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
    
# BUILD THE NETWORK

class Transformer(nn.Module):
    """
    A standard Transformer architecture. Base for this and many
    other models.
    """
    def __init__(self, decoder, tgt_embed, generator, Nq, Na):
        """
        Args:
            decoder (nn.Module): Attention decoder
            tgt_embed (nn.Module): Embedding layer to map the vocab idxs in
                                   each sample e.g. [1, X, ..., 2] to fixed-
                                   size embedding vectors followed by adding
                                   positional encodings
            generator (nn.Module): Linear layer (d_model -> vocab) + 
                                   log_softmax(...) to compute the probs. 
                                   over vocab
            Nq (int): No. of qubits
            Na (int): No. of POVM outcomes = len(vocab)-3
        """
        super(Transformer, self).__init__()
        self.decoder = decoder
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.Nq = Nq
        self.Na = Na
        
    def forward(self, tgt, tgt_mask):
        """Take in and process masked target sequences.

        Args:
           tgt (torch.Tensor): Batch of {start_token + n_q measurements}
                               samples passed through embedding and positional
                               encoding layers
                               dims = (batch_size, n_qubits+1, d_model)
           tgt_mask (torch.Tensor): batch_size copies of subsequent mask to
                                    only receive information from previous
                                    measurements
                                    dims=(batch_size, n_qubits+1, n_qubits+1)
        Returns;
            (torch.Tensor): dims=(batch_size, n_qubits+1, d_model)
        """
        return self.decoder(self.tgt_embed(tgt), tgt_mask)

    def p(self, a_vec, ret_tensor=False):
        """
        Args:
            a_vec (list) : n-nary representation of an int i b/w 0 and N_a**N_q
                           list of size N_q
            ret_tensor (bool) : ?

        Returns:
            p (float) : ?
        """
        outcome = list(a_vec)
        
        # Add 3 to each element of a_vec list
        for nq in range(self.Nq):
            outcome[nq] += 3

        # tensor of dim (Nq + 2, 1) e.g. for i = 0, Na=6, Nq=3 -> trg =
        # [1, 3, 3, 3, 2]
        trg = torch.tensor([[1] + outcome + [2]])
        
        # Prepare input to be passed into attention
        trg_mask = Batch.make_std_mask(trg, 0)
        out = self.forward(trg, trg_mask)

        log_p = self.generator(out)
        p_tensor = torch.exp(log_p)
        
        p = 1.
        
        for nq in range(self.Nq):
            p *= p_tensor[0,nq,outcome[nq]].item()

        if ret_tensor:
            return (p, p_tensor)
        else:
            return p
    
    def generate_next(self, a_vec):

        a_vec = list(a_vec)
        
        for a_ind in range(len(a_vec)):
            a_vec[a_ind] += 3
        trg = torch.tensor([[1] + a_vec ] )
        trg_mask = Batch.make_std_mask(trg, 0)
        out=self.forward(trg, trg_mask)
        log_p=self.generator(out)
        p_tensor = torch.exp(log_p)

        p_vec = p_tensor[0,-1].detach().numpy()[3:]
        
        p_vec = np.array(p_vec)/sum(p_vec) # RENORMALIZE DUE TO NUMERICAL ERRORS
        next_a = np.random.choice(np.arange(len(p_vec)), size=None, p=p_vec)

        return next_a

    def samples(self, Ns):
        Nq = self.Nq
        outcomes = np.zeros((Ns, Nq), dtype=int)
        for ns in range(Ns):
            outcome = []
            for nq in range(Nq):
                o = self.generate_next(outcome)
                outcome = outcome + [o]
            outcomes[ns] = np.array(outcome)

        return outcomes

def sample(model, Ns, device):
    model.to(device)

    Nq = model.Nq
    outcomes = torch.ones(Ns, Nq+1, dtype=int).to(device)
    for ns in range(Ns):
        outcome = outcomes[[ns], :1]
        for i in range(Nq):
            log_p = model.generator(model.forward(outcome, Batch.make_std_mask(outcome, 0)))
            p_tensor = torch.exp(log_p)
            next_a = torch.multinomial(p_tensor.data[0,0], 1)
            outcomes[ns, i+1] = next_a
            outcome = outcomes[[ns], :(i+2)]

    model.to('cpu')
    outcomes.to('cpu')
    return outcomes[:, 1:]-3


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        """
        Args:
            d_model (int) : Embedding dimension
            vocab (int) : Size of the vocabulary, set to POVM.Na
        """
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input sequence embedding+pe representation passed
                              through the Decoder,
                              dims=(batch_size, n_qubits+1, d_model)
        Returns:
            (torch.Tensor): Log prob of each qubit to be each of the values in
                            vocab, dims=(batch_size, n_qubits+1, vocab)
        """
        return F.log_softmax(self.proj(x), dim=-1)

# BASIC BUILDING BLOCKS
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        """
        Args:
            size (int): d_model, model or embedding dimension
            dropout (float): Prob. for dropout
        """
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
    
# / BUILDING BLOCKS

# DECODER
class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, tgt_mask):
        for layer in self.layers:
            x = layer(x, tgt_mask)
        return self.norm(x)
    
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        """
        Args:
            size (int): d_model, model or embedding dimension
            self_attn (nn.Module): Multi-headed attention
            feed_forward (nn.Module): PositionwiseFeedForward
            dropout (float): Prob. for dropout
        """
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, tgt_mask):
        """
        Args:
            x (torch.Tensor): Embedding+PositionalEncoding(x), dims=(
                batch_size, n_qubits+1, d_model)
            tgt_mask (torch.Tensor): Subsequent masking e.g.
            [[True, False, ..., False]
             [True, True, ...,  False]
             [True, True, ...,  True]]
            dims=(batch_size, n_qubits+1, n_qubits+1)
        Returns:
           (torch.Tensor): dims=?
        """
        "Follow Figure 1 (right) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        return self.sublayer[2](x, self.feed_forward)
    
# / DECODER

# MASK FOR SEQUENTIAL GENERATIVE MODELLING
def subsequent_mask(size):
    """annotated-transformer - Mask out subsequent positions.

    Args:
        size (int) : (n_qubits+1)

    Returns:
        (torch.Tensor) : dims=(1, n_qubits+1, n_qubits+1), square matrix
        containing True on diagonal and lower-triangular, and False on
        upper-triangular e.g.
        [[[ True, False, False],
         [ True,  True, False],
         [ True,  True,  True]]]
    """
    # (1, n_qubits+1, n_qubits+1)
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
# / MASK


# ATTENTION MODULE

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    # d_k should be d_model // h
    d_k = query.size(-1)
    # QK^T/sqrt(d_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.0):
        """
        Args:
            h (int): No. of heads
            d_model (int): Embedding or model dimension
            dropout (float): Prob. for dropout, default=0.0
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        # Floor function but should not be needed due to the assertion above
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query (torch.Tensor): dims=(batch_size, n_qubits+1, d_model), =trg
            key (torch.Tensor): dims=(batch_size, n_qubits+1, d_model), =trg
            value (torch.Tensor): dims=(batch_size, n_qubits+1, d_model), =trg
            mask (torch.Tensor): dims=(batch_size, n_qubits+1, n_qubits+1)

        Return:
            x (torch.Tensor): dims=(batch_size, n_qubits+1, d_model)
        """
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            # (batch_size, n_qubits+1, n_qubits+1) to (., 1, ., .)
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
    
# / ATTENTION


# FEED FORWARD MODULE
    
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.0):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
    
# / FEED FORWARD

# EMBEDDING
class Embeddings(nn.Module):
    """Simple lookup-table based embedding
    Takes as input list of idxs corresponding to words and outputs embedding
    vectors of size d_model for each word"""
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): dims=(batch_size*n_qubits+1)
        Returns:
            (torch.Tensor): dims=()
        """
        return self.lut(x) * math.sqrt(self.d_model)
# / EMBEDDING

# POSITIONAL ENCODING
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        """
        Args:
            d_model (int): Model dimension
            dropout (float): Prob. for dropout
            max_len (int): max. sequence length
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # GHZ state: Permutation Invariant
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)
# / POSITIONAL

# DATA BATCHING


class Batch:
    """annoted-transfomer - Object for holding a batch of data with mask 
    during training."""
    def __init__(self, trg=None, pad=0):
        """
        Args:
            trg (torch.Tensor): dims=(batch_size, n_qubits+2)
            pad (int): padding integer
        """
        if trg is not None:
            self.trg = trg[:, :-1]  # trg[:, :trg.size(1)-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        """Create a mask to hide padding and future words.

        Args:
            tgt (torch.Tensor) : dims=(batch_size, n_qubits+1), for each
                                 sample, we have [1, X, ..., X] i.e. does not
                                 include the end token 2
            pdf (int): padding integer
        Returns:
            tgt_mask (torch.Tensor) : dims=(batch_size, n_qubits+1, n_qubits+1)
                                      tensor of bools, where for each sample in
                                      the batch, we have a 
                                      (n_qubit+1, n_qubit+1) matrix usually 
                                      containing True on diagonal and lower-T 
                                      and False on upper-T e.g.
                                      [[ True, False, False],
                                      [ True,  True, False],
                                      [True,  True,  True]]
        """
        # tgt_mask.size() = (batch_size, 1, n_qubits+1)
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

# / BATCHING





# OPTIMIZATION: changes learning rate. increases linearly for n=warmup steps, then decays as sqrt(step)
# class NoamOpt:
#     "Optim wrapper that implements rate."
#     def __init__(self, model_size, factor, warmup, optimizer):
#         self.optimizer = optimizer
#         self._step = 0
#         self.warmup = warmup
#         self.factor = factor
#         self.model_size = model_size
#         self._rate = 0
        
#     def step(self):
#         "Update parameters and rate"
#         self._step += 1
#         rate = self.rate()
#         for p in self.optimizer.param_groups:
#             p['lr'] = rate
#         self._rate = rate
#         self.optimizer.step()
        
#     def rate(self, step = None):
#         "Implement `lrate` above"
#         if step is None:
#             step = self._step
#         # return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))
#         return self.factor*0.01
        
# def get_std_opt(model):
#     return NoamOpt(model.tgt_embed[0].d_model, 2, 4000,
#             torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

# / OPTIMIZATION

# LOSS FUNCTION
def LossFunction(x, y):
    # print(torch.tensor(y, dtype=torch.long))
    # loss_NLL = nn.NLLLoss(size_average=False)(x,torch.tensor(y,dtype=torch.long))
    loss_KL = nn.KLDivLoss(reduction='sum')(x,y)
    loss_L1 = 0.*nn.L1Loss(reduction='sum')(x,y)
    loss_L2 = 0.*nn.MSELoss(reduction='sum')(x,y)
    print(loss_KL)
    return loss_KL+loss_L1+loss_L2
# / LOSS FUNCTION


# LABEL SMOOTHING ?
class LabelSmoothing(nn.Module):
    """annotated-transformer - Implement label smoothing. This allows us the
    model less confidence since even if the model assigns the correct probs. 
    to the right idx in [0, vocab-1], the loss would still be non-zero. 
    """
    def __init__(self, size, padding_idx, smoothing=0.0):
        """
        Args:
            size (int): Vocabulary size, no. of POVM outcomes + 3
            padding_idx (int): int value used to denote patting to make
                               sequences the same length
            smoothing (float): Amount of label smoothing to use to convert the
                               one-hot target to smoother distribution
        """
        super(LabelSmoothing, self).__init__()
        self.criterion = LossFunction  # nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        """
        Args:
            x (torch.Tensor): dims=(batch_size*n_qubits+1, vocab),
                              output logprobs generated by the decoder
            target (torch.Tensor): dims=(batch_size*n_qubits+1), measurement
                                   POVM token sequences e.g. [X, X, ..., 2, X,
                                   X, ..., 2, ..., 2]
        Returns:
            nn.KLDivLoss(x, y) where x is the decoder output and y is the
            smoothed one-hot vectors
        """
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        # Distributes self.smoothing over n_qubits-1
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.size(0) > 0:
            print('LabelSmoothing: A wild Padding Character has appeared!')
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))
# / LABEL SMOOTHING


# COMPUTE LOSS AND BACK PROPAGATE
class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, optimizer=None):
        """
        Args:
            generator: model generator that take as input 
                       (batch_size, *, d_model)-dim tensor and outputs log 
                       softmax probs over vocab (*, vocab)
            criterion: LabelSmoothing that computes the KLDivLoss b/w decoder
                       output and smoothed one-hot targets of size
                       (batch_size*?*vocab)
            optimizer: nn.optim.optimizer e.g. adam
        """
        self.generator = generator
        self.criterion = criterion
        self.optimizer = optimizer

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss.item() * norm
    
    
# / LOSS, BACKPROP


def data_to_torch(data):
    """Convert data np.ndarray into torch.Tensor

    Args:
        data (np.ndarray): dims=(n_samples, n_qubits), POVM train/test data

    Returns:
        torch.Tensor, dims=(n_samples, n_qubits+2)
    """
    # Padding = 0, Start-of-line character = 1, End-of-line character = 2,
    # Tokens = {3, ..., povm.Na+3}

    data = np.array(data)

    n_samples = len(data)
    n_qubits = len(data[0])
    data_np = np.zeros((n_samples, n_qubits+2), dtype=int)

    # Add start/end tokens (1, 2) to POVM measurements
    for n in range(n_samples):
        data_np[n, 0] = 1
        data_np[n, -1] = 2
        data_np[n, 1:-1] = data[n]+3

    np.random.shuffle(data_np)
    return torch.from_numpy(data_np)

def data_gen(data, batch_size):
    """Create a iterator/generator and batch the dataset
    
    Yields:
        Batch object with attributes trg, trg_y, trg_mask
    """
    n_samples = len(data)
    n_batches = int(n_samples / batch_size)

    # Data batching
    for batch_idx in range(n_batches):
        # dims=(batch_size, n_qubits+2)
        data_tgt = data[batch_idx*batch_size:min((batch_idx+1)*batch_size,
                                                 n_samples)]
        tgt = Variable(data_tgt, requires_grad=False)
        # Batch object with attributes trg (batch_size, n_qubits+1), e.g.
        # [1, X, ..., X] for each sample, trg_y (batch_size, n_qubits+1) e.g.
        # [X, ..., X, 2] for each sample, trg_mask (batch_size, n_qubits+1,
        # n_qubits+1) e.g. [[ True, False, False], [ True,  True, False],
        # [True,  True,  True]] for each sample if n_q = 2
        yield Batch(tgt, 0)
    # / batch


# MAKE MODEL. SET HYPERPARAMETERS.

def make_model(Nq, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.):
    """
    annotated-transformer - Helper: Construct a model from hyperparameters.

    Args:
        Nq (int): No. of qubits
        tgt_vocab (int): No. of POVM outcomes (e.g. 6 for Pauli-6 POVM) + 3
        N (int): No. of attention decoder layers
        d_model (int): Embedding dimension
        d_ff (int): Dimensionality of the inner-layer in the position-wise
                    feed-forward networks
        h (int): No. of parallel attention layers, or heads
        dropout (float): p for the dropout layer
    """
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = Transformer(
        Decoder(DecoderLayer(d_model, c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab), Nq, tgt_vocab-3)

    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model
# / MAKE MODEL

# RUN ONE EPOCH
def run_epoch(data_iter, model, loss_compute, verbose=True):
    """annotated-transformer - Run one epoch, Standard Training and Logging
    Function

    Args:
        data_iter: iterator over dset, yields a Batch object each call
        model (nn.Module): Transfomer NN object
        loss_compute (SimpleLossCompute): Loss compute and train function that
                                          computes the loss and performs
                                          backward and step
        verbose (bool): Print loss, tokens at each batch, default=True
    Returns:
        total_loss/total_tokens (float): ratio of accumulated loss and no.
                                         of token over all batches
    """
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        # trg size=(batch_size, n_qubits+1)
        # trg_mask size=(batch_size, n_qubits+1, n_qubits+1)
        out = model.forward(batch.trg, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens.item())

        ntokens = batch.ntokens.item()
        total_loss += loss
        total_tokens += ntokens
        tokens += ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            if verbose:
                print(f"Epoch Step: {i} Loss: {loss/ntokens}",
                      f"Tokens per Sec: {tokens/elapsed}")
            start = time.time()
            tokens = 0
    return total_loss / total_tokens

# / ONE EPOCH


def InitializeModel(Nq, Nlayer=2, dmodel=128, Nh=4, Na=4, dropout=0.):
    """Initialize Transformer model

    Args:
        Nq (int): No. of qubits
        Nlayer (int): No. of attention decoder layers, default=2
        dmodel (int): Embedding dimension, default=128
        Nh (int): No. of parallel attention layers, or heads
        Na (int): No. of possible POVM outcomes, default=4
        dropout (float): p for the dropout layer, default=0.

    Return:
        model (nn.Module): Initialized transformer NN
    """
    # Initialize Model
    model = make_model(Nq=Nq, tgt_vocab=Na+3, N=Nlayer, d_model=dmodel,
                       d_ff=4*dmodel, h=Nh, dropout=dropout)

    return model

def TrainModel(model, train_data_np, test_data_np, device, smoothing=0.0,
               lr=0.001, batch_size=100, Nep=20):
    """Training function

    Args:
        model (nn.Module): Initialized transformer NN with only a decoder
        train_data_np (np.ndarray): Training data, dims=?
        test_data_np (np.ndarray): Test data, dims=?
        device (torch.device): CPU or GPU
        smoothing (float): label smoothing where instead of one-hot target
                           distribution, the correct token has prob 1-smoothing
                           and rest of smoothing is distributed throughout
                           vocab, default=0.0
        lr (float): Learning rate, default=0.001
        batch_size (int): Batch size
        Nep (int): No. of epochs

    Returns:
        model (nn.Module): Trained transformer NN which can be used to generate
                           the POVMProbTable in fidelity.py
        loss (np.ndarray): dims=?

    """
    # Train Model
    loss = np.zeros((2, Nep))

    criterion = LabelSmoothing(size=model.Na+3, padding_idx=0,
                               smoothing=smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98),
                                 eps=1e-9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=Nep,
                                                           eta_min=0.)

    train_data = data_to_torch(train_data_np).to(device)
    test_data = data_to_torch(test_data_np).to(device)

    for epoch in range(Nep):

        # Run a single epoch
        model.train()
        loss[0, epoch] = run_epoch(
            data_gen(train_data, batch_size),
            model,
            SimpleLossCompute(model.generator, criterion, optimizer),
            verbose=False)

        # Test the model at the end of each epoch
        model.eval()
        loss[1, epoch] = run_epoch(
            data_gen(test_data, batch_size),
            model,
            SimpleLossCompute(model.generator, criterion, None),
            verbose=False)

        print(epoch+1, ':', loss[1, epoch])
        scheduler.step()

    train_data.to('cpu')
    test_data.to('cpu')
    return model, loss
