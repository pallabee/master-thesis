import torch
import numpy as np
from utils import PlaceHolder


def sum_except_batch(x):

    return x.reshape(x.size(0), -1).sum(dim=-1)

def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask.long())).abs().max().item() < 1e-4, \
        'Variables not masked properly.'

def cosine_beta_schedule_discrete(timesteps, s=0.008):
    """ Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ. """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = 1 - alphas
    return betas.squeeze()

def sample_discrete_features(probF):
    ''' Sample features from multinomial distribution with given probabilities (probX, probE, proby)
        :param probX: bs, n, dx_out        node features
        :param probE: bs, n, n, de_out     edge features
        :param proby: bs, dy_out           global features.
    '''
    bs1, n1, _ = probF.shape

    # # Flatten the probability tensor to sample with multinomial
    probFeat = probF.reshape(bs1 * n1, -1)       # (bs * n, dx_out)

    # Sample F
    #https://discuss.pytorch.org/t/sampling-from-a-tensor-in-torch/97112/2
    F_t = probFeat.multinomial(num_samples=probF.shape[2])

    F_t = F_t.reshape(bs1, n1, -1)

    return PlaceHolder(Feat=F_t)

def compute_posterior_distribution(M, M_t, Qt_M, Qsb_M, Qtb_M):
    ''' M: X or E
        Compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T
    '''
    # Flatten feature tensors
    M = M.flatten(start_dim=1, end_dim=-2).to(torch.float32)        # (bs, N, d) with N = n or n * n
    M_t = M_t.flatten(start_dim=1, end_dim=-2).to(torch.float32)    # same

    Qt_M_T = torch.transpose(Qt_M, -2, -1)      # (bs, d, d)

    left_term = M_t @ Qt_M_T   # (bs, N, d)
    right_term = M @ Qsb_M     # (bs, N, d)
    product = left_term * right_term    # (bs, N, d)

    denom = M @ Qtb_M     # (bs, N, d) @ (bs, d, d) = (bs, N, d)
    denom = (denom * M_t).sum(dim=-1)   # (bs, N, d) * (bs, N, d) + sum = (bs, N)
    # denom = product.sum(dim=-1)
    # denom[denom == 0.] = 1

    prob = product / denom.unsqueeze(-1)    # (bs, N, d)

    return prob

def posterior_distributions(Feat, F_t, Qt, Qsb, Qtb):
    prob_Feat = compute_posterior_distribution(M=Feat, M_t=F_t, Qt_M=Qt.Feat, Qsb_M=Qsb.Feat, Qtb_M=Qtb.Feat)

    return PlaceHolder(Feat=prob_Feat)




