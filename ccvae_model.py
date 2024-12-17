import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)
    
class Encoder(nn.Module):
    def __init__(self, input_size, z_dim, hidden_dim, bidirectional=True):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.hidden_factor = 2 if bidirectional else 1

        self.encoder = nn.LSTM(input_size, hidden_dim, bidirectional=True, batch_first=True)

        self.locs = nn.Linear(hidden_dim*self.hidden_factor, z_dim)
        self.scales = nn.Linear(hidden_dim*self.hidden_factor, z_dim)
    
    def forward(self, x):
        batch_size = x.size(0)
    
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)  
        elif x.dim() == 2:
            x = x.unsqueeze(2)  
        
        output, (hidden, c_n) = self.encoder(x)
        hidden = hidden.view(batch_size, self.hidden_dim*self.hidden_factor)

        locs = self.locs(hidden)
        scales = torch.clamp(F.softplus(self.scales(hidden)), min=1e-3)
        return locs, scales

class Decoder(nn.Module):
    def __init__(self, input_size, z_dim, hidden_dim, vocab_size, bidirectional=True):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.bidirectional = bidirectional
        self.hidden_factor = 2 if bidirectional else 1

        # Linear layer to transform z_dim to hidden_dim
        self.linear = nn.Linear(z_dim, hidden_dim * self.hidden_factor)

        # LSTM to generate sequences
        self.lstm = nn.LSTM(input_size, hidden_dim, bidirectional=bidirectional, batch_first=True)
        self.output = nn.Linear(hidden_dim * self.hidden_factor, self.vocab_size+1)

    def forward(self, z, xs):
        batch_size = z.size(0)
        hidden = self.linear(z).view(self.hidden_factor, batch_size, self.hidden_dim)
        c_0 = torch.zeros(self.hidden_factor, batch_size, self.hidden_dim)

        decoder_hidden = (hidden, c_0)

        outputs, _ = self.lstm(xs, decoder_hidden)
        outputs = self.output(outputs)
        # Apply log softmax to get log probabilities over the vocabulary
        logp = nn.functional.log_softmax(outputs, dim=-1)
        return logp

class Diagonal(nn.Module):
    def __init__(self, dim):
        super(Diagonal, self).__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(self.dim))
        self.bias = nn.Parameter(torch.zeros(self.dim))

    def forward(self, x):
        return x * self.weight + self.bias

class Classifier(nn.Module):
    def __init__(self, dim):
        super(Classifier, self).__init__()
        self.dim = dim
        self.diag = Diagonal(self.dim)

    def forward(self, x):
        return self.diag(x)

class CondPrior(nn.Module):
    def __init__(self, dim):
        super(CondPrior, self).__init__()
        self.dim = dim
        self.diag_loc_true = nn.Parameter(torch.zeros(self.dim))
        self.diag_loc_false = nn.Parameter(torch.zeros(self.dim))
        self.diag_scale_true = nn.Parameter(torch.ones(self.dim))
        self.diag_scale_false = nn.Parameter(torch.ones(self.dim))

    def forward(self, x):
        x = x.unsqueeze(1) 
        loc = x * self.diag_loc_true + (1 - x) * self.diag_loc_false
        scale = x * self.diag_scale_true + (1 - x) * self.diag_scale_false
        return loc, torch.clamp(F.softplus(scale), min=1e-3)
    


# CCVAE model

def compute_kl(locs_q, scale_q, locs_p=None, scale_p=None):
    """
    Computes the KL(q||p)
    """
    if locs_p is None:
        locs_p = torch.zeros_like(locs_q)
    if scale_p is None:
        scale_p = torch.ones_like(scale_q)

    dist_q = dist.Normal(locs_q, scale_q)
    dist_p = dist.Normal(locs_p, scale_p)
    return dist.kl.kl_divergence(dist_q, dist_p).sum(dim=-1)

def sentence_log_likelihood(recon, xs):
    vocab_size = recon.shape[-1] 
    target = xs.squeeze(-1)  
    target = target.long()

    recon = recon.view(-1, vocab_size)  
    target = target.view(-1)  

    criterion = nn.CrossEntropyLoss()

    loss = criterion(recon, target)
    return loss


class CCVAE(nn.Module):
    """
    CCVAE
    """
    def __init__(self, input_size, hidden_dim, vocab_size, z_dim, num_classes, use_cuda, prior_fn, device='cpu'):
        super(CCVAE, self).__init__()
        self.device = device
        self.z_dim = z_dim
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.z_classify = num_classes
        self.z_style = z_dim - num_classes
        self.use_cuda = use_cuda
        self.num_classes = num_classes
        self.ones = torch.ones(1, self.z_style).to(device)
        self.zeros = torch.zeros(1, self.z_style).to(device)
        self.y_prior_params = prior_fn.to(device)

        self.classifier = Classifier(self.num_classes)

        self.encoder = Encoder(self.input_size, self.z_dim, self.hidden_dim)
        self.decoder = Decoder(self.input_size, self.z_dim, self.hidden_dim, self.vocab_size)

        self.cond_prior = CondPrior(self.num_classes)

    def sup(self, x, y):
        y = y.float()
        bs = x.shape[0]
        
        post_params = self.encoder(x)
        
        z = dist.Normal(*post_params).rsample()
        zc, zs = z.split([self.z_classify, self.z_style], 1)
        qyzc = dist.Bernoulli(logits=self.classifier(zc))
        log_qyzc = qyzc.log_prob(y.view(-1, 1)).sum(dim=-1)

        locs_p_zc, scales_p_zc = self.cond_prior(y)
    
        prior_params = (torch.cat([locs_p_zc, self.zeros.expand(bs, -1)], dim=1), 
                        torch.cat([scales_p_zc, self.ones.expand(bs, -1)], dim=1))
        kl = compute_kl(*post_params, *prior_params)

        log_py = dist.Bernoulli(self.y_prior_params.expand(bs, -1)).log_prob(y.view(-1, 1)).sum(dim=-1)

        recon = self.decoder(z, x)

        log_qyx = self.classifier_loss(x, y)
        log_pxz = sentence_log_likelihood(recon, x)

        log_qyzc_ = dist.Bernoulli(logits=self.classifier(zc.detach())).log_prob(y.view(-1, 1)).sum(dim=-1)
        w = torch.exp(log_qyzc_ - log_qyx)+ 1e-8
        elbo = (w * (-log_pxz - kl - log_qyzc) + log_py + log_qyx).mean()
        return -elbo

    def classifier_loss(self, x, y, k=100):
        """
        Computes the classifier loss.
        """
        zc, _ = dist.Normal(*self.encoder(x)).rsample(torch.tensor([k], device=self.device)).split([self.z_classify, self.z_style], -1)
        logits = self.classifier(zc.view(-1, self.z_classify))
        d = dist.Bernoulli(logits=logits)
        y = y.unsqueeze(0).unsqueeze(-1)
        y = y.expand(k, -1, -1).contiguous().view(-1, self.num_classes)
        lqy_z = d.log_prob(y).view(k, x.shape[0], self.num_classes).sum(dim=-1)
        lqy_x = torch.logsumexp(lqy_z, dim=0) - np.log(k)
        return lqy_x

    def classifier_acc(self, x, y=None, k=1):
        zc, _ = dist.Normal(*self.encoder(x)).rsample(torch.tensor([k], device=self.device)).split([self.z_classify, self.z_style], -1)
        logits = self.classifier(zc.view(-1, self.z_classify)).view(-1, self.num_classes)
        y = y.unsqueeze(0).unsqueeze(-1)
        y = y.expand(k, -1, -1).contiguous().view(-1, self.num_classes)
        preds = torch.round(torch.sigmoid(logits))
        acc = (preds.eq(y)).float().mean()
        return acc
    
    def accuracy(self, data_loader, *args, **kwargs):
        acc = 0.0
        for (_, x, y) in data_loader:
            x, y = x.to(self.device), y.to(self.device)
            batch_acc = self.classifier_acc(x, y)
            acc += batch_acc
        return acc / len(data_loader)