import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

class Encoder(nn.Module):
    def __init__(self, input_size, z_dim, hidden_dim, num_classes, bidirectional=True):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.hidden_factor = 2 if bidirectional else 1
        
        self.encoder = nn.LSTM(input_size, hidden_dim, bidirectional=True, batch_first=True)

        self.locs = nn.Linear(hidden_dim*self.hidden_factor, z_dim)
        self.scales = nn.Linear(hidden_dim*self.hidden_factor, z_dim)

    def forward(self, x, y):
        batch_size = x.size(0)
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0) 
        elif x.dim() == 2:
            x = x.unsqueeze(2) 

        y = y.view(batch_size,self.num_classes,self.input_size)
        x_cat_y = torch.cat((x,y), 1)

        output, (hidden, c_n) = self.encoder(x_cat_y)
       
        hidden = hidden.view(batch_size, self.hidden_dim*self.hidden_factor)

        locs = self.locs(hidden)
        scales = torch.clamp(F.softplus(self.scales(hidden)), min=1e-3)
        return locs, scales
    

class Decoder(nn.Module):
    def __init__(self, input_size, z_dim, num_classes, hidden_dim, vocab_size, seq_len,bidirectional=True):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.bidirectional = bidirectional
        self.hidden_factor = 2 if bidirectional else 1

        # Linear layer to transform z_dim to hidden_dim
        self.linear = nn.Linear(z_dim + num_classes, hidden_dim * self.hidden_factor)

        # LSTM to generate sequences
        self.lstm = nn.LSTM(input_size, hidden_dim, bidirectional=True, batch_first=True)
        self.output = nn.Linear(hidden_dim * self.hidden_factor, self.vocab_size + 1)
        
    def forward(self, z, ysoft, xs):
        batch_size = z.size(0)
        z_cat_ysoft = torch.cat((z,ysoft), dim=1)
        hidden = self.linear(z_cat_ysoft).view(self.hidden_factor, batch_size, self.hidden_dim)
        c_0 = torch.zeros(self.hidden_factor, batch_size, self.hidden_dim) 
        
        decoder_hidden = (hidden, c_0)
        outputs, _ = self.lstm(xs, decoder_hidden)

        outputs = self.output(outputs)
        logp = nn.functional.log_softmax(outputs, dim=-1)
        return logp
    
class Classifier(nn.Module):
    def __init__(self, input_size, hidden_dim, num_classes):
        super(Classifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim,num_classes)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0) 
        elif x.dim() == 2:
            x = x.unsqueeze(2)  
            
        _, (h,c) = self.lstm(x)
        logits = self.linear(h[-1])
        preds = torch.sigmoid(logits)
        return logits, preds
    

def sentence_log_likelihood(recon, xs):
    vocab_size = recon.shape[-1] 
    target = xs.squeeze(-1)  
    target = target.long()

    recon = recon.view(-1, vocab_size)  
    target = target.view(-1)  

    criterion = nn.CrossEntropyLoss()

    loss = criterion(recon, target)
    return loss

class M2(nn.Module):
    def __init__(self,input_size, z_dim, num_classes, hidden_dim, seq_len, vocab_size, batch_size):
        super(M2, self).__init__()
        self.input_size = input_size
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.encoder = Encoder(self.input_size, self.z_dim, self.hidden_dim, num_classes)
        self.decoder = Decoder(self.input_size, self.z_dim, self.num_classes, self.hidden_dim, self.vocab_size, self.seq_len)
        self.classifier = Classifier(self.input_size, self.hidden_dim, self.num_classes)

        self.cat_loss = nn.BCEWithLogitsLoss(reduce=False)
    
    def sup(self, x, y):
        logits, preds = self.classifier(x)
        locs_z, scales_z = self.encoder(x,y)
        z = dist.Normal(locs_z, scales_z).rsample()
        recons = self.decoder(z,preds,x)
        
        llk = sentence_log_likelihood(recons,x)
        
        cat = self.cat_loss(logits.squeeze(1),y.float())
        kld_norm = torch.sum(0.5 * ( locs_z**2 + scales_z - 1 - torch.log(scales_z)),-1)
        
        loss = (llk + cat + kld_norm).mean()

        return loss, llk, cat, kld_norm
    
    def classifier_acc(self, x, y=None, k=1):
        _, preds = self.classifier(x)
        preds = torch.round(preds)
        acc = (preds.eq(y)).float().mean()
        return acc
    
    def accuracy(self, data_loader, *args, **kwargs):
        acc = 0.0
        for (_, x, y) in data_loader:
            if self.use_cuda:
                x, y = x.cuda(), y.cuda()
            batch_acc = self.classifier_acc(x, y)
            acc += batch_acc
        return acc / len(data_loader)