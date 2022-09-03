from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from pretrainModel import pretraining

class AE(nn.Module):
    def __init__(self, n_input, n_enc_1, n_dec_1, n_z):
        super(AE, self).__init__()

        # encoder
        self.enc_1 = nn.Linear(n_input, n_enc_1)
        self.middle_layer = nn.Linear(n_enc_1, n_z)

        # decoder
        self.dec_1 = nn.Linear(n_z, n_dec_1)
        self.out = nn.Linear(n_dec_1, n_input)

    def forward(self, x):
        # encoder
        enc_h1 = F.relu(self.enc_1(x))


        middle_layer = self.middle_layer(enc_h1)
        # decoder
        dec_h1 = F.relu(self.dec_1(middle_layer))

        x_bar = self.out(dec_h1)

        return x_bar, middle_layer


class MultiDL(nn.Module):
    def __init__(self, n_input, n_enc_1, n_z, n_dec_1, n_clusters, alpha=1):
        super(MultiDL, self).__init__()

        self.alpha = alpha
        self.ae_list = nn.ModuleList([AE(i, n_enc_1, n_dec_1, n_z) for i in n_input])
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z * len(n_input)))
        nn.init.xavier_normal_(self.cluster_layer)

    def forward(self, x):

        recon_x = []
        z_con = []
        for ind in range(len(x)):
            x_bar, z = self.ae_list[ind](x[ind])
            recon_x.append(x_bar)
            z_con.append(z)

        z_con = torch.cat(z_con, dim=1)

        # cluster
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(z_con.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return recon_x, q

    def pretrain_AE(self, opts, dataset, pretrainFlag=True):
        if pretrainFlag:
            pretraining(self.ae_list, opts, dataset)

        self.ae_list.load_state_dict(torch.load(opts.pretrain_path))
        print('load pretrained AE from', opts.pretrain_path)

    def getHidden_concate(self, x):
        z_con = []
        for ind in range(len(x)):
            _, z = self.ae_list[ind](x[ind])
            z_con.append(z)

        z_con = torch.cat(z_con, dim=1)


        return z_con

    def getHidden_separate(self, x):

        return [self.ae_list[ind](x[ind])[1] for ind in range(len(x))]