from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConstrastiveLoss(nn.Module):
    def __init__(self, opts):
        super(ConstrastiveLoss, self).__init__()
        self.temperature = opts.temperature
        self.device = opts.device
        self.batch_size = opts.batch_size
        self.view_num = opts.view_num

        self.mask = self.mask_positive_samples(self.view_num, self.batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_measure = nn.CosineSimilarity(dim=2)

    def mask_positive_samples(self, view_num,batch_size):
        mat_dim = view_num * batch_size
        mask = torch.ones(mat_dim, mat_dim)
        for i in range(view_num):
            for j in range(view_num):
                mask[(i * batch_size):((i + 1) * batch_size),
                            (j * batch_size):((j + 1) * batch_size)].fill_diagonal_(0)

        mask = mask.bool()
        return mask


    def forward(self, input):
        view_num = len(input)

        batch_size = input[0].shape[0]
        if batch_size < self.batch_size:
            mask = self.mask_positive_samples(view_num, batch_size)
        else:
            mask = self.mask

        N = view_num * batch_size
        input_cat = torch.cat(input, dim=0)
        sim = self.similarity_measure(input_cat.unsqueeze(1), input_cat.unsqueeze(0)) / self.temperature

        total_loss = 0
        for i in range(view_num):
            negative_samples_mask = mask[(i * batch_size):((i + 1) * batch_size),]
            temp_sim = sim[(i * batch_size):((i + 1) * batch_size),]
            negative_samples = temp_sim[negative_samples_mask].reshape(batch_size, -1)
            for j in [x for x in range(view_num) if x != i]:
                sim_i_j = torch.diag(sim[(i * batch_size):((i + 1) * batch_size),
                                            (j * batch_size):((j + 1) * batch_size)]).reshape(batch_size, 1)
                labels = torch.zeros(batch_size).to(sim.device).long()
                logits = torch.cat((sim_i_j, negative_samples), dim=1)
                loss = self.criterion(logits, labels)
                total_loss += loss /batch_size


        return total_loss









def AE_loss(input, recon_input):
    ae_loss = 0
    for i in range(len(input)):
        ae_loss += F.mse_loss(input[i], recon_input[i])

    return ae_loss

def KL_loss(q, p):
    return F.kl_div(q, p)


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()




