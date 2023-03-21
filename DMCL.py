from __future__ import print_function, division
import argparse
import os.path
import numpy as np

from scipy.io import savemat
from sklearn.cluster import KMeans
import torch
import random
from torch.optim import Adam
from torch.utils.data import DataLoader

from processDatasets import CancerDataset
from networkModels import MultiDL
from modelLoss import AE_loss, target_distribution, KL_loss, ConstrastiveLoss

def parse_option():
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--n_clusters', default=n_cluster, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--n_z', default=32, type=int)
    parser.add_argument('--hid_dim',default =64,type = int)
    parser.add_argument('--dataset', type=str, default=dataset_name)
    parser.add_argument('--pretrain_path', type=str, default='pretrain_models')
    parser.add_argument('--pretrain_epoch', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=74)
    parser.add_argument('--early_stopping_pretraining', type=int, default=20)
    parser.add_argument('--early_stopping_training', type=int, default=50)
    parser.add_argument('--max_iter',type = int,default=2000)
    parser.add_argument('--result_models',type = str,default='result_models')
    parser.add_argument('--result_mat',type = str,default='result')
    parser.add_argument(
        '--gamma',
        default= 1,
        type=float,
        help='coefficient of clustering loss')
    parser.add_argument(
        '--beta',
        default= 1,
        type=float,
        help='coefficient of constrastive loss')
    parser.add_argument('--update_interval', default=5, type=int)
    parser.add_argument('--tol', default=0.00001, type=float)
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    args.device = torch.device('cuda' if args.cuda else 'cpu')
    if not os.path.isdir(args.result_models):
        os.makedirs(args.result_models)
    if not os.path.isdir(args.result_mat+f'/{args.dataset}'):
        os.makedirs(args.result_mat+f'/{args.dataset}')
    args.result_models= args.result_models +f'/{args.dataset}_{args.n_clusters}'
    args.result_mat = args.result_mat+f'/{args.dataset}'+f'/{args.dataset}_{args.n_clusters}'
    return args
def set_seed(seed=0):
    # credits to CC and https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def train_model(require_pretrain=False,fix_seed = True):
    if fix_seed:
        set_seed(opts.seed)

    dataset = CancerDataset(opts.dataset+'.mat', 'data')

    if opts.batch_size == 0:
        opts.batch_size = len(dataset)
    n_input = [x.shape[1] for x in dataset.data]

    opts.view_num = len(n_input)

    model = MultiDL(n_input, n_enc_1=opts.hid_dim, n_z=opts.n_z, n_dec_1=opts.hid_dim, n_clusters=opts.n_clusters).to(opts.device)

    model.pretrain_AE(opts, dataset, require_pretrain)

    train_loader = DataLoader(dataset, batch_size=opts.batch_size, shuffle=True, drop_last=False)
    optimizer = Adam(model.parameters(), lr=opts.lr)

    # cluster parameter initiate
    data = [curData.to(opts.device) for curData in dataset.data]
    hidden = model.getHidden_concate(data)

    kmeans = KMeans(n_clusters=opts.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(hidden.detach().cpu().numpy())
    hidden = None
    y_pred_last = y_pred

    with torch.no_grad():
        model.cluster_layer.copy_(torch.tensor(kmeans.cluster_centers_).to(opts.device))

    # initialize constrastive loss
    constLoss = ConstrastiveLoss(opts)

    model.train()
    is_break = False
    is_kl = False
    min_loss = 1e6
    patience = 0
    is_early = False

    for epoch in range(opts.max_iter):

        if epoch % opts.update_interval == 0:

            _, temp_q = model(data)

            # update target distribution p
            temp_q = temp_q.detach()
            p = target_distribution(temp_q)

            y_pred = temp_q.cpu().numpy().argmax(1)
            delta_label = np.sum(y_pred != y_pred_last).astype(
                np.float32) / y_pred.shape[0]
            y_pred_last = y_pred

            if epoch > 0 and delta_label < opts.tol:
                print('delta_label {:.4f}'.format(delta_label), '< tol',
                      opts.tol)
                is_kl = True
            else:
                is_kl = False

        for batch_idx, (x, _, idx) in enumerate(train_loader):
            x = [x_dat.to(opts.device) for x_dat in x]
            idx = idx.long().to(opts.device)

            recon_x, q = model(x)

            recon_loss = AE_loss(x, recon_x)
            kl_loss = KL_loss(q.log(), p[idx])

            cl_loss = constLoss(model.getHidden_separate(x))

            loss = recon_loss + opts.gamma * kl_loss + opts.beta * cl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch {} loss={:.4f}'.format(epoch, loss / (batch_idx + 1)))
        if (loss < min_loss):
            min_loss = loss
            patience = 0
        else:
            patience += 1

        if patience == opts.early_stopping_pretraining:
            print("Pretraning has reached early stopping")
            print(f"Min loss is {min_loss}")
            is_early = True
        unique_y_pred = np.unique(y_pred)
        if len(unique_y_pred) == opts.n_clusters:
            is_true_cluster = True
        else:
            is_true_cluster = False
        if (is_early == True and is_true_cluster == True):
            is_break = True
            break
        elif (is_kl == True and is_true_cluster == True):
            is_break = True
            break
    print(y_pred)
    torch.save(model.state_dict(), opts.result_models)
    opts.result_mat = opts.result_mat +f'_{is_break}'
    savemat(opts.result_mat+'.mat',{'label':y_pred})
if __name__ == '__main__':
    n_cluster = 4
    dataset_name = 'liver'
    opts = parse_option()
    print(opts)
    train_model(require_pretrain=True, fix_seed = True)


