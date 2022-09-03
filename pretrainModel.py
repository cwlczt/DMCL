import torch
from torch.utils.data import DataLoader
from modelLoss import AE_loss
from torch.optim import Adam

#pretraining(self.ae_list, opts, dataset)
def pretraining(model, opts, dataset):
    """
    pretrain AEs
    model here is a ModuleList consisting of multiple AEs
    """
    train_loader = DataLoader(dataset, batch_size=opts.batch_size, shuffle=True)
    print(model)
    model.train()
    optimizer = Adam(model.parameters(), lr=opts.lr)
    min_loss = 1e6
    patience = 0
    for epoch in range(opts.pretrain_epoch):
        total_loss = 0.
        for batch_idx, (x, _, idx) in enumerate(train_loader):
            x = [x_dat.to(opts.device) for x_dat in x]

            optimizer.zero_grad()

            recon_x = []
            for ind in range(len(x)):
                x_bar, _ = model[ind](x[ind])

                recon_x.append(x_bar)

            loss = AE_loss(x, recon_x)

            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        print('epoch {} loss={:.4f}'.format(epoch,
                                            total_loss / (batch_idx + 1)))
        if(total_loss < min_loss):
            min_loss = total_loss
            patience = 0
        else:
            patience += 1

        if patience == opts.early_stopping_pretraining:
            print("Pretraning has reached early stopping")
            print(f"Min loss is {min_loss}")
            break

    torch.save(model.state_dict(), opts.pretrain_path)

    print(f'Pretrained model saved to {opts.pretrain_path}')