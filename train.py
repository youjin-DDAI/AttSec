import argparse
import time
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import seed_everything, preprocessing, T5_seg_att_collate_fn, T5_seg_att_train_collate_fn, scatter_compute

from Model.DSSPDataset import DSSPDataset
from Model.MaskSecondary_RPE_2dfeature_att_multi import MASKSecondary


def compute_loss(pred, lengths, batch_labels, batch_seg_labels, batch_index, batch_seg_index, loss_fn):
    diag, pred, inter_pred = pred
    batch_size = len(lengths)
    preds = []

    for i in range(len(lengths)):
        preds.append(pred[i][:lengths[i]])

    preds = torch.cat(preds)
    batch_index = batch_index.cuda()

    pred_loss = loss_fn(preds, batch_labels.cuda())
    pred_loss = scatter_compute(pred_loss, batch_size, batch_index, 'mean')

    # acc
    _, p = torch.max(preds, dim=-1)
    correct = p.eq(batch_labels.cuda()).float()
    acc = scatter_compute(correct, batch_size, batch_index, 'mean')

    return pred_loss.mean(), 0, acc.mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Proteinnet')
    parser.add_argument('--str_type', type=str, default='dssp8')
    parser.add_argument('--v', type=bool, default=False)
    args = parser.parse_args()

    # seed setting
    seed = 41
    seed_everything(seed)

    # data load
    if args.dataset == 'Proteinnet':
        train_file = 'Dataset/SPOT/train.csv'
        valid_file = 'Dataset/SPOT/validation.csv'
        data_type = 'SPOT'

    elif args.dataset == 'Netsurf':
        train_file = 'Dataset/Netsurf/Train_HHblits.csv'
        valid_file = 'Dataset/Netsurf/Validation_HHblits.csv'
        data_type = 'Netsurf'


    if args.str_type == 'dssp8':
        dssp_dim = 8

    elif args.str_type == 'dssp3':
        dssp_dim = 3

    # preprocessing
    train_seqs, train_labels, train_lengths = preprocessing(train_file, args.str_type, data_type, train=False)
    valid_seqs, valid_labels, valid_lengths = preprocessing(valid_file, args.str_type, data_type)

    train_dataset = DSSPDataset(train_seqs, train_labels, train_lengths)
    valid_dataset = DSSPDataset(valid_seqs, valid_labels, valid_lengths)


    train_collate_fn = T5_seg_att_train_collate_fn(dssp_dim)
    valid_collate_fn = T5_seg_att_collate_fn(dssp_dim)
    batch_size = 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=valid_collate_fn)

    # loss
    loss_fn = nn.CrossEntropyLoss(reduction='none')

    # model
    dropout = 0.3
    model = MASKSecondary(dropout, dssp_dim)
    model = model.cuda()

    # initialization
    params = [p for p in model.parameters() if p.requires_grad]
    for p in model.parameters():
        if p.dim() > 1 and p.requires_grad == True:
            nn.init.xavier_uniform_(p)

    print('# of trainable parameters....{}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # parameter
    optim = torch.optim.Adam(params, lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=int(len(train_seqs)//batch_size), eta_min=0.00001)
    num_epoch = 50
    iter = 0
    best_val_loss = 1000
    start_epoch = 1


    for epoch in tqdm(range(start_epoch, num_epoch + 1)):
        model.train()
        start_train = time.time()
        for batch in train_loader:
            batch_index, batch_seg_index, lengths, rep_pack, att_pack, batch_labels, batch_seg_label, mask = batch
            iter += 1
            optim.zero_grad()
            pred, attention = model(rep_pack.cuda(), att_pack, mask)
            pred_loss, inter_loss, acc = compute_loss(pred, lengths, batch_labels, batch_seg_label, batch_index, batch_seg_index, loss_fn)

            loss = pred_loss

            loss.backward()
            optim.step()
            scheduler.step()
            if iter % 50 == 0:
                print("[train] %d ==> loss : %.3f, acc : %.3f" % (iter, loss.item(), acc))

        print("train_time_per_epoch : %.3f" % ((time.time() - start_train) / 3600))
        torch.cuda.empty_cache()
        model.eval()
        iter_val = 0
        loss_val = 0
        pred_loss_val = 0
        acc_val = 0
        start_val = time.time()

        with torch.no_grad():
            for batch in valid_loader:
                iter_val += 1
                batch_index, batch_seg_index, lengths, rep_pack, att_pack, batch_labels, batch_seg_label, mask = batch
                pred, _ = model(rep_pack.cuda(), att_pack, mask)
                pred_loss, inter_loss, acc = compute_loss(pred, lengths, batch_labels, batch_seg_label, batch_index, batch_seg_index, loss_fn)
                loss_val += pred_loss.item()
                pred_loss_val += pred_loss.item()
                acc_val += acc

            loss_val /= iter_val
            acc_val /= iter_val
            pred_loss_val /= iter_val

            print('[val] loss : %.3f, acc : %.3f' % (pred_loss_val, acc_val))
            print("time for validation: ", (time.time() - start_val) / 60)

            if best_val_loss > pred_loss_val:
                print('Updated best model with loss : %.3f, acc : %.3f' % (pred_loss_val, acc_val))
                best_val_loss = pred_loss_val



if __name__ == "__main__":
    main()
