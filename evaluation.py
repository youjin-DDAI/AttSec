import argparse

import torch
from torch.utils.data import DataLoader

from Model.DSSPDataset import DSSPDataset
from utils import T5_collate_fn, preprocessing


def main():
    parser = argparse.ArgumentParser('Script for evaluating AttSec.')
    parser.add_argument('--model', type=str, default='MASKSecondary_Proteinnet_dssp8_T5_emb_dropout_0.3_16_RPE_seg_feature_block3.jit.pt')
    parser.add_argument('--train_data_type', type=str, default='Proteinnet')
    parser.add_argument('--str_type', type=str, default='dssp8')
    args = parser.parse_args()

    if args.train_data_type == 'Proteinnet':
        dataset = {'SPOT-2016.csv': 'Proteinnet', 'SPOT-2016-HQ.csv': 'Proteinnet', 'SPOT-2018.csv': 'Proteinnet', 'SPOT-2018-HQ.csv': 'Proteinnet', 'test2018.csv': 'Proteinnet'}

    else:
        dataset = {'CASP12_HHblits.csv': 'Netsurf', 'CB513_HHblits.csv': 'Netsurf', 'TS115_HHblits.csv': 'Netsurf', 'NEW364.csv': 'Netsurf'}

    if args.str_type == 'dssp8':
        dssp_dim = 8
    else:
        dssp_dim = 3

    checkpoint = 'Checkpoints/' + args.model

    model = torch.jit.load(checkpoint)
    model = model.cuda()

    results = []
    for dataset, encoder in dataset.items():
        file = 'Dataset/' + args.train_data_type + '/' + dataset
        seqs, labels, lengths = preprocessing(file, str_type=args.str_type, data_type=encoder, train=False)
        evaluation_data = DSSPDataset(seqs, labels, lengths)
        collate_fn = T5_collate_fn(dssp_dim)
        dataloader = DataLoader(evaluation_data, batch_size=1, shuffle=False, collate_fn=collate_fn)

        normal_list = list(range(len(seqs)))
        model.eval()
        acc = 0
        iter = 0
        sample_size = len(normal_list)
        with torch.no_grad():
            for batch in dataloader:
                iter += 1
                batch_index, lengths, rep_pack, batch_labels, mask = batch
                pred = model(rep_pack.cuda(), mask)
                pred = pred.squeeze(0)
                _, p = pred.max(dim=-1)
                correct = p.eq(batch_labels.cuda()).float()
                acc += (correct.sum()/len(p))
        results.append((dataset, sample_size, acc/sample_size))
        print('acc: %.3f' % (acc / sample_size))

    print(checkpoint)

    for result in results:
        print(f'{result[0]} || ({result[1]}) || accuracy : {result[2]} ')


if __name__=="__main__":

    main()





