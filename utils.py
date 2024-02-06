import os
import random
from random import seed

import numpy as np
import pandas as pd
import torch
from torch_scatter import scatter_mean, scatter_add
from Model.alphabets import SecStr8, CaspSecStr8, Str3

from transformers import T5EncoderModel, T5Tokenizer, logging
import re


def create_padding_mask(x, mask_index):
    mask = torch.eq(x,mask_index).int()
    return mask.unsqueeze(1).unsqueeze(2).cuda()


def tokenize(seq, alphabet):
    a = alphabet.all_toks
    seq_encoded = np.array([a.index(s) for s in seq])
    return seq_encoded


def process_token(seq, alphabet):
    seq_encoded = tokenize(seq,alphabet)
    tokens = torch.empty(
        (
            1,
            len(seq) + int(alphabet.prepend_bos) + int(alphabet.append_eos),
        ),
        dtype=torch.int64,
    )

    tokens.fill_(alphabet.padding_idx)

    if alphabet.prepend_bos:
        tokens[0,0] = alphabet.cls_idx
    seq = torch.tensor(seq_encoded, dtype=torch.int64)
    tokens[0,int(alphabet.prepend_bos): len(seq_encoded)+ int(alphabet.prepend_bos),] = seq

    if alphabet.append_eos:
        tokens[0, len(seq_encoded) + int(alphabet.prepend_bos)] = alphabet.eos_idx
    return tokens


def save_checkpoint(epoch, model, optim, savename, best_val_loss):
    state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optim.state_dict(),
        'best_val_loss': best_val_loss,

    }
    torch.save(state, savename)

    return


def seed_everything(s):
    seed(s)
    os.environ['PYTHONHASHSEED'] = str(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True#False


def scatter_compute(src, out_size, index, type):
    out = torch.zeros(int(out_size)).cuda()
    if type == 'mean':
        output = scatter_mean(src, index, 0, out, 0)
    elif type == 'sum':
        output = scatter_add(src, index, 0, out, 0)

    return output


def preprocessing(f, str_type='dssp8', data_type='Proteinnet', train=True):

    data = pd.read_csv(f)
    dssp_encoder = SecStr8

    seq_list = data.loc[:, 'seq'].tolist()
    label_list = data.loc[:, str_type].tolist()

    if data_type == 'Netsurf':
        dssp_encoder = CaspSecStr8
        seq_list = [s.replace(' ','') for s in seq_list]
        label_list = [s.replace(' ','') for s in label_list]

    elif data_type == 'Casp13':
        seq_list = [s.replace(' ', '') for s in seq_list]
        label_list = [s.replace(' ', '') for s in label_list]

    if str_type == 'dssp3':
        dssp_encoder = Str3

    seqs = []
    labels = []
    lengths = []


    for i in range(len(seq_list)):
        if pd.isna(seq_list[i]):
            continue

        if train:
            if len(seq_list[i]) > 900:
                s = seq_list[i][:900]
                l = label_list[i][:900]
                leng = 900
            else:
                s = seq_list[i]
                l = label_list[i]
                leng = len(l)

        else:
            s = seq_list[i]
            l = label_list[i]
            leng = len(l)

        l = torch.from_numpy(dssp_encoder.encode(l.encode())).long()
        seqs.append(s)
        labels.append(l)
        lengths.append(leng)

    return seqs, labels, lengths


def random_start(l, cut):
    start = random.randint(0, l-cut-1)
    return start


class T5_collate_fn(object):
    def __init__(self, dssp_dim):
        logging.set_verbosity_warning()
        logging.set_verbosity_error()
        self.tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
        self.model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
        self.dssp_dim = dssp_dim

    def __call__(self, batch):
        seqs = [b[0] for b in batch]
        labels = [b[1] for b in batch]
        lengths = [b[2] for b in batch]

        device = 'cuda'
        self.model = self.model.to(device)
        self.model = self.model.eval()

        reps = []
        batch_index = []
        for i, s in enumerate(seqs):
            s = ' '.join(s)
            s = re.sub(r"[UZOB]", "X", s)
            ids = self.tokenizer.batch_encode_plus([s])
            input_ids = torch.tensor(ids['input_ids']).to(device)
            attention_mask = torch.tensor(ids['attention_mask']).to(device)
            with torch.no_grad():
                embedding = self.model(input_ids=input_ids, attention_mask=attention_mask)
            embedding = embedding.last_hidden_state.squeeze(0)
            reps.append(embedding[:-1, :])
            batch_index.append(torch.tensor([i] * lengths[i]))

        batch_index = torch.cat(batch_index)
        batch_labels = torch.cat(labels)
        rep_pack = torch.nn.utils.rnn.pad_sequence(reps, batch_first=True, padding_value=0)
        seq_pad = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=21)
        mask = create_padding_mask(seq_pad, 21)

        return batch_index, lengths, rep_pack, batch_labels, mask


class T5_train_collate_fn(object):
    def __init__(self, dssp_dim):
        self.tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False )
        self.model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
        self.dssp_dim = dssp_dim

    def __call__(self, batch):
        seqs = [b[0] for b in batch]
        labels = [b[1] for b in batch]
        lengths = [b[2] for b in batch]

        device = 'cuda'
        self.model = self.model.to(device)
        self.model = self.model.eval()

        reps = []
        batch_index = []

        cut = 500
        for i, s in enumerate(seqs):
            if lengths[i]>cut:
                start = random_start(lengths[i], cut)
                s = s[start:start+cut]
                lengths[i]=cut
                labels[i]=labels[i][start:start+cut]
            s = ' '.join(s)
            s = re.sub(r"[UZOB]", "X", s)

            ids = self.tokenizer.batch_encode_plus([s])
            input_ids = torch.tensor(ids['input_ids']).to(device)
            attention_mask = torch.tensor(ids['attention_mask']).to(device)
            with torch.no_grad():
                embedding = self.model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)

            embedding = embedding.last_hidden_state.squeeze(0)
            reps.append(embedding[:-1, :])
            batch_index.append(torch.tensor([i] * lengths[i]))

        batch_index = torch.cat(batch_index)
        batch_labels = torch.cat(labels)
        rep_pack = torch.nn.utils.rnn.pad_sequence(reps, batch_first=True, padding_value=0)

        seq_pad = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=21)
        mask = create_padding_mask(seq_pad, 21)

        return batch_index, lengths, rep_pack, batch_labels, mask

class T5_seg_att_collate_fn(object):
    def __init__(self, dssp_dim):
        self.tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False )
        self.model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
        self.dssp_dim = dssp_dim

    def __call__(self, batch):
        seqs = [b[0] for b in batch]
        labels = [b[1] for b in batch]
        lengths = [b[2] for b in batch]

        device = 'cuda'
        self.model = self.model.to(device)
        self.model = self.model.eval()

        reps = []
        atts = []
        batch_seg_label = []
        batch_index = []
        batch_seg_index = []

        cut = 500
        for i, s in enumerate(seqs):
            """if lengths[i]>cut:
                s = s[:500]
                lengths[i]=500
                labels[i]=labels[i][:500]"""
            s = ' '.join(s)
            s = re.sub(r"[UZOB]", "X", s)
            ids = self.tokenizer.batch_encode_plus([s])
            input_ids = torch.tensor(ids['input_ids']).to(device)
            attention_mask = torch.tensor(ids['attention_mask']).to(device)
            with torch.no_grad():
                embedding = self.model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
            attention = embedding.attentions[-1].squeeze(0)
            embedding = embedding.last_hidden_state.squeeze(0)
            atts.append(attention)
            reps.append(embedding[:-1, :])
            batch_index.append(torch.tensor([i] * lengths[i]))
            l = labels[i]
            l_mask = make_gt_seg(l, self.dssp_dim).reshape(-1)
            batch_seg_label.append(l_mask)
            batch_seg_index.append(torch.tensor([i]*l_mask.size(0)))

        batch_index = torch.cat(batch_index)
        batch_seg_index = torch.cat(batch_seg_index)#None
        batch_labels = torch.cat(labels)
        batch_seg_label = torch.cat(batch_seg_label)
        rep_pack = torch.nn.utils.rnn.pad_sequence(reps, batch_first=True, padding_value=0)
        att_pack = attention_padding(atts, lengths)#None
        seq_pad = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=21)
        mask = create_padding_mask(seq_pad, 21)

        return batch_index, batch_seg_index, lengths, rep_pack, att_pack, batch_labels, batch_seg_label, mask


class T5_seg_att_train_collate_fn(object):
    def __init__(self, dssp_dim):
        self.tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False )
        self.model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
        self.dssp_dim = dssp_dim

    def __call__(self, batch):
        seqs = [b[0] for b in batch]
        labels = [b[1] for b in batch]
        lengths = [b[2] for b in batch]

        device = 'cuda'
        self.model = self.model.to(device)
        self.model = self.model.eval()

        reps = []
        atts = []
        batch_seg_label = []
        batch_index = []
        batch_seg_index = []
        cut = 500 
        for i, s in enumerate(seqs):
            if lengths[i]>cut:
                start = random_start(lengths[i], cut)
                s = s[start:start+cut]
                lengths[i]=cut
                labels[i]=labels[i][start:start+cut]
            s = ' '.join(s)
            s = re.sub(r"[UZOB]", "X", s)

            ids = self.tokenizer.batch_encode_plus([s])
            input_ids = torch.tensor(ids['input_ids']).to(device)
            attention_mask = torch.tensor(ids['attention_mask']).to(device)
            with torch.no_grad():
                embedding = self.model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
            attention = embedding.attentions[-1].squeeze(0)
            embedding = embedding.last_hidden_state.squeeze(0)
            atts.append(attention)
            reps.append(embedding[:-1, :])
            batch_index.append(torch.tensor([i] * lengths[i]))
            l = labels[i]
            l_mask = make_gt_seg(l, self.dssp_dim).reshape(-1)
            batch_seg_label.append(l_mask)
            batch_seg_index.append(torch.tensor([i]*l_mask.size(0)))

        batch_index = torch.cat(batch_index)
        batch_seg_index = torch.cat(batch_seg_index)#None
        batch_labels = torch.cat(labels)
        batch_seg_label = torch.cat(batch_seg_label)
        rep_pack = torch.nn.utils.rnn.pad_sequence(reps, batch_first=True, padding_value=0)
        att_pack = attention_padding(atts, lengths) #None
        seq_pad = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=21)
        mask = create_padding_mask(seq_pad, 21)

        return batch_index, batch_seg_index, lengths, rep_pack, att_pack, batch_labels, batch_seg_label, mask

def attention_padding(atts, lengths):
    max_l = max(lengths)
    frame = torch.zeros(len(atts), atts[0].size(0), max_l, max_l).cuda()
    for i, att in enumerate(atts):
        frame[i, :, :lengths[i], :lengths[i]] = att[:,:-1,:-1]

    return frame


def make_gt_seg(l, num_class):
    l = nn.functional.one_hot(l, num_classes=num_class)
    l = l.transpose(-1, -2)
    l1 = l.unsqueeze(1)
    l2 = l.unsqueeze(2)
    l_ = l1 == l2
    tril = torch.ones(num_class, l_.size(-1), l_.size(
        -1)).tril().bool()  # torch.ones(7,l_.size(-1), l_.size(-1)).tril().bool().cuda()
    consecutive = torch.cumprod((tril + l_).bool(), axis=-1)
    l_upper = torch.triu(l.unsqueeze(-1).repeat(1, 1, l_.size(-1)) * consecutive)
    l_mask = (l_upper + l_upper.transpose(-1, -2)).bool().int()
    l_mask_all = l_mask.sum(dim=0)
    no_interaction = (1-l_mask_all)*(num_class)
    l_mask = torch.argmax(l_mask, dim=0)
    l_mask += no_interaction

    return l_mask

