from torch.utils.data import Dataset


class DSSPDataset(Dataset):
    def __init__(self, seqs, labels, lengths):
        self.seqs = seqs
        self.labels = labels
        self.lengths = lengths

    def __getitem__(self, index):
        return self.seqs[index], self.labels[index], self.lengths[index]

    def __len__(self):
        return len(self.seqs)