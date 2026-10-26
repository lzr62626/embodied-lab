"""Wrap article7 windows with frozen training statistics; no fitting in getitem."""
from torch.utils.data import Dataset
from normalization import FIELDS


class NormalizedWindowDataset(Dataset):
    def __init__(self, base, normalizer):
        normalizer.verify_context(base.manifest, base.splits)
        self.base = base
        self.normalizer = normalizer

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        sample = self.base[index]  # The verified reader already returns independent storage.
        for field in FIELDS:
            sample['observations'][field] = self.normalizer.transform(field, sample['observations'][field])
        return sample
