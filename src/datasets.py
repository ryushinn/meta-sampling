import fastmerl_torch
from sampler import sample_on_merl_with_rejection, uniform_sampler

import torch
from torch.utils.data import Dataset

# TODO: avoid using _device
_device = "cuda" if torch.cuda.is_available() else "cpu"


# TODO: make it a bulk-loading dataset
class MerlDataset(Dataset):
    def __init__(self, merl, splr, nsamples, batch_size):
        if isinstance(merl, str):
            brdf = fastmerl_torch.Merl(merl)
        else:
            brdf = merl
        # self.merl = merl
        self.sampler = splr

        self.nsamples = nsamples
        # current batch index
        self.cbi = 0
        self.bs = batch_size
        self.num_batches = nsamples // batch_size
        assert self.num_batches > 0

        rangles, rvectors, brdf_vals = sample_on_merl_with_rejection(
            brdf, self.sampler, nsamples
        )

        self.rangles = rangles.to(_device)
        self.rvectors = rvectors.to(_device)
        self.brdf_vals = brdf_vals.to(_device)

    def __len__(self):
        return self.nsamples

    def __getitem__(self, indices):
        return (
            self.rangles[indices, :],
            self.rvectors[indices, :],
            self.brdf_vals[indices, :],
        )

    def shuffle(self):
        p = torch.randperm(self.nsamples)
        self.rangles = self.rangles[p, :]
        self.rvectors = self.rvectors[p, :]
        self.brdf_vals = self.brdf_vals[p, :]

    def next(self):
        if not (self.cbi < self.num_batches):
            self.cbi = 0
            self.shuffle()
        left = self.cbi * self.bs
        right = left + self.bs
        self.cbi += 1
        return (
            self.rangles[left:right, :],
            self.rvectors[left:right, :],
            self.brdf_vals[left:right, :],
        )

    def get_all(self):
        return self.rangles, self.rvectors, self.brdf_vals


def custom_collate(batch):
    tasks_train = []
    tasks_test = []
    for _task_train, _task_test in batch:
        tasks_train.append(_task_train)
        tasks_test.append(_task_test)
    return tasks_train, tasks_test


class MerlTaskset(Dataset):
    def __init__(self, merlPaths, n_test_samples=512):
        merls = []
        for path in merlPaths:
            merl = fastmerl_torch.Merl(path)
            merls.append(merl)

        test_sampler = uniform_sampler()
        task_test = []
        for merl in merls:
            dataset = MerlDataset(merl, test_sampler, 25000, batch_size=n_test_samples)
            task_test.append(dataset)
        self.task_test = task_test

        self.task_train = merls

        for merl in self.task_train:
            merl.to_(_device)

    def __len__(self):
        return len(self.task_train)

    def __getitem__(self, idx):
        return self.task_train[idx], self.task_test[idx]
