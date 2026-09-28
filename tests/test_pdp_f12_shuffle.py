"""F12: on one device the training order changes every epoch and is reproducible with the seed."""

from types import SimpleNamespace

import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset

from main import make_train_loader


class Indices(Dataset):
    def __len__(self):
        return 12

    def __getitem__(self, i):
        return i

    @staticmethod
    def collate_fn(items):
        return torch.tensor(items)


class Recorder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.seen = []

    def training_step(self, batch, batch_idx):
        self.seen.extend(batch.tolist())
        return (self.w * batch.float()).sum()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def epoch_orders(shuffle, seed=42):
    pl.seed_everything(seed, workers=True)
    args = SimpleNamespace(batch_size=4, num_workers=0, shuffle=shuffle)
    model = Recorder()
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_epochs=2, logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(model, make_train_loader(Indices(), args))
    return model.seen[:12], model.seen[12:]


def test_order_differs_between_epochs_and_is_reproducible():
    first, second = epoch_orders(shuffle=1)
    assert sorted(first) == sorted(second) == list(range(12))
    assert first != second
    assert epoch_orders(shuffle=1) == (first, second)


def test_original_behaviour_without_shuffle():
    first, second = epoch_orders(shuffle=0)
    assert first == second == list(range(12))
