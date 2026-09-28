"""R1: an interrupted task resumes from last.ckpt with its epoch, optimizer/scheduler and prototypes."""

import json
import math

import pytest
import torch

import engine
from pdp_helpers import run_main

pytestmark = pytest.mark.slow

ARGS = ("--n_tasks", "1", "--epochs", "3", "--eff_batch_size", "2", "--ckpt_every_minutes", "0")


class Interrupted(Exception):
    pass


def test_resume_after_interruption_in_the_last_epoch(tmp_path, monkeypatch):
    original_step = engine.local_trainer.training_step

    def failing_step(self, batch, batch_idx):
        if self.current_epoch == 2 and batch_idx == 2:  # after the prototype memory got its first update
            raise Interrupted
        return original_step(self, batch, batch_idx)

    monkeypatch.setattr(engine.local_trainer, "training_step", failing_step)
    with pytest.raises(Interrupted):
        run_main(tmp_path, monkeypatch, *ARGS)

    task_dir = tmp_path / "run" / "task_1"
    saved = torch.load(task_dir / "last.ckpt", map_location="cpu")
    assert (task_dir / "last.ckpt.prev").exists()
    assert saved["epoch"] == 2 and saved["global_step"] > 0
    assert sum(saved["pdp_state"]["class_cache_count"].values()) > 0

    resumed = {}

    def on_train_start(self):
        resumed.update(epoch=self.current_epoch, step=self.trainer.global_step,
                       scheduler=self.trainer.lr_scheduler_configs[0].scheduler.last_epoch,
                       cache=dict(self.class_cache_count))

    monkeypatch.setattr(engine.local_trainer, "training_step", original_step)
    monkeypatch.setattr(engine.local_trainer, "on_train_start", on_train_start, raising=False)
    monkeypatch.setattr(engine.local_trainer, "on_train_end",
                        lambda self: resumed.update(final_step=self.trainer.global_step), raising=False)
    run_main(tmp_path, monkeypatch, *ARGS)

    # The interrupted epoch continues with its remaining number of batches (in a new shuffled order:
    # the exact data order is not restored), so the total number of optimizer steps is unchanged.
    n_images = len(json.loads((tmp_path / "data/tasks/train_task_1.json").read_text())["images"])
    assert resumed["final_step"] == 3 * math.ceil(n_images / 2)

    assert resumed["epoch"] == saved["epoch"]
    assert resumed["step"] == saved["global_step"]
    assert resumed["scheduler"] == saved["lr_schedulers"][0]["last_epoch"] == 2
    assert resumed["cache"] == saved["pdp_state"]["class_cache_count"]
    assert (task_dir / "task_final.pth").exists()
    assert not list(task_dir.glob("last.ckpt*"))
