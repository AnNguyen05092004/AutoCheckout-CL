"""Resume checkpoints for Spot VMs (R1).

A Spot VM can be stopped at any moment with at most 30 s notice, so the training state of the
current task is kept in <task dir>/last.ckpt: written at the end of every training epoch and,
within an epoch, every `every_minutes`, between batches, after an optimizer step. The file is written to a
temporary name and renamed, and the previous checkpoint is kept as last.ckpt.prev: Lightning's own
saving writes straight into the target file, so a stop during a write would leave it truncated.
"""

import os
import time

import pytorch_lightning as pl

LAST = 'last.ckpt'


def resume_path(task_dir):
    """last.ckpt of the task if training was interrupted, else None."""
    path = os.path.join(task_dir, LAST)
    return path if os.path.exists(path) else None


def remove_resume_checkpoints(task_dir):
    """Called once the task's final weights are saved; resume checkpoints are 0.5 GiB each."""
    for name in (LAST, LAST + '.prev', LAST + '.tmp'):
        path = os.path.join(task_dir, name)
        if os.path.exists(path):
            os.remove(path)


class ResumeCheckpoint(pl.Callback):
    def __init__(self, task_dir, every_minutes=30):
        self.path = os.path.join(task_dir, LAST)
        self.every_seconds = every_minutes * 60
        self._last_save = time.monotonic()
        self._global_step = None

    def on_train_start(self, trainer, pl_module):
        self._global_step = trainer.global_step
        self._last_save = time.monotonic()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Saved at the start of the next batch, when Lightning has already counted the previous
        # batch as completed: a checkpoint written in on_train_batch_end makes the resumed run
        # repeat that batch.
        stepped = trainer.global_step != self._global_step  # the previous batch ran an optimizer step
        self._global_step = trainer.global_step
        if stepped and time.monotonic() - self._last_save >= self.every_seconds:
            self.save(trainer)

    def on_train_epoch_end(self, trainer, pl_module):
        self.save(trainer)

    def save(self, trainer):
        tmp = self.path + '.tmp'
        trainer.save_checkpoint(tmp)
        if os.path.exists(self.path):
            os.replace(self.path, self.path + '.prev')
        os.replace(tmp, self.path)
        self._last_save = time.monotonic()
