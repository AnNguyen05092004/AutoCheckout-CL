"""F5: the teacher is the frozen model of the previous task and infers in two passes with its prompts."""

import pytest
import torch

from pdp_helpers import make_batch, make_trainer, pdp_args, use_tiny_detr

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def task2_trainer(tmp_path_factory):
    monkeypatch = pytest.MonkeyPatch()
    use_tiny_detr(monkeypatch)
    args = pdp_args(tmp_path_factory.mktemp("f5"))
    trainer = make_trainer(args, task_id=2)
    trainer.set_teacher()
    yield trainer
    monkeypatch.undo()


def two_pass(model, batch):
    first = model(pixel_values=batch["pixel_values"], pixel_mask=batch["pixel_mask"], train=False)
    return model(pixel_values=batch["pixel_values"], pixel_mask=batch["pixel_mask"],
                 query=first.last_hidden_state, train=False)


def test_teacher_is_frozen_and_not_part_of_the_module(task2_trainer):
    teacher = task2_trainer.teacher
    assert not any(p.requires_grad for p in teacher.parameters())
    assert not any(name.startswith("teacher") for name in task2_trainer.state_dict())
    assert sum(p.numel() for p in task2_trainer.parameters()) == sum(p.numel() for p in teacher.parameters())
    # teacher uses the prompts of task 1 only; the student those of tasks 1-2
    assert teacher.model.prompts.task_count == 0
    assert task2_trainer.model.model.prompts.task_count == 1


def test_teacher_matches_student_with_same_weights_and_task(task2_trainer):
    trainer = task2_trainer
    batch = make_batch([[0], [1, 2]])
    trainer.model.eval()
    trainer.model.model.prompts.set_task_id(0)
    try:
        with torch.no_grad():
            expected = two_pass(trainer.model, batch)
        got = trainer.teacher_outputs(batch["pixel_values"], batch["pixel_mask"])
    finally:
        trainer.model.model.prompts.set_task_id(1)
    torch.testing.assert_close(got.logits, expected.logits)
    torch.testing.assert_close(got.last_hidden_state, expected.last_hidden_state)


def test_original_single_pass_without_prompts(task2_trainer):
    trainer = task2_trainer
    batch = make_batch([[0]])
    trainer.args.teacher_prompts = 0
    try:
        got = trainer.teacher_outputs(batch["pixel_values"], batch["pixel_mask"])
    finally:
        trainer.args.teacher_prompts = 1
    with torch.no_grad():
        expected = trainer.teacher(pixel_values=batch["pixel_values"], pixel_mask=batch["pixel_mask"],
                                   train=False)
    torch.testing.assert_close(got.logits, expected.logits)


def test_missing_previous_checkpoint_is_an_error(task2_trainer, tmp_path):
    with pytest.raises(FileNotFoundError):
        task2_trainer.resume(str(tmp_path / "task_final.pth"))
