"""Speed and memory of the PDP model on random images (plan 3.5 first measurement, R4 GPU smoke, V5).

    cd pdp && python benchmark.py <model arguments of an experiment> --bench_mode train --bench_steps 20
    cd pdp && python benchmark.py <model arguments> --bench_mode infer --bench_sizes 640 800

train: a full PDP step of a task >= 2 (teacher two passes + PPG, student query pass, training pass,
backward, optimizer step) on random 800x800 images with random boxes; reports seconds per image and
peak GPU memory, to choose --batch_size and to budget GPU hours.
infer: two-pass inference latency per image at each size, batch 1 (the checkout use case).
Also prints the storage added per class (private prompts, prototype memory).
"""

import argparse
import json
import time

import torch

import main as pdp_main
from engine import local_trainer
from inference import predict_batch


def random_batch(batch_size, size, n_objects, device, generator):
    boxes = torch.rand(batch_size, n_objects, 4, generator=generator) * 0.3
    boxes[..., :2] += 0.2
    labels = [{'class_labels': torch.randint(0, 100, (n_objects,), generator=generator).to(device),
               'boxes': b.to(device), 'orig_size': torch.tensor([size, size], device=device)} for b in boxes]
    pixels = torch.randn(batch_size, 3, size, size, generator=generator).to(device)
    return {'pixel_values': pixels, 'pixel_mask': torch.ones(batch_size, size, size, dtype=torch.long, device=device),
            'labels': labels}


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def bench_train(args, device):
    """A task-2 step: teacher (copy of the model) + PPG + student, like training with pseudo-labels."""
    trainer = local_trainer(train_loader=None, val_loader=None, test_dataset=None, args=args,
                            local_evaluator=argparse.Namespace(), task_id=2)
    if args.use_prompts:
        trainer.model.model.prompts.set_task_id(1)
    trainer.resume()
    if args.pseudo != 'none':
        trainer.set_teacher()
    trainer.to(device)
    if trainer.teacher is not None:
        trainer.teacher.to(device)
    trainer.train()
    optimizer = trainer.configure_optimizers()['optimizer']
    generator = torch.Generator().manual_seed(0)
    timings = []
    for step in range(args.bench_steps):
        batch = random_batch(args.batch_size, args.bench_train_size, 12, device, generator)
        synchronize(device)
        start = time.perf_counter()
        loss, _ = trainer.common_step(batch, step)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        synchronize(device)
        timings.append(time.perf_counter() - start)
    steady = timings[2:] or timings  # skip warm-up (kernel build, allocator)
    return {'mode': 'train', 'batch_size': args.batch_size, 'size': args.bench_train_size,
            'seconds_per_image': round(sum(steady) / len(steady) / args.batch_size, 4)}


def bench_infer(args, device):
    trainer = local_trainer(train_loader=None, val_loader=None, test_dataset=None, args=args,
                            local_evaluator=argparse.Namespace(), task_id=1)
    model = trainer.model.to(device).eval()
    generator = torch.Generator().manual_seed(0)
    result = {'mode': 'infer'}
    for size in args.bench_sizes:
        timings = []
        for _ in range(args.bench_steps):
            batch = random_batch(1, size, 1, device, generator)
            synchronize(device)
            start = time.perf_counter()
            predict_batch(model, batch['pixel_values'], batch['pixel_mask'],
                          torch.tensor([[size, size]], device=device), use_prompts=bool(args.use_prompts),
                          local_query=bool(args.local_query), seen_classes=100)
            synchronize(device)
            timings.append(time.perf_counter() - start)
        steady = timings[2:] or timings
        result[f'ms_per_image_{size}'] = round(1000 * sum(steady) / len(steady), 1)
    return result


def storage_per_class(args):
    """Bytes added per class: private prompts (p, k, a of every layer) and the prototype memory (V5)."""
    layers = 6
    prompt = layers * (args.prompt_len * 256 + 256 + 256) * 4
    memory = 100 * 256 * 4  # up to 100 cached query vectors of d_model 256, float32
    return {'prompt_bytes_per_class': prompt, 'prototype_memory_bytes_per_class': memory}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(parents=[pdp_main.get_args_parser()])
    parser.add_argument('--bench_mode', choices=['train', 'infer'], default='train')
    parser.add_argument('--bench_steps', type=int, default=20)
    parser.add_argument('--bench_sizes', type=int, nargs='+', default=[640, 800])
    parser.add_argument('--bench_train_size', type=int, default=800)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    pdp_main.check_kernel(args)
    pdp_main.setup_task_info(args)
    args.log_file = open('/dev/null', 'w')
    device = torch.device('cuda' if args.accelerator == 'gpu' else 'cpu')
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    result = bench_train(args, device) if args.bench_mode == 'train' else bench_infer(args, device)
    if device.type == 'cuda':
        result['peak_gpu_memory_gb'] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        result['gpu'] = torch.cuda.get_device_name(0)
    result.update(storage_per_class(args))
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
