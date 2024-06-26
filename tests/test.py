from functools import partial
import os
from statistics import mean
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from decent_dp.utils import initialize_dist
rank, world_size = initialize_dist()

from tqdm import tqdm
from loguru import logger
import torch
import torchvision
from torch.utils.data import DistributedSampler, DataLoader
import torch.distributed as dist
from decent_dp.ddp import DecentralizedDataParallel as ddp
from decent_dp.optim import optim_fn_adam

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(784, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 10),
)

optim_fn = partial(optim_fn_adam, beta1=0.974, lr=1e-3 * world_size)
model = ddp(model,
            optim_fn,
            lr_scheduler_fn=None,
            topology='ring',
            profile_mode=True)

train_dataset = torchvision.datasets.MNIST(
    train=True,
    download=True,
    root='.',
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
)
valid_dataset = torchvision.datasets.MNIST(
    train=False,
    download=True,
    root='.',
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
)
train_sampler = DistributedSampler(train_dataset)
valid_sampler = DistributedSampler(valid_dataset, shuffle=False)

train_ds = DataLoader(train_dataset,
                      batch_size=256 // world_size, sampler=train_sampler)
valid_ds = DataLoader(valid_dataset, batch_size=256 // world_size, sampler=valid_sampler)

loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.025)

for epoch in range(20):
    train_sampler.set_epoch(epoch)
    model.train()
    avg_loss = 0.0
    avg_acc = 0.0
    with tqdm(train_ds, desc=f'Epoch {epoch + 1}', disable=rank!=0) as t:
        for step, (data, target) in enumerate(t):
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            avg_loss += loss.item()
            avg_acc += (output.argmax(1) == target).float().mean().item()
            t.set_postfix({'loss': f'{avg_loss / (step + 1):.5f}', 'acc': f'{avg_acc / (step + 1):.5f}'})
    if rank == 0:
        logger.info(f'Training loss: {avg_loss / (step + 1):.5f}, accuracy: {avg_acc / (step + 1):.5f}')
    
    model.global_avg()

    with torch.no_grad():
        model.eval()
        avg_loss = 0.0
        avg_acc = 0.0
        for step, (data, target) in enumerate(valid_ds):
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = loss_fn(output, target)
            avg_loss += loss.item()
            avg_acc += (output.argmax(1) == target).float().mean().item()
        if rank == 0:
            logger.info(f'Validation loss: {avg_loss / (step + 1):.5f}, accuracy: {avg_acc / (step + 1):.5f}')

time_stats = model.get_time_stats()
if rank == 0:
    logger.info(f'Average Iteration Time: {mean(time_stats["iter"]):.6f}')
    logger.info(f'Average Non-overlap Comm Time: {mean(time_stats["non_overlap_comm"]):.6f}')
