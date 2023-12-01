import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)

    # Create model, criterion, optimizer, data_loader as before
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank])


    cleanup()

