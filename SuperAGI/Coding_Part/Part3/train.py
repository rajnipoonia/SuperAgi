from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

def train(rank, world_size):
    setup(rank, world_size)

    # FSDP setup requires wrapping model with FSDP
    model = FSDP(model)

    cleanup()

