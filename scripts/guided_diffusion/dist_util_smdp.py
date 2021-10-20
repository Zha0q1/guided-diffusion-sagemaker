"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
# from mpi4py import MPI   ## SageMaker
import torch as th
# import torch.distributed as dist  ## SageMaker

## 1 SageMaker ##
import smdistributed.dataparallel.torch.distributed as dist

if not dist.is_initialized():
    dist.init_process_group()

import importlib  ## SageMaker

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3


## 2 SageMaker ##
def setup_dist():
    import json
    
    backend = "gloo" if not th.cuda.is_available() else "nccl"
    hostname = json.loads(os.environ['SM_HOSTS'])[0]
        
    rank = str(dist.get_rank())
    world_size = str(dist.get_world_size())
    local_rank = str(dist.get_local_rank())
    os.environ["SMDDP"] = "1"

    os.environ["CUDA_VISIBLE_DEVICES"] = local_rank
    os.environ["RANK"] = rank
    os.environ["LOCAL_RANK"] = local_rank
    os.environ["WORLD_SIZE"] = world_size
    os.environ["MASTER_ADDR"] = hostname
    
    
# def setup_dist_bak():
#     """
#     Setup a distributed process group.
#     """
#     if dist.is_initialized():
#         return
#     os.environ["CUDA_VISIBLE_DEVICES"] = f"{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}"

#     comm = MPI.COMM_WORLD
#     backend = "gloo" if not th.cuda.is_available() else "nccl"

#     if backend == "gloo":
#         hostname = "localhost"
#     else:
#         hostname = socket.gethostbyname(socket.getfqdn())
#     os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
#     os.environ["RANK"] = str(comm.rank)
#     os.environ["WORLD_SIZE"] = str(comm.size)

#     port = comm.bcast(_find_free_port(), root=0)
#     os.environ["MASTER_PORT"] = str(port)
#     dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    from mpi4py import MPI  ## 3. SageMaker ##
    
    chunk_size = 2 ** 30  # MPI has a relatively small size limit
#     if MPI.COMM_WORLD.Get_rank() == 0:  ## 4. SageMaker ##
    if os.environ["RANK"] == "0":
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
        num_chunks = len(data) // chunk_size
        if len(data) % chunk_size:
            num_chunks += 1
        MPI.COMM_WORLD.bcast(num_chunks)
        for i in range(0, len(data), chunk_size):
            MPI.COMM_WORLD.bcast(data[i : i + chunk_size])
    else:
        num_chunks = MPI.COMM_WORLD.bcast(None)
        data = bytes()
        for _ in range(num_chunks):
            data += MPI.COMM_WORLD.bcast(None)

    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
