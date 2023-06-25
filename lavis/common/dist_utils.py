"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import datetime
import functools
import os
import colorsys
import datetime
import torch
import torch.distributed as dist
import timm.models.hub as timm_hub
import socket
import subprocess
import deepspeed
import time

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def find_free_port():
    s = socket.socket()
    s.bind(('', 0))  # Bind to a free port provided by the host.
    return s.getsockname()[1]  # Return the port number assigned.


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        if int(os.environ["RANK"]) == 0:
            print('this task is not running on cluster!')
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.dist_url = 'env://'
        os.environ['LOCAL_SIZE'] = str(torch.cuda.device_count())
        # addr = socket.gethostname()
        if os.environ.get('MASTER_ADDR_OVERRIDE', None):
            os.environ['MASTER_ADDR'] = os.environ['MASTER_ADDR_OVERRIDE']
        if os.environ.get('MASTER_PORT_OVERRIDE', None):
            os.environ['MASTER_PORT'] = os.environ['MASTER_PORT_OVERRIDE']

        addr = os.environ['MASTER_ADDR']
        # addr = socket.gethostbyname(os.environ['MASTER_ADDR'])

    elif 'SLURM_PROCID' in os.environ:
        proc_id = int(os.environ['SLURM_PROCID'])
        if proc_id == 0:
            print('Init dist using slurm!')
            print("Job Id is {} on {} ".format(os.environ["SLURM_JOBID"], os.environ['SLURM_NODELIST']))
        ntasks = int(os.environ['SLURM_NTASKS'])
        # node_list = os.environ['SLURM_NODELIST']
        node_list = os.environ['SLURM_STEP_NODELIST']
        # node_list = os.environ['SLURM_STEP_NODELIST']
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput('scontrol show hostname {} | head -n1'.format(node_list))
        jobid = os.environ["SLURM_JOBID"]
        hostfile = "dist_url_" + jobid + ".txt"
        if proc_id == 0:
            args.tcp_port = str(find_free_port())
            print('write port {} to file: {} '.format(args.tcp_port, hostfile))
            with open(hostfile, "w") as f:
                f.write(args.tcp_port)
        else:
            print('read port from file: {}'.format(hostfile))
            while not os.path.exists(hostfile):
                time.sleep(1)
            time.sleep(2)
            with open(hostfile, "r") as f:
                args.tcp_port = f.read()

        os.environ['MASTER_PORT'] = str(args.tcp_port)
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
        os.environ['LOCAL_SIZE'] = str(num_gpus)
        args.dist_url = 'env://'
        args.world_size = ntasks
        args.rank = proc_id
        args.gpu = proc_id % num_gpus
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('rank: {} addr: {}  port: {}'.format(args.rank, addr, os.environ['MASTER_PORT']))
    print(f' rank {proc_id} world size: {args.world_size}, local rank: {args.gpu}, local size: {os.environ["LOCAL_SIZE"]}')
    # torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    # import pdb; pdb.set_trace()
    deepspeed.init_distributed(dist_backend=args.dist_backend,  init_method=args.dist_url)
    torch.distributed.barrier()
    if 'SLURM_PROCID' in os.environ and args.rank == 0:
        if os.path.isfile(hostfile):
            os.remove(hostfile)
            # init method


def get_dist_info():
    if torch.__version__ < "1.0":
        initialized = dist._initialized
    else:
        initialized = dist.is_initialized()
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:  # non-distributed training
        rank = 0
        world_size = 1
    return rank, world_size


def main_process(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper


def download_cached_file(url, check_hash=True, progress=False):
    """
    Download a file from a URL and cache it locally. If the file already exists, it is not downloaded again.
    If distributed, only the main process downloads the file, and the other processes wait for the file to be downloaded.
    """

    def get_cached_file_path():
        # a hack to sync the file path across processes
        parts = torch.hub.urlparse(url)
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(timm_hub.get_cache_dir(), filename)

        return cached_file

    if is_main_process():
        timm_hub.download_cached_file(url, check_hash, progress)

    if is_dist_avail_and_initialized():
        dist.barrier()

    return get_cached_file_path()
