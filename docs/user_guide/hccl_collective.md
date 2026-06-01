# HCCL Collective Communication

> _Last updated: 05/30/2026_

ray-ascend provides HCCL (Huawei Collective Communication Library) support for
distributed collective operations across Ray actors.

> **Note**: HCCL collective backend requires Ray >= 2.56.

## Available Collective Operations

- **broadcast**: Send data from one rank to all ranks
- **allreduce**: Combine data from all ranks and distribute the result
- **allgather**: Gather data from all ranks to each rank
- **reduce**: Combine data from all ranks to one rank
- **reducescatter**: Combine data and scatter the result to all ranks
- **send/recv**: Point-to-point communication
- **barrier**: Synchronize all ranks

## Quick Example

```python
import ray
import torch
from ray.util import collective
from ray_ascend import register_hccl_collective_backend

ray.init()
register_hccl_collective_backend()

@ray.remote(resources={"NPU": 1})
class Worker:
    def __init__(self):
        register_hccl_collective_backend()

    def do_allreduce(self, data):
        tensor = torch.tensor(data, dtype=torch.float32).npu()
        collective.allreduce(tensor, group_name="my_hccl_group")
        return tensor.cpu().tolist()

world_size = 2
actors = [Worker.remote() for _ in range(world_size)]

collective.create_collective_group(
    actors,
    world_size,
    list(range(world_size)),
    backend="HCCL",
    group_name="my_hccl_group",
)

results = ray.get([
    actors[i].do_allreduce.remote([1.0 * (i + 1), 2.0 * (i + 1)])
    for i in range(world_size)
])
print("Allreduce results:", results)

ray.shutdown()
```

## Point-to-Point Communication

HCCL supports send/recv operations between specific ranks in a collective group:

```python
import ray
import torch
from ray.util import collective
from ray_ascend import register_hccl_collective_backend

ray.init()
register_hccl_collective_backend()

@ray.remote(resources={"NPU": 1})
class Worker:
    def __init__(self):
        register_hccl_collective_backend()

    def send_tensor(self, data, dst_rank):
        tensor = torch.tensor(data, dtype=torch.float32).npu()
        collective.send(tensor, dst_rank=dst_rank, group_name="p2p_group")

    def recv_tensor(self, shape, src_rank):
        tensor = torch.zeros(shape, dtype=torch.float32).npu()
        collective.recv(tensor, src_rank=src_rank, group_name="p2p_group")
        return tensor.cpu().tolist()

world_size = 2
actors = [Worker.remote() for _ in range(world_size)]

collective.create_collective_group(
    actors,
    world_size,
    list(range(world_size)),
    backend="HCCL",
    group_name="p2p_group",
)

# Rank 0 sends to rank 1
ray.get(actors[0].send_tensor.remote([7.0, 8.0, 9.0], dst_rank=1))
result = ray.get(actors[1].recv_tensor.remote((3,), src_rank=0))
print("Received:", result)  # [7.0, 8.0, 9.0]

ray.shutdown()
```

## Supported Tensor Types

HCCL supports common PyTorch types:

- `int8`, `int16`, `int32`, `int64`
- `uint8`, `uint16`, `uint32`, `uint64`
- `float16`, `float32`, `float64`
- `bfloat16`

## Supported Reduce Operations

- `SUM`
- `PRODUCT`
- `MAX`
- `MIN`
