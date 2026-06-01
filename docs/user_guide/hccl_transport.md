# HCCL Tensor Transport

> _Last updated: 05/30/2026_

HCCL tensor transport enables zero-copy transfer of NPU tensors between Ray actors via
HCCS (Huawei Cache Coherence System).

> **Note**: HCCL tensor transport requires Ray >= 2.56.

## Quick Example

```python
import ray
import torch
from ray.util.collective import create_collective_group
from ray_ascend import register_hccl_tensor_transport

ray.init()
register_hccl_tensor_transport()

@ray.remote(resources={"NPU": 1})
class RayActor:
    def __init__(self):
        register_hccl_tensor_transport()

    @ray.method(tensor_transport="HCCL")
    def random_tensor(self):
        return torch.zeros(1024, device="npu")

    def sum(self, tensor: torch.Tensor):
        return torch.sum(tensor)

sender, receiver = RayActor.remote(), RayActor.remote()
group = create_collective_group([sender, receiver], backend="HCCL")

tensor = sender.random_tensor.remote()
result = receiver.sum.remote(tensor)
print(ray.get(result))

ray.shutdown()
```

## How It Works

`register_hccl_tensor_transport()` registers both the HCCL collective backend and the
HCCL tensor transport. It must be called in the driver process and in each actor's
`__init__`.

Under the hood, HCCL tensor transport uses Ray's `CollectiveTensorTransport`
infrastructure, which reuses the HCCL collective communicator for point-to-point tensor
transfers. A collective group must be created between the sender and receiver actors
before using `@ray.method(tensor_transport="HCCL")`.

## Supported Device Types

- **NPU**: Tensors on Ascend NPU devices (via HCCS)
