# API Reference

> _Last updated: 06/02/2026_

______________________________________________________________________

## register_yr_tensor_transport

Register YR tensor transport for Ray and initialize YR backend.

```python
from ray_ascend import register_yr_tensor_transport

register_yr_tensor_transport(["npu", "cpu"])
```

**Parameters:**

- `devices` – List of device types to support. Can be:
    - `["npu"]` for NPU tensors only
    - `["npu", "cpu"]` for NPU and CPU tensors
    - `["cpu"]` for CPU tensors only

**Returns:** `None`

**Raises:**

- `ImportError` – YR tensor transport requires the `[yr]` extra dependency. Install
  with: `pip install "ray-ascend[yr]"`
- `ValueError` – When `devices` is `None`. Specify a list of device types, e.g.,
  `["npu", "cpu"]`

**Notes:**

- Requires Ray >= 2.55
- Must be called in both the driver process and each actor's `__init__`
- Environment variables should be set in the driver process before calling
- YR backend initialization happens once across the cluster via a named actor
- The `tensor_transport` argument is case-insensitive. Both `"YR"` and `"yr"` work, as
  do `"HCCL"` and `"hccl"`.

**Environment Variables:**

| Variable               | Default     | Description                                 |
| ---------------------- | ----------- | ------------------------------------------- |
| `YR_DS_INIT_MODE`      | `metastore` | Initialization mode (`metastore` or `etcd`) |
| `YR_DS_WORKER_PORT`    | `31501`     | openYuanRong Datasystem worker port         |
| `YR_DS_METASTORE_PORT` | `2379`      | Metastore service port                      |
| `YR_DS_ETCD_ADDRESS`   | -           | Etcd address (required for etcd mode)       |

**Examples:**

```python
import os
import ray
import torch
from ray_ascend import register_yr_tensor_transport

os.environ["YR_DS_INIT_MODE"] = "metastore"
os.environ["YR_DS_WORKER_PORT"] = "31501"

ray.init()

register_yr_tensor_transport(["npu", "cpu"])

@ray.remote(resources={"NPU": 1})
class RayActor:
    def __init__(self):
        register_yr_tensor_transport(["npu", "cpu"])

    @ray.method(tensor_transport="YR")
    def transfer_npu_tensor_via_hccs(self):
        return torch.zeros(1024, device="npu")

    @ray.method(tensor_transport="YR")
    def transfer_cpu_tensor_via_rdma(self):
        return torch.zeros(1024)

actor = RayActor.remote()

npu_tensor = ray.get(actor.transfer_npu_tensor_via_hccs.remote())
print(f"NPU tensor transferred via HCCS: device={npu_tensor.device}, shape={npu_tensor.shape}")

cpu_tensor = ray.get(actor.transfer_cpu_tensor_via_rdma.remote())
print(f"CPU tensor transferred via RDMA: device={cpu_tensor.device}, shape={cpu_tensor.shape}")

ray.shutdown()
```

## cleanup_yr_resources

Clean up all YR resources. Shuts down the YR manager and releases all associated
resources.

```python
from ray_ascend.utils import cleanup_yr_resources

cleanup_yr_resources()
```

**Parameters:** `None`

**Returns:** `None`

**Raises:** `None`

**Notes:**

- `ray stop` can also clean up YR workers

**Examples:**

```python
import ray
from ray_ascend import register_yr_tensor_transport
from ray_ascend.utils import cleanup_yr_resources

ray.init()
register_yr_tensor_transport(["npu", "cpu"])

# ... your Ray application ...

cleanup_yr_resources()
ray.shutdown()
```

## register_hccl_collective_backend

Register HCCL collective backend for Ray.

```python
from ray_ascend import register_hccl_collective_backend

register_hccl_collective_backend()
```

**Parameters:** `None`

**Returns:** `None`

**Raises:**

- `RuntimeError` – Requires Ray >= 2.56. Upgrade with: `pip install 'ray>=2.56'`

**Notes:**

- Requires Ray >= 2.56
- Must be called in both the driver process and each actor's `__init__`
- After registration, use Ray's `ray.util.collective` interface with `backend="HCCL"`

**Examples:**

```python
import ray
import torch
import torch_npu
from ray.util import collective
from ray_ascend import register_hccl_collective_backend

ray.init()
register_hccl_collective_backend()

@ray.remote(resources={"NPU": 1})
class RayActor:
    def __init__(self):
        register_hccl_collective_backend()

    def init_data(self):
        self.data = torch.tensor([1.0]).npu()

    def get_data(self):
        return self.data.cpu()

    def allreduce(self):
        collective.allreduce(self.data, group_name="my_group")

actors = [RayActor.remote() for _ in range(4)]

collective.create_collective_group(
    actors,
    len(actors),
    list(range(0, len(actors))),
    backend="HCCL",
    group_name="my_group",
)

ray.get([actor.init_data.remote() for actor in actors])

ray.get([actor.allreduce.remote() for actor in actors])

results = ray.get([actor.get_data.remote() for actor in actors])
print(results)
```

## register_hccl_tensor_transport

Register HCCL backend and tensor transport for Ray.

```python
from ray_ascend import register_hccl_tensor_transport

register_hccl_tensor_transport()
```

**Parameters:** `None`

**Returns:** `None`

**Raises:**

- `RuntimeError` – Requires Ray >= 2.56. Upgrade with: `pip install 'ray>=2.56'`

**Notes:**

- Requires Ray >= 2.56
- Must be called in both the driver process and each actor's `__init__`
- Internally calls `register_hccl_collective_backend()` and registers tensor transport
- Use with `@ray.method(tensor_transport="HCCL")` for zero-copy tensor transfer

**Examples:**

```python
import ray
import torch
import torch_npu
from ray.util import collective
from ray_ascend import register_hccl_tensor_transport

ray.init()
register_hccl_tensor_transport()

@ray.remote(resources={"NPU": 1})
class RayActor:
    def __init__(self):
        register_hccl_tensor_transport()

    @ray.method(tensor_transport="HCCL")
    def transfer_npu_tensor(self):
        return torch.tensor([1, 2, 3]).npu()

    def sum(self, tensor: torch.Tensor):
        return torch.sum(tensor)

actors = [RayActor.remote(), RayActor.remote()]

collective.create_collective_group(
    actors,
    len(actors),
    list(range(0, len(actors))),
    backend="HCCL",
    group_name="hccl_group",
)

sender, receiver = actors[0], actors[1]

tensor = sender.transfer_npu_tensor.remote()
result = ray.get(receiver.sum.remote(tensor))
print(f"Sum of transferred tensor: {result}")
```
