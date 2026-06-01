# API Reference

> _Last updated: 05/30/2026_

## register_hccl_collective_backend

Register HCCL collective backend for Ray. Requires Ray >= 2.56.

```python
from ray_ascend import register_hccl_collective_backend

register_hccl_collective_backend()
```

Must be called in both the driver process and each actor's `__init__`.

## register_hccl_tensor_transport

Register HCCL backend and tensor transport for Ray. Requires Ray >= 2.56.

```python
from ray_ascend import register_hccl_tensor_transport

register_hccl_tensor_transport()
```

Must be called in both the driver process and each actor's `__init__`.

## register_yr_tensor_transport

Register YR tensor transport for Ray and initialize YR backend.

```python
from ray_ascend import register_yr_tensor_transport

register_yr_tensor_transport(["npu", "cpu"])
```

Must be called in both the driver process and each actor's `__init__`.

### Environment Variables

| Variable               | Default     | Description                                 |
| ---------------------- | ----------- | ------------------------------------------- |
| `YR_DS_INIT_MODE`      | `metastore` | Initialization mode (`metastore` or `etcd`) |
| `YR_DS_WORKER_PORT`    | `31501`     | YR DS worker port                           |
| `YR_DS_METASTORE_PORT` | `2379`      | Metastore service port                      |
| `YR_DS_ETCD_ADDRESS`   | -           | Etcd address (required for etcd mode)       |
