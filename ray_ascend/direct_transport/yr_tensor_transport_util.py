import struct
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import List

import torch

try:
    from yr.datasystem import KVClient

    YR_AVAILABLE = True
except ImportError:
    KVClient = None
    YR_AVAILABLE = False
    warnings.warn(
        "The 'yr_tensor_transport' feature requires optional dependencies"
        "'datasystem', Install with: 'pip install openyuanrong-datasystem'",
        RuntimeWarning,
    )

try:
    import torch_npu  # noqa: F401
    from yr.datasystem import DsTensorClient

    NPU_AVAILABLE = True
except ImportError:
    DsTensorClient = None
    NPU_AVAILABLE = False
    warnings.warn(
        "The 'yr_tensor_transport' feature requires optional dependencies "
        "'torch_npu'. CPU-only paths can still work, but NPU transport "
        "will be unavailable. Install with: 'pip install torch-npu'",
        RuntimeWarning,
    )


from abc import ABC, abstractmethod

from ray_ascend.utils.serial_utils import _decoder, _encoder


def raise_if_failed(failed_keys: List[str], action: str) -> None:
    """Raise RuntimeError if any keys failed.

    Args:
        failed_keys: List of keys that failed the operation.
        action: Description of the action (e.g., "put", "get", "delete").
    """
    if failed_keys:
        raise RuntimeError(f"Failed to {action} keys: {failed_keys}")


class BaseDSAdapter(ABC):
    """Base class for YR DS client adapters with batch processing support."""

    MAX_KEYS_PER_BATCH: int = 10000

    @abstractmethod
    def init(self) -> None:
        """Initialize the DS client connection."""
        pass

    def put(self, keys: List[str], tensors: List["torch.Tensor"]) -> None:
        """Store multiple objects with batch processing.

        Args:
            keys: List of keys to store.
            tensors: List of tensors to store.
        """
        batch_size = self.MAX_KEYS_PER_BATCH
        for i in range(0, len(keys), batch_size):
            self._put_batch(keys[i : i + batch_size], tensors[i : i + batch_size])

    @abstractmethod
    def _put_batch(self, keys: List[str], tensors: List["torch.Tensor"]) -> None:
        """Process a single batch of put operations.

        Args:
            keys: List of keys for this batch.
            tensors: List of tensors for this batch.
        """
        pass

    def get(self, keys: List[str], tensors: List["torch.Tensor"]) -> None:
        """Retrieve multiple objects with batch processing.

        Args:
            keys: List of keys to retrieve.
            tensors: List of tensors to populate with retrieved data.
        """
        batch_size = self.MAX_KEYS_PER_BATCH
        for i in range(0, len(keys), batch_size):
            self._get_batch(keys[i : i + batch_size], tensors[i : i + batch_size])

    @abstractmethod
    def _get_batch(self, keys: List[str], tensors: List["torch.Tensor"]) -> None:
        """Process a single batch of get operations.

        Args:
            keys: List of keys for this batch.
            tensors: List of tensors for this batch.
        """
        pass

    def delete(self, keys: List[str]) -> None:
        """Delete multiple keys with batch processing.

        Args:
            keys: List of keys to delete.
        """
        batch_size = self.MAX_KEYS_PER_BATCH
        for i in range(0, len(keys), batch_size):
            self._delete_batch(keys[i : i + batch_size])

    @abstractmethod
    def _delete_batch(self, keys: List[str]) -> None:
        """Process a single batch of delete operations.

        Args:
            keys: List of keys for this batch.
        """
        pass


class CPUClientAdapter(BaseDSAdapter):
    """DS client adapter for CPU tensors using structured binary packing."""

    # Header: number of entries (uint32, little-endian)
    HEADER_FMT: str = "<I"
    HEADER_SIZE: int = struct.calcsize(HEADER_FMT)
    # Entry: (payload_offset: uint32, payload_size: uint32)
    ENTRY_FMT: str = "<II"
    ENTRY_SIZE: int = struct.calcsize(ENTRY_FMT)

    DS_MAX_WORKERS: int = 4

    def __init__(self, host: str, port: int):
        """Initialize CPUClientAdapter with DS server address.

        Args:
            host: DS server host address.
            port: DS server port.

        Raises:
            RuntimeError: If 'datasystem' dependency is not installed.
        """
        if not YR_AVAILABLE:
            raise RuntimeError(
                "Missing optional dependency 'datasystem'. Install with: "
                "'pip install openyuanrong-datasystem' to use CPUClientAdapter."
            )
        self._client = KVClient(host=host, port=port)
        self.local_tensors: List["torch.Tensor"] = []

    def init(self) -> None:
        """Initialize the KV client connection."""
        self._client.init()

    @classmethod
    def calc_packed_size(cls, items: List[memoryview]) -> int:
        """Calculate the total size (in bytes) required to pack a list of memoryview items
        into the structured binary format used by pack_into.

        Args:
            items: List of memoryview objects to be packed.

        Returns:
            Total buffer size in bytes.
        """
        return (
            cls.HEADER_SIZE
            + len(items) * cls.ENTRY_SIZE
            + sum(item.nbytes for item in items)
        )

    @classmethod
    def pack_into(cls, target: memoryview, items: List[memoryview]) -> None:
        """Pack multiple contiguous buffers into a single buffer.
            ┌───────────────┐
            │ item_count    │  uint32
            ├───────────────┤
            │ entries       │  N * item entries
            ├───────────────┤
            │ payload blob  │  N * concatenated buffers
            └───────────────┘

        Args:
            target: A writable memoryview returned by StateValueBuffer.MutableData().
                It must be large enough to accommodate the total number of bytes of HEADER + ENTRY_TABLE + all items.
                This buffer is usually mapped to shared memory or Zero-Copy memory area.
            items: List of read-only memory views (e.g., from serialized objects).
                Each item must support the buffer protocol and be readable as raw bytes.

        """
        struct.pack_into(cls.HEADER_FMT, target, 0, len(items))

        entry_offset = cls.HEADER_SIZE
        payload_offset = cls.HEADER_SIZE + len(items) * cls.ENTRY_SIZE

        target_tensor = torch.frombuffer(target, dtype=torch.uint8)

        for item in items:
            struct.pack_into(
                cls.ENTRY_FMT, target, entry_offset, payload_offset, item.nbytes
            )
            src_tensor = torch.frombuffer(item, dtype=torch.uint8)
            target_tensor[payload_offset : payload_offset + item.nbytes].copy_(
                src_tensor
            )
            entry_offset += cls.ENTRY_SIZE
            payload_offset += item.nbytes

    @classmethod
    def unpack_from(cls, source: memoryview) -> List[memoryview]:
        """Unpack multiple contiguous buffers from a single packed buffer.

        Args:
            source: The packed source buffer.

        Returns:
            List of unpacked contiguous buffers.
        """
        mv = memoryview(source)
        item_count = struct.unpack_from(cls.HEADER_FMT, mv, 0)[0]
        offsets = []
        for i in range(item_count):
            offset, length = struct.unpack_from(
                cls.ENTRY_FMT, mv, cls.HEADER_SIZE + i * cls.ENTRY_SIZE
            )
            offsets.append((offset, length))
        return [mv[offset : offset + length] for offset, length in offsets]

    def _put_batch(self, keys: List[str], tensors: List["torch.Tensor"]) -> None:
        """Process a single batch of put operations.

        Args:
            keys: List of keys for this batch.
            tensors: List of tensors for this batch.

        Raises:
            RuntimeError: If any keys fail to be put.
        """
        items_list = [[memoryview(b) for b in _encoder.encode(obj)] for obj in tensors]
        packed_sizes = [self.calc_packed_size(items) for items in items_list]
        buffers = self._client.mcreate(keys, packed_sizes)
        tasks = [
            (target.MutableData(), item)
            for target, item in zip(buffers, items_list, strict=True)
        ]
        num_workers = min(self.DS_MAX_WORKERS, len(tasks))
        if num_workers == 1:
            for p in tasks:
                self.pack_into(*p)
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                list(executor.map(lambda p: self.pack_into(*p), tasks))
        self._client.mset_buffer(buffers)

    def _get_batch(self, keys: List[str], tensors: List["torch.Tensor"]) -> None:
        """Process a single batch of get operations.

        Args:
            keys: List of keys for this batch.
            tensors: List of tensors to populate with retrieved data.

        Raises:
            RuntimeError: If any key fails to be retrieved.
        """
        buffers = self._client.get_buffers(keys)
        for i, buffer in enumerate(buffers):
            if buffer is None:
                raise RuntimeError(f"Failed to get key: {keys[i]}")
            tensors[i] = _decoder.decode(self.unpack_from(buffer))

    def _delete_batch(self, keys: List[str]) -> None:
        """Process a single batch of delete operations.

        Args:
            keys: List of keys for this batch.

        Raises:
            RuntimeError: If any keys fail to be deleted.
        """
        failed_keys = self._client.delete(keys=keys)
        raise_if_failed(failed_keys, "delete")

    def health_check(self) -> bool:
        """Check if the DS client is healthy.

        Returns:
            True if the client is healthy, False otherwise.
        """
        is_healthy: bool = self._client.health_check().is_ok()
        return is_healthy


class NPUClientAdapter(BaseDSAdapter):
    """DS client adapter for NPU tensors using device-direct operations."""

    def __init__(self, host: str, port: int):
        """Initialize NPUClientAdapter with DS server address.

        Args:
            host: DS server host address.
            port: DS server port.

        Raises:
            RuntimeError: If 'datasystem' or NPU support is not installed.
        """
        if not NPU_AVAILABLE:
            raise RuntimeError(
                "Missing optional dependency 'datasystem' or NPU support. Install with: "
                "'pip install torch-npu' and 'pip install openyuanrong-datasystem' "
                "to ensure NPU support is available."
            )
        self._client = DsTensorClient(
            host=host,
            port=port,
            device_id=0,
            connect_timeout_ms=60000,
        )

    def init(self) -> None:
        """Initialize the DsTensorClient connection."""
        self._client.init()

    def _put_batch(self, keys: List[str], tensors: List["torch.Tensor"]) -> None:
        """Process a single batch of put operations for NPU tensors.

        Args:
            keys: List of keys for this batch.
            tensors: List of NPU tensors for this batch.

        Raises:
            RuntimeError: If any keys fail to be put.
        """
        failed_keys = self._client.dev_mset(keys=keys, tensors=tensors)
        raise_if_failed(failed_keys, "put")

    def _get_batch(self, keys: List[str], tensors: List["torch.Tensor"]) -> None:
        """Process a single batch of get operations for NPU tensors.

        Args:
            keys: List of keys for this batch.
            tensors: List of NPU tensors to populate with retrieved data.

        Raises:
            RuntimeError: If any keys fail to be retrieved.
        """
        failed_keys = self._client.dev_mget(keys=keys, tensors=tensors)
        raise_if_failed(failed_keys, "get")

    def _delete_batch(self, keys: List[str]) -> None:
        """Process a single batch of delete operations for NPU tensors.

        Args:
            keys: List of keys for this batch.

        Raises:
            RuntimeError: If any keys fail to be deleted.
        """
        failed_keys = self._client.dev_delete(keys=keys)
        raise_if_failed(failed_keys, "delete")
