import pytest

pytest.importorskip("hixl", reason="HIXL tests require the hixl wheel")
pytest.importorskip("torch_npu", reason="HIXL tests require torch_npu + NPU hardware")

import pickle
import threading

import hixl
import ray
import torch

from ray.exceptions import RayDirectTransportError

from ray_ascend import register_hixl_tensor_transport
from ray_ascend.direct_transport.hixl_tensor_transport import (
    HixlCommunicatorMetadata,
    HixlFetchRequest,
    HixlTensorDesc,
    HixlTensorTransport,
    HixlTransportMetadata,
)
from ray.experimental.rdt.tensor_transport_manager import (
    CommunicatorMetadata,
    FetchRequest,
    TensorTransportManager,
    TensorTransportMetadata,
)

DEFAULT_NPU_COUNT = 2

@pytest.fixture(scope="session")
def ray_cluster_with_npu():
    """Ray cluster with NPU resources. Mirrors tests/conftest.py, kept local."""
    if not torch.npu.is_available():
        pytest.skip("NPU hardware not available")
    if torch.npu.device_count() < DEFAULT_NPU_COUNT:
        pytest.skip(
            f"HIXL transfer tests need {DEFAULT_NPU_COUNT} NPU devices, "
            f"only {torch.npu.device_count()} available"
        )
    if not ray.is_initialized():
        try:
            ray.init(
                ignore_reinit_error=True, resources={"NPU": DEFAULT_NPU_COUNT}
            )
        except ValueError:
            ray.init(ignore_reinit_error=True)
    yield
    if ray.is_initialized():
        ray.shutdown()


@pytest.fixture
def transport():
    """A bare HixlTensorTransport. teardown finalizes the engine if initialized."""
    t = HixlTensorTransport()
    yield t
    if t._hixl_initialized and t._hixl_engine is not None:
        try:
            t._hixl_engine.finalize()
        except Exception:
            pass
        t._hixl_initialized = False
        t._hixl_engine = None

class TestTransportProperties:
    """Verify static properties and class identity without hardware."""

    def test_tensor_transport_backend(self, transport):
        assert transport.tensor_transport_backend() == "HIXL"

    def test_is_one_sided(self):
        assert HixlTensorTransport.is_one_sided() is True

    def test_can_abort_transport(self):
        assert HixlTensorTransport.can_abort_transport() is True

    def test_inherits_tensor_transport_manager(self):
        assert issubclass(HixlTensorTransport, TensorTransportManager)

    def test_send_multiple_tensors_is_not_implemented(self, transport):
        with pytest.raises(NotImplementedError, match="one-sided"):
            transport.send_multiple_tensors(
                [],
                HixlTransportMetadata(tensor_meta=[], tensor_device=None),
                HixlCommunicatorMetadata(),
            )


class TestDataClasses:
    """Verify data class definitions, inheritance, and field layout."""

    def test_communicator_metadata_inherits(self):
        assert issubclass(HixlCommunicatorMetadata, CommunicatorMetadata)

    def test_transport_metadata_inherits(self):
        assert issubclass(HixlTransportMetadata, TensorTransportMetadata)

    def test_transport_metadata_fields(self):
        meta = HixlTransportMetadata(
            tensor_meta=[((2, 3), torch.float32)],
            tensor_device="npu",
            hixl_serialized_mem_descs=b"fake",
            hixl_engine_id="10.0.0.1:12345",
            hixl_mem_generation=0,
        )
        assert meta.hixl_serialized_mem_descs == b"fake"
        assert meta.hixl_engine_id == "10.0.0.1:12345"
        assert meta.hixl_mem_generation == 0

    def test_transport_metadata_no_duplicate_base_fields(self):
        base_fields = list(TensorTransportMetadata.__dataclass_fields__)
        child_fields = list(HixlTransportMetadata.__dataclass_fields__)
        new_fields = [f for f in child_fields if f not in base_fields]
        assert "hixl_serialized_mem_descs" in new_fields
        assert "hixl_engine_id" in new_fields
        assert "hixl_mem_generation" in new_fields

    def test_tensor_desc_fields(self):
        desc = HixlTensorDesc(
            mem_handle=42, nbytes=1024, mem_type_str="npu", metadata_count=1
        )
        assert desc.mem_handle == 42
        assert desc.nbytes == 1024
        assert desc.mem_type_str == "npu"
        assert desc.metadata_count == 1

    def test_fetch_request_inherits(self):
        assert issubclass(HixlFetchRequest, FetchRequest)

    def test_fetch_request_custom_fields(self):
        req = HixlFetchRequest(
            obj_id="test_obj",
            tensors=[],
            transfer_req=123,
            remote_engine_id="10.0.0.1:12345",
            remove_tensor_descs=True,
            transport=None,
        )
        assert req.transfer_req == 123
        assert req.remote_engine_id == "10.0.0.1:12345"
        assert req.remove_tensor_descs is True

class TestNpuMemoryRegistration:
    """Register/deregister NPU tensors against the real hixl engine."""

    @pytest.fixture(autouse=True)
    def _require_cluster(self, ray_cluster_with_npu):
        pass

    def test_register_new_npu_tensor(self, transport):
        transport._ensure_hixl_initialized()
        t = torch.randn(2, 3, device="npu")
        transport._add_tensor_descs([t])

        key = t.untyped_storage().data_ptr()
        assert key in transport._tensor_desc_cache
        desc = transport._tensor_desc_cache[key]
        assert desc.metadata_count == 1
        assert desc.mem_type_str == "npu"
        assert desc.mem_handle != 0

    def test_register_same_tensor_twice_bumps_ref_count(self, transport):
        transport._ensure_hixl_initialized()
        t = torch.randn(2, 3, device="npu")
        transport._add_tensor_descs([t])
        transport._add_tensor_descs([t])

        key = t.untyped_storage().data_ptr()
        assert transport._tensor_desc_cache[key].metadata_count == 2

    def test_partial_deregister_keeps_registration(self, transport):
        transport._ensure_hixl_initialized()
        t = torch.randn(2, 3, device="npu")
        transport._add_tensor_descs([t])
        transport._add_tensor_descs([t])

        transport._remove_tensor_descs([t])
        key = t.untyped_storage().data_ptr()
        assert key in transport._tensor_desc_cache
        assert transport._tensor_desc_cache[key].metadata_count == 1

    def test_full_deregister_bumps_meta_version(self, transport):
        transport._ensure_hixl_initialized()
        initial = transport._hixl_mem_generation

        t = torch.randn(2, 3, device="npu")
        transport._add_tensor_descs([t])
        transport._remove_tensor_descs([t])

        assert transport._hixl_mem_generation > initial

    def test_partial_deregister_does_not_bump_meta_version(self, transport):
        transport._ensure_hixl_initialized()
        initial = transport._hixl_mem_generation

        t = torch.randn(2, 3, device="npu")
        transport._add_tensor_descs([t])
        transport._add_tensor_descs([t])
        transport._remove_tensor_descs([t])
        assert transport._hixl_mem_generation == initial

    def test_tensor_memory_registered(self, transport):
        transport._ensure_hixl_initialized()
        t = torch.randn(2, 3, device="npu")
        assert transport._tensor_memory_registered(t) is False
        transport._add_tensor_descs([t])
        assert transport._tensor_memory_registered(t) is True

class TestMetadataExtraction:
    """extract_tensor_transport_metadata: register + serialize + store."""

    @pytest.fixture(autouse=True)
    def _require_cluster(self, ray_cluster_with_npu):
        pass

    def test_basic_extraction_npu(self, transport):
        transport._ensure_hixl_initialized()
        tensors = [torch.randn(2, 3, device="npu")]
        meta = transport.extract_tensor_transport_metadata("obj1", tensors)

        assert isinstance(meta, HixlTransportMetadata)
        assert meta.tensor_device == "npu"
        assert len(meta.tensor_meta) == 1
        assert meta.hixl_serialized_mem_descs is not None
        assert meta.hixl_engine_id == transport._local_engine_id
        assert meta.hixl_mem_generation == transport._hixl_mem_generation

    def test_metadata_stored_in_managed_meta(self, transport):
        transport._ensure_hixl_initialized()
        tensors = [torch.randn(2, 3, device="npu")]
        meta = transport.extract_tensor_transport_metadata("obj1", tensors)
        assert transport._get_meta("obj1") == meta

    def test_serialized_mem_descs_format(self, transport):
        transport._ensure_hixl_initialized()
        tensors = [torch.randn(2, 3, device="npu")]
        meta = transport.extract_tensor_transport_metadata("obj1", tensors)

        descs = pickle.loads(meta.hixl_serialized_mem_descs)
        assert len(descs) == 1
        _, nbytes, mem_type = descs[0]
        assert mem_type == "npu"
        assert nbytes == tensors[0].untyped_storage().nbytes()

    def test_multiple_tensors_serialization(self, transport):
        transport._ensure_hixl_initialized()
        tensors = [
            torch.randn(2, 3, device="npu"),
            torch.randn(4, device="npu"),
        ]
        meta = transport.extract_tensor_transport_metadata("obj1", tensors)
        descs = pickle.loads(meta.hixl_serialized_mem_descs)
        assert len(descs) == 2

    def test_contiguous_check_raises(self, transport):
        transport._ensure_hixl_initialized()
        t = torch.randn(2, 4, device="npu").t()
        with pytest.raises(ValueError, match="contiguous"):
            transport.extract_tensor_transport_metadata("obj1", [t])

    def test_empty_object_returns_none_fields(self, transport):
        transport._ensure_hixl_initialized()
        meta = transport.extract_tensor_transport_metadata("obj1", [])
        assert meta.hixl_serialized_mem_descs is None
        assert meta.hixl_engine_id is None
        assert meta.hixl_mem_generation is None
        assert meta.tensor_meta == []
        assert meta.tensor_device is None

    def test_get_communicator_metadata(self, transport):
        comm = transport.get_communicator_metadata(None, None)
        assert isinstance(comm, HixlCommunicatorMetadata)

class TestGarbageCollection:
    """garbage_collect: pop metadata, decrement ref count, deregister at zero."""

    @pytest.fixture(autouse=True)
    def _require_cluster(self, ray_cluster_with_npu):
        pass

    def test_gc_removes_meta_and_deregisters(self, transport):
        transport._ensure_hixl_initialized()
        tensors = [torch.randn(2, 3, device="npu")]
        meta = transport.extract_tensor_transport_metadata("obj1", tensors)

        transport.garbage_collect("obj1", meta, tensors)
        assert transport._get_meta("obj1") is None
        key = tensors[0].untyped_storage().data_ptr()
        assert key not in transport._tensor_desc_cache

    def test_gc_unknown_obj_id_is_noop(self, transport):
        transport._ensure_hixl_initialized()
        meta = HixlTransportMetadata(
            tensor_meta=[], tensor_device=None, hixl_serialized_mem_descs=None
        )
        transport.garbage_collect("unknown_obj", meta, [])  # must not raise

    def test_gc_shared_tensor_keeps_registration(self, transport):
        transport._ensure_hixl_initialized()
        t = torch.randn(2, 3, device="npu")
        meta1 = transport.extract_tensor_transport_metadata("obj1", [t])
        meta2 = transport.extract_tensor_transport_metadata("obj2", [t])

        transport.garbage_collect("obj1", meta1, [t])
        key = t.untyped_storage().data_ptr()
        assert key in transport._tensor_desc_cache
        assert transport._tensor_desc_cache[key].metadata_count == 1

    def test_gc_both_metadatas_fully_deregisters(self, transport):
        transport._ensure_hixl_initialized()
        t = torch.randn(2, 3, device="npu")
        meta1 = transport.extract_tensor_transport_metadata("obj1", [t])
        meta2 = transport.extract_tensor_transport_metadata("obj2", [t])

        transport.garbage_collect("obj1", meta1, [t])
        transport.garbage_collect("obj2", meta2, [t])
        key = t.untyped_storage().data_ptr()
        assert key not in transport._tensor_desc_cache

class TestAbortTransport:
    """abort_transport marks an obj_id so fetch raises and the wait loop exits.

    get_transfer_status is monkeypatched to always return WAITING (real RDMA
    completes too fast to pin the WAITING branch); everything else stays real.
    """

    @pytest.fixture(autouse=True)
    def _require_cluster(self, ray_cluster_with_npu):
        pass

    def test_abort_marks_obj_id(self, transport):
        transport.abort_transport("obj1", HixlCommunicatorMetadata())
        assert "obj1" in transport._aborted_transfer_obj_ids

    def test_aborted_fetch_raises_error(self, transport):
        transport._ensure_hixl_initialized()
        transport.abort_transport("obj1", HixlCommunicatorMetadata())
        tensors = [torch.randn(2, 3, device="npu")]
        meta = transport.extract_tensor_transport_metadata("obj1", tensors)

        with pytest.raises(RuntimeError, match="aborted"):
            transport.fetch_multiple_tensors("obj1", meta, HixlCommunicatorMetadata())

    def _waiting_status_fn(self):
        """get_transfer_status stub always returning (SUCCESS, WAITING).

        Keeps the loop in WAITING forever so only abort/timeout can exit it.
        """
        return lambda _transfer_req: (hixl.SUCCESS, hixl.TransferStatus.WAITING)

    def _make_fetch_request(self, transport, obj_id="obj_abort"):
        """Build a minimal HixlFetchRequest on a registered tensor.

        transfer_req is opaque to the stubbed get_transfer_status, so a
        placeholder int suffices.
        """
        transport._ensure_hixl_initialized()
        t = torch.randn(2, 3, device="npu")
        return HixlFetchRequest(
            obj_id=obj_id,
            tensors=[t],
            transfer_req=0,
            remote_engine_id="ignored",
            remove_tensor_descs=False,
            transport=None,
        )

    def test_aborted_wait_exits_immediately_without_timeout(self, transport):
        """Aborted + no timeout: must raise on the first WAITING tick, without
        evaluating the timeout (deadline is None). Guards against regressing to
        timeout-first order, which would miss the abort here.

        Poll-loop RuntimeErrors are wrapped as RayDirectTransportError (not a
        RuntimeError), so we assert on that.
        """
        req = self._make_fetch_request(transport, obj_id="obj_abort_only")
        transport.abort_transport("obj_abort_only", HixlCommunicatorMetadata())

        transport._hixl_engine.get_transfer_status = self._waiting_status_fn()
        with pytest.raises(RayDirectTransportError, match="aborted"):
            transport.wait_fetch_complete(req, timeout=-1)

    def test_abort_takes_priority_over_timeout(self, transport):
        """Abort + timeout both holding must report 'aborted', not 'timed out'.
        A user cancel must not be misreported as a passive timeout. Forced by
        aborting AND timeout=0 (deadline already past on the first tick).
        """
        req = self._make_fetch_request(transport, obj_id="obj_both")
        transport.abort_transport("obj_both", HixlCommunicatorMetadata())

        transport._hixl_engine.get_transfer_status = self._waiting_status_fn()
        with pytest.raises(RayDirectTransportError, match="aborted"):
            transport.wait_fetch_complete(req, timeout=0)

    def test_wait_exits_on_timeout_when_not_aborted(self, transport):
        """No abort + expired timeout must still raise TimeoutError. Ensures the
        abort-first swap didn't break the timeout path.
        """
        req = self._make_fetch_request(transport, obj_id="obj_timeout")

        transport._hixl_engine.get_transfer_status = self._waiting_status_fn()
        with pytest.raises(TimeoutError, match="timed out"):
            transport.wait_fetch_complete(req, timeout=0)

    def test_abort_removes_obj_id_from_set(self, transport):
        """wait_fetch_complete must remove obj_id from the aborted set on surfacing
        the abort; a stale entry would spuriously abort a reused obj_id.
        """
        req = self._make_fetch_request(transport, obj_id="obj_remove")
        transport.abort_transport("obj_remove", HixlCommunicatorMetadata())
        assert "obj_remove" in transport._aborted_transfer_obj_ids

        transport._hixl_engine.get_transfer_status = self._waiting_status_fn()
        with pytest.raises(RayDirectTransportError, match="aborted"):
            transport.wait_fetch_complete(req, timeout=-1)

        assert "obj_remove" not in transport._aborted_transfer_obj_ids

    def _waiting_status_with_signal(self, entered: threading.Event):
        """get_transfer_status stub: always WAITING, signals on first call.

        The first call means the worker is inside the poll loop; `entered` is
        the main thread's cue to call abort_transport.
        """
        def _stub(_transfer_req):
            entered.set()
            return (hixl.SUCCESS, hixl.TransferStatus.WAITING)

        return _stub

    def test_abort_from_another_thread_exits_in_flight_wait(self, transport):
        """abort_transport from another thread must interrupt a wait_fetch_complete
        already spinning in the WAITING branch.

        Worker calls wait_fetch_complete(timeout=-1); the stub keeps it in
        WAITING forever. Once the worker enters the loop (signalled by the
        first stub call), the main thread aborts; the worker must raise
        RayDirectTransportError('aborted') without hanging.
        """
        req = self._make_fetch_request(transport, obj_id="obj_xthread")

        entered = threading.Event()
        transport._hixl_engine.get_transfer_status = (
            self._waiting_status_with_signal(entered)
        )

        result: dict = {}

        def _wait():
            try:
                transport.wait_fetch_complete(req, timeout=-1)
            except RayDirectTransportError as e:
                result["exc"] = e
            except BaseException as e:
                result["exc"] = e

        worker = threading.Thread(target=_wait, name="hixl-wait-abort")
        worker.start()
        try:
            assert entered.wait(timeout=5), (
                "wait_fetch_complete never entered its poll loop (stub not "
                "called) — cannot exercise in-flight abort"
            )
            transport.abort_transport("obj_xthread", HixlCommunicatorMetadata())
            worker.join(timeout=5)
        finally:
            if worker.is_alive():
                transport.abort_transport(
                    "obj_xthread", HixlCommunicatorMetadata()
                )
                worker.join(timeout=2)

        assert not worker.is_alive(), (
            "wait_fetch_complete did not exit after cross-thread abort — "
            "the in-flight WAITING loop ignored the abort"
        )
        assert isinstance(result.get("exc"), RayDirectTransportError), (
            f"expected RayDirectTransportError('aborted'), "
            f"got {result.get('exc')!r}"
        )
        assert "aborted" in str(result["exc"])

class TestRemoteEngineCache:
    """LRU eviction, version-mismatch reconnect, and reuse semantics."""

    @pytest.fixture(autouse=True)
    def _require_cluster(self, ray_cluster_with_npu):
        pass

    def test_version_match_reuses_connection(self, transport):
        transport._ensure_hixl_initialized()
        local_id = transport._local_engine_id

        transport._connect_remote_engine(local_id, 0)
        assert local_id in transport._remote_engines
        size_after_first = len(transport._remote_engines)

        transport._connect_remote_engine(local_id, 0)
        assert len(transport._remote_engines) == size_after_first

    def test_version_mismatch_reconnects(self, transport):
        transport._ensure_hixl_initialized()
        local_id = transport._local_engine_id

        transport._connect_remote_engine(local_id, 0)
        transport._connect_remote_engine(local_id, 5)
        assert transport._remote_engines[local_id] == 5

    def test_lru_eviction(self, transport, monkeypatch):
        transport._ensure_hixl_initialized()
        import ray_ascend.direct_transport.hixl_tensor_transport as hixl_mod

        original = hixl_mod.HIXL_REMOTE_ENGINE_CACHE_MAXSIZE
        monkeypatch.setattr(hixl_mod, "HIXL_REMOTE_ENGINE_CACHE_MAXSIZE", 2)
        local_id = transport._local_engine_id

        try:
            ids = ["e_A", "e_B", local_id]
            for i, eid in enumerate(ids):
                try:
                    transport._connect_remote_engine(eid, i)
                except RuntimeError:
                    pass
            assert local_id in transport._remote_engines
        finally:
            monkeypatch.setattr(hixl_mod, "HIXL_REMOTE_ENGINE_CACHE_MAXSIZE", original)

@ray.remote(resources={"NPU": 1})
class _HixlHealthCheckActor:
    def __init__(self):
        register_hixl_tensor_transport(["npu", "cpu"])

    def health(self):
        from ray.experimental.rdt.util import get_tensor_transport_manager

        try:
            manager = get_tensor_transport_manager("HIXL")
            manager._ensure_hixl_initialized()
            return True
        except Exception:
            return False

@ray.remote(resources={"NPU": 1})
class _HixlSourceActor:
    def __init__(self):
        register_hixl_tensor_transport(["npu", "cpu"])

    @ray.method(tensor_transport="HIXL")
    def make_tensor(self):
        return torch.arange(12, dtype=torch.float32, device="npu").reshape(3, 4)

    def get_cache_state(self):
        """Introspect the actor-side HIXL cache for assertions.

        Returns engine id, tensor_desc_cache size, remote engine cache size,
        and init flag. The driver can't reach actor attributes, so this
        surfaces the state the cache-reuse test needs.
        """
        from ray.experimental.rdt.util import get_tensor_transport_manager

        mgr = get_tensor_transport_manager("HIXL")
        return {
            "driver_engine_id": mgr._local_engine_id,
            "tensor_desc_cache_size": len(mgr._tensor_desc_cache),
            "remote_engines_size": len(mgr._remote_engines),
            "hixl_initialized": mgr._hixl_initialized,
        }


class TestEndToEndTransfer:
    """Full RDMA READ between two NPU actors."""

    @pytest.fixture(autouse=True)
    def _require_cluster(self, ray_cluster_with_npu):
        pass

    def test_tensor_transport_via_rdt(self):
        """HIXL-decorated remote method returns a tensor transported via HIXL.
        Verifies content + shape + device."""
        source = _HixlSourceActor.remote()
        ref = source.make_tensor.remote()
        tensor = ray.get(ref)

        assert tensor.device.type == "npu"
        assert tensor.shape == (3, 4)
        expected = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        assert torch.equal(tensor.cpu(), expected)

    def test_two_source_tensors_transferred(self):
        """Two sequential HIXL transfers reuse actor-side state. Verifies across
        two make_tensor calls:
          - Engine stays initialized (no re-init per transfer).
          - tensor_desc_cache grows by one per transfer (distinct storages —
            expected RDT contract, not a leak).
          - remote engine cache stays at 1 (sole receiver -> reused connection).
        """
        source = _HixlSourceActor.remote()
        ref1 = source.make_tensor.remote()
        t1 = ray.get(ref1)
        state_after_first = ray.get(source.get_cache_state.remote())

        ref2 = source.make_tensor.remote()
        t2 = ray.get(ref2)
        state_after_second = ray.get(source.get_cache_state.remote())

        expected = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        assert torch.equal(t1.cpu(), expected)
        assert torch.equal(t2.cpu(), expected)

        assert state_after_first["hixl_initialized"] is True
        assert state_after_second["hixl_initialized"] is True

        assert state_after_first["tensor_desc_cache_size"] == 1
        assert state_after_second["tensor_desc_cache_size"] == 2

        assert state_after_first["remote_engines_size"] == 1
        assert state_after_second["remote_engines_size"] == 1
