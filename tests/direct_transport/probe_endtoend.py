"""Minimal probe to localize where B-group (EndToEnd) transfer breaks.

Runs on the 8-NPU machine. It mirrors what TestEndToEndTransfer does but
strips Ray out of the picture and reports the FIRST HIXL call that fails,
so we know whether the break is in init / connect / transfer / status.

Run from the ray-ascend repo dir:
    python3 tests/direct_transport/probe_endtoend.py

Three stages, each prints OK or the HIXL status/exception:
  1. SOURCE  : init engine + register a tensor (mirrors source actor side)
  2. DRIVER  : init a *second* engine in the SAME process, connect to source,
              transfer_async a READ, poll get_transfer_status
  3. VERIFY  : compare bytes

If stage 1 fails -> init/register problem (same class as the old 107002).
If stage 2 connect fails -> the connect path is the break (103900-ish).
If stage 2 transfer/status fails -> RDMA READ itself fails.
"""

import pickle

import hixl
import torch
import torch_npu  # noqa: F401  ensures torch.npu exists

# Two ports so the two in-process engines listen on distinct sockets.
# Single-process, two engines: if this *also* fails, the problem is not
# Ray-specific (no cross-process, no actor) and we've isolated it to HIXL.
SRC_ENGINE = "127.0.0.1:17001"
DST_ENGINE = "127.0.0.1:17002"


def stage_source():
    print("\n=== STAGE 1: SOURCE (init + register_mem) ===")
    torch.npu.set_device(0)
    src = hixl.Hixl()
    st = src.initialize(SRC_ENGINE, {})
    print(f"  src.initialize -> status={st} (SUCCESS={hixl.SUCCESS})")
    assert st == hixl.SUCCESS, f"src init failed status={st}"

    t = torch.arange(12, dtype=torch.float32, device="npu").reshape(3, 4)
    addr = t.untyped_storage().data_ptr()
    nbytes = t.untyped_storage().nbytes()
    print(f"  tensor addr=0x{addr:x} nbytes={nbytes}")

    st, handle = src.register_mem(hixl.MemDesc(addr, nbytes), hixl.MemType.MEM_DEVICE)
    print(f"  src.register_mem -> status={st} handle=0x{handle:x}")
    assert st == hixl.SUCCESS, f"src register_mem failed status={st}"

    # Serialize the remote address the way the transport does (raw data_ptr).
    meta = pickle.dumps([(addr, nbytes, "npu")])
    print("  STAGE 1 OK")
    return src, t, addr, nbytes, handle, meta


def stage_driver(src_engine_id, meta):
    print("\n=== STAGE 2: DRIVER (2nd engine, connect, transfer_async) ===")
    torch.npu.set_device(0)
    dst = hixl.Hixl()
    st = dst.initialize(DST_ENGINE, {})
    print(f"  dst.initialize -> status={st} (SUCCESS={hixl.SUCCESS})")
    assert st == hixl.SUCCESS, f"dst init failed status={st}"

    # Pre-allocate the receive buffer the way fetch_multiple_tensors does.
    recv = torch.empty(12, dtype=torch.float32, device="npu").reshape(3, 4)
    local_addr = recv.untyped_storage().data_ptr()
    local_nbytes = recv.untyped_storage().nbytes()
    print(f"  recv local_addr=0x{local_addr:x} nbytes={local_nbytes}")

    st, handle = dst.register_mem(
        hixl.MemDesc(local_addr, local_nbytes), hixl.MemType.MEM_DEVICE
    )
    print(f"  dst.register_mem -> status={st} handle=0x{handle:x}")
    assert st == hixl.SUCCESS, f"dst register_mem failed status={st}"

    print(f"  dst.connect({src_engine_id}) ...")
    st = dst.connect(src_engine_id)
    print(
        f"  dst.connect -> status={st} (SUCCESS={hixl.SUCCESS} "
        f"ALREADY_CONNECTED={getattr(hixl,'ALREADY_CONNECTED','?')})"
    )
    assert st == hixl.SUCCESS or st == getattr(
        hixl, "ALREADY_CONNECTED", -1
    ), f"dst connect failed status={st}"

    remote_mem_descs = pickle.loads(meta)
    remote_addr, remote_nbytes, _ = remote_mem_descs[0]
    assert local_nbytes == remote_nbytes, "size mismatch"
    op_descs = [hixl.TransferOpDesc(local_addr, remote_addr, remote_nbytes)]
    print(
        f"  dst.transfer_async(READ, local=0x{local_addr:x}, "
        f"remote=0x{remote_addr:x}, {remote_nbytes}B) ..."
    )
    st, transfer_req = dst.transfer_async(src_engine_id, hixl.TransferOp.READ, op_descs)
    print(f"  dst.transfer_async -> status={st} transfer_req={transfer_req}")
    assert st == hixl.SUCCESS, f"transfer_async failed status={st}"

    print("  polling get_transfer_status ...")
    import time

    deadline = time.monotonic() + 10
    last = None
    while time.monotonic() < deadline:
        st, ts = dst.get_transfer_status(transfer_req)
        last = (st, ts)
        if ts == hixl.TransferStatus.COMPLETED:
            print(f"  COMPLETED (status={st})")
            break
        if ts in (hixl.TransferStatus.FAILED, hixl.TransferStatus.TIMEOUT):
            print(f"  !!! transfer ended badly: status={st} ts={ts}")
            break
        time.sleep(0.01)
    else:
        print(f"  !!! timed out polling; last={last}")
    print("  STAGE 2 done")
    return recv


def main():
    print("env: torch", torch.__version__, "npus", torch.npu.device_count())
    src, src_t, addr, nbytes, src_handle, meta = stage_source()
    try:
        recv = stage_driver(SRC_ENGINE, meta)
        print("\n=== STAGE 3: VERIFY ===")
        got = recv.cpu().reshape(-1).tolist()
        expect = [float(x) for x in range(12)]
        ok = all(abs(a - b) < 1e-6 for a, b in zip(got, expect))
        print(f"  recv={got[:12]}")
        print(f"  expect={expect}")
        print(
            "  RESULT:",
            (
                "OK ✓ end-to-end RDMA works"
                if ok
                else "FAIL ✗ data mismatch (transfer ran but wrong bytes)"
            ),
        )
    finally:
        try:
            src.deregister_mem(src_handle)
        except Exception as e:
            print("  src.deregister_mem err:", e)
        try:
            src.finalize()
        except Exception as e:
            print("  src.finalize err:", e)


if __name__ == "__main__":
    main()
