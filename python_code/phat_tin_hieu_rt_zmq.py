import argparse
import time

import numpy as np
import zmq

from gnss_utils import generateCAcode


def _make_code(prn_start: int, prn_end: int) -> np.ndarray:
    if prn_end < prn_start:
        raise ValueError("prn_end must be >= prn_start")
    return np.concatenate([generateCAcode(i) for i in range(prn_start, prn_end + 1)])


def _synthesize_iq_complex64(
    *,
    prn_start: int,
    prn_end: int,
    tau: int,
    Rc: float,
    fs: float,
    ft: float,
    n: np.ndarray,
    amplitude: float,
) -> np.ndarray:
    cacodes = _make_code(prn_start, prn_end)
    idx = (np.floor((n + tau) / fs * Rc).astype(np.int64)) % 10230
    r = cacodes[idx].astype(np.float32)
    phase = (2 * np.pi * ft * n / fs).astype(np.float32)
    i = (amplitude * r * np.cos(phase)).astype(np.float32)
    q = (amplitude * r * np.sin(phase)).astype(np.float32)
    return (i + 1j * q).astype(np.complex64)


def _write_interleaved_int16(path: str, iq: np.ndarray) -> None:
    i16 = np.empty(iq.size * 2, dtype=np.int16)
    i16[0::2] = np.round(np.real(iq)).astype(np.int16)
    i16[1::2] = np.round(np.imag(iq)).astype(np.int16)
    with open(path, "wb") as f:
        f.write(i16.tobytes())

def _write_complex64(path: str, iq: np.ndarray) -> None:
    iq.astype(np.complex64).tofile(path)


def _publish_zmq_realtime(
    *,
    address: str,
    iq: np.ndarray,
    fs: float,
    chunk: int,
    repeat: bool,
    pace: bool,
) -> None:
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.bind(address)
    time.sleep(0.2)

    i = 0
    t0 = time.perf_counter()
    sent = 0
    try:
        while True:
            if i + chunk > iq.size:
                if not repeat:
                    break
                i = 0
            block = iq[i : i + chunk]
            i += chunk

            sock.send(block.tobytes())
            sent += block.size

            if pace:
                target = sent / fs
                now = time.perf_counter() - t0
                dt = target - now
                if dt > 0:
                    time.sleep(dt)
    finally:
        sock.close(0)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phát tín hiệu GNSS giả lập (file hoặc ZMQ realtime).")
    p.add_argument("--mode", choices=["FILE", "ZMQ"], default="ZMQ", help="Chế độ phát.")
    p.add_argument("--address", default="tcp://127.0.0.1:5556", help="ZMQ bind address (PUB).")
    p.add_argument("--fs", type=float, default=2e6, help="Sample rate (Hz).")
    p.add_argument("--Rc", type=float, default=1.023e6, help="Chip rate (Hz).")
    p.add_argument("--ft", type=float, default=0, help="Doppler offset (Hz).")
    p.add_argument("--time_s", type=float, default=1.0, help="Thời lượng tín hiệu (giây).")
    p.add_argument("--chunk", type=int, default=10000, help="Số mẫu gửi mỗi gói (ZMQ).")
    p.add_argument("--pace", action="store_true", help="Pace theo fs để giống realtime.")
    p.add_argument("--repeat", action="store_true", help="Lặp lại tín hiệu khi hết data.")
    p.add_argument("--amplitude", type=float, default=1024.0, help="Biên độ I/Q (float).")
    p.add_argument("--tau", type=int, default=40, help="Độ trễ (samples) áp lên mã.")
    p.add_argument("--prn_start", type=int, default=11, help="PRN bắt đầu (mặc định TX1: 11-20).")
    p.add_argument("--prn_end", type=int, default=20, help="PRN kết thúc (mặc định TX1: 11-20).")
    p.add_argument("--out", default="data_tx1_prn_11_20_10MHz.bin", help="File output (mode=FILE).")
    args = p.parse_args(argv)

    N = int(args.fs * args.time_s)
    n = np.arange(N, dtype=np.float32)
    iq = _synthesize_iq_complex64(
        prn_start=args.prn_start,
        prn_end=args.prn_end,
        tau=args.tau,
        Rc=args.Rc,
        fs=args.fs,
        ft=args.ft,
        n=n,
        amplitude=args.amplitude,
    )

    if args.mode == "FILE":
        _write_complex64(args.out, iq)
        # _write_interleaved_int16(args.out, iq)
        # print(f"Đã lưu file int16 IQ xen kẽ: {args.out} (N={N}, fs={args.fs})")
        return 0

    print(f"ZMQ PUB bind: {args.address} | dtype=complex64 | chunk={args.chunk} | fs={args.fs}")
    _publish_zmq_realtime(
        address=args.address,
        iq=iq,
        fs=args.fs,
        chunk=args.chunk,
        repeat=args.repeat,
        pace=args.pace,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

