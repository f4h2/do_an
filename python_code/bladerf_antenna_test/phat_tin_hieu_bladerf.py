"""
phat_tin_hieu_bladerf.py
========================
Phát tín hiệu GNSS giả lập trực tiếp qua BladeRF x40 dùng libbladeRF.
Tương đương phat_tin_hieu_rt_zmq.py nhưng thay ZMQ bằng BladeRF TX.

Cấu trúc tín hiệu:
  - Ghép C/A code nhiều PRN liên tiếp (mặc định PRN 11–20)
  - Điều chế BPSK lên sóng mang ft (baseband offset, Hz)
  - Định dạng SC16_Q11 interleaved int16 cho BladeRF

Chạy:
    conda activate gnss_env
    python phat_tin_hieu_bladerf.py
    python phat_tin_hieu_bladerf.py --prn_start 21 --prn_end 30 --ft 100

Argparse:
    --freq_hz   : Tần số RF sóng mang (Hz), mặc định 1575.42e6 (GPS L1)
    --fs        : Sample rate (Hz), mặc định 2e6
    --ft        : Doppler offset baseband (Hz), mặc định 0
    --tau       : Độ trễ code (samples), mặc định 40
    --prn_start : PRN bắt đầu, mặc định 11
    --prn_end   : PRN kết thúc, mặc định 20
    --amplitude : Biên độ (0–2047 cho SC16_Q11), mặc định 1024
    --tx_gain   : Gain TX dB, mặc định 60
    --time_s    : Thời lượng 1 vòng tín hiệu (giây), mặc định 1.0
"""

import argparse
import sys
import time

import numpy as np

# --- Import gnss_utils từ thư mục cha ---
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gnss_utils import generateCAcode

try:
    from bladerf import _bladerf
    HAS_BLADERF = True
except ImportError:
    HAS_BLADERF = False
    print("[WARN] Không tìm thấy module bladerf. Chỉ có thể xuất file.")


# ══════════════════════════════════════════════════════════════════════════════
# Tổng hợp tín hiệu IQ
# ══════════════════════════════════════════════════════════════════════════════

def make_ca_code(prn_start: int, prn_end: int) -> np.ndarray:
    """Ghép C/A code nhiều PRN, trả về mảng ±1 float32."""
    return np.concatenate([generateCAcode(i) for i in range(prn_start, prn_end + 1)]).astype(np.float32)


def synthesize_iq_complex64(
    prn_start: int,
    prn_end: int,
    tau: int,
    Rc: float,
    fs: float,
    ft: float,
    N: int,
    amplitude: float,
) -> np.ndarray:
    """
    Tổng hợp N mẫu IQ complex64.
    Công thức: s(n) = amplitude * code[(n+tau)/fs*Rc % L] * exp(j*2π*ft*n/fs)
    """
    total_chips = (prn_end - prn_start + 1) * 1023
    cacodes = make_ca_code(prn_start, prn_end)

    n = np.arange(N, dtype=np.float64)
    idx = (np.floor((n + tau) / fs * Rc).astype(np.int64)) % total_chips
    r = cacodes[idx].astype(np.float32)

    phase = (2 * np.pi * ft * n / fs).astype(np.float32)
    I = amplitude * r * np.cos(phase)
    Q = amplitude * r * np.sin(phase)
    return (I + 1j * Q).astype(np.complex64)


def complex64_to_sc16q11(iq: np.ndarray) -> np.ndarray:
    """
    Chuyển mảng complex64 sang int16 interleaved [I0,Q0,I1,Q1,...] (SC16_Q11).
    """
    buf = np.empty(iq.size * 2, dtype=np.int16)
    buf[0::2] = np.clip(np.round(np.real(iq)), -2047, 2047).astype(np.int16)
    buf[1::2] = np.clip(np.round(np.imag(iq)), -2047, 2047).astype(np.int16)
    return buf


# ══════════════════════════════════════════════════════════════════════════════
# BladeRF TX
# ══════════════════════════════════════════════════════════════════════════════

def open_bladerf_tx(freq_hz: float, fs: float, bw: float, tx_gain: int, serial: str = ""):
    dev_id = f"*:serial={serial}" if serial else ""
    b = _bladerf.BladeRF(dev_id) if dev_id else _bladerf.BladeRF()
    print(f"[INFO] Thiết bị: {b.get_board_name()}  serial={serial or 'auto'}")
    ch = _bladerf.CHANNEL_TX(0)
    b.set_sample_rate(ch, fs)
    b.set_frequency(ch, freq_hz)
    b.set_bandwidth(ch, bw)
    b.set_gain(ch, tx_gain)
    print(f"[INFO] TX  fc={freq_hz/1e6:.3f} MHz  fs={fs/1e6:.2f} MHz  gain={tx_gain} dB")
    return b


def transmit_loop(b, iq_sc16: np.ndarray):
    """Phát lặp liên tục đến khi nhấn Ctrl+C."""
    ch = _bladerf.CHANNEL_TX(0)
    b.sync_config(
        layout=_bladerf.ChannelLayout.TX_X1,
        fmt=_bladerf.Format.SC16_Q11,
        num_buffers=16,
        buffer_size=8192,
        num_transfers=8,
        stream_timeout=3500,
    )
    b.enable_module(ch, True)
    num_samples = len(iq_sc16) // 2
    print(f"[TX]  Bắt đầu phát {num_samples} mẫu/vòng... (Ctrl+C để dừng)")
    try:
        while True:
            b.sync_tx(iq_sc16, num_samples)
    except KeyboardInterrupt:
        print("\n[TX]  Nhận Ctrl+C — dừng phát.")
    finally:
        b.enable_module(ch, False)
        b.close()
        print("[TX]  Đã đóng BladeRF.")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(argv=None):
    p = argparse.ArgumentParser(description="Phát tín hiệu GNSS qua BladeRF TX (libbladeRF).")
    p.add_argument("--freq_hz",   type=float, default=1575.42e6, help="Tần số RF sóng mang (Hz). Mặc định GPS L1 = 1575.42 MHz.")
    p.add_argument("--fs",        type=float, default=2e6,     help="Sample rate (Hz).")
    p.add_argument("--Rc",        type=float, default=1.023e6, help="Chip rate (Hz).")
    p.add_argument("--ft",        type=float, default=0.0,     help="Doppler offset baseband (Hz).")
    p.add_argument("--time_s",    type=float, default=1.0,     help="Thời lượng 1 vòng tín hiệu (giây).")
    p.add_argument("--tau",       type=int,   default=40,      help="Độ trễ code (samples).")
    p.add_argument("--prn_start", type=int,   default=11,      help="PRN bắt đầu.")
    p.add_argument("--prn_end",   type=int,   default=20,      help="PRN kết thúc.")
    p.add_argument("--amplitude", type=float, default=1024.0,  help="Biên độ IQ (0–2047).")
    p.add_argument("--tx_gain",   type=int,   default=60,      help="Gain TX (dB).")
    p.add_argument("--serial",    default="", help="Serial BladeRF TX.")
    p.add_argument("--save_file", default="",                  help="Nếu đặt, lưu tín hiệu ra file .bin rồi thoát.")
    args = p.parse_args(argv)

    N = int(args.fs * args.time_s)
    bw = args.fs * 0.8

    print("══════════════════════════════════════════════════════")
    print("  phat_tin_hieu_bladerf.py  —  BladeRF TX")
    print(f"  PRN {args.prn_start}–{args.prn_end}  |  fs={args.fs/1e6:.2f} MHz"
          f"  |  fc={args.freq_hz/1e6:.3f} MHz")
    print(f"  ft={args.ft} Hz  |  tau={args.tau}  |  N={N} samples/vòng")
    print("══════════════════════════════════════════════════════\n")

    # Tổng hợp tín hiệu
    print("[INFO] Đang tổng hợp tín hiệu IQ...")
    iq = synthesize_iq_complex64(
        prn_start=args.prn_start,
        prn_end=args.prn_end,
        tau=args.tau,
        Rc=args.Rc,
        fs=args.fs,
        ft=args.ft,
        N=N,
        amplitude=args.amplitude,
    )
    iq_sc16 = complex64_to_sc16q11(iq)
    print(f"[INFO] Tín hiệu: {N} mẫu complex64 → {len(iq_sc16)} int16 SC16_Q11")

    # Lưu file nếu yêu cầu
    if args.save_file:
        iq.tofile(args.save_file)
        print(f"[INFO] Đã lưu file complex64: {args.save_file}")
        return 0

    # Phát qua BladeRF
    if not HAS_BLADERF:
        print("[ERROR] Module bladerf chưa cài. Dùng --save_file để xuất file thay thế.")
        return 1

    try:
        b = open_bladerf_tx(args.freq_hz, args.fs, bw, args.tx_gain, args.serial)
    except Exception as e:
        print(f"[ERROR] Không mở được BladeRF: {e}")
        return 1

    transmit_loop(b, iq_sc16)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
