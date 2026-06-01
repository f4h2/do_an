"""
phat_thu_bladerf_gui.py
=======================
Full-duplex TX + RX trên 1 BladeRF x40 duy nhất (cùng process, cùng device).
TX phát tín hiệu GNSS C/A, RX thu đồng thời và tính tương quan realtime + GUI.

Không cần GNU Radio, không cần ZMQ, không cần 2 thiết bị.

Sơ đồ:
    BladeRF x40
    ├─ TX1 (ăng ten phát) → sóng RF →
    └─ RX1 (ăng ten thu)  ← sóng RF ←

Chạy:
    conda activate gnss_env
    python phat_thu_bladerf_gui.py
    python phat_thu_bladerf_gui.py --freq_hz 433e6 --prn1_start 11 --prn1_end 20

Argparse:
    --freq_hz    : Tần số RF sóng mang (Hz), mặc định 1575.42e6 (GPS L1)
    --fs         : Sample rate (Hz), mặc định 2e6
    --ft         : Doppler offset bù khi tương quan (Hz), mặc định 0
    --tau        : Độ trễ code TX (samples), mặc định 40
    --prn1_start : PRN nhóm 1 bắt đầu, mặc định 11
    --prn1_end   : PRN nhóm 1 kết thúc, mặc định 20
    --prn2_start : PRN nhóm 2 bắt đầu, mặc định 21
    --prn2_end   : PRN nhóm 2 kết thúc, mặc định 30
    --tx_prn_start / --tx_prn_end : PRN dùng để phát, mặc định 11–20
    --amplitude  : Biên độ TX (0–2047), mặc định 1024
    --tx_gain    : Gain TX (dB), mặc định 60
    --rx_gain    : Gain RX (dB), mặc định 60
    --chunk_ms   : Số ms mỗi chunk xử lý, mặc định 5
    --save       : Lưu IQ thu được ra file .bin (complex64)
"""

import argparse
import os
import sys
import threading
import time
import queue

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gnss_utils import generateCAcode, calcDistance

try:
    from bladerf import _bladerf
    HAS_BLADERF = True
except ImportError:
    HAS_BLADERF = False
    print("[WARN] Không tìm thấy module bladerf — chạy ở chế độ mô phỏng.")


# ══════════════════════════════════════════════════════════════════════════════
# Tổng hợp tín hiệu
# ══════════════════════════════════════════════════════════════════════════════

def make_ca_code(prn_start: int, prn_end: int) -> np.ndarray:
    return np.concatenate([generateCAcode(i) for i in range(prn_start, prn_end + 1)]).astype(np.float32)


def synthesize_tx_sc16(prn_start, prn_end, tau, Rc, fs, ft, time_s, amplitude) -> np.ndarray:
    """Tổng hợp tín hiệu TX, trả về int16 interleaved SC16_Q11."""
    total_chips = (prn_end - prn_start + 1) * 1023
    cacodes = make_ca_code(prn_start, prn_end)
    N = int(fs * time_s)
    n = np.arange(N, dtype=np.float64)
    idx = (np.floor((n + tau) / fs * Rc).astype(np.int64)) % total_chips
    r = cacodes[idx].astype(np.float32)
    phase = (2 * np.pi * ft * n / fs).astype(np.float32)
    I = amplitude * r * np.cos(phase)
    Q = amplitude * r * np.sin(phase)
    iq = (I + 1j * Q).astype(np.complex64)
    buf = np.empty(N * 2, dtype=np.int16)
    buf[0::2] = np.clip(np.round(I), -2047, 2047).astype(np.int16)
    buf[1::2] = np.clip(np.round(Q), -2047, 2047).astype(np.int16)
    return buf


def make_local_ref_fft(prn_start, prn_end, fs, Rc, Nfft) -> np.ndarray:
    """FFT của mã tham chiếu nội bộ (Nfft điểm)."""
    total_chips = (prn_end - prn_start + 1) * 1023
    cacodes = make_ca_code(prn_start, prn_end)
    idx = np.floor(np.arange(Nfft) / fs * Rc).astype(int) % total_chips
    return np.fft.fft(cacodes[idx].astype(np.complex64))


def sc16q11_to_complex64(buf: np.ndarray) -> np.ndarray:
    return (buf[0::2].astype(np.float32) + 1j * buf[1::2].astype(np.float32)).astype(np.complex64)


# ══════════════════════════════════════════════════════════════════════════════
# TX thread — dùng device handle b đã mở sẵn
# ══════════════════════════════════════════════════════════════════════════════

def tx_worker(b, iq_sc16: np.ndarray, stop_event: threading.Event):
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
    print(f"[TX]  Bắt đầu phát {num_samples} mẫu/vòng...")
    try:
        while not stop_event.is_set():
            b.sync_tx(iq_sc16, num_samples)
    except Exception as e:
        if not stop_event.is_set():
            print(f"[TX]  Lỗi: {e}")
    finally:
        b.enable_module(ch, False)
        print("[TX]  Dừng phát.")


# ══════════════════════════════════════════════════════════════════════════════
# RX thread — dùng device handle b đã mở sẵn
# ══════════════════════════════════════════════════════════════════════════════

def rx_worker(b, chunk_samples: int, rx_queue: queue.Queue, stop_event: threading.Event):
    ch = _bladerf.CHANNEL_RX(0)
    b.sync_config(
        layout=_bladerf.ChannelLayout.RX_X1,
        fmt=_bladerf.Format.SC16_Q11,
        num_buffers=16,
        buffer_size=8192,
        num_transfers=8,
        stream_timeout=3500,
    )
    b.enable_module(ch, True)
    buf = np.zeros(chunk_samples * 2, dtype=np.int16)
    print(f"[RX]  Bắt đầu thu {chunk_samples} mẫu/chunk...")
    try:
        while not stop_event.is_set():
            b.sync_rx(buf, chunk_samples)
            chunk_iq = sc16q11_to_complex64(buf)
            if not rx_queue.full():
                rx_queue.put(chunk_iq.copy())
    except Exception as e:
        if not stop_event.is_set():
            print(f"[RX]  Lỗi: {e}")
    finally:
        b.enable_module(ch, False)
        print("[RX]  Dừng thu.")


# ══════════════════════════════════════════════════════════════════════════════
# Mô phỏng (không có BladeRF)
# ══════════════════════════════════════════════════════════════════════════════

def sim_worker(iq_sc16, chunk_samples, fs, rx_queue, stop_event):
    """Giả lập RX = TX + trễ + noise, đẩy vào queue."""
    tx_cplx = sc16q11_to_complex64(iq_sc16)
    N_tx = len(tx_cplx)
    DELAY = int(fs * 3e-6)
    ptr = 0
    while not stop_event.is_set():
        chunk = np.roll(tx_cplx, DELAY)[ptr % N_tx: ptr % N_tx + chunk_samples]
        if len(chunk) < chunk_samples:
            ptr = 0
            continue
        noise = (np.random.randn(chunk_samples) + 1j * np.random.randn(chunk_samples)).astype(np.complex64) * 30
        if not rx_queue.full():
            rx_queue.put(chunk + noise)
        ptr += chunk_samples
        time.sleep(chunk_samples / fs)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(argv=None):
    p = argparse.ArgumentParser(description="Full-duplex TX+RX trên 1 BladeRF + GUI tương quan.")
    p.add_argument("--freq_hz",      type=float, default=1575.42e6, help="Tần số RF sóng mang (Hz). Mặc định GPS L1 = 1575.42 MHz.")
    p.add_argument("--fs",           type=float, default=2e6,     help="Sample rate (Hz).")
    p.add_argument("--Rc",           type=float, default=1.023e6, help="Chip rate (Hz).")
    p.add_argument("--ft",           type=float, default=0.0,     help="Doppler offset bù (Hz).")
    p.add_argument("--tau",          type=int,   default=40,      help="Độ trễ code TX (samples).")
    p.add_argument("--tx_prn_start", type=int,   default=11,      help="PRN TX bắt đầu.")
    p.add_argument("--tx_prn_end",   type=int,   default=20,      help="PRN TX kết thúc.")
    p.add_argument("--prn1_start",   type=int,   default=11,      help="PRN tham chiếu nhóm 1 bắt đầu.")
    p.add_argument("--prn1_end",     type=int,   default=20,      help="PRN tham chiếu nhóm 1 kết thúc.")
    p.add_argument("--prn2_start",   type=int,   default=21,      help="PRN tham chiếu nhóm 2 bắt đầu.")
    p.add_argument("--prn2_end",     type=int,   default=30,      help="PRN tham chiếu nhóm 2 kết thúc.")
    p.add_argument("--amplitude",    type=float, default=1024.0,  help="Biên độ TX (0–2047).")
    p.add_argument("--tx_gain",      type=int,   default=60,      help="Gain TX (dB).")
    p.add_argument("--rx_gain",      type=int,   default=60,      help="Gain RX (dB).")
    p.add_argument("--chunk_ms",     type=float, default=5.0,     help="Số ms mỗi chunk.")
    p.add_argument("--time_s",       type=float, default=1.0,     help="Thời lượng 1 vòng TX (giây).")
    p.add_argument("--serial",       default="",                  help="Serial BladeRF (bỏ trống = tự chọn thiết bị đầu tiên).")
    p.add_argument("--save",         default="",                  help="Lưu IQ RX ra file .bin.")
    args = p.parse_args(argv)

    fs       = args.fs
    Rc       = args.Rc
    ft       = args.ft
    Nchunk   = int(fs * args.chunk_ms / 1000)
    Nfft     = Nchunk * 2
    bw       = fs * 0.8
    speedOfLight = 299792458

    print("══════════════════════════════════════════════════════════")
    print("  phat_thu_bladerf_gui.py  —  Full-Duplex TX+RX BladeRF")
    print(f"  fc={args.freq_hz/1e6:.3f} MHz  |  fs={fs/1e6:.2f} MHz")
    print(f"  TX PRN {args.tx_prn_start}–{args.tx_prn_end}"
          f"  |  RX ref nhóm1 PRN {args.prn1_start}–{args.prn1_end}"
          f"  /  nhóm2 PRN {args.prn2_start}–{args.prn2_end}")
    print(f"  chunk={Nchunk} mẫu ({args.chunk_ms} ms)  |  Nfft={Nfft}")
    print("══════════════════════════════════════════════════════════\n")

    # ── Tổng hợp tín hiệu TX ──────────────────────────────────────────────
    print("[INFO] Đang tổng hợp tín hiệu TX...")
    iq_sc16 = synthesize_tx_sc16(
        prn_start=args.tx_prn_start,
        prn_end=args.tx_prn_end,
        tau=args.tau,
        Rc=Rc,
        fs=fs,
        ft=ft,
        time_s=args.time_s,
        amplitude=args.amplitude,
    )
    print(f"[INFO] TX: {len(iq_sc16)//2} mẫu SC16_Q11")

    # ── Tạo mã tham chiếu nội bộ ──────────────────────────────────────────
    print("[INFO] Đang tổng hợp mã tham chiếu nội bộ...")
    F_local1 = make_local_ref_fft(args.prn1_start, args.prn1_end, fs, Rc, Nfft)
    F_local2 = make_local_ref_fft(args.prn2_start, args.prn2_end, fs, Rc, Nfft)
    print("[INFO] Hoàn thành.\n")

    # ── Toạ độ TDOA ──────────────────────────────────────────────────────
    TX1 = [20.9896100, 105.7110745]
    TX2 = [20.9924397, 105.7106347]
    RX  = [20.9911114, 105.7107914]
    T1  = calcDistance(TX1[0], TX1[1], RX[0], RX[1])
    T2  = calcDistance(TX2[0], TX2[1], RX[0], RX[1])
    T0  = (T1 - T2) / speedOfLight * fs

    # ── Khởi tạo BladeRF (1 device cho cả TX và RX) ───────────────────────
    stop_event = threading.Event()
    rx_queue   = queue.Queue(maxsize=50)
    b = None

    if HAS_BLADERF:
        try:
            dev_id = f"*:serial={args.serial}" if args.serial else ""
            b = _bladerf.BladeRF(dev_id) if dev_id else _bladerf.BladeRF()
            print(f"[INFO] Thiết bị: {b.get_board_name()}  serial={args.serial or 'auto'}")

            for ch, is_tx in [(_bladerf.CHANNEL_RX(0), False), (_bladerf.CHANNEL_TX(0), True)]:
                b.set_sample_rate(ch, fs)
                b.set_frequency(ch, args.freq_hz)
                b.set_bandwidth(ch, bw)
                b.set_gain(ch, args.tx_gain if is_tx else args.rx_gain)

            print(f"[INFO] TX gain={args.tx_gain} dB  |  RX gain={args.rx_gain} dB")

            # Khởi động TX thread
            t_tx = threading.Thread(
                target=tx_worker, args=(b, iq_sc16, stop_event), daemon=True
            )
            t_tx.start()
            time.sleep(0.3)  # đợi TX ổn định

            # Khởi động RX thread
            t_rx = threading.Thread(
                target=rx_worker, args=(b, Nchunk, rx_queue, stop_event), daemon=True
            )
            t_rx.start()

        except Exception as e:
            print(f"[ERROR] Không mở được BladeRF: {e}")
            print("[INFO]  Chuyển sang chế độ mô phỏng...")
            b = None

    if b is None:
        print("[SIM]  Chạy mô phỏng (TX → trễ 3µs → RX)...")
        t_sim = threading.Thread(
            target=sim_worker,
            args=(iq_sc16, Nchunk, fs, rx_queue, stop_event),
            daemon=True,
        )
        t_sim.start()

    # ── Mở file lưu nếu cần ───────────────────────────────────────────────
    save_fh = open(args.save, "wb") if args.save else None

    # ── Khởi tạo GUI ─────────────────────────────────────────────────────
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9))
    fig.suptitle(
        f"BladeRF x40 — Full-Duplex TX+RX  |  "
        f"fc={args.freq_hz/1e6:.3f} MHz  |  fs={fs/1e6:.2f} MHz",
        fontsize=12, color="white",
    )

    x_axis = np.arange(Nfft)
    line1, = ax1.plot(x_axis, np.zeros(Nfft), color="#1f77b4", lw=0.6, alpha=0.85,
                      label=f"PRN {args.prn1_start}–{args.prn1_end}")
    line2, = ax2.plot(x_axis, np.zeros(Nfft), color="#ff7f0e", lw=0.6, alpha=0.85,
                      label=f"PRN {args.prn2_start}–{args.prn2_end}")
    peak1, = ax1.plot([0], [0], "ro", ms=8, label="Peak 1")
    peak2, = ax2.plot([0], [0], "go", ms=8, label="Peak 2")
    txt1   = ax1.text(0.01, 0.91, "", transform=ax1.transAxes, color="cyan",  fontsize=9)
    txt2   = ax2.text(0.01, 0.91, "", transform=ax2.transAxes, color="yellow", fontsize=9)

    ax1.set_title(f"Correlation — Nhóm 1  (PRN {args.prn1_start}–{args.prn1_end})", color="white")
    ax2.set_title(f"Correlation — Nhóm 2  (PRN {args.prn2_start}–{args.prn2_end})", color="white")
    for ax in [ax1, ax2]:
        ax.set_xlabel("Delay (samples)")
        ax.set_ylabel("Magnitude")
        ax.set_xlim(0, Nfft)
        ax.set_ylim(0, 100)
        ax.grid(True, ls="--", alpha=0.35)
        ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()

    buffer_iq = np.array([], dtype=np.complex64)
    n_arr     = np.arange(Nchunk, dtype=np.float32)

    def update(_frame):
        nonlocal buffer_iq

        chunk = None
        try:
            chunk = rx_queue.get_nowait()
        except queue.Empty:
            return line1, line2, peak1, peak2, txt1, txt2

        if save_fh:
            chunk.astype(np.complex64).tofile(save_fh)

        buffer_iq = np.concatenate((buffer_iq, chunk))
        if len(buffer_iq) < Nchunk:
            return line1, line2, peak1, peak2, txt1, txt2

        IQ = buffer_iq[:Nchunk]
        buffer_iq = buffer_iq[Nchunk:]

        # Bù Doppler
        IQ_d = IQ * np.exp(1j * 2 * np.pi * ft * n_arr / fs).astype(np.complex64)

        # Zero-pad + FFT
        IQ_padded = np.zeros(Nfft, dtype=np.complex64)
        IQ_padded[:Nchunk] = IQ_d
        F_IQ = np.fft.fft(IQ_padded)

        # Cross-correlation
        mag1 = np.abs(np.fft.ifft(F_local1 * np.conj(F_IQ)))
        mag2 = np.abs(np.fft.ifft(F_local2 * np.conj(F_IQ)))

        tau1 = int(np.argmax(mag1)); v1 = mag1[tau1]
        tau2 = int(np.argmax(mag2)); v2 = mag2[tau2]

        # Cập nhật đồ thị
        line1.set_ydata(mag1)
        line2.set_ydata(mag2)
        peak1.set_data([tau1], [v1])
        peak2.set_data([tau2], [v2])
        txt1.set_text(f"Peak @ sample {tau1}  |  Mag = {v1:.1f}")
        txt2.set_text(f"Peak @ sample {tau2}  |  Mag = {v2:.1f}")

        # Auto-scale Y
        top = max(v1, v2, 10) * 1.5
        if top > ax1.get_ylim()[1] * 0.85 or top < ax1.get_ylim()[1] * 0.1:
            ax1.set_ylim(0, top); ax2.set_ylim(0, top)

        # TDOA
        Delta_T = (tau1 - tau2) - T0
        Delta_m = Delta_T / fs * speedOfLight
        print(f"[CORR] Peak1={tau1:5d} ({v1:7.1f})  "
              f"Peak2={tau2:5d} ({v2:7.1f})  "
              f"TDOA_diff={Delta_m:+.2f} m")

        return line1, line2, peak1, peak2, txt1, txt2

    ani = animation.FuncAnimation(
        fig, update,
        interval=max(int(args.chunk_ms), 50),
        blit=False,
        cache_frame_data=False,
    )

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        if b:
            try:
                b.close()
            except Exception:
                pass
        if save_fh:
            save_fh.close()
            print(f"[INFO] Đã lưu IQ: {args.save}")
        print("[INFO] Thoát.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
