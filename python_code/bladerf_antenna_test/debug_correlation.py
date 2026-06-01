"""
debug_correlation.py
====================
Script chẩn đoán: thu vài chunk IQ từ BladeRF, vẽ:
  1. Phổ tần (FFT magnitude) — kiểm tra xem tín hiệu có đến RX không
  2. Bản đồ 2D tương quan (frequency offset × delay) — tìm Δf thực tế

Chạy trong lúc phat_tin_hieu_bladerf.py đang phát:
    python debug_correlation.py --freq_hz 433e6 --prn_start 11 --prn_end 20

Hoặc đọc từ file đã lưu:
    python debug_correlation.py --file captured.bin --prn_start 11 --prn_end 20
"""

import argparse
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gnss_utils import generateCAcode

try:
    from bladerf import _bladerf
    HAS_BLADERF = True
except ImportError:
    HAS_BLADERF = False

# ─────────────────────────────────────────────────────────────────────────────

def sc16q11_to_complex64(buf):
    return (buf[0::2].astype(np.float32) + 1j * buf[1::2].astype(np.float32)).astype(np.complex64)

def make_local_ref_fft(prn_start, prn_end, fs, Rc, Nfft):
    total_chips = (prn_end - prn_start + 1) * 1023
    cacodes = np.concatenate([generateCAcode(i) for i in range(prn_start, prn_end + 1)])
    idx = np.floor(np.arange(Nfft) / fs * Rc).astype(int) % total_chips
    return np.fft.fft(cacodes[idx].astype(np.complex64))

# ─────────────────────────────────────────────────────────────────────────────

def capture_iq(freq_hz, fs, bw, rx_gain, n_samples, serial=""):
    """Thu n_samples mẫu từ BladeRF."""
    dev_id = f"*:serial={serial}" if serial else ""
    b = _bladerf.BladeRF(dev_id) if dev_id else _bladerf.BladeRF()
    print(f"[RX] Thiết bị: {b.get_board_name()}  serial={serial or 'auto'}")

    ch = _bladerf.CHANNEL_RX(0)
    b.set_sample_rate(ch, fs)
    b.set_frequency(ch, freq_hz)
    b.set_bandwidth(ch, bw)
    b.set_gain(ch, rx_gain)
    b.sync_config(
        layout=_bladerf.ChannelLayout.RX_X1,
        fmt=_bladerf.Format.SC16_Q11,
        num_buffers=16, buffer_size=8192,
        num_transfers=8, stream_timeout=3500,
    )
    b.enable_module(ch, True)

    print(f"[RX] Thu {n_samples} mẫu...")
    buf = np.zeros(n_samples * 2, dtype=np.int16)
    # Bỏ vài chunk đầu để ổn định
    for _ in range(4):
        b.sync_rx(buf, n_samples)
    b.sync_rx(buf, n_samples)

    b.enable_module(ch, False)
    b.close()
    return sc16q11_to_complex64(buf)

# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--freq_hz",   type=float, default=433e6)
    p.add_argument("--fs",        type=float, default=2e6)
    p.add_argument("--Rc",        type=float, default=1.023e6)
    p.add_argument("--rx_gain",   type=int,   default=60)
    p.add_argument("--prn_start", type=int,   default=11)
    p.add_argument("--prn_end",   type=int,   default=20)
    p.add_argument("--chunk_ms",  type=float, default=10.0)
    p.add_argument("--ft_range",  type=float, default=30000.0,
                   help="Bán kính quét tần số (Hz)")
    p.add_argument("--ft_step",   type=float, default=100.0,
                   help="Bước quét tần số (Hz)")
    p.add_argument("--serial",    default="a0e5ffb5f1c28a2d57f5f5d9d13372ed")
    p.add_argument("--file",      default="",
                   help="Đọc từ file .bin thay vì BladeRF")
    args = p.parse_args()

    fs     = args.fs
    Rc     = args.Rc
    Nchunk = int(fs * args.chunk_ms / 1000)
    Nfft   = Nchunk * 2
    bw     = fs * 0.8
    df     = fs / Nfft

    print(f"Nchunk={Nchunk}  Nfft={Nfft}  df={df:.1f} Hz/bin")

    # ── Thu hoặc đọc IQ ──────────────────────────────────────────────────
    if args.file:
        raw = np.fromfile(args.file, dtype=np.complex64)
        IQ  = raw[:Nchunk]
        print(f"[FILE] Đọc {len(IQ)} mẫu từ {args.file}")
    else:
        if not HAS_BLADERF:
            print("[ERROR] Không có bladerf. Dùng --file.")
            return
        IQ = capture_iq(args.freq_hz, fs, bw, args.rx_gain, Nchunk, args.serial)

    print(f"\n[STAT] IQ power = {np.mean(np.abs(IQ)**2):.1f}  "
          f"max(|IQ|) = {np.max(np.abs(IQ)):.1f}")
    if np.mean(np.abs(IQ)**2) < 1.0:
        print("[WARN] ⚠ Tín hiệu rất yếu! Kiểm tra ăng ten và rx_gain.")

    # ── Tham chiếu nội bộ ────────────────────────────────────────────────
    F_local = make_local_ref_fft(args.prn_start, args.prn_end, fs, Rc, Nfft)

    # ── FFT của IQ ───────────────────────────────────────────────────────
    IQ_padded = np.zeros(Nfft, dtype=np.complex64)
    IQ_padded[:Nchunk] = IQ
    F_IQ = np.fft.fft(IQ_padded)

    # ── Quét 2D: frequency offset × delay ───────────────────────────────
    ft_arr = np.arange(-args.ft_range, args.ft_range + args.ft_step,
                       args.ft_step, dtype=np.float32)
    n_ft  = len(ft_arr)
    # Chỉ lưu đỉnh mỗi ft (tiết kiệm RAM), và bản đồ 2D thu nhỏ
    # Lưu full 2D với delay giảm mẫu (lấy mỗi 10 mẫu)
    DSAMPLE = 10
    n_delay = Nfft // DSAMPLE
    corr_map = np.zeros((n_ft, n_delay), dtype=np.float32)
    peak_per_ft = np.zeros(n_ft, dtype=np.float32)
    tau_per_ft  = np.zeros(n_ft, dtype=np.int32)

    print(f"\n[SCAN] Quét {n_ft} giá trị ft từ {ft_arr[0]:.0f} đến {ft_arr[-1]:.0f} Hz...")
    for i, ft in enumerate(ft_arr):
        k     = int(round(ft / df))
        F_shifted = np.roll(F_IQ, k)
        mag = np.abs(np.fft.ifft(F_local * np.conj(F_shifted))).astype(np.float32)
        corr_map[i] = mag[::DSAMPLE]
        tau_per_ft[i]  = int(np.argmax(mag))
        peak_per_ft[i] = float(mag[tau_per_ft[i]])
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n_ft}  best so far: ft={ft_arr[np.argmax(peak_per_ft[:i+1])]:.0f} Hz  "
                  f"peak={peak_per_ft[:i+1].max():.1f}")

    best_idx = int(np.argmax(peak_per_ft))
    best_ft  = ft_arr[best_idx]
    best_tau = tau_per_ft[best_idx]
    best_mag = peak_per_ft[best_idx]
    print(f"\n[RESULT] ★ ft_best={best_ft:+.0f} Hz  tau_best={best_tau}  peak={best_mag:.1f}")

    if best_mag < 20:
        print("[WARN] ⚠ Đỉnh tương quan thấp (<20). Có thể:")
        print("         • Ăng ten TX/RX chưa kết nối hoặc quá xa")
        print("         • Lệch tần > ft_range (thử tăng --ft_range 60000)")
        print("         • rx_gain quá thấp (thử --rx_gain 70 hoặc 80)")
        print("         • PRN TX ≠ PRN tham chiếu RX")
    else:
        print(f"[OK] Tương quan tốt! Dùng --ft {best_ft:.0f} hoặc để freqsearch tự tìm.")

    # ── VẼ ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Debug Correlation — fc={args.freq_hz/1e6:.3f} MHz  "
                 f"PRN {args.prn_start}–{args.prn_end}  fs={fs/1e6:.1f} MHz", fontsize=12)

    # 1. Phổ tần số của IQ thu
    ax = axes[0, 0]
    freqs = np.fft.fftshift(np.fft.fftfreq(Nchunk, 1/fs)) / 1e3
    spec  = np.fft.fftshift(np.abs(F_IQ[:Nchunk]))
    ax.plot(freqs, 20*np.log10(spec + 1e-6), lw=0.5, color="#1f77b4")
    ax.set_title("Phổ IQ thu (baseband)")
    ax.set_xlabel("Tần số (kHz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.grid(True, ls="--", alpha=0.4)

    # 2. Đỉnh tương quan theo ft
    ax = axes[0, 1]
    ax.plot(ft_arr / 1e3, peak_per_ft, lw=0.8, color="#ff7f0e")
    ax.axvline(best_ft / 1e3, color="red", lw=1.5, ls="--", label=f"ft_best={best_ft:.0f} Hz")
    ax.set_title("Peak tương quan theo frequency offset")
    ax.set_xlabel("Frequency offset ft (kHz)")
    ax.set_ylabel("Peak magnitude")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.4)

    # 3. Bản đồ 2D tương quan (ft × delay)
    ax = axes[1, 0]
    extent = [0, Nfft * DSAMPLE, ft_arr[-1]/1e3, ft_arr[0]/1e3]
    im = ax.imshow(corr_map, aspect="auto", extent=extent,
                   cmap="viridis", interpolation="nearest")
    ax.axhline(best_ft / 1e3, color="red", lw=1, ls="--")
    ax.axvline(best_tau, color="red", lw=1, ls="--")
    ax.set_title(f"2D Correlation Map  (★ ft={best_ft:.0f} Hz, τ={best_tau})")
    ax.set_xlabel("Delay τ (samples)")
    ax.set_ylabel("Frequency offset (kHz)")
    plt.colorbar(im, ax=ax)

    # 4. Tương quan tại ft_best
    ax = axes[1, 1]
    k_best   = int(round(best_ft / df))
    F_best   = np.roll(F_IQ, k_best)
    mag_best = np.abs(np.fft.ifft(F_local * np.conj(F_best))).astype(np.float32)
    ax.plot(np.arange(Nfft), mag_best, lw=0.6, color="#2ca02c")
    ax.axvline(best_tau, color="red", lw=1.5, ls="--",
               label=f"τ={best_tau}  mag={best_mag:.1f}")
    ax.set_title(f"Tương quan tại ft_best={best_ft:.0f} Hz")
    ax.set_xlabel("Delay τ (samples)")
    ax.set_ylabel("Magnitude")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.4)
    ax.set_xlim(0, Nfft)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
