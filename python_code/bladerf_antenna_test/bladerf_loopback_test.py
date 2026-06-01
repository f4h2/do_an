"""
bladerf_loopback_test.py
========================
Thu phát tín hiệu đồng thời trên một BladeRF x40 duy nhất (full-duplex),
sau đó tính tương quan (cross-correlation) và vẽ đồ thị — KHÔNG cần GNU Radio UI.

Kết nối phần cứng:
  - Gắn ăng ten vào cổng TX1 (phát) và một ăng ten khác vào cổng RX1 (thu).
  - Đặt hai ăng ten gần nhau (cách nhau 10–50 cm, cùng phân cực).
  - Điều chỉnh TX_GAIN / RX_GAIN sao cho tín hiệu thu không bị bão hoà.

Cài đặt:
    pip install bladerf numpy matplotlib scipy

Chạy:
    python bladerf_loopback_test.py
"""

import numpy as np
import matplotlib.pyplot as plt
import threading
import time
import sys

# ── Tuỳ chỉnh thông số ────────────────────────────────────────────────────────
FREQ_HZ        = 1575.42e6  # Tần số sóng mang GPS L1 (Hz). Đổi sang 433e6 nếu test không có Faraday cage
SAMPLE_RATE    = 10e6       # Sample rate (sps)
BANDWIDTH      = 5e6        # Bandwidth (Hz)
TX_GAIN        = 60         # Gain TX (dB) — tăng nếu khoảng cách ăng ten xa
RX_GAIN        = 60         # Gain RX (dB) — giảm nếu tín hiệu bị bão hoà (clipping)
PRN            = 1          # Số hiệu PRN C/A code (1–32)
NUM_MS         = 10         # Số mili-giây tín hiệu cần thu (và phát)

# Dùng 2 ăng ten vật lý (TX1 và RX1) → luôn để False.
# Chỉ đặt True nếu test nội bộ không cần ăng ten.
RF_LOOPBACK    = False

# ─────────────────────────────────────────────────────────────────────────────

try:
    from bladerf import _bladerf
    HAS_BLADERF = True
except ImportError:
    HAS_BLADERF = False
    print("[WARN] Không tìm thấy module bladerf — chạy ở chế độ mô phỏng (simulation).")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Tạo tín hiệu TX: BPSK điều chế bằng C/A code PRN
# ══════════════════════════════════════════════════════════════════════════════

def generate_ca_code(prn: int) -> np.ndarray:
    """Tạo C/A code (GPS) cho PRN chỉ định; trả về mảng ±1 dài 1023."""
    g2_taps = [
        [2,6],[3,7],[4,8],[5,9],[1,9],[2,10],[1,8],[2,9],[3,10],[2,3],
        [3,4],[5,6],[6,7],[7,8],[8,9],[9,10],[1,4],[2,5],[3,6],[4,7],
        [5,8],[6,9],[1,3],[4,6],[5,7],[6,8],[7,9],[8,10],[1,6],[2,7],
        [3,8],[4,9],
    ]
    if not (1 <= prn <= 32):
        raise ValueError("PRN phải trong khoảng 1–32")
    t1, t2 = g2_taps[prn - 1]
    t1 -= 1; t2 -= 1
    g1 = [1] * 10
    g2 = [1] * 10
    code = np.empty(1023, dtype=np.int8)
    for i in range(1023):
        out = g1[9] ^ (g2[t1] ^ g2[t2])
        code[i] = 1 if out == 0 else -1
        fb1 = g1[2] ^ g1[9]
        fb2 = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]
        g1 = [fb1] + g1[:-1]
        g2 = [fb2] + g2[:-1]
    return code.astype(np.float32)


def build_tx_signal(prn: int, fs: float, duration_ms: float) -> np.ndarray:
    """
    Tạo tín hiệu baseband IQ BPSK-CA.
    Carrier offset = 0 (baseband); pha điều chế bằng C/A code.
    Trả về mảng int16 interleaved [I0, Q0, I1, Q1, ...].
    """
    Rc  = 1.023e6          # chip rate GPS C/A
    Nc  = 1023             # độ dài 1 chu kỳ C/A
    N   = int(fs * duration_ms / 1000)

    ca  = generate_ca_code(prn)          # ±1, dài 1023
    n   = np.arange(N)
    idx = (n * Rc / fs).astype(int) % Nc
    baseband = ca[idx]                   # ±1 float32

    # Scale thành int16 (±2047 ~ 50 % full-scale SC16_Q11)
    SCALE = 2047
    I_int16 = (baseband * SCALE).astype(np.int16)
    Q_int16 = np.zeros(N, dtype=np.int16)

    # Interleave I, Q
    iq = np.empty(N * 2, dtype=np.int16)
    iq[0::2] = I_int16
    iq[1::2] = Q_int16
    return iq


def iq_to_complex(iq_int16: np.ndarray) -> np.ndarray:
    """Chuyển mảng int16 interleaved [I,Q,...] sang complex64."""
    I = iq_int16[0::2].astype(np.float32)
    Q = iq_int16[1::2].astype(np.float32)
    return (I + 1j * Q).astype(np.complex64)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Khởi tạo BladeRF
# ══════════════════════════════════════════════════════════════════════════════

def open_bladerf(fs, freq, bw, tx_gain, rx_gain, loopback=False):
    b = _bladerf.BladeRF()
    print(f"[INFO] Thiết bị: {b.get_board_name()}")

    for ch, is_tx in [(_bladerf.CHANNEL_RX(0), False), (_bladerf.CHANNEL_TX(0), True)]:
        b.set_sample_rate(ch, fs)
        b.set_frequency(ch, freq)
        b.set_bandwidth(ch, bw)
        b.set_gain(ch, tx_gain if is_tx else rx_gain)

    if loopback:
        # RF loopback nội bộ chip — chỉ dùng khi không có ăng ten/cáp
        b.set_loopback(_bladerf.Loopback.RF_LNA1)
        print("[INFO] Đã bật RF loopback nội bộ (LNA1).")
    else:
        print("[INFO] Chế độ ăng ten: TX1 → không khí → RX1")

    return b


# ══════════════════════════════════════════════════════════════════════════════
# 3. TX / RX thread
# ══════════════════════════════════════════════════════════════════════════════

_rx_result: np.ndarray = None
_rx_done   = threading.Event()
_tx_stop   = threading.Event()


def _tx_thread(b, iq_tx: np.ndarray):
    """Phát lặp tín hiệu cho đến khi _tx_stop được set."""
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
    print("[TX]  Bắt đầu phát...")
    try:
        while not _tx_stop.is_set():
            b.sync_tx(iq_tx, len(iq_tx) // 2)
    finally:
        b.enable_module(ch, False)
        print("[TX]  Dừng phát.")


def _rx_thread(b, num_samples: int):
    """Thu đúng num_samples mẫu, lưu vào _rx_result rồi báo _rx_done."""
    global _rx_result
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
    print("[RX]  Bắt đầu thu...")
    buf = np.zeros(num_samples * 2, dtype=np.int16)
    try:
        b.sync_rx(buf, num_samples)
        _rx_result = buf.copy()
        print(f"[RX]  Thu xong {num_samples} mẫu.")
    except Exception as e:
        print(f"[RX]  Lỗi: {e}")
    finally:
        b.enable_module(ch, False)
        _rx_done.set()


def transceive(b, iq_tx: np.ndarray, num_rx_samples: int) -> np.ndarray:
    """Phát và thu đồng thời; trả về dữ liệu RX dạng int16 interleaved."""
    global _rx_result
    _rx_result = None
    _rx_done.clear()
    _tx_stop.clear()

    t_tx = threading.Thread(target=_tx_thread, args=(b, iq_tx), daemon=True)
    t_rx = threading.Thread(target=_rx_thread, args=(b, num_rx_samples), daemon=True)

    t_tx.start()
    # Đợi TX ổn định trước khi mở RX (qua không khí cần thêm thời gian lock PLL)
    time.sleep(0.2)
    t_rx.start()

    _rx_done.wait(timeout=10)   # chờ RX tối đa 10 s
    _tx_stop.set()
    t_tx.join(timeout=3)
    t_rx.join(timeout=3)

    return _rx_result


# ══════════════════════════════════════════════════════════════════════════════
# 4. Correlation + plot
# ══════════════════════════════════════════════════════════════════════════════

def correlate_and_plot(tx_iq: np.ndarray, rx_iq: np.ndarray,
                       fs: float, prn: int):
    """
    Tính cross-correlation giữa tín hiệu TX và RX (dùng FFT).
    Vẽ 3 đồ thị:
      1. Phổ tần số TX và RX
      2. Tương quan toàn bộ
      3. Phóng to quanh đỉnh tương quan
    """
    tx = iq_to_complex(tx_iq)
    rx = iq_to_complex(rx_iq)

    N = min(len(tx), len(rx))
    tx = tx[:N]
    rx = rx[:N]

    # ── Cross-correlation via FFT ──────────────────────────────────────────
    F_tx   = np.fft.fft(tx)
    F_rx   = np.fft.fft(rx)
    corr   = np.abs(np.fft.ifft(F_rx * np.conj(F_tx)))

    peak_idx = int(np.argmax(corr))
    peak_val = corr[peak_idx]
    delay_us = peak_idx / fs * 1e6
    delay_m  = delay_us * 1e-6 * 3e8   # khoảng cách tương đương (m)

    print("\n══════════════ KẾT QUẢ TƯƠNG QUAN ══════════════")
    print(f"  Đỉnh tại sample index : {peak_idx}")
    print(f"  Giá trị đỉnh          : {peak_val:.2f}")
    print(f"  Độ trễ                : {delay_us:.3f} µs")
    print(f"  Khoảng cách tương đương: {delay_m:.1f} m")
    print("═════════════════════════════════════════════════\n")

    # ── Tính phổ (Power Spectral Density) ─────────────────────────────────
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1/fs)) / 1e6   # MHz
    psd_tx = 20 * np.log10(np.abs(np.fft.fftshift(F_tx)) / N + 1e-12)
    psd_rx = 20 * np.log10(np.abs(np.fft.fftshift(F_rx)) / N + 1e-12)

    # ── Vẽ đồ thị ─────────────────────────────────────────────────────────
    plt.style.use("dark_background")
    fig, axes = plt.subplots(3, 1, figsize=(13, 11))
    fig.suptitle(f"BladeRF x40 — Loopback Test  |  PRN {prn}  |  "
                 f"fs={fs/1e6:.1f} MHz  |  fc={FREQ_HZ/1e6:.2f} MHz",
                 fontsize=13, color="white")

    # --- 1. Phổ TX & RX ---
    ax = axes[0]
    ax.plot(freqs, psd_tx, color="#00d2ff", linewidth=0.8, alpha=0.85, label="TX spectrum")
    ax.plot(freqs, psd_rx, color="#ff9f00", linewidth=0.8, alpha=0.85, label="RX spectrum")
    ax.set_title("Phổ tần số (Power Spectral Density)", color="white")
    ax.set_xlabel("Tần số (MHz)")
    ax.set_ylabel("Biên độ (dB)")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.3)

    # --- 2. Tương quan toàn bộ ---
    t_axis_us = np.arange(N) / fs * 1e6
    ax = axes[1]
    ax.plot(t_axis_us, corr, color="#00ff99", linewidth=0.7, alpha=0.9)
    ax.plot(delay_us, peak_val, "ro", markersize=8,
            label=f"Peak  @  {delay_us:.3f} µs  (idx={peak_idx})")
    ax.set_title("Cross-correlation TX × RX", color="white")
    ax.set_xlabel("Độ trễ (µs)")
    ax.set_ylabel("Biên độ")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.3)

    # --- 3. Phóng to quanh đỉnh ---
    zoom = max(200, int(fs * 5e-6))   # ±5 µs hoặc ít nhất 200 sample
    s = max(0, peak_idx - zoom)
    e = min(N, peak_idx + zoom)
    t_zoom = t_axis_us[s:e]
    ax = axes[2]
    ax.plot(t_zoom, corr[s:e], color="#ff4f7b", linewidth=1.5)
    ax.axvline(x=delay_us, color="white", linestyle=":", linewidth=1.2,
               label=f"Đỉnh @ {delay_us:.3f} µs")
    ax.set_title(f"Phóng to đỉnh tương quan (±{zoom} samples ≈ ±5 µs)", color="white")
    ax.set_xlabel("Độ trễ (µs)")
    ax.set_ylabel("Biên độ")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    out_png = f"loopback_corr_prn{prn}.png"
    plt.savefig(out_png, dpi=150)
    print(f"[INFO] Đã lưu đồ thị: {out_png}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 5. Main
# ══════════════════════════════════════════════════════════════════════════════

def run_simulation(fs, num_ms, prn):
    """
    Chế độ mô phỏng khi không có BladeRF:
    TX → thêm noise + trễ nhân tạo → RX, rồi tính tương quan.
    """
    print("[SIM]  Không có BladeRF — chạy mô phỏng với trễ giả lập.")
    iq_tx  = build_tx_signal(prn, fs, num_ms)
    N      = len(iq_tx) // 2
    tx_cplx = iq_to_complex(iq_tx)

    DELAY_SAMPLES = int(fs * 3e-6)   # giả lập trễ 3 µs
    noise  = (np.random.randn(N) + 1j * np.random.randn(N)).astype(np.complex64) * 50
    rx_cplx = np.roll(tx_cplx, DELAY_SAMPLES) + noise

    I_rx = np.clip(rx_cplx.real, -32767, 32767).astype(np.int16)
    Q_rx = np.clip(rx_cplx.imag, -32767, 32767).astype(np.int16)
    iq_rx = np.empty(N * 2, dtype=np.int16)
    iq_rx[0::2] = I_rx
    iq_rx[1::2] = Q_rx

    print(f"[SIM]  Trễ giả lập: {DELAY_SAMPLES} samples = {DELAY_SAMPLES/fs*1e6:.2f} µs")
    correlate_and_plot(iq_tx, iq_rx, fs, prn)


def main():
    fs      = SAMPLE_RATE
    num_ms  = NUM_MS
    prn     = PRN
    N       = int(fs * num_ms / 1000)

    print("══════════════════════════════════════════════════")
    print("  BladeRF x40  –  Two-Antenna Over-the-Air Test")
    print(f"  PRN={prn}  |  fs={fs/1e6:.1f} MHz  |  fc={FREQ_HZ/1e6:.3f} MHz")
    print(f"  Thời gian thu: {num_ms} ms  ({N} samples)")
    print("══════════════════════════════════════════════════\n")

    # ── Tạo tín hiệu TX ───────────────────────────────────────────────────
    print("[INFO] Đang tạo tín hiệu TX (C/A PRN BPSK)...")
    iq_tx = build_tx_signal(prn, fs, num_ms)
    print(f"[INFO] TX signal: {len(iq_tx)//2} samples, {num_ms} ms")

    if not HAS_BLADERF:
        run_simulation(fs, num_ms, prn)
        return

    # ── Mở BladeRF ────────────────────────────────────────────────────────
    try:
        b = open_bladerf(fs, FREQ_HZ, BANDWIDTH, TX_GAIN, RX_GAIN, RF_LOOPBACK)
    except Exception as e:
        print(f"[ERROR] Không thể mở BladeRF: {e}")
        print("[INFO]  Chuyển sang chế độ mô phỏng...")
        run_simulation(fs, num_ms, prn)
        return

    # ── Thu / Phát đồng thời ──────────────────────────────────────────────
    try:
        iq_rx = transceive(b, iq_tx, N)
    finally:
        try:
            b.close()
        except Exception:
            pass

    if iq_rx is None or len(iq_rx) < N * 2:
        print("[ERROR] Không thu được đủ dữ liệu!")
        return

    # ── Lưu raw data (tuỳ chọn) ───────────────────────────────────────────
    np.save(f"rx_raw_prn{prn}.npy", iq_rx)
    np.save(f"tx_raw_prn{prn}.npy", iq_tx)
    print(f"[INFO] Đã lưu raw data: tx_raw_prn{prn}.npy, rx_raw_prn{prn}.npy")

    # ── Tính tương quan và vẽ đồ thị ──────────────────────────────────────
    correlate_and_plot(iq_tx, iq_rx, fs, prn)


if __name__ == "__main__":
    main()
