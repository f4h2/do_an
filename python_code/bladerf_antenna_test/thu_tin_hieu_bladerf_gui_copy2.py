"""
thu_tin_hieu_bladerf_gui.py
============================
Thu tín hiệu GNSS realtime từ BladeRF x40, tính tương quan với mã C/A nội,
hiển thị đồ thị realtime — KHÔNG dùng GNU Radio, KHÔNG dùng ZMQ.
Tương đương thu_tin_hieu_rt_gui.py nhưng thay ZMQ bằng BladeRF RX (libbladeRF).

Sơ đồ kết nối:
    [phat_tin_hieu_bladerf.py chạy trên cùng máy]
         TX1 (ăng ten phát)  →  sóng RF  →  RX1 (ăng ten thu)
                                                    ↓
                                       thu_tin_hieu_bladerf_gui.py
                                       tính tương quan + vẽ đồ thị

Chạy (2 terminal):
    Terminal 1:  python phat_tin_hieu_bladerf.py
    Terminal 2:  python thu_tin_hieu_bladerf_gui.py

Argparse:
    --freq_hz   : Tần số RF sóng mang (Hz), mặc định 1575.42e6 (GPS L1)
    --fs        : Sample rate (Hz), mặc định 2e6
    --ft        : Doppler offset để bù (Hz), mặc định 0
    --rx_gain   : Gain RX (dB), mặc định 60
    --prn1_start: PRN bắt đầu nhóm 1, mặc định 11
    --prn1_end  : PRN kết thúc nhóm 1, mặc định 20
    --prn2_start: PRN bắt đầu nhóm 2, mặc định 21
    --prn2_end  : PRN kết thúc nhóm 2, mặc định 30
    --chunk_ms  : Số ms mỗi lần xử lý, mặc định 5
    --save      : Lưu dữ liệu thu vào file .bin
"""

import argparse
import sys
import threading
import time
import os

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
    print("[WARN] Không tìm thấy module bladerf — chạy ở chế độ đọc file.")


# ══════════════════════════════════════════════════════════════════════════════
# Hàm tiện ích
# ══════════════════════════════════════════════════════════════════════════════

def make_local_ref(prn_start: int, prn_end: int, fs: float, Rc: float, Nfft: int) -> np.ndarray:
    """
    Tổng hợp tín hiệu tham chiếu nội bộ (C/A code) và tính FFT sẵn.
    Trả về FFT của mã tham chiếu (Nfft điểm, complex64).
    """
    total_chips = (prn_end - prn_start + 1) * 1023
    cacodes = np.concatenate([generateCAcode(i) for i in range(prn_start, prn_end + 1)])
    n_local = np.arange(Nfft)
    idx = np.floor(n_local / fs * Rc).astype(int) % total_chips
    local_fs = cacodes[idx].astype(np.complex64)
    return np.fft.fft(local_fs)


def sc16q11_to_complex64(buf: np.ndarray) -> np.ndarray:
    """Chuyển int16 interleaved [I,Q,...] sang complex64."""
    I = buf[0::2].astype(np.float32)
    Q = buf[1::2].astype(np.float32)
    return (I + 1j * Q).astype(np.complex64)


def latlon_to_xy(ref_lat: float, ref_lon: float, lat: float, lon: float):
    """
    Chuyển (lat, lon) sang toạ độ Cartesian (x, y) đơn vị mét,
    lấy (ref_lat, ref_lon) làm gốc toạ độ (flat-Earth).
    """
    R = 6371000.0  # bán kính Trái Đất (m)
    x = np.radians(lon - ref_lon) * np.cos(np.radians((lat + ref_lat) / 2)) * R
    y = np.radians(lat - ref_lat) * R
    return x, y


def trilaterate_2d(tx1_xy, tx2_xy, r1: float, r2: float, hint_xy=None):
    """
    Tính toạ độ (x, y) của RX từ 2 trạm phát TX1, TX2 (đơn vị mét)
    với khoảng cách r1 (đến TX1) và r2 (đến TX2).
    Trả về (x, y) nghiệm gần với hint_xy nhất (nếu có),
    hoặc None nếu không có nghiệm thực.
    """
    x1, y1 = tx1_xy
    x2, y2 = tx2_xy
    d = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    if d == 0:
        return None
    a = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
    h2 = r1 ** 2 - a ** 2
    if h2 < 0:
        return None  # không có nghiệm thực
    h = np.sqrt(h2)
    # điểm giữa trên đường nối TX1→TX2
    mx = x1 + a * (x2 - x1) / d
    my = y1 + a * (y2 - y1) / d
    # vector vuông góc
    px = h * (y2 - y1) / d
    py = h * (x2 - x1) / d
    sol1 = (mx + px, my - py)
    sol2 = (mx - px, my + py)
    if hint_xy is None:
        return sol1
    # chọn nghiệm gần hint nhất
    d1 = (sol1[0] - hint_xy[0]) ** 2 + (sol1[1] - hint_xy[1]) ** 2
    d2 = (sol2[0] - hint_xy[0]) ** 2 + (sol2[1] - hint_xy[1]) ** 2
    return sol1 if d1 <= d2 else sol2


# ══════════════════════════════════════════════════════════════════════════════
# BladeRF RX (chạy trong thread riêng)
# ══════════════════════════════════════════════════════════════════════════════

class BladeRFReceiver:
    """
    Thread thu liên tục từ BladeRF RX, đẩy chunk complex64 vào queue.
    """

    def __init__(self, freq_hz, fs, bw, rx_gain, chunk_samples, queue_maxsize=100, serial=""):
        import queue
        self._q = queue.Queue(maxsize=queue_maxsize)
        self._stop = threading.Event()
        self._freq   = freq_hz
        self._fs     = fs
        self._bw     = bw
        self._gain   = rx_gain
        self._chunk  = chunk_samples
        self._serial = serial
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

    def get_chunk(self, timeout=1.0):
        """Lấy 1 chunk complex64 từ queue, None nếu timeout."""
        import queue
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def _run(self):
        try:
            dev_id = f"*:serial={self._serial}" if self._serial else ""
            b = _bladerf.BladeRF(dev_id) if dev_id else _bladerf.BladeRF()
            print(f"[RX]  Thiết bị: {b.get_board_name()}  serial={self._serial or 'auto'}")
        except Exception as e:
            print(f"[RX]  Lỗi mở BladeRF: {e}")
            return

        ch = _bladerf.CHANNEL_RX(0)
        b.set_sample_rate(ch, self._fs)
        b.set_frequency(ch, self._freq)
        b.set_bandwidth(ch, self._bw)
        b.set_gain(ch, self._gain)
        print(f"[RX]  fc={self._freq/1e6:.3f} MHz  fs={self._fs/1e6:.2f} MHz  gain={self._gain} dB")

        b.sync_config(
            layout=_bladerf.ChannelLayout.RX_X1,
            fmt=_bladerf.Format.SC16_Q11,
            num_buffers=16,
            buffer_size=8192,
            num_transfers=8,
            stream_timeout=3500,
        )
        b.enable_module(ch, True)
        print(f"[RX]  Bắt đầu thu {self._chunk} mẫu/chunk...")

        buf = np.zeros(self._chunk * 2, dtype=np.int16)
        try:
            while not self._stop.is_set():
                b.sync_rx(buf, self._chunk)
                chunk_iq = sc16q11_to_complex64(buf)
                if not self._q.full():
                    self._q.put(chunk_iq.copy())
        except Exception as e:
            if not self._stop.is_set():
                print(f"[RX]  Lỗi: {e}")
        finally:
            b.enable_module(ch, False)
            b.close()
            print("[RX]  Đã đóng BladeRF.")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(argv=None):
    p = argparse.ArgumentParser(description="Thu tín hiệu GNSS realtime từ BladeRF RX + GUI tương quan.")
    p.add_argument("--freq_hz",    type=float, default=1575.42e6, help="Tần số RF sóng mang (Hz). Mặc định GPS L1 = 1575.42 MHz.")
    p.add_argument("--fs",         type=float, default=2e6,     help="Sample rate (Hz).")
    p.add_argument("--Rc",         type=float, default=1.023e6, help="Chip rate (Hz).")
    p.add_argument("--ft",         type=float, default=0.0,     help="Doppler offset bù (Hz).")
    p.add_argument("--rx_gain",    type=int,   default=60,      help="Gain RX (dB).")
    p.add_argument("--chunk_ms",   type=float, default=5.0,     help="Số ms mỗi chunk xử lý.")
    p.add_argument("--prn1_start", type=int,   default=11,      help="PRN nhóm 1 bắt đầu.")
    p.add_argument("--prn1_end",   type=int,   default=20,      help="PRN nhóm 1 kết thúc.")
    p.add_argument("--prn2_start", type=int,   default=21,      help="PRN nhóm 2 bắt đầu.")
    p.add_argument("--prn2_end",   type=int,   default=30,      help="PRN nhóm 2 kết thúc.")
    p.add_argument("--serial",     default="a0e5ffb5f1c28a2d57f5f5d9d13372ed", help="Serial BladeRF RX.")
    p.add_argument("--save",       default="", help="Lưu IQ thu được ra file .bin (complex64).")
    p.add_argument("--file",       default="", help="Đọc từ file .bin thay vì BladeRF (debug).")
    args = p.parse_args(argv)

    fs         = args.fs
    Rc         = args.Rc
    ft         = args.ft
    Nchunk     = int(fs * args.chunk_ms / 1000)   # số mẫu mỗi chunk (ví dụ 5ms → 10000 mẫu ở 2MHz)
    Nfft       = Nchunk * 2                        # zero-pad x2 để tương quan đủ chu kỳ
    bw         = fs * 0.8
    speedOfLight = 299792458

    print("══════════════════════════════════════════════════════")
    print("  thu_tin_hieu_bladerf_gui.py  —  BladeRF RX + GUI")
    print(f"  fc={args.freq_hz/1e6:.3f} MHz  |  fs={fs/1e6:.2f} MHz  |  gain={args.rx_gain} dB")
    print(f"  Nhóm 1: PRN {args.prn1_start}–{args.prn1_end}  |  "
          f"Nhóm 2: PRN {args.prn2_start}–{args.prn2_end}")
    print(f"  chunk={Nchunk} mẫu ({args.chunk_ms} ms)  |  Nfft={Nfft}")
    print("══════════════════════════════════════════════════════\n")

    # ── Tạo tham chiếu nội bộ ──────────────────────────────────────────────
    print("[INFO] Đang tổng hợp mã tham chiếu nội bộ...")
    F_local1 = make_local_ref(args.prn1_start, args.prn1_end, fs, Rc, Nfft)
    F_local2 = make_local_ref(args.prn2_start, args.prn2_end, fs, Rc, Nfft)
    print("[INFO] Hoàn thành.")

    # ── Toạ độ TDOA (giữ nguyên từ thu_tin_hieu_rt_gui.py) ─────────────────
    # TX1 = [20.9896100, 105.7110745]
    # TX2 = [20.9924397, 105.7106347]
    # RX  = [20.9911114, 105.7107914]
    # T1  = calcDistance(TX1[0], TX1[1], RX[0], RX[1])
    # T2  = calcDistance(TX2[0], TX2[1], RX[0], RX[1])
    T1_init = 3.92
    T2_init = 2.94
    D_init  = 6.86

    T1 = T1_init
    T2 = T2_init
    T0  = (T1 - T2) / speedOfLight * fs
    # D   = calcDistance(TX1[0], TX1[1], TX2[0], TX2[1])
    D   = D_init

    use_file = bool(args.file)
    receiver = None
    file_iq  = None
    file_ptr = 0

    if use_file:
        if not os.path.exists(args.file):
            print(f"[ERROR] Không tìm thấy file: {args.file}")
            return 1
        raw = np.fromfile(args.file, dtype=np.complex64)
        file_iq = raw
        print(f"[FILE] Đã nạp {len(file_iq)} mẫu từ {args.file}")
    else:
        if not HAS_BLADERF:
            print("[ERROR] Module bladerf chưa cài. Dùng --file để đọc từ file.")
            return 1
        receiver = BladeRFReceiver(
            freq_hz=args.freq_hz,
            fs=fs,
            bw=bw,
            rx_gain=args.rx_gain,
            chunk_samples=Nchunk,
            serial=args.serial,
        )
        receiver.start()
        time.sleep(0.5)   # đợi BladeRF khởi động

    # ── Mở file lưu (tùy chọn) ─────────────────────────────────────────────
    save_fh = open(args.save, "wb") if args.save else None

    # ── Khởi tạo GUI ───────────────────────────────────────────────────────
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(
        f"BladeRF RX — Realtime Correlation  |  "
        f"fc={args.freq_hz/1e6:.3f} MHz  |  fs={fs/1e6:.2f} MHz",
        fontsize=12, color="white"
    )

    x_axis = np.arange(Nfft)
    line1, = ax1.plot(x_axis, np.zeros(Nfft), color="#1f77b4", lw=0.6, alpha=0.8, label=f"PRN {args.prn1_start}–{args.prn1_end}")
    line2, = ax2.plot(x_axis, np.zeros(Nfft), color="#ff7f0e", lw=0.6, alpha=0.8, label=f"PRN {args.prn2_start}–{args.prn2_end}")
    peak1, = ax1.plot([0], [0], "ro", ms=7, label="Peak 1")
    peak2, = ax2.plot([0], [0], "go", ms=7, label="Peak 2")
    txt1   = ax1.text(0.01, 0.92, "", transform=ax1.transAxes, color="white", fontsize=9)
    txt2   = ax2.text(0.01, 0.92, "", transform=ax2.transAxes, color="white", fontsize=9)

    ax1.set_title(f"Correlation — Nhóm 1 (PRN {args.prn1_start}–{args.prn1_end})", color="white")
    ax2.set_title(f"Correlation — Nhóm 2 (PRN {args.prn2_start}–{args.prn2_end})", color="white")
    for ax in [ax1, ax2]:
        ax.set_xlabel("Delay (samples)")
        ax.set_ylabel("Magnitude")
        ax.set_xlim(0, Nfft)
        ax.set_ylim(0, 100)
        ax.grid(True, ls="--", alpha=0.4)
        ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()

    # ── Biến trạng thái chia sẻ giữa update callback và vòng lặp ──────────
    buffer_iq = np.array([], dtype=np.complex64)
    n_arr     = np.arange(Nchunk, dtype=np.float32)
    running   = [True]
    delta_t_fixed = [None]

    def update(_frame):
        nonlocal buffer_iq, file_ptr

        # Lấy chunk từ BladeRF hoặc file
        if use_file:
            if file_ptr + Nchunk > len(file_iq):
                file_ptr = 0
            chunk = file_iq[file_ptr: file_ptr + Nchunk]
            file_ptr += Nchunk
        else:
            chunk = receiver.get_chunk(timeout=0.1)
            if chunk is None:
                return line1, line2, peak1, peak2, txt1, txt2

        # Lưu nếu cần
        if save_fh:
            chunk.astype(np.complex64).tofile(save_fh)

        buffer_iq = np.concatenate((buffer_iq, chunk))

        if len(buffer_iq) < Nchunk:
            return line1, line2, peak1, peak2, txt1, txt2

        IQ = buffer_iq[:Nchunk]
        buffer_iq = buffer_iq[Nchunk:]

        # Bù Doppler
        IQ_shifted = IQ * np.exp(1j * 2 * np.pi * ft * n_arr / fs).astype(np.complex64)

        # Zero-pad + FFT
        IQ_padded = np.zeros(Nfft, dtype=np.complex64)
        IQ_padded[:Nchunk] = IQ_shifted
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
        txt1.set_text(f"Peak @ {tau1}  |  Mag={v1:.1f}")
        txt2.set_text(f"Peak @ {tau2}  |  Mag={v2:.1f}")

        # Auto scale Y
        top = max(v1, v2, 10) * 1.5
        if top > ax1.get_ylim()[1] * 0.8 or top < ax1.get_ylim()[1] * 0.1:
            ax1.set_ylim(0, top)
            ax2.set_ylim(0, top)

        if delta_t_fixed[0] is None:
            delta_t_fixed[0] = (tau1 - tau2) - T0
            print(f"[INFO] Delta_T được khóa tại lần tính đầu: {delta_t_fixed[0]:.6f} samples")

        Delta_T  = delta_t_fixed[0]
        Delta_m  = Delta_T / fs * speedOfLight          # TDOA hiệu chỉnh (m)
        Delta_C  = (tau1 - tau2) / fs * speedOfLight    # TDOA thô (m)
        X        = (-Delta_C + D + Delta_m) / 2         # T1_est: khoảng cách TX1→RX (m)
        X2       = D - X                                # T2_est: khoảng cách TX2→RX (m)
        print(f"[CORR] Peak1={tau1:5d} ({v1:6.1f})  Peak2={tau2:5d} ({v2:6.1f})"
              f"  TDOA={Delta_m:+.2f} m"
              f"  |  T1 được tính ra bằng: {X:.3f} m  T2 được tính ra bằng: {X2:.3f} m")

        return line1, line2, peak1, peak2, txt1, txt2

    ani = animation.FuncAnimation(
        fig, update, interval=int(args.chunk_ms), blit=False, cache_frame_data=False
    )

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        running[0] = False
        if receiver:
            receiver.stop()
        if save_fh:
            save_fh.close()
            print(f"[INFO] Đã lưu IQ: {args.save}")
        print("[INFO] Thoát.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
