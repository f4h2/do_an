"""
thu_bladerf_freqsearch_gui.py
==============================
Thu tín hiệu GNSS realtime từ BladeRF x40, tính tương quan với mã C/A nội,
**tự động tìm lệch tần (frequency search)** để bù sai số TCXO giữa 2 thiết bị.
Hiển thị đồ thị realtime.

Vấn đề giải quyết:
    Khi TX và RX là 2 BladeRF khác nhau, TCXO mỗi máy có sai số vài ppm,
    dẫn đến lệch tần số sóng mang hàng trăm đến hàng nghìn Hz.
    File này quét dải tần ±ft_range Hz theo bước ft_step Hz, chọn ft cho
    đỉnh tương quan cao nhất, rồi dần hội tụ và bám theo lệch tần thực tế.

Chạy (2 terminal):
    Terminal 1:  python phat_tin_hieu_bladerf.py --freq_hz 433e6 --prn_start 11 --prn_end 20
    Terminal 2:  python thu_bladerf_freqsearch_gui.py --freq_hz 433e6

Argparse:
    --freq_hz      : Tần số RF sóng mang (Hz), mặc định 1575.42e6
    --fs           : Sample rate (Hz), mặc định 2e6
    --Rc           : Chip rate (Hz), mặc định 1.023e6
    --rx_gain      : Gain RX (dB), mặc định 60
    --prn1_start   : PRN bắt đầu nhóm 1, mặc định 11
    --prn1_end     : PRN kết thúc nhóm 1, mặc định 20
    --prn2_start   : PRN bắt đầu nhóm 2, mặc định 21
    --prn2_end     : PRN kết thúc nhóm 2, mặc định 30
    --chunk_ms     : Số ms mỗi chunk, mặc định 10
    --ft_range     : Bán kính tìm kiếm tần số (Hz), mặc định 20000
    --ft_step      : Bước tìm kiếm tần số (Hz), mặc định 200
    --ft_refine    : Sau khi tìm thô, tinh chỉnh trong ±ft_step với bước ft_step/10
    --ft_ema_alpha : Hệ số EMA bám lệch tần (0–1, nhỏ = chậm nhưng ổn định), mặc định 0.15
    --serial       : Serial BladeRF RX
    --save         : Lưu IQ thu được ra file .bin
    --file         : Đọc từ file .bin thay vì BladeRF (debug)
"""

import argparse
import os
import sys
import threading
import time

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
    """FFT của mã C/A tham chiếu nội bộ (Nfft điểm, complex64)."""
    total_chips = (prn_end - prn_start + 1) * 1023
    cacodes = np.concatenate([generateCAcode(i) for i in range(prn_start, prn_end + 1)])
    idx = np.floor(np.arange(Nfft) / fs * Rc).astype(int) % total_chips
    return np.fft.fft(cacodes[idx].astype(np.complex64))


def sc16q11_to_complex64(buf: np.ndarray) -> np.ndarray:
    return (buf[0::2].astype(np.float32) + 1j * buf[1::2].astype(np.float32)).astype(np.complex64)


def best_corr_over_freqs(
    IQ: np.ndarray,
    F_local: np.ndarray,
    n_arr: np.ndarray,
    fs: float,
    ft_candidates: np.ndarray,
    Nfft: int,
    Nchunk: int,
) -> tuple[float, np.ndarray, int, float]:
    """
    Quét qua danh sách ft_candidates, tính tương quan cho mỗi ft,
    trả về (ft_best, mag_best, tau_best, peak_best).

    Để tăng tốc, dùng FFT shift thay vì nhân exp cho mỗi ft riêng:
      - Dịch IQ trong miền tần số tương đương nhân exp(-j2π ft n/fs)
      - Mỗi ft ứng với shift bin k = round(ft / (fs/Nfft))
    """
    # FFT của IQ zero-padded (chưa bù tần)
    IQ_padded = np.zeros(Nfft, dtype=np.complex64)
    IQ_padded[:Nchunk] = IQ
    F_IQ_base = np.fft.fft(IQ_padded)   # shape (Nfft,)

    df = fs / Nfft  # Hz per bin

    best_peak = -1.0
    best_ft   = ft_candidates[0]
    best_mag  = np.zeros(Nfft, dtype=np.float32)
    best_tau  = 0

    for ft in ft_candidates:
        # Dịch phổ: nhân exp(-j2π ft n/fs) ≡ dịch tròn F_IQ sang phải k bin
        k = int(round(ft / df))
        F_IQ_shifted = np.roll(F_IQ_base, k)

        mag = np.abs(np.fft.ifft(F_local * np.conj(F_IQ_shifted))).astype(np.float32)
        peak_val = float(np.max(mag))

        if peak_val > best_peak:
            best_peak = peak_val
            best_ft   = ft
            best_mag  = mag
            best_tau  = int(np.argmax(mag))

    return best_ft, best_mag, best_tau, best_peak


# ══════════════════════════════════════════════════════════════════════════════
# BladeRF RX thread
# ══════════════════════════════════════════════════════════════════════════════

class BladeRFReceiver:
    def __init__(self, freq_hz, fs, bw, rx_gain, chunk_samples, queue_maxsize=100, serial=""):
        import queue
        self._q      = queue.Queue(maxsize=queue_maxsize)
        self._stop   = threading.Event()
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
                ciq = sc16q11_to_complex64(buf)
                if not self._q.full():
                    self._q.put(ciq.copy())
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
    p = argparse.ArgumentParser(
        description="Thu GNSS BladeRF RX với frequency search tự động bù lệch TCXO."
    )
    p.add_argument("--freq_hz",      type=float, default=1575.42e6)
    p.add_argument("--fs",           type=float, default=2e6)
    p.add_argument("--Rc",           type=float, default=1.023e6)
    p.add_argument("--rx_gain",      type=int,   default=60)
    p.add_argument("--prn1_start",   type=int,   default=11)
    p.add_argument("--prn1_end",     type=int,   default=20)
    p.add_argument("--prn2_start",   type=int,   default=21)
    p.add_argument("--prn2_end",     type=int,   default=30)
    p.add_argument("--chunk_ms",     type=float, default=10.0,
                   help="Chunk size (ms). Lớn hơn → phân giải tần số tốt hơn, nhưng chậm hơn.")
    p.add_argument("--ft_range",     type=float, default=20000.0,
                   help="Bán kính tìm kiếm tần số ban đầu (Hz). Mặc định ±20 kHz.")
    p.add_argument("--ft_step",      type=float, default=200.0,
                   help="Bước tìm kiếm tần số (Hz). Mặc định 200 Hz.")
    p.add_argument("--ft_refine",    action="store_true", default=True,
                   help="Tinh chỉnh ft sau tìm thô (bước = ft_step/10).")
    p.add_argument("--ft_ema_alpha", type=float, default=0.15,
                   help="Hệ số EMA bám ft (0–1). Nhỏ = chậm nhưng ổn định. Mặc định 0.15.")
    p.add_argument("--lock_chunks",  type=int,   default=8,
                   help="Số chunk liên tiếp ft_best ổn định để khoá ft_est. Mặc định 8.")
    p.add_argument("--lock_tol",     type=float, default=None,
                   help="Ngưỡng ổn định để khoá (Hz). Mặc định = ft_step.")
    p.add_argument("--recheck",      type=int,   default=100,
                   help="Sau khi khoá, cứ N chunk chạy coarse search 1 lần. Mặc định 100.")
    p.add_argument("--serial",       default="a0e5ffb5f1c28a2d57f5f5d9d13372ed")
    p.add_argument("--save",         default="")
    p.add_argument("--file",         default="")
    args = p.parse_args(argv)

    fs    = args.fs
    Rc    = args.Rc
    alpha = args.ft_ema_alpha
    Nchunk = int(fs * args.chunk_ms / 1000)
    Nfft   = Nchunk * 2
    bw     = fs * 0.8
    speedOfLight = 299792458

    print("══════════════════════════════════════════════════════════════")
    print("  thu_bladerf_freqsearch_gui.py  —  BladeRF RX + Freq Search")
    print(f"  fc={args.freq_hz/1e6:.3f} MHz  fs={fs/1e6:.2f} MHz  gain={args.rx_gain} dB")
    print(f"  Nhóm 1: PRN {args.prn1_start}–{args.prn1_end}  |  "
          f"Nhóm 2: PRN {args.prn2_start}–{args.prn2_end}")
    print(f"  Nchunk={Nchunk} ({args.chunk_ms} ms)  Nfft={Nfft}")
    print(f"  Freq search: ±{args.ft_range:.0f} Hz  bước={args.ft_step:.0f} Hz  "
          f"refine={'ON' if args.ft_refine else 'OFF'}  EMA α={alpha}")
    print("══════════════════════════════════════════════════════════════\n")

    # ── Dải tần tìm kiếm ban đầu ──────────────────────────────────────────
    ft_candidates_coarse = np.arange(
        -args.ft_range, args.ft_range + args.ft_step, args.ft_step, dtype=np.float32
    )
    print(f"[INFO] Số ứng viên tần số (thô): {len(ft_candidates_coarse)}")

    # ── Tạo tham chiếu nội bộ ─────────────────────────────────────────────
    print("[INFO] Đang tổng hợp mã tham chiếu nội bộ...")
    F_local1 = make_local_ref(args.prn1_start, args.prn1_end, fs, Rc, Nfft)
    F_local2 = make_local_ref(args.prn2_start, args.prn2_end, fs, Rc, Nfft)
    print("[INFO] Hoàn thành.")

    # ── Toạ độ TDOA ───────────────────────────────────────────────────────
    TX1 = [20.9896100, 105.7110745]
    TX2 = [20.9924397, 105.7106347]
    RX  = [20.9911114, 105.7107914]
    T1  = calcDistance(TX1[0], TX1[1], RX[0], RX[1])
    T2  = calcDistance(TX2[0], TX2[1], RX[0], RX[1])
    T0  = (T1 - T2) / speedOfLight * fs

    # ── Nguồn dữ liệu ─────────────────────────────────────────────────────
    use_file = bool(args.file)
    receiver = None
    file_iq  = None
    file_ptr = 0

    if use_file:
        if not os.path.exists(args.file):
            print(f"[ERROR] Không tìm thấy file: {args.file}")
            return 1
        file_iq = np.fromfile(args.file, dtype=np.complex64)
        print(f"[FILE] Đã nạp {len(file_iq)} mẫu từ {args.file}")
    else:
        if not HAS_BLADERF:
            print("[ERROR] Module bladerf chưa cài. Dùng --file để chạy thử.")
            return 1
        receiver = BladeRFReceiver(
            freq_hz=args.freq_hz, fs=fs, bw=bw, rx_gain=args.rx_gain,
            chunk_samples=Nchunk, serial=args.serial,
        )
        receiver.start()
        time.sleep(0.5)

    save_fh = open(args.save, "wb") if args.save else None

    # ── GUI ───────────────────────────────────────────────────────────────
    plt.style.use("dark_background")
    fig, axes = plt.subplots(3, 1, figsize=(13, 10),
                             gridspec_kw={"height_ratios": [3, 3, 2]})
    ax1, ax2, ax3 = axes
    title_obj = fig.suptitle(
        f"BladeRF RX — Freq Search  |  fc={args.freq_hz/1e6:.3f} MHz  |  "
        f"ft_est=0 Hz",
        fontsize=11, color="white"
    )

    x_axis = np.arange(Nfft)

    line1, = ax1.plot(x_axis, np.zeros(Nfft), color="#1f77b4", lw=0.6, alpha=0.85,
                      label=f"PRN {args.prn1_start}–{args.prn1_end}")
    line2, = ax2.plot(x_axis, np.zeros(Nfft), color="#ff7f0e", lw=0.6, alpha=0.85,
                      label=f"PRN {args.prn2_start}–{args.prn2_end}")
    peak1, = ax1.plot([0], [0], "ro", ms=7, label="Peak 1")
    peak2, = ax2.plot([0], [0], "go", ms=7, label="Peak 2")
    txt1   = ax1.text(0.01, 0.90, "", transform=ax1.transAxes, color="white", fontsize=9)
    txt2   = ax2.text(0.01, 0.90, "", transform=ax2.transAxes, color="white", fontsize=9)

    ax1.set_title(f"Correlation — Nhóm 1 (PRN {args.prn1_start}–{args.prn1_end})", color="white")
    ax2.set_title(f"Correlation — Nhóm 2 (PRN {args.prn2_start}–{args.prn2_end})", color="white")
    for ax in [ax1, ax2]:
        ax.set_xlabel("Delay (samples)")
        ax.set_ylabel("Magnitude")
        ax.set_xlim(0, Nfft)
        ax.set_ylim(0, 100)
        ax.grid(True, ls="--", alpha=0.4)
        ax.legend(loc="upper right", fontsize=9)

    # Đồ thị lịch sử ft_estimated
    FT_HIST_LEN = 200
    ft_history  = np.zeros(FT_HIST_LEN, dtype=np.float32)
    ft_hist_x   = np.arange(FT_HIST_LEN)
    line_ft, = ax3.plot(ft_hist_x, ft_history, color="#2ca02c", lw=1.2, label="ft_estimated (Hz)")
    ax3.set_title("Ước lượng lệch tần theo thời gian", color="white")
    ax3.set_xlabel("Chunk #")
    ax3.set_ylabel("ft (Hz)")
    ax3.set_xlim(0, FT_HIST_LEN)
    ax3.set_ylim(-args.ft_range, args.ft_range)
    ax3.axhline(0, color="gray", lw=0.8, ls="--")
    ax3.grid(True, ls="--", alpha=0.4)
    ax3.legend(loc="upper right", fontsize=9)

    plt.tight_layout()

    # ── Trạng thái bộ nhớ ─────────────────────────────────────────────────
    lock_tol    = args.lock_tol if args.lock_tol is not None else args.ft_step
    buffer_iq   = np.array([], dtype=np.complex64)
    n_arr       = np.arange(Nchunk, dtype=np.float32)
    ft_est      = [0.0]          # lệch tần ước lượng hiện tại (EMA)
    use_coarse  = [True]         # True → quét toàn bộ; False → chỉ quét quanh ft_est
    ft_locked   = [False]        # True → dừng search, dùng ft_est cố định
    ft_recent   = []             # lịch sử ft_best gần đây để kiểm tra hội tụ
    chunk_count = [0]
    RECHECK     = args.recheck   # khi locked, cứ N chunk coarse search 1 lần

    def update(_frame):
        nonlocal buffer_iq, file_ptr

        # ── Lấy chunk ─────────────────────────────────────────────────────
        if use_file:
            if file_ptr + Nchunk > len(file_iq):
                file_ptr = 0
            chunk = file_iq[file_ptr: file_ptr + Nchunk]
            file_ptr += Nchunk
        else:
            chunk = receiver.get_chunk(timeout=0.1)
            if chunk is None:
                return line1, line2, peak1, peak2, txt1, txt2, line_ft

        if save_fh:
            chunk.astype(np.complex64).tofile(save_fh)

        buffer_iq = np.concatenate((buffer_iq, chunk))
        if len(buffer_iq) < Nchunk:
            return line1, line2, peak1, peak2, txt1, txt2, line_ft

        IQ = buffer_iq[:Nchunk]
        buffer_iq = buffer_iq[Nchunk:]
        chunk_count[0] += 1

        # ── Chuẩn bị FFT chung ────────────────────────────────────────────
        IQ_padded = np.zeros(Nfft, dtype=np.complex64)
        IQ_padded[:Nchunk] = IQ
        F_IQ_base = np.fft.fft(IQ_padded)
        df = fs / Nfft

        # ── Quyết định có search hay không ───────────────────────────────
        # Khi đã locked, chỉ recheck mỗi RECHECK chunk
        need_search = (not ft_locked[0]) or (chunk_count[0] % RECHECK == 0)

        if need_search:
            do_coarse = use_coarse[0] or ft_locked[0]  # recheck luôn dùng coarse

            if do_coarse:
                candidates = ft_candidates_coarse
            else:
                fine_half = args.ft_step * 2
                fine_step = args.ft_step / 5
                candidates = np.arange(
                    ft_est[0] - fine_half,
                    ft_est[0] + fine_half + fine_step,
                    fine_step, dtype=np.float32
                )

            ft_best1, _, _, _ = best_corr_over_freqs(
                IQ, F_local1, n_arr, fs, candidates, Nfft, Nchunk
            )

            # Tinh chỉnh
            if args.ft_refine and do_coarse:
                refine_step = args.ft_step / 10.0
                refine_cands = np.arange(
                    ft_best1 - args.ft_step,
                    ft_best1 + args.ft_step + refine_step,
                    refine_step, dtype=np.float32
                )
                ft_best1, _, _, _ = best_corr_over_freqs(
                    IQ, F_local1, n_arr, fs, refine_cands, Nfft, Nchunk
                )

            # EMA bám ft
            ft_est[0] = alpha * ft_best1 + (1 - alpha) * ft_est[0]
            use_coarse[0] = False

            # ── Kiểm tra hội tụ → khoá ft_est ────────────────────────────
            ft_recent.append(ft_best1)
            if len(ft_recent) > args.lock_chunks:
                ft_recent.pop(0)
            if (not ft_locked[0]) and len(ft_recent) == args.lock_chunks:
                if max(ft_recent) - min(ft_recent) <= lock_tol:
                    ft_locked[0] = True
                    print(f"[LOCK] ft_est khoá tại {ft_est[0]:+.1f} Hz  "
                          f"(ổn định trong {args.lock_chunks} chunk liên tiếp)")

        # ── Tính tương quan cả 2 nhóm bằng ft_est cố định ────────────────
        k_est     = int(round(ft_est[0] / df))
        F_IQ_corr = np.roll(F_IQ_base, k_est)

        mag1 = np.abs(np.fft.ifft(F_local1 * np.conj(F_IQ_corr))).astype(np.float32)
        mag2 = np.abs(np.fft.ifft(F_local2 * np.conj(F_IQ_corr))).astype(np.float32)
        tau1 = int(np.argmax(mag1)); v1 = float(mag1[tau1])
        tau2 = int(np.argmax(mag2)); v2 = float(mag2[tau2])

        # Nếu peak quá yếu sau khi đã locked → mở khoá để search lại
        if ft_locked[0] and max(v1, v2) < 10:
            ft_locked[0] = False
            use_coarse[0] = True
            ft_recent.clear()
            print("[UNLOCK] Peak mất — search lại...")

        lock_status = "LOCKED" if ft_locked[0] else ("search" if need_search else "---")

        # ── Cập nhật đồ thị tương quan ─────────────────────────────────────
        line1.set_ydata(mag1)
        line2.set_ydata(mag2)
        peak1.set_data([tau1], [v1])
        peak2.set_data([tau2], [v2])
        txt1.set_text(f"Peak @ {tau1}  |  Mag={v1:.1f}  |  [{lock_status}] ft={ft_est[0]:+.0f} Hz")
        txt2.set_text(f"Peak @ {tau2}  |  Mag={v2:.1f}  |  ft_est={ft_est[0]:+.0f} Hz")

        top = max(v1, v2, 10) * 1.5
        if top > ax1.get_ylim()[1] * 0.85 or top < ax1.get_ylim()[1] * 0.1:
            ax1.set_ylim(0, top)
            ax2.set_ylim(0, top)

        # ── Cập nhật đồ thị lịch sử ft ─────────────────────────────────────
        ft_history[:-1] = ft_history[1:]
        ft_history[-1]  = ft_est[0]
        line_ft.set_ydata(ft_history)
        ft_range_now = max(abs(ft_est[0]) * 2, args.ft_range * 0.1, 1000)
        ax3.set_ylim(-ft_range_now, ft_range_now)

        # ── TDOA ───────────────────────────────────────────────────────────
        Delta_T = (tau1 - tau2) - T0
        Delta_m = Delta_T / fs * speedOfLight

        title_obj.set_text(
            f"BladeRF RX  [{lock_status}]  |  fc={args.freq_hz/1e6:.3f} MHz  |  "
            f"ft_est={ft_est[0]:+.0f} Hz  |  TDOA={Delta_m:+.2f} m"
        )

        print(
            f"[{chunk_count[0]:4d}] {lock_status:8s}  "
            f"ft_est={ft_est[0]:+7.0f} Hz  "
            f"Peak1={tau1:5d}({v1:6.1f})  Peak2={tau2:5d}({v2:6.1f})  "
            f"TDOA={Delta_m:+.2f} m"
        )

        return line1, line2, peak1, peak2, txt1, txt2, line_ft

    ani = animation.FuncAnimation(
        fig, update, interval=int(args.chunk_ms), blit=False, cache_frame_data=False
    )

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        if receiver:
            receiver.stop()
        if save_fh:
            save_fh.close()
            print(f"[INFO] Đã lưu IQ: {args.save}")
        print(f"[INFO] ft_est cuối: {ft_est[0]:+.1f} Hz")
        print("[INFO] Thoát.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
