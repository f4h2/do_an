"""
thu_tin_hieu_bladerf_gui_copy.py
=================================
Thu tín hiệu GNSS realtime từ BladeRF x40 + GUI nhập toạ độ TX1/TX2/RX,
hiển thị realtime tương quan + T1_est, T2_est, toạ độ hiện tại.

Sơ đồ kết nối:
    [phat_tin_hieu_bladerf.py chạy trên cùng máy]
         TX1 (ăng ten phát)  →  sóng RF  →  RX1 (ăng ten thu)
                                                    ↓
                               thu_tin_hieu_bladerf_gui_copy.py
                               nhập toạ độ + tính TDOA + vẽ đồ thị

Chạy:
    Terminal 1:  python phat_tin_hieu_bladerf.py
    Terminal 2:  python thu_tin_hieu_bladerf_gui_copy.py
"""

import argparse
import sys
import threading
import time
import os

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.widgets import TextBox, Button

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
    total_chips = (prn_end - prn_start + 1) * 1023
    cacodes = np.concatenate([generateCAcode(i) for i in range(prn_start, prn_end + 1)])
    n_local = np.arange(Nfft)
    idx = np.floor(n_local / fs * Rc).astype(int) % total_chips
    local_fs = cacodes[idx].astype(np.complex64)
    return np.fft.fft(local_fs)


def sc16q11_to_complex64(buf: np.ndarray) -> np.ndarray:
    I = buf[0::2].astype(np.float32)
    Q = buf[1::2].astype(np.float32)
    return (I + 1j * Q).astype(np.complex64)


def latlon_to_xy(ref_lat: float, ref_lon: float, lat: float, lon: float):
    """Chuyển (lat, lon) sang Cartesian (x, y) mét, lấy (ref_lat, ref_lon) làm gốc."""
    R = 6371000.0
    x = np.radians(lon - ref_lon) * np.cos(np.radians((lat + ref_lat) / 2)) * R
    y = np.radians(lat - ref_lat) * R
    return x, y


def trilaterate_2d(tx1_xy, tx2_xy, r1: float, r2: float, hint_xy=None):
    """
    Tính (x, y) của RX từ 2 trạm phát TX1, TX2 (m) với khoảng cách r1, r2.
    Chọn nghiệm gần hint_xy nhất. Trả về None nếu không có nghiệm thực.
    """
    x1, y1 = tx1_xy
    x2, y2 = tx2_xy
    d = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    if d == 0:
        return None
    a = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
    h2 = r1 ** 2 - a ** 2
    if h2 < 0:
        return None
    h = np.sqrt(h2)
    mx = x1 + a * (x2 - x1) / d
    my = y1 + a * (y2 - y1) / d
    px = h * (y2 - y1) / d
    py = h * (x2 - x1) / d
    sol1 = (mx + px, my - py)
    sol2 = (mx - px, my + py)
    if hint_xy is None:
        return sol1
    d1 = (sol1[0] - hint_xy[0]) ** 2 + (sol1[1] - hint_xy[1]) ** 2
    d2 = (sol2[0] - hint_xy[0]) ** 2 + (sol2[1] - hint_xy[1]) ** 2
    return sol1 if d1 <= d2 else sol2


# ══════════════════════════════════════════════════════════════════════════════
# BladeRF RX
# ══════════════════════════════════════════════════════════════════════════════

class BladeRFReceiver:
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
    p = argparse.ArgumentParser(description="Thu GNSS realtime BladeRF + GUI nhập toạ độ.")
    p.add_argument("--freq_hz",    type=float, default=1575.42e6)
    p.add_argument("--fs",         type=float, default=2e6)
    p.add_argument("--Rc",         type=float, default=1.023e6)
    p.add_argument("--ft",         type=float, default=0.0)
    p.add_argument("--rx_gain",    type=int,   default=60)
    p.add_argument("--chunk_ms",   type=float, default=5.0)
    p.add_argument("--prn1_start", type=int,   default=11)
    p.add_argument("--prn1_end",   type=int,   default=20)
    p.add_argument("--prn2_start", type=int,   default=21)
    p.add_argument("--prn2_end",   type=int,   default=30)
    p.add_argument("--serial",     default="a0e5ffb5f1c28a2d57f5f5d9d13372ed")
    p.add_argument("--save",       default="")
    p.add_argument("--file",       default="")
    args = p.parse_args(argv)

    fs           = args.fs
    Rc           = args.Rc
    ft           = args.ft
    Nchunk       = int(fs * args.chunk_ms / 1000)
    Nfft         = Nchunk * 2
    bw           = fs * 0.8
    speedOfLight = 299792458

    # ── Mã tham chiếu ─────────────────────────────────────────────────────
    print("[INFO] Đang tổng hợp mã tham chiếu nội bộ...")
    F_local1 = make_local_ref(args.prn1_start, args.prn1_end, fs, Rc, Nfft)
    F_local2 = make_local_ref(args.prn2_start, args.prn2_end, fs, Rc, Nfft)
    print("[INFO] Hoàn thành.")

    # ── Toạ độ mặc định (có thể chỉnh trong GUI) ──────────────────────────
    coords = {
        "TX1_lat": 20.9896100,
        "TX1_lon": 105.7110745,
        "TX2_lat": 20.9924397,
        "TX2_lon": 105.7106347,
        "RX_lat":  20.9911114,
        "RX_lon":  105.7107914,
    }

    # Trạng thái tính toán (shared với animation callback)
    state = {
        "T0": 0.0,
        "D":  0.0,
        "T1": 0.0,
        "T2": 0.0,
        "T1_est": 0.0,
        "T2_est": 0.0,
        "tau1": 0,
        "tau2": 0,
        "v1": 0.0,
        "v2": 0.0,
        "Delta_m": 0.0,
        "x": 0.0,
        "y": 0.0,
        "tx1_xy": (0.0, 0.0),
        "tx2_xy": (0.0, 0.0),
        "rx_hint": (0.0, 0.0),
    }

    def recalc_geometry():
        TX1 = [coords["TX1_lat"], coords["TX1_lon"]]
        TX2 = [coords["TX2_lat"], coords["TX2_lon"]]
        RX  = [coords["RX_lat"],  coords["RX_lon"]]
        T1  = calcDistance(TX1[0], TX1[1], RX[0], RX[1])
        T2  = calcDistance(TX2[0], TX2[1], RX[0], RX[1])
        T0  = (T1 - T2) / speedOfLight * fs
        D   = calcDistance(TX1[0], TX1[1], TX2[0], TX2[1])
        tx1_xy  = (0.0, 0.0)
        tx2_xy  = latlon_to_xy(TX1[0], TX1[1], TX2[0], TX2[1])
        rx_hint = latlon_to_xy(TX1[0], TX1[1], RX[0], RX[1])
        state["T0"]      = T0
        state["D"]       = D
        state["T1"]      = T1
        state["T2"]      = T2
        state["tx1_xy"]  = tx1_xy
        state["tx2_xy"]  = tx2_xy
        state["rx_hint"] = rx_hint
        print(f"[GEO]  T1={T1:.2f} m  T2={T2:.2f} m  D={D:.2f} m  T0={T0:.4f} samples")
        print(f"[GEO]  TX1=(0, 0) m  TX2=({tx2_xy[0]:.2f}, {tx2_xy[1]:.2f}) m"
              f"  RX hint=({rx_hint[0]:.2f}, {rx_hint[1]:.2f}) m")

    recalc_geometry()

    # ── Nguồn dữ liệu ─────────────────────────────────────────────────────
    use_file = bool(args.file)
    receiver = None
    file_iq  = None
    file_ptr = [0]

    if use_file:
        if not os.path.exists(args.file):
            print(f"[ERROR] Không tìm thấy file: {args.file}")
            return 1
        file_iq = np.fromfile(args.file, dtype=np.complex64)
        print(f"[FILE] Đã nạp {len(file_iq)} mẫu từ {args.file}")
    else:
        if not HAS_BLADERF:
            print("[ERROR] Module bladerf chưa cài. Dùng --file để đọc từ file.")
            return 1
        receiver = BladeRFReceiver(
            freq_hz=args.freq_hz, fs=fs, bw=bw, rx_gain=args.rx_gain,
            chunk_samples=Nchunk, serial=args.serial,
        )
        receiver.start()
        time.sleep(0.5)

    save_fh = open(args.save, "wb") if args.save else None

    # ══════════════════════════════════════════════════════════════════════
    # Xây dựng GUI
    # ══════════════════════════════════════════════════════════════════════
    plt.style.use("dark_background")

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"BladeRF RX — Realtime TDOA  |  fc={args.freq_hz/1e6:.3f} MHz  |  fs={fs/1e6:.2f} MHz",
        fontsize=12, color="white",
    )

    # GridSpec: cột trái (nhập toạ độ + kết quả) | cột phải (2 đồ thị)
    gs = gridspec.GridSpec(
        nrows=2, ncols=2,
        left=0.05, right=0.98, top=0.93, bottom=0.30,
        wspace=0.35, hspace=0.45,
        width_ratios=[1, 2],
    )

    ax_info  = fig.add_subplot(gs[:, 0])   # toàn bộ cột trái: thông tin + kết quả
    ax_corr1 = fig.add_subplot(gs[0, 1])   # tương quan nhóm 1
    ax_corr2 = fig.add_subplot(gs[1, 1])   # tương quan nhóm 2

    # Tắt trục ax_info, dùng làm bảng text
    ax_info.set_facecolor("#111111")
    ax_info.set_xticks([])
    ax_info.set_yticks([])
    ax_info.set_title("Thông số & Kết quả", color="cyan", fontsize=10)

    info_text = ax_info.text(
        0.05, 0.97, "", transform=ax_info.transAxes,
        color="white", fontsize=9, va="top", family="monospace",
    )

    # Đường tương quan
    x_axis = np.arange(Nfft)
    line1, = ax_corr1.plot(x_axis, np.zeros(Nfft), color="#1f77b4", lw=0.6, alpha=0.8)
    line2, = ax_corr2.plot(x_axis, np.zeros(Nfft), color="#ff7f0e", lw=0.6, alpha=0.8)
    peak1, = ax_corr1.plot([0], [0], "ro", ms=7, label="Peak 1")
    peak2, = ax_corr2.plot([0], [0], "go", ms=7, label="Peak 2")
    txt1   = ax_corr1.text(0.01, 0.92, "", transform=ax_corr1.transAxes, color="white", fontsize=8)
    txt2   = ax_corr2.text(0.01, 0.92, "", transform=ax_corr2.transAxes, color="white", fontsize=8)

    ax_corr1.set_title(f"Correlation — PRN {args.prn1_start}–{args.prn1_end}", color="white")
    ax_corr2.set_title(f"Correlation — PRN {args.prn2_start}–{args.prn2_end}", color="white")
    for ax in [ax_corr1, ax_corr2]:
        ax.set_xlabel("Delay (samples)")
        ax.set_ylabel("Magnitude")
        ax.set_xlim(0, Nfft)
        ax.set_ylim(0, 100)
        ax.grid(True, ls="--", alpha=0.4)
        ax.legend(loc="upper right", fontsize=8)

    # ── TextBox nhập toạ độ (dưới cùng figure) ───────────────────────────
    #  6 ô: TX1_lat, TX1_lon, TX2_lat, TX2_lon, RX_lat, RX_lon
    #  + 1 nút "Cập nhật"
    box_h   = 0.038
    box_gap = 0.006
    label_x = 0.02
    box_x   = 0.18
    box_w   = 0.22
    col2_lx = 0.52
    col2_bx = 0.68

    fields = [
        ("TX1_lat", "TX1 Lat:",  label_x, box_x,   0.22),
        ("TX1_lon", "TX1 Lon:",  label_x, box_x,   0.22 - (box_h + box_gap)),
        ("TX2_lat", "TX2 Lat:",  col2_lx, col2_bx, 0.22),
        ("TX2_lon", "TX2 Lon:",  col2_lx, col2_bx, 0.22 - (box_h + box_gap)),
        ("RX_lat",  "RX  Lat:",  label_x, box_x,   0.22 - 2*(box_h + box_gap)),
        ("RX_lon",  "RX  Lon:",  col2_lx, col2_bx, 0.22 - 2*(box_h + box_gap)),
    ]

    textboxes = {}
    for key, label, lx, bx, by in fields:
        fig.text(lx, by + 0.008, label, color="cyan", fontsize=9, transform=fig.transFigure)
        ax_tb = fig.add_axes([bx, by, box_w, box_h])
        tb = TextBox(ax_tb, "", initial=str(coords[key]),
                     color="#222222", hovercolor="#333333",
                     label_pad=0.02)
        tb.label.set_color("cyan")
        tb.text_disp.set_color("white")
        tb.text_disp.set_fontsize(9)
        textboxes[key] = tb

    # Nút "Cập nhật toạ độ"
    ax_btn = fig.add_axes([0.40, 0.22 - 2*(box_h + box_gap) - 0.005, 0.18, box_h + 0.01])
    btn_update = Button(ax_btn, "Cập nhật toạ độ", color="#1a6b3c", hovercolor="#25a05a")
    btn_update.label.set_fontsize(10)
    btn_update.label.set_color("white")

    def on_update(_event):
        try:
            for key, tb in textboxes.items():
                coords[key] = float(tb.text)
            recalc_geometry()
            print(f"[GUI]  Đã cập nhật toạ độ: TX1=({coords['TX1_lat']}, {coords['TX1_lon']})"
                  f"  TX2=({coords['TX2_lat']}, {coords['TX2_lon']})"
                  f"  RX=({coords['RX_lat']}, {coords['RX_lon']})")
        except ValueError as e:
            print(f"[GUI]  Lỗi giá trị toạ độ: {e}")

    btn_update.on_clicked(on_update)

    # ══════════════════════════════════════════════════════════════════════
    # Animation callback
    # ══════════════════════════════════════════════════════════════════════
    buffer_iq = np.array([], dtype=np.complex64)
    n_arr     = np.arange(Nchunk, dtype=np.float32)

    def update(_frame):
        nonlocal buffer_iq

        # Lấy chunk
        if use_file:
            if file_ptr[0] + Nchunk > len(file_iq):
                file_ptr[0] = 0
            chunk = file_iq[file_ptr[0]: file_ptr[0] + Nchunk]
            file_ptr[0] += Nchunk
        else:
            chunk = receiver.get_chunk(timeout=0.1)
            if chunk is None:
                return line1, line2, peak1, peak2, txt1, txt2

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

        tau1 = int(np.argmax(mag1)); v1 = float(mag1[tau1])
        tau2 = int(np.argmax(mag2)); v2 = float(mag2[tau2])

        # Cập nhật đồ thị tương quan
        line1.set_ydata(mag1)
        line2.set_ydata(mag2)
        peak1.set_data([tau1], [v1])
        peak2.set_data([tau2], [v2])
        txt1.set_text(f"Peak @ {tau1}  |  Mag={v1:.1f}")
        txt2.set_text(f"Peak @ {tau2}  |  Mag={v2:.1f}")

        top = max(v1, v2, 10) * 1.5
        if top > ax_corr1.get_ylim()[1] * 0.8 or top < ax_corr1.get_ylim()[1] * 0.1:
            ax_corr1.set_ylim(0, top)
            ax_corr2.set_ylim(0, top)

        # TDOA + vị trí
        T0      = state["T0"]
        D       = state["D"]
        Delta_T = (tau1 - tau2) - T0
        Delta_m = Delta_T / fs * speedOfLight          # TDOA hiệu chỉnh (m)
        Delta_C = (tau1 - tau2) / fs * speedOfLight    # TDOA thô (m)
        X       = (-Delta_C + D + Delta_m) / 2         # T1_est (m)
        X2      = D - X                                 # T2_est (m)
        # Tính x, y (trilateration)
        xy = trilaterate_2d(state["tx1_xy"], state["tx2_xy"], X, X2, hint_xy=state["rx_hint"])
        rx_x = xy[0] if xy is not None else float("nan")
        rx_y = xy[1] if xy is not None else float("nan")

        state.update({"tau1": tau1, "tau2": tau2, "v1": v1, "v2": v2,
                      "Delta_m": Delta_m, "T1_est": X, "T2_est": X2,
                      "x": rx_x, "y": rx_y})

        print(f"[CORR] Peak1={tau1:5d} ({v1:6.1f})  Peak2={tau2:5d} ({v2:6.1f})"
              f"  TDOA={Delta_m:+.2f} m  T1_est={X:.3f} m  T2_est={X2:.3f} m"
              f"  |  x={rx_x:+.3f} m  y={rx_y:+.3f} m")

        # Cập nhật bảng thông tin
        info_str = (
            f"── Toạ độ hiện tại ──────────────\n"
            f"  TX1 : {coords['TX1_lat']:.7f}\n"
            f"        {coords['TX1_lon']:.7f}\n"
            f"  TX2 : {coords['TX2_lat']:.7f}\n"
            f"        {coords['TX2_lon']:.7f}\n"
            f"  RX  : {coords['RX_lat']:.7f}\n"
            f"        {coords['RX_lon']:.7f}\n"
            f"\n"
            f"── Hình học ─────────────────────\n"
            f"  T1 thực : {state['T1']:>10.3f} m\n"
            f"  T2 thực : {state['T2']:>10.3f} m\n"
            f"  D       : {D:>10.3f} m\n"
            f"  T0      : {T0:>10.4f} samples\n"
            f"\n"
            f"── Kết quả realtime ─────────────\n"
            f"  tau1    : {tau1:>10d} samples\n"
            f"  tau2    : {tau2:>10d} samples\n"
            f"  TDOA    : {Delta_m:>+10.3f} m\n"
            f"\n"
            f"  T1_est  : {X:>10.3f} m\n"
            f"  T2_est  : {X2:>10.3f} m\n"
            f"\n"
            f"  x       : {rx_x:>+10.3f} m\n"
            f"  y       : {rx_y:>+10.3f} m\n"
            f"\n"
            f"  ΔT1     : {X  - state['T1']:>+10.3f} m\n"
            f"  ΔT2     : {X2 - state['T2']:>+10.3f} m\n"
        )
        info_text.set_text(info_str)

        return line1, line2, peak1, peak2, txt1, txt2

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
        print("[INFO] Thoát.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
