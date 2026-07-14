"""
thu_tin_hieu_bladerf_newton.py
==============================
Thu tín hiệu từ phat_tin_hieu_bladerf.py (4 trạm phát, 1 trạm thu),
tương quan từng mã C/A, tính pseudorange và giải vị trí bằng Newton
(tương đương receiver.m, mở rộng NumTr=4).

Sơ đồ:
    TX1..TX4 (mỗi máy chạy phat_tin_hieu_bladerf.py, PRN khác nhau)
         ↓ RF
    RX (script này) → tương quan → pseudorange → Newton → (x, y)

Chạy:
    python thu_tin_hieu_bladerf_newton.py --freq_hz 433e6 --serial <RX_SERIAL>

    python thu_tin_hieu_bladerf_newton.py --file capture.bin --input_fmt sc16

    python thu_tin_hieu_bladerf_newton.py --tx_prns 1,2,3,4

    python thu_tin_hieu_bladerf_newton.py \\
        --tx_prn_ranges 11:20,21:30,31:32,1:10
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import numpy as np
from bladerf import _bladerf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gnss_utils import calcDistance, generateCAcode

try:

    HAS_BLADERF = True
except ImportError:
    HAS_BLADERF = False
    print("[WARN] Không tìm thấy module bladerf — chỉ chạy được với --file.")


SPEED_OF_LIGHT = 299792458.0


def parse_prn_ranges(specs: list[str]) -> list[tuple[int, int]]:
    out = []
    for spec in specs:
        if ":" in spec:
            a, b = spec.split(":", 1)
            out.append((int(a), int(b)))
        else:
            prn = int(spec)
            out.append((prn, prn))
    return out


def make_prn_code(prn_start: int, prn_end: int) -> np.ndarray:
    codes = np.concatenate([generateCAcode(i) for i in range(prn_start, prn_end + 1)])
    return np.concatenate([codes, codes])


def make_local_fft_ref(prn_start: int, prn_end: int, fs: float, rc: float, nfft: int) -> np.ndarray:
    total_chips = (prn_end - prn_start + 1) * 1023
    cacodes = np.concatenate([generateCAcode(i) for i in range(prn_start, prn_end + 1)])
    n_local = np.arange(nfft)
    idx = np.floor(n_local / fs * rc).astype(int) % total_chips
    return np.fft.fft(cacodes[idx].astype(np.complex64))


def sc16q11_to_complex64(buf: np.ndarray) -> np.ndarray:
    i = buf[0::2].astype(np.float32)
    q = buf[1::2].astype(np.float32)
    return (i + 1j * q).astype(np.complex64)


def int16_iq_file_to_complex64(raw: np.ndarray) -> np.ndarray:
    i = raw[0::2].astype(np.float32)
    q = raw[1::2].astype(np.float32)
    return (i + 1j * q).astype(np.complex64)


def latlon_to_xy(ref_lat: float, ref_lon: float, lat: float, lon: float) -> tuple[float, float]:
    r_earth = 6371000.0
    x = np.radians(lon - ref_lon) * np.cos(np.radians((lat + ref_lat) / 2.0)) * r_earth
    y = np.radians(lat - ref_lat) * r_earth
    return float(x), float(y)


def correlate_fft(
    iq: np.ndarray,
    f_local: np.ndarray,
    ft: float,
    fs: float,
    nfft: int,
    iq_padded: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    n = np.arange(iq.size, dtype=np.float32)
    iq_shifted = iq * np.exp(1j * 2 * np.pi * ft * n / fs)
    iq_padded.fill(0)
    iq_padded[: iq.size] = iq_shifted.astype(np.complex64, copy=False)
    f_iq = np.fft.fft(iq_padded)
    mag = np.abs(np.fft.ifft(f_local * np.conj(f_iq)))
    tau = int(np.argmax(mag))
    return mag, tau, float(mag[tau])


def correlate_time_domain_receiver_m(
    signal_i: np.ndarray,
    prn_code: np.ndarray,
    fs: float,
    rc: float,
    num_processed: int,
) -> tuple[np.ndarray, int]:
    code_phase = rc / fs
    tau_max = int(1023 / rc * fs)
    xcorr = np.zeros(tau_max + 1, dtype=np.float64)
    sig = signal_i[:num_processed]
    for tau in range(tau_max + 1):
        tcode = (np.arange(num_processed) + tau) * code_phase
        tcode2 = np.floor(np.mod(tcode, 1023)).astype(int)
        code = prn_code[tcode2]
        xcorr[tau] = np.sum(code * sig) ** 2
    tau_peak = int(np.argmax(xcorr))
    return xcorr, tau_peak


def delays_to_pseudorange(taus: np.ndarray, fs: float) -> np.ndarray:
    """Pseudorange (m) = độ trễ (s) × c, với độ trễ = tau / fs."""
    return taus.astype(float) / fs * SPEED_OF_LIGHT


def newton_position(
    tx_xy: np.ndarray,
    obs: np.ndarray,
    pos0: np.ndarray | None = None,
    num_iterations: int = 10,
) -> np.ndarray:
    pos = np.zeros(3) if pos0 is None else pos0.astype(float).copy()
    n = tx_xy.shape[0]
    for _ in range(num_iterations):
        f_vec = np.zeros(n, dtype=float)
        j_mat = np.zeros((n, 3), dtype=float)
        for i in range(n):
            dx = tx_xy[i, 0] - pos[0]
            dy = tx_xy[i, 1] - pos[1]
            geom = np.hypot(dx, dy)
            f_vec[i] = obs[i] - geom - pos[2]
            denom = obs[i] if obs[i] != 0 else 1.0
            j_mat[i, 0] = -dx / denom
            j_mat[i, 1] = -dy / denom
            j_mat[i, 2] = 1.0
        delta, _, _, _ = np.linalg.lstsq(j_mat, f_vec, rcond=None)
        pos += delta
    return pos


class BladeRFReceiver:
    def __init__(self, freq_hz, fs, bw, rx_gain, chunk_samples, serial=""):
        import queue

        self._q = queue.Queue(maxsize=100)
        self._stop_event = threading.Event()
        self._freq = freq_hz
        self._fs = fs
        self._bw = bw
        self._gain = rx_gain
        self._chunk = chunk_samples
        self._serial = serial
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
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
            print(f"[RX] Thiết bị: {b.get_board_name()}  serial={self._serial or 'auto'}")
        except Exception as exc:
            print(f"[RX] Lỗi mở BladeRF: {exc}")
            return

        ch = _bladerf.CHANNEL_RX(0)
        b.set_sample_rate(ch, self._fs)
        b.set_frequency(ch, self._freq)
        b.set_bandwidth(ch, self._bw)
        b.set_gain(ch, self._gain)
        print(f"[RX] fc={self._freq/1e6:.3f} MHz  fs={self._fs/1e6:.2f} MHz  gain={self._gain} dB")

        b.sync_config(
            layout=_bladerf.ChannelLayout.RX_X1,
            fmt=_bladerf.Format.SC16_Q11,
            num_buffers=16,
            buffer_size=8192,
            num_transfers=8,
            stream_timeout=3500,
        )
        b.enable_module(ch, True)
        print(f"[RX] Thu {self._chunk} mẫu/chunk...")

        buf = np.zeros(self._chunk * 2, dtype=np.int16)
        try:
            while not self._stop_event.is_set():
                b.sync_rx(buf, self._chunk)
                chunk = sc16q11_to_complex64(buf)
                if not self._q.full():
                    self._q.put(chunk.copy())
        except Exception as exc:
            if not self._stop_event.is_set():
                print(f"[RX] Lỗi: {exc}")
        finally:
            b.enable_module(ch, False)
            b.close()
            print("[RX] Đã đóng BladeRF.")


def build_default_coords() -> dict[str, float]:
    return {
        "TX1_lat": 20.9896100,
        "TX1_lon": 105.7110745,
        "TX2_lat": 20.9924397,
        "TX2_lon": 105.7106347,
        "TX3_lat": 20.9905000,
        "TX3_lon": 105.7125000,
        "TX4_lat": 20.9918000,
        "TX4_lon": 105.7098000,
        "RX_lat": 20.9911114,
        "RX_lon": 105.7107914,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Thu 4 TX / 1 RX BladeRF + Newton positioning.")
    p.add_argument("--freq_hz", type=float, default=1575.42e6)
    p.add_argument("--fs", type=float, default=2e6)
    p.add_argument("--Rc", type=float, default=1.023e6)
    p.add_argument("--ft", type=float, default=0.0, help="Bù Doppler baseban (Hz).")
    p.add_argument("--rx_gain", type=int, default=60)
    p.add_argument("--chunk_ms", type=float, default=5.0)
    p.add_argument("--serial", default="")
    p.add_argument("--file", default="", help="Đọc IQ từ file thay vì BladeRF.")
    p.add_argument("--input_fmt", choices=["sc16", "complex64"], default="sc16")
    p.add_argument(
        "--corr_mode",
        choices=["fft", "time"],
        default="fft",
        help="fft: IQ phức (phat_tin_hieu_bladerf). time: I-only như receiver.m.",
    )
    p.add_argument(
        "--tx_prns",
        default="1,2,3,4",
        help="PRN đơn cho 4 TX, ví dụ 1,2,3,4.",
    )
    p.add_argument(
        "--tx_prn_ranges",
        default="",
        help="Ghi đè --tx_prns, ví dụ 11:20,21:30,31:32,1:10.",
    )
    p.add_argument("--newton_iters", type=int, default=10)
    p.add_argument("--num_processed", type=int, default=0, help="Số mẫu cho corr time-domain (0=auto).")
    args = p.parse_args(argv)

    fs = args.fs
    rc = args.Rc
    nchunk = int(fs * args.chunk_ms / 1000)
    nfft = nchunk * 2
    bw = fs * 0.8
    num_tr = 4

    if args.tx_prn_ranges:
        prn_ranges = parse_prn_ranges(args.tx_prn_ranges.split(","))
    else:
        prns = [int(x) for x in args.tx_prns.split(",")]
        if len(prns) != num_tr:
            print(f"[ERROR] Cần đúng {num_tr} PRN, nhận {len(prns)}.")
            return 1
        prn_ranges = [(prn, prn) for prn in prns]

    if len(prn_ranges) != num_tr:
        print(f"[ERROR] Cần đúng {num_tr} dải PRN.")
        return 1

    print("[INFO] Đang tạo mã tham chiếu...")
    f_locals = []
    prn_codes = []
    for i, (ps, pe) in enumerate(prn_ranges, start=1):
        f_locals.append(make_local_fft_ref(ps, pe, fs, rc, nfft))
        prn_codes.append(make_prn_code(ps, pe))
        print(f"  TX{i}: PRN {ps}" + (f"–{pe}" if pe != ps else ""))

    coords = build_default_coords()
    state = {
        "tx_xy": np.zeros((num_tr, 2)),
        "rx_hint": (0.0, 0.0),
        "true_dist": np.zeros(num_tr),
        "taus": np.zeros(num_tr, dtype=int),
        "mags": np.zeros(num_tr),
        "pseudorange": np.zeros(num_tr),
        "pos": np.zeros(3),
    }

    def recalc_geometry():
        tx_latlons = [
            (coords["TX1_lat"], coords["TX1_lon"]),
            (coords["TX2_lat"], coords["TX2_lon"]),
            (coords["TX3_lat"], coords["TX3_lon"]),
            (coords["TX4_lat"], coords["TX4_lon"]),
        ]
        rx_lat, rx_lon = coords["RX_lat"], coords["RX_lon"]
        ref_lat, ref_lon = tx_latlons[0]
        tx_xy = []
        true_dist = []
        for lat, lon in tx_latlons:
            tx_xy.append(latlon_to_xy(ref_lat, ref_lon, lat, lon))
            true_dist.append(calcDistance(lat, lon, rx_lat, rx_lon))
        state["tx_xy"] = np.array(tx_xy, dtype=float)
        state["rx_hint"] = latlon_to_xy(ref_lat, ref_lon, rx_lat, rx_lon)
        state["true_dist"] = np.array(true_dist, dtype=float)
        print("[GEO] Khoảng cách thực từ RX đến các TX (m):")
        for i, d in enumerate(state["true_dist"], start=1):
            print(f"  TX{i}: {d:.3f}")

    recalc_geometry()

    use_file = bool(args.file)
    receiver = None
    file_iq = None
    file_ptr = [0]

    if use_file:
        if not os.path.exists(args.file):
            print(f"[ERROR] Không tìm thấy file: {args.file}")
            return 1
        if args.input_fmt == "complex64":
            file_iq = np.fromfile(args.file, dtype=np.complex64)
        else:
            raw = np.fromfile(args.file, dtype=np.int16)
            file_iq = int16_iq_file_to_complex64(raw)
        print(f"[FILE] Đã nạp {len(file_iq)} mẫu từ {args.file}")
    else:
        if not HAS_BLADERF:
            print("[ERROR] Module bladerf chưa cài. Dùng --file.")
            return 1
        receiver = BladeRFReceiver(
            freq_hz=args.freq_hz,
            fs=fs,
            bw=bw,
            rx_gain=args.rx_gain,
            chunk_samples=nchunk,
            serial=args.serial,
        )
        receiver.start()
        time.sleep(0.5)

    num_processed = args.num_processed or min(nchunk, int(0.005 * fs))

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(
        f"4 TX / 1 RX — Newton positioning  |  fc={args.freq_hz/1e6:.3f} MHz  fs={fs/1e6:.2f} MHz",
        fontsize=12,
        color="white",
    )

    gs = gridspec.GridSpec(
        nrows=4,
        ncols=2,
        left=0.06,
        right=0.98,
        top=0.92,
        bottom=0.28,
        wspace=0.30,
        hspace=0.45,
        height_ratios=[0.7, 1.0, 1.0, 1.0],
    )

    ax_info = fig.add_subplot(gs[0, 0])
    ax_map = fig.add_subplot(gs[1:, 0])
    ax_info.set_facecolor("#111111")
    ax_info.set_xticks([])
    ax_info.set_yticks([])
    ax_info.set_title("Kết quả Newton", color="cyan", fontsize=10)
    info_text = ax_info.text(
        0.04, 0.97, "", transform=ax_info.transAxes,
        color="white", fontsize=8.5, va="top", family="monospace",
    )

    ax_map.set_title("Bản đồ local (m)", color="white")
    ax_map.set_xlabel("x (m)")
    ax_map.set_ylabel("y (m)")
    ax_map.grid(True, ls="--", alpha=0.35)
    tx_scatter = ax_map.scatter([], [], c="cyan", s=80, marker="^", label="TX")
    rx_true_scatter = ax_map.scatter([], [], c="lime", s=80, marker="o", label="RX thực")
    rx_est_scatter = ax_map.scatter([], [], c="red", s=80, marker="x", label="RX Newton")
    ax_map.legend(loc="upper right", fontsize=8)

    corr_axes = []
    corr_lines = []
    corr_peaks = []
    corr_texts = []
    x_axis = np.arange(nfft if args.corr_mode == "fft" else int(1023 / rc * fs) + 1)

    for i in range(num_tr):
        ax = fig.add_subplot(gs[i, 1])
        ps, pe = prn_ranges[i]
        prn_label = f"{ps}" if ps == pe else f"{ps}-{pe}"
        ax.set_title(f"TX{i+1} — PRN {prn_label}", color="white", fontsize=9)
        ax.set_xlabel("Delay (samples)")
        ax.set_ylabel("|corr|")
        ax.grid(True, ls="--", alpha=0.35)
        line, = ax.plot(x_axis, np.zeros_like(x_axis), lw=0.6)
        peak, = ax.plot([0], [0], "ro", ms=5)
        txt = ax.text(0.02, 0.90, "", transform=ax.transAxes, color="white", fontsize=8)
        corr_axes.append(ax)
        corr_lines.append(line)
        corr_peaks.append(peak)
        corr_texts.append(txt)

    field_defs = [
        ("TX1_lat", "TX1 Lat:", 0.02, 0.18, 0.24),
        ("TX1_lon", "TX1 Lon:", 0.02, 0.18, 0.24 - 0.042),
        ("TX2_lat", "TX2 Lat:", 0.52, 0.68, 0.24),
        ("TX2_lon", "TX2 Lon:", 0.52, 0.68, 0.24 - 0.042),
        ("TX3_lat", "TX3 Lat:", 0.02, 0.18, 0.24 - 2 * 0.042),
        ("TX3_lon", "TX3 Lon:", 0.52, 0.68, 0.24 - 2 * 0.042),
        ("TX4_lat", "TX4 Lat:", 0.02, 0.18, 0.24 - 3 * 0.042),
        ("TX4_lon", "TX4 Lon:", 0.52, 0.68, 0.24 - 3 * 0.042),
        ("RX_lat", "RX  Lat:", 0.02, 0.18, 0.24 - 4 * 0.042),
        ("RX_lon", "RX  Lon:", 0.52, 0.68, 0.24 - 4 * 0.042),
    ]
    textboxes = {}
    for key, label, lx, bx, by in field_defs:
        fig.text(lx, by + 0.008, label, color="cyan", fontsize=8, transform=fig.transFigure)
        ax_tb = fig.add_axes([bx, by, 0.22, 0.032])
        tb = TextBox(ax_tb, "", initial=str(coords[key]), color="#1f1f1f", hovercolor="#333333")
        tb.text_disp.set_color("white")
        tb.text_disp.set_fontsize(8)
        textboxes[key] = tb

    ax_btn = fig.add_axes([0.38, 0.24 - 4 * 0.042 - 0.01, 0.20, 0.035])
    btn_update = Button(ax_btn, "Cập nhật toạ độ", color="#1a6b3c", hovercolor="#25a05a")
    btn_update.label.set_color("white")

    def on_update(_event):
        try:
            for key, tb in textboxes.items():
                coords[key] = float(tb.text)
            recalc_geometry()
        except ValueError as exc:
            print(f"[GUI] Lỗi toạ độ: {exc}")

    btn_update.on_clicked(on_update)

    buffer_iq = np.array([], dtype=np.complex64)
    iq_padded = np.zeros(nfft, dtype=np.complex64)

    def update_map():
        tx_xy = state["tx_xy"]
        rx_hint = state["rx_hint"]
        pos = state["pos"]
        tx_scatter.set_offsets(tx_xy)
        rx_true_scatter.set_offsets([rx_hint])
        rx_est_scatter.set_offsets([[pos[0], pos[1]]])
        all_x = np.r_[tx_xy[:, 0], rx_hint[0], pos[0]]
        all_y = np.r_[tx_xy[:, 1], rx_hint[1], pos[1]]
        pad = 20.0
        ax_map.set_xlim(all_x.min() - pad, all_x.max() + pad)
        ax_map.set_ylim(all_y.min() - pad, all_y.max() + pad)

    def update(_frame):
        nonlocal buffer_iq

        if use_file:
            if file_ptr[0] + nchunk > len(file_iq):
                file_ptr[0] = 0
            chunk = file_iq[file_ptr[0] : file_ptr[0] + nchunk]
            file_ptr[0] += nchunk
        else:
            chunk = receiver.get_chunk(timeout=0.1)
            if chunk is None:
                return corr_lines

        buffer_iq = np.concatenate((buffer_iq, chunk))
        if len(buffer_iq) < nchunk:
            return corr_lines

        iq = buffer_iq[:nchunk]
        buffer_iq = buffer_iq[nchunk:]

        taus = np.zeros(num_tr, dtype=int)
        mags = np.zeros(num_tr)
        corr_profiles = []

        for i in range(num_tr):
            if args.corr_mode == "fft":
                profile, tau, peak = correlate_fft(
                    iq, f_locals[i], args.ft, fs, nfft, iq_padded
                )
            else:
                profile, tau = correlate_time_domain_receiver_m(
                    np.real(iq), prn_codes[i], fs, rc, num_processed
                )
                peak = float(profile[tau])
            taus[i] = tau
            mags[i] = peak
            corr_profiles.append(profile)

        pseudorange = delays_to_pseudorange(taus, fs)
        pos = newton_position(
            state["tx_xy"],
            pseudorange,
            pos0=np.array([state["rx_hint"][0], state["rx_hint"][1], 0.0]),
            num_iterations=args.newton_iters,
        )

        state["taus"] = taus
        state["mags"] = mags
        state["pseudorange"] = pseudorange
        state["pos"] = pos

        for i in range(num_tr):
            profile = corr_profiles[i]
            corr_lines[i].set_ydata(profile)
            corr_lines[i].set_xdata(np.arange(profile.size))
            corr_peaks[i].set_data([taus[i]], [mags[i]])
            corr_texts[i].set_text(f"tau={taus[i]}  mag={mags[i]:.1f}")
            top = max(mags[i], 10) * 1.4
            corr_axes[i].set_xlim(0, profile.size)
            corr_axes[i].set_ylim(0, top)

        update_map()

        err_x = pos[0] - state["rx_hint"][0]
        err_y = pos[1] - state["rx_hint"][1]
        err_norm = np.hypot(err_x, err_y)

        info = ["── Pseudorange & Newton ──"]
        for i in range(num_tr):
            info.append(
                f"TX{i+1}: tau={taus[i]:5d}  P={pseudorange[i]:8.2f} m  "
                f"true={state['true_dist'][i]:8.2f} m"
            )
        info.extend(
            [
                "",
                f"x_est = {pos[0]:+.3f} m",
                f"y_est = {pos[1]:+.3f} m",
                f"b_est = {pos[2]:+.3f} m",
                "",
                f"x_true= {state['rx_hint'][0]:+.3f} m",
                f"y_true= {state['rx_hint'][1]:+.3f} m",
                "",
                f"Err   = {err_norm:.3f} m",
            ]
        )
        info_text.set_text("\n".join(info))

        print(
            f"[NEWTON] pos=({pos[0]:+.2f}, {pos[1]:+.2f}) b={pos[2]:+.2f} "
            f"| err={err_norm:.2f} m | taus={taus.tolist()}"
        )

        if len(buffer_iq) > nchunk * 50:
            buffer_iq = np.array([], dtype=np.complex64)

        return corr_lines

    _ani = animation.FuncAnimation(
        fig, update, interval=int(args.chunk_ms), blit=False, cache_frame_data=False
    )

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        if receiver:
            receiver.stop()
        print("[INFO] Thoát.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
