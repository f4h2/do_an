"""
web_newton_server.py
====================
Flask server cho 2 bước:
  1) Hiệu chuẩn đồng bộ 3 TX: khóa Delta_m cho 2 cặp (TX1–TX2, TX1–TX3)
     trong giả thiết TX2/TX3 cùng 1 vị trí (co-located) khi hiệu chuẩn.
  2) Định vị RX bằng Newton với pseudorange đã bù Delta_m (2D: x, y, clock bias).

Chỉ hiển thị đồ thị tương quan (Chart.js) trên web.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np
from flask import Flask, jsonify, render_template, request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gnss_utils import calcDistance, generateCAcode  # noqa: E402

try:
    from bladerf import _bladerf  # type: ignore

    HAS_BLADERF = True
except Exception:
    HAS_BLADERF = False

SPEED_OF_LIGHT = 299792458.0


def parse_prn_ranges(specs: list[str]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for spec in specs:
        if ":" in spec:
            a, b = spec.split(":", 1)
            out.append((int(a), int(b)))
        else:
            prn = int(spec)
            out.append((prn, prn))
    return out


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


def newton_position(tx_xy: np.ndarray, obs: np.ndarray, pos0: np.ndarray | None = None, num_iterations: int = 10) -> np.ndarray:
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
    def __init__(self, freq_hz: float, fs: float, bw: float, rx_gain: int, chunk_samples: int, serial: str = ""):
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

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=5)

    def get_chunk(self, timeout: float = 1.0) -> np.ndarray | None:
        import queue

        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def _run(self) -> None:
        if not HAS_BLADERF:
            return
        try:
            dev_id = f"*:serial={self._serial}" if self._serial else ""
            b = _bladerf.BladeRF(dev_id) if dev_id else _bladerf.BladeRF()
        except Exception:
            return

        ch = _bladerf.CHANNEL_RX(0)
        b.set_sample_rate(ch, self._fs)
        b.set_frequency(ch, self._freq)
        b.set_bandwidth(ch, self._bw)
        b.set_gain(ch, self._gain)

        b.sync_config(
            layout=_bladerf.ChannelLayout.RX_X1,
            fmt=_bladerf.Format.SC16_Q11,
            num_buffers=16,
            buffer_size=8192,
            num_transfers=8,
            stream_timeout=3500,
        )
        b.enable_module(ch, True)

        buf = np.zeros(self._chunk * 2, dtype=np.int16)
        try:
            while not self._stop_event.is_set():
                b.sync_rx(buf, self._chunk)
                chunk = sc16q11_to_complex64(buf)
                if not self._q.full():
                    self._q.put(chunk.copy())
        finally:
            b.enable_module(ch, False)
            b.close()


def build_default_coords() -> dict[str, float]:
    return {
        "TX1_lat": 20.9896100,
        "TX1_lon": 105.7110745,
        "TX2_lat": 20.9924397,
        "TX2_lon": 105.7106347,
        "TX3_lat": 20.9905000,
        "TX3_lon": 105.7125000,
        "RX_lat": 20.9911114,
        "RX_lon": 105.7107914,
    }


@dataclass
class PairCalib:
    tx_a: int                          # Chỉ số TX mốc trong cặp (thường 0 = TX1)
    tx_b: int                          # Chỉ số TX còn lại (1=TX2 hoặc 2=TX3)
    dist_rx_tx_a: float = 0.0          # Khoảng cách thật RX→TXa (m), nhập lúc hiệu chuẩn
    dist_rx_tx_b: float = 0.0          # Khoảng cách thật RX→TXb (m)
    dist_tx_tx: float = 0.0            # Khoảng cách giữa 2 TX (m), tham chiếu hình học
    t0_samples: float | None = None    # T0 = (d_a - d_b)/c * fs — chênh lệch mẫu do hình học
    tau_diff: int | None = None        # τ_a - τ_b đo từ tương quan (samples)
    delta_t_fixed: float | None = None # (τ_a - τ_b) - T0 — bias thời gian còn lại (samples)
    delta_m: float | None = None       # delta_t_fixed đổi ra mét: bias đồng bộ giữa TXa và TXb


class AppState:
    def __init__(self) -> None:
        self.lock = threading.Lock()  # Khoá chia sẻ state giữa thread xử lý RF và Flask
        self.mode = "calibration"  # "calibration" | "positioning" — bước hiện tại của web UI

        # --- Tham số tín hiệu / xử lý ---
        self.fs = 2e6              # Sample rate (Hz)
        self.rc = 1.023e6          # Chip rate C/A (Hz)
        self.ft = 0.0              # Offset Doppler baseband khi tương quan (Hz)
        self.nchunk = 10_000       # Số mẫu IQ mỗi khung xử lý
        self.nfft = 20_000         # Độ dài FFT (thường = 2 * nchunk, có zero-pad)
        self.newton_iters = 10     # Số vòng lặp Newton
        self.num_tr = 3            # Số trạm phát (TX)

        # --- Mã tham chiếu ---
        self.prn_ranges: list[tuple[int, int]] = [(1, 1), (2, 2), (3, 3)]  # (prn_start, prn_end) từng TX
        self.f_locals: list[np.ndarray] = []  # FFT mã cục bộ đã resample, 1 mảng / TX

        # --- Hình học ---
        self.coords = build_default_coords()  # Lat/lon TX1..TX3 và RX (dict)
        self.tx_xy = np.zeros((self.num_tr, 2), dtype=float)  # Toạ độ TX local (m), gốc TX1
        self.rx_hint = (0.0, 0.0)  # Toạ độ RX thật local (m) — khởi tạo Newton + tính sai số
        self.true_dist = np.zeros(self.num_tr, dtype=float)  # Khoảng cách Haversine RX→từng TX (m)

        # --- Kết quả tương quan realtime ---
        self.corr_profiles: list[np.ndarray] = [np.zeros(self.nfft, dtype=float) for _ in range(self.num_tr)]  # Đường |corr| từng TX
        self.taus = np.zeros(self.num_tr, dtype=int)   # Chỉ số đỉnh tương quan τ (samples) từng TX
        self.peaks = np.zeros(self.num_tr, dtype=float)  # Biên độ đỉnh tương quan từng TX

        # --- Hiệu chuẩn đồng bộ ---
        # 2 cặp: TX1 so với TX2 / TX3
        self.pairs = [PairCalib(0, 1), PairCalib(0, 2)]
        self.delta_m_locked = False  # True sau khi POST /api/calibration/lock thành công
        # Bias (m) cộng vào pseudorange: TX1=0, TX2/TX3 = Δ_m(TX1−TXi)
        self.delta_m_by_tx = np.zeros(self.num_tr, dtype=float)

        # Kết quả Newton gần nhất: pos (x,y,b), pseudorange, err_norm, ...
        self.newton_out: dict | None = None

    def recalc_geometry(self) -> None:
        tx_latlons = [
            (self.coords["TX1_lat"], self.coords["TX1_lon"]),
            (self.coords["TX2_lat"], self.coords["TX2_lon"]),
            (self.coords["TX3_lat"], self.coords["TX3_lon"]),
        ]
        rx_lat, rx_lon = self.coords["RX_lat"], self.coords["RX_lon"]
        ref_lat, ref_lon = tx_latlons[0]
        tx_xy = []
        true_dist = []
        for lat, lon in tx_latlons:
            tx_xy.append(latlon_to_xy(ref_lat, ref_lon, lat, lon))
            true_dist.append(calcDistance(lat, lon, rx_lat, rx_lon))
        self.tx_xy = np.array(tx_xy, dtype=float)
        self.rx_hint = latlon_to_xy(ref_lat, ref_lon, rx_lat, rx_lon)
        self.true_dist = np.array(true_dist, dtype=float)


def downsample_profile(y: np.ndarray, max_points: int = 1200) -> tuple[list[int], list[float]]:
    n = int(y.size)
    if n <= max_points:
        return list(range(n)), y.astype(float).tolist()
    step = max(1, n // max_points)
    idx = np.arange(0, n, step, dtype=int)
    return idx.tolist(), y[idx].astype(float).tolist()


def compute_t0_samples(pair: PairCalib, fs: float) -> float:
    # T0 (samples) = (T_a - T_b)/c * fs, với T_a/T_b là khoảng cách RX→TXa/TXb (m)
    return (pair.dist_rx_tx_a - pair.dist_rx_tx_b) / SPEED_OF_LIGHT * fs


def build_app(state: AppState, args: argparse.Namespace) -> Flask:
    app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "templates"))

    @app.get("/")
    def calibration_page():
        return render_template("calibration.html")

    @app.get("/positioning")
    def positioning_page():
        return render_template("positioning.html")

    @app.post("/api/calibration/distances")
    def set_distances():
        body = request.get_json(force=True, silent=True) or {}
        pairs = body.get("pairs", [])
        with state.lock:
            for p_in in pairs:
                a = int(p_in.get("tx_a", -1))
                b = int(p_in.get("tx_b", -1))
                for p in state.pairs:
                    if p.tx_a == a and p.tx_b == b:
                        p.dist_rx_tx_a = float(p_in.get("dist_rx_tx_a", p.dist_rx_tx_a))
                        p.dist_rx_tx_b = float(p_in.get("dist_rx_tx_b", p.dist_rx_tx_b))
                        p.dist_tx_tx = float(p_in.get("dist_tx_tx", p.dist_tx_tx))
                        p.t0_samples = compute_t0_samples(p, state.fs)
        return jsonify({"ok": True})

    @app.post("/api/calibration/lock")
    def lock_delta_m():
        with state.lock:
            if state.taus.size != state.num_tr:
                return jsonify({"locked": False})
            for p in state.pairs:
                p.t0_samples = compute_t0_samples(p, state.fs)
                p.tau_diff = int(state.taus[p.tx_a] - state.taus[p.tx_b])
                # ΔT_fixed (samples) = (tau_a - tau_b) - T0
                p.delta_t_fixed = float(p.tau_diff) - float(p.t0_samples)
                p.delta_m = p.delta_t_fixed / state.fs * SPEED_OF_LIGHT

            # Bù vào pseudorange theo mốc TX1: P_b_corr = P_b + Delta_m(TX1-TXb)
            state.delta_m_by_tx[:] = 0.0
            for p in state.pairs:
                if p.delta_m is not None:
                    state.delta_m_by_tx[p.tx_b] = float(p.delta_m)
            state.delta_m_locked = True
            return jsonify({"locked": True, "delta_m_by_tx": state.delta_m_by_tx.tolist()})

    @app.post("/api/next")
    def next_step():
        with state.lock:
            if not state.delta_m_locked:
                return jsonify({"ok": False, "error": "delta_m_not_locked"})
            state.mode = "positioning"
        return jsonify({"ok": True})

    @app.post("/api/geometry")
    def set_geometry():
        body = request.get_json(force=True, silent=True) or {}
        coords = body.get("coords", {})
        with state.lock:
            for k, v in coords.items():
                if k in state.coords:
                    state.coords[k] = float(v)
            state.recalc_geometry()
        return jsonify({"ok": True})

    @app.get("/api/status")
    def status():
        with state.lock:
            corr = []
            for i in range(state.num_tr):
                x, y = downsample_profile(state.corr_profiles[i])
                ps, pe = state.prn_ranges[i]
                label = f"TX{i+1} PRN {ps}" + (f"–{pe}" if pe != ps else "")
                corr.append(
                    {
                        "label": label,
                        "x": x,
                        "y": y,
                        "tau": int(state.taus[i]),
                        "peak": float(state.peaks[i]),
                    }
                )

            pairs_out = [
                {
                    "tx_a": p.tx_a,
                    "tx_b": p.tx_b,
                    "tau_diff": p.tau_diff,
                    "t0_samples": p.t0_samples,
                    "delta_t_fixed": p.delta_t_fixed,
                    "delta_m": p.delta_m,
                }
                for p in state.pairs
            ]

            return jsonify(
                {
                    "mode": state.mode,
                    "delta_m_locked": state.delta_m_locked,
                    "pairs": pairs_out,
                    "corr": corr,
                    "coords": {k: float(v) for k, v in state.coords.items()},
                    "tx_xy": state.tx_xy.tolist(),
                    "rx_hint": [float(state.rx_hint[0]), float(state.rx_hint[1])],
                    "newton": state.newton_out,
                }
            )

    return app


def processing_loop(state: AppState, args: argparse.Namespace) -> None:
    fs = state.fs
    nfft = state.nfft
    nchunk = state.nchunk
    iq_padded = np.zeros(nfft, dtype=np.complex64)

    use_file = bool(args.file)
    receiver: BladeRFReceiver | None = None
    file_iq: np.ndarray | None = None
    file_ptr = 0

    if use_file:
        if not os.path.exists(args.file):
            raise FileNotFoundError(args.file)
        if args.input_fmt == "complex64":
            file_iq = np.fromfile(args.file, dtype=np.complex64)
        else:
            raw = np.fromfile(args.file, dtype=np.int16)
            file_iq = int16_iq_file_to_complex64(raw)
    else:
        if not HAS_BLADERF:
            raise RuntimeError("bladerf module not available (use --file).")
        receiver = BladeRFReceiver(
            freq_hz=args.freq_hz,
            fs=fs,
            bw=fs * 0.8,
            rx_gain=args.rx_gain,
            chunk_samples=nchunk,
            serial=args.serial or "",
        )
        receiver.start()
        time.sleep(0.5)

    buffer_iq = np.array([], dtype=np.complex64)

    try:
        while True:
            if use_file:
                assert file_iq is not None
                if file_ptr + nchunk > len(file_iq):
                    file_ptr = 0
                chunk = file_iq[file_ptr : file_ptr + nchunk]
                file_ptr += nchunk
            else:
                assert receiver is not None
                chunk = receiver.get_chunk(timeout=0.2)
                if chunk is None:
                    continue

            buffer_iq = np.concatenate((buffer_iq, chunk))
            if buffer_iq.size < nchunk:
                continue
            iq = buffer_iq[:nchunk]
            buffer_iq = buffer_iq[nchunk:]

            corr_profiles: list[np.ndarray] = []
            taus = np.zeros(state.num_tr, dtype=int)
            peaks = np.zeros(state.num_tr, dtype=float)

            with state.lock:
                f_locals = state.f_locals
                ft = state.ft
                tx_xy = state.tx_xy.copy()
                rx_hint = np.array([state.rx_hint[0], state.rx_hint[1], 0.0], dtype=float)
                delta_m_by_tx = state.delta_m_by_tx.copy()
                do_newton = state.delta_m_locked and state.mode == "positioning"

            for i in range(state.num_tr):
                profile, tau, peak = correlate_fft(iq, f_locals[i], ft, fs, nfft, iq_padded)
                corr_profiles.append(profile.astype(float))
                taus[i] = tau
                peaks[i] = peak

            newton_out = None
            if do_newton:
                pseudorange = taus.astype(float) / fs * SPEED_OF_LIGHT
                # bù Delta_m để đưa TX2/TX3 về cùng mốc TX1
                pseudorange = pseudorange + delta_m_by_tx
                pos = newton_position(tx_xy, pseudorange, pos0=rx_hint, num_iterations=state.newton_iters)
                err_x = pos[0] - rx_hint[0]
                err_y = pos[1] - rx_hint[1]
                err_norm = float(np.hypot(err_x, err_y))
                newton_out = {
                    "taus": taus.tolist(),
                    "pseudorange": pseudorange.tolist(),
                    "pos": pos.tolist(),
                    "true_dist": state.true_dist.tolist(),
                    "err_norm": err_norm,
                }

            with state.lock:
                state.corr_profiles = corr_profiles
                state.taus = taus
                state.peaks = peaks
                state.newton_out = newton_out

            if buffer_iq.size > nchunk * 50:
                buffer_iq = np.array([], dtype=np.complex64)
    finally:
        if receiver is not None:
            receiver.stop()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Web UI: calibration Delta_m + Newton positioning (3 TX).")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--freq_hz", type=float, default=1575.42e6)
    p.add_argument("--fs", type=float, default=2e6)
    p.add_argument("--Rc", type=float, default=1.023e6)
    p.add_argument("--ft", type=float, default=0.0)
    p.add_argument("--rx_gain", type=int, default=60)
    p.add_argument("--chunk_ms", type=float, default=5.0)
    p.add_argument("--serial", default="")
    p.add_argument("--file", default="")
    p.add_argument("--input_fmt", choices=["sc16", "complex64"], default="sc16")
    p.add_argument("--tx_prns", default="1,2,3")
    p.add_argument("--tx_prn_ranges", default="")
    p.add_argument("--newton_iters", type=int, default=10)
    args = p.parse_args(argv)

    state = AppState()
    state.fs = float(args.fs)
    state.rc = float(args.Rc)
    state.ft = float(args.ft)
    state.nchunk = int(state.fs * float(args.chunk_ms) / 1000.0)
    state.nfft = state.nchunk * 2
    state.newton_iters = int(args.newton_iters)
    state.recalc_geometry()

    if args.tx_prn_ranges:
        prn_ranges = parse_prn_ranges(args.tx_prn_ranges.split(","))
    else:
        prns = [int(x) for x in args.tx_prns.split(",")]
        if len(prns) != state.num_tr:
            raise SystemExit(f"Cần đúng {state.num_tr} PRN, nhận {len(prns)}.")
        prn_ranges = [(prn, prn) for prn in prns]
    if len(prn_ranges) != state.num_tr:
        raise SystemExit(f"Cần đúng {state.num_tr} dải PRN.")
    state.prn_ranges = prn_ranges
    state.f_locals = [make_local_fft_ref(ps, pe, state.fs, state.rc, state.nfft) for (ps, pe) in prn_ranges]
    state.corr_profiles = [np.zeros(state.nfft, dtype=float) for _ in range(state.num_tr)]

    t = threading.Thread(target=processing_loop, args=(state, args), daemon=True)
    t.start()

    app = build_app(state, args)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# python web_newton_server.py --serial a0e5ffb5f1c28a2d57f5f5d9d13372ed --freq_hz 433e6 --tx_prn_ranges 11:15,16:20,26:30
