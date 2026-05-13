import numpy as np
import zmq
import time
import matplotlib.pyplot as plt
import os
from gnss_utils import generateCAcode, calcDistance


def _parse_float(s: str, default: float) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


def _parse_int(s: str, default: int) -> int:
    try:
        return int(s)
    except (TypeError, ValueError):
        return default


def _build_freq_grid(f_min: float, f_max: float, step: float) -> np.ndarray:
    if step <= 0:
        step = 500.0
    n = int(np.floor((f_max - f_min) / step + 1e-9)) + 1
    return f_min + step * np.arange(n, dtype=np.float64)


def _correlate_fft(
    IQ: np.ndarray,
    f_hz: float,
    n_arr: np.ndarray,
    fs: float,
    IQ_padded: np.ndarray,
    f_local1: np.ndarray,
    f_local2: np.ndarray,
    Nmax: int,
):
    IQ_shifted = IQ * np.exp(1j * 2 * np.pi * f_hz * n_arr / fs)
    IQ_padded.fill(0)
    IQ_padded[:Nmax] = IQ_shifted.astype(np.complex64, copy=False)
    f_IQ = np.fft.fft(IQ_padded)
    mag1 = np.abs(np.fft.ifft(f_local1 * np.conj(f_IQ)))
    mag2 = np.abs(np.fft.ifft(f_local2 * np.conj(f_IQ)))
    taue1 = int(np.argmax(mag1))
    taue2 = int(np.argmax(mag2))
    max_val1 = float(mag1[taue1])
    max_val2 = float(mag2[taue2])
    return mag1, mag2, taue1, taue2, max_val1, max_val2


def _freq_search(
    IQ: np.ndarray,
    f_candidates: np.ndarray,
    n_arr: np.ndarray,
    fs: float,
    IQ_padded: np.ndarray,
    f_local1: np.ndarray,
    f_local2: np.ndarray,
    Nmax: int,
):
    best_f = float(f_candidates[0])
    best_score = -1.0
    best = None
    for f_c in f_candidates:
        mag1, mag2, taue1, taue2, m1, m2 = _correlate_fft(
            IQ, float(f_c), n_arr, fs, IQ_padded, f_local1, f_local2, Nmax
        )
        score = m1 + m2
        if score > best_score:
            best_score = score
            best_f = float(f_c)
            best = (mag1, mag2, taue1, taue2, m1, m2)
    return best_f, best


def thu_tin_hieu_rt_gui():
    Rc = 1.023e6
    fs = 10e6
    ft = _parse_float(os.environ.get("FT_NOMINAL", "10"), 10.0)
    speedOfLight = 299792458

    FREQ_SEARCH = os.environ.get("FREQ_SEARCH", "1").strip() not in ("0", "false", "False", "no", "NO")
    f_min = _parse_float(os.environ.get("F_SEARCH_MIN", "-5000"), -5000.0)
    f_max = _parse_float(os.environ.get("F_SEARCH_MAX", "5000"), 5000.0)
    f_step = _parse_float(os.environ.get("F_SEARCH_STEP", "500"), 500.0)
    f_candidates = _build_freq_grid(f_min, f_max, f_step)
    f_candidates = np.unique(np.concatenate([f_candidates, np.array([ft], dtype=np.float64)]))
    f_candidates.sort()
    FREQ_SEARCH_PERIOD = max(1, _parse_int(os.environ.get("FREQ_SEARCH_PERIOD", "5"), 5))

    INPUT_MODE = os.environ.get("INPUT_MODE", "ZMQ").upper()
    FILE_PATH = "data_tx1_prn_11_20_10MHz.bin"

    Nmax = 10000

    print(f"--- CHẾ ĐỘ ĐẦU VÀO: {INPUT_MODE} ---")
    if FREQ_SEARCH:
        print(
            f"Quét tần số: [{f_min:.0f}, {f_max:.0f}] Hz, bước {f_step:.0f} Hz "
            f"({len(f_candidates)} ứng viên), lặp lại mỗi {FREQ_SEARCH_PERIOD} khung"
        )
    else:
        print(f"Không quét tần số; dùng FT_NOMINAL = {ft} Hz")

    cacodes1 = np.concatenate([generateCAcode(i) for i in range(11, 21)])
    cacodes2 = np.concatenate([generateCAcode(i) for i in range(21, 31)])

    n_local = np.arange(100000)
    idx_local = np.floor(n_local / fs * Rc).astype(int) % 10230
    local1_fs = cacodes1[idx_local].astype(np.complex64)
    local2_fs = cacodes2[idx_local].astype(np.complex64)

    f_local1 = np.fft.fft(local1_fs)
    f_local2 = np.fft.fft(local2_fs)

    n_arr = np.arange(Nmax)

    TX1 = [20.9896100, 105.7110745]
    TX2 = [20.9924397, 105.7106347]
    RX = [20.9911114, 105.7107914]

    T1 = calcDistance(TX1[0], TX1[1], RX[0], RX[1])
    T2 = calcDistance(TX2[0], TX2[1], RX[0], RX[1])
    T0 = (T1 - T2) / speedOfLight * fs
    D = calcDistance(TX1[0], TX1[1], TX2[0], TX2[1])

    socket = None
    file_iq_data = None
    file_ptr = 0

    if INPUT_MODE == "ZMQ":
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.CONFLATE, 1)
        socket.connect("tcp://127.0.0.1:5555")
        socket.setsockopt_string(zmq.SUBSCRIBE, "")
        print("Đã kết nối ZMQ: tcp://127.0.0.1:5555")
    else:
        if not os.path.exists(FILE_PATH):
            print(f"LỖI: Không tìm thấy file {FILE_PATH}")
            return
        raw_data = np.fromfile(FILE_PATH, dtype=np.int16)
        I = raw_data[0::2].astype(np.float32)
        Q = raw_data[1::2].astype(np.float32)
        file_iq_data = (I + 1j * Q).astype(np.complex64)
        print(f"Đã nạp file: {FILE_PATH} ({len(file_iq_data)} mẫu)")

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    line1, = ax1.plot(np.zeros(100000), color="#1f77b4", linewidth=0.5, alpha=0.7, label="Signal TX1")
    line2, = ax2.plot(np.zeros(100000), color="#ff7f0e", linewidth=0.5, alpha=0.7, label="Signal TX2")

    peak_dot1, = ax1.plot([0], [0], "ro", markersize=6, label="Peak TX1")
    peak_dot2, = ax2.plot([0], [0], "go", markersize=6, label="Peak TX2")

    ax1.set_title("Real-time Correlation Map - TX1", fontsize=12)
    ax2.set_title("Real-time Correlation Map - TX2", fontsize=12)

    for ax in [ax1, ax2]:
        ax.set_ylim(0, 100)
        ax.set_xlabel("Delay Samples")
        ax.set_ylabel("Magnitude")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right")

    plt.tight_layout()
    buffer_iq = np.array([], dtype=np.complex64)
    IQ_padded = np.zeros(100000, dtype=np.complex64)
    tracked_f = float(ft)
    frame_idx = 0

    try:
        while True:
            if INPUT_MODE == "ZMQ":
                raw_msg = socket.recv()
                chunk_iq = np.frombuffer(raw_msg, dtype=np.complex64)
            else:
                if file_ptr + Nmax > len(file_iq_data):
                    file_ptr = 0
                chunk_iq = file_iq_data[file_ptr : file_ptr + Nmax]
                file_ptr += Nmax
                time.sleep(0.001)

            buffer_iq = np.concatenate((buffer_iq, chunk_iq))

            if len(buffer_iq) >= Nmax:
                IQ = buffer_iq[:Nmax]
                buffer_iq = buffer_iq[Nmax:]

                start_time = time.time()

                if FREQ_SEARCH and (frame_idx % FREQ_SEARCH_PERIOD == 0):
                    tracked_f, pack = _freq_search(
                        IQ, f_candidates, n_arr, fs, IQ_padded, f_local1, f_local2, Nmax
                    )
                    mag1, mag2, taue1, taue2, max_val1, max_val2 = pack
                elif FREQ_SEARCH:
                    mag1, mag2, taue1, taue2, max_val1, max_val2 = _correlate_fft(
                        IQ, tracked_f, n_arr, fs, IQ_padded, f_local1, f_local2, Nmax
                    )
                else:
                    mag1, mag2, taue1, taue2, max_val1, max_val2 = _correlate_fft(
                        IQ, ft, n_arr, fs, IQ_padded, f_local1, f_local2, Nmax
                    )
                frame_idx += 1

                line1.set_ydata(mag1)
                line2.set_ydata(mag2)
                peak_dot1.set_data([taue1], [max_val1])
                peak_dot2.set_data([taue2], [max_val2])

                current_top = max(max_val1, max_val2, 10)
                if current_top > ax1.get_ylim()[1] * 0.8 or current_top < ax1.get_ylim()[1] * 0.2:
                    new_limit = current_top * 1.5
                    ax1.set_ylim(0, new_limit)
                    ax2.set_ylim(0, new_limit)

                plt.pause(0.001)

                Delta_T = (taue1 - taue2) - T0
                Delta_T_m = Delta_T / fs * speedOfLight

                process_time = time.time() - start_time
                f_used = tracked_f if FREQ_SEARCH else ft
                print(
                    f"Peaks: {taue1:5d}, {taue2:5d} | Mags: {max_val1:6.1f}, {max_val2:6.1f} "
                    f"| f_used: {f_used:7.1f} Hz | Diff: {Delta_T_m:8.2f}m | t_proc: {process_time:.3f}s"
                )

                if len(buffer_iq) > Nmax * 100:
                    buffer_iq = np.array([], dtype=np.complex64)

    except KeyboardInterrupt:
        print("\n>>> Đã dừng.")
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    thu_tin_hieu_rt_gui()

