import argparse
import time
import os
import numpy as np
import zmq
import matplotlib.pyplot as plt
from gnss_utils import generateCAcode, calcDistance


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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Thu tín hiệu GNSS giả lập (ZMQ hoặc file) với dò Doppler.")
    p.add_argument("--mode", choices=["FILE", "ZMQ"], default="ZMQ", help="Chế độ đầu vào.")
    p.add_argument("--address", default="tcp://127.0.0.1:5555", help="ZMQ connect address (SUB).")
    p.add_argument("--fs", type=float, default=2e6, help="Sample rate (Hz).")
    p.add_argument("--Rc", type=float, default=1.023e6, help="Chip rate (Hz).")
    p.add_argument("--ft", type=float, default=10.0, help="Doppler nominal (Hz).")
    p.add_argument("--chunk", type=int, default=10000, help="Số mẫu mỗi khung xử lý.")
    p.add_argument("--f-min", type=float, default=-5000.0, help="Tần số quét tối thiểu (Hz).")
    p.add_argument("--f-max", type=float, default=5000.0, help="Tần số quét tối đa (Hz).")
    p.add_argument("--f-step", type=float, default=500.0, help="Bước quét tần số (Hz).")
    p.add_argument("--search-period", type=int, default=5, help="Chu kỳ quét lại Doppler (khung).")
    p.add_argument("--no-search", action="store_true", help="Vô hiệu hóa quét tần số.")
    p.add_argument("--input", default="data_tx1_prn_11_20_10MHz.bin", help="File input (nếu mode=FILE).")
    
    args = p.parse_args(argv)

    speedOfLight = 299792458
    Nmax = args.chunk
    fs = args.fs
    Rc = args.Rc
    ft = args.ft

    use_freq_search = not args.no_search
    f_candidates = _build_freq_grid(args.f_min, args.f_max, args.f_step)
    # Ensure nominal ft is included in candidates
    f_candidates = np.unique(np.concatenate([f_candidates, np.array([ft], dtype=np.float64)]))
    f_candidates.sort()

    print(f"--- CHẾ ĐỘ ĐẦU VÀO: {args.mode} ---")
    if use_freq_search:
        print(f"Quét tần số: [{args.f_min:.0f}, {args.f_max:.0f}] Hz, bước {args.f_step:.0f} Hz "
              f"({len(f_candidates)} ứng viên), chu kỳ {args.search_period} khung")
    else:
        print(f"Cố định Doppler: {ft} Hz")

    # Tạo mã giả ngẫu nhiên local
    cacodes1 = np.concatenate([generateCAcode(i) for i in range(11, 21)])
    cacodes2 = np.concatenate([generateCAcode(i) for i in range(21, 31)])

    # Buffer local dài 10ms để FFT correlation (tương ứng 10230 chips)
    # Tuy nhiên Nmax hiện tại là 10000 (1ms ở 10MHz). 
    # Mã CA lặp lại mỗi 1ms (1023 chips). 
    # Ở 10MHz, 1ms là 10000 mẫu.
    n_local = np.arange(20000)  # 10ms ở 2MHz (2e6 * 0.01)
    idx_local = np.floor(n_local / fs * Rc).astype(int) % 10230
    local1_fs = cacodes1[idx_local].astype(np.complex64)
    local2_fs = cacodes2[idx_local].astype(np.complex64)

    f_local1 = np.fft.fft(local1_fs)
    f_local2 = np.fft.fft(local2_fs)

    n_arr = np.arange(Nmax)

    # Tọa độ giả lập
    TX1 = [20.9896100, 105.7110745]
    TX2 = [20.9924397, 105.7106347]
    RX = [20.9911114, 105.7107914]

    T1 = calcDistance(TX1[0], TX1[1], RX[0], RX[1])
    T2 = calcDistance(TX2[0], TX2[1], RX[0], RX[1])
    T0 = (T1 - T2) / speedOfLight * fs

    socket = None
    file_iq_data = None
    file_ptr = 0

    if args.mode == "ZMQ":
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.CONFLATE, 1)
        socket.connect(args.address)
        socket.setsockopt_string(zmq.SUBSCRIBE, "")
        print(f"Đã kết nối ZMQ SUB: {args.address}")
    else:
        if not os.path.exists(args.input):
            print(f"LỖI: Không tìm thấy file {args.input}")
            return 1
        raw_data = np.fromfile(args.input, dtype=np.int16)
        I = raw_data[0::2].astype(np.float32)
        Q = raw_data[1::2].astype(np.float32)
        file_iq_data = (I + 1j * Q).astype(np.complex64)
        print(f"Đã nạp file: {args.input} ({len(file_iq_data)} mẫu)")

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    line1, = ax1.plot(np.zeros(20000), color="#1f77b4", linewidth=0.5, alpha=0.7, label="Signal TX1")
    line2, = ax2.plot(np.zeros(20000), color="#ff7f0e", linewidth=0.5, alpha=0.7, label="Signal TX2")

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
    IQ_padded = np.zeros(20000, dtype=np.complex64)
    tracked_f = float(ft)
    frame_idx = 0

    try:
        while True:
            if args.mode == "ZMQ":
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

                if use_freq_search and (frame_idx % args.search_period == 0):
                    tracked_f, pack = _freq_search(
                        IQ, f_candidates, n_arr, fs, IQ_padded, f_local1, f_local2, Nmax
                    )
                    mag1, mag2, taue1, taue2, max_val1, max_val2 = pack
                elif use_freq_search:
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
                f_used = tracked_f if use_freq_search else ft
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
    finally:
        if socket:
            socket.close()
    
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
