import numpy as np
import zmq
import time
import matplotlib.pyplot as plt
import os
from gnss_utils import generateCAcode, calcDistance


def thu_tin_hieu_rt_gui():
    # --- Cấu hình tham số ---
    Rc = 1.023e6      # Chip rate
    fs = 2e6         # Sampling frequency
    ft = 10           # Doppler offset (Hz)
    speedOfLight = 299792458

    # Lấy chế độ đầu vào từ biến môi trường (ZMQ hoặc FILE)
    INPUT_MODE = os.environ.get('INPUT_MODE', 'ZMQ').upper()
    FILE_PATH = 'data_tx1_prn_11_20_10MHz.bin'
    
    # Cấu hình lưu file
    SAVE_DATA = True  # Set True để lưu tín hiệu thu được
    SAVE_FILENAME = 'captured_signal_rt.bin'

    # Số mẫu xử lý mỗi lần (1ms dữ liệu tại fs=2MHz là 2,000 mẫu)
    # Tuy nhiên code cũ đang dùng Nmax=10000 (5ms tại 2MHz)
    Nmax = 10000

    # 1. KHỞI TẠO: Sinh và Resample mã CA một lần duy nhất
    print(f"--- CHẾ ĐỘ ĐẦU VÀO: {INPUT_MODE} ---")
    if SAVE_DATA and INPUT_MODE == 'ZMQ':
        print(f"--- CHẾ ĐỘ LƯU FILE: ĐANG BẬT ({SAVE_FILENAME}) ---")
    
    print("Đang khởi tạo mã CA và thực hiện resampling (10ms)...")
    cacodes1 = np.concatenate([generateCAcode(i) for i in range(11, 21)])
    cacodes2 = np.concatenate([generateCAcode(i) for i in range(21, 31)])

    # fs = 2e6, Rc = 1.023e6. 10ms = 20,000 mẫu. 
    # Mã CA 10ms là 10230 chips.
    n_local = np.arange(20000)  # 10ms ở 2MHz
    idx_local = np.floor(n_local / fs * Rc).astype(int) % 10230
    local1_fs = cacodes1[idx_local].astype(np.complex64)
    local2_fs = cacodes2[idx_local].astype(np.complex64)

    f_local1 = np.fft.fft(local1_fs)
    f_local2 = np.fft.fft(local2_fs)

    n_arr = np.arange(Nmax)

    # Tọa độ (giữ nguyên)
    TX1 = [20.9896100, 105.7110745]
    TX2 = [20.9924397, 105.7106347]
    RX = [20.9911114, 105.7107914]

    T1 = calcDistance(TX1[0], TX1[1], RX[0], RX[1])
    T2 = calcDistance(TX2[0], TX2[1], RX[0], RX[1])
    T0 = (T1 - T2) / speedOfLight * fs
    D = calcDistance(TX1[0], TX1[1], TX2[0], TX2[1])

    # 2. THIẾT LẬP NGUỒN DỮ LIỆU
    socket = None
    file_iq_data = None
    file_ptr = 0
    save_file_handle = None

    if INPUT_MODE == 'ZMQ':
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.CONFLATE, 1)
        socket.connect("tcp://127.0.0.1:5555")
        socket.setsockopt_string(zmq.SUBSCRIBE, "")
        print("Đã kết nối ZMQ: tcp://127.0.0.1:5555")
        
        if SAVE_DATA:
            save_file_handle = open(SAVE_FILENAME, 'wb')
    else:
        if not os.path.exists(FILE_PATH):
            print(f"LỖI: Không tìm thấy file {FILE_PATH}")
            return
        # Đọc toàn bộ file và chuyển sang complex64
        # Giả sử file đầu vào là int16 (I, Q xen kẽ)
        raw_data = np.fromfile(FILE_PATH, dtype=np.int16)
        I = raw_data[0::2].astype(np.float32)
        Q = raw_data[1::2].astype(np.float32)
        file_iq_data = (I + 1j*Q).astype(np.complex64)
        print(f"Đã nạp file: {FILE_PATH} ({len(file_iq_data)} mẫu)")

    # 3. THIẾT LẬP ĐỒ THỊ
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    line1, = ax1.plot(np.zeros(20000), color='#1f77b4', linewidth=0.5, alpha=0.7, label='Signal TX1')
    line2, = ax2.plot(np.zeros(20000), color='#ff7f0e', linewidth=0.5, alpha=0.7, label='Signal TX2')

    peak_dot1, = ax1.plot([0], [0], 'ro', markersize=6, label='Peak TX1')
    peak_dot2, = ax2.plot([0], [0], 'go', markersize=6, label='Peak TX2')

    ax1.set_title("Real-time Correlation Map - TX1", fontsize=12)
    ax2.set_title("Real-time Correlation Map - TX2", fontsize=12)

    for ax in [ax1, ax2]:
        ax.set_ylim(0, 100)
        ax.set_xlabel("Delay Samples")
        ax.set_ylabel("Magnitude")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right')

    plt.tight_layout()
    buffer_iq = np.array([], dtype=np.complex64)

    try:
        while True:
            if INPUT_MODE == 'ZMQ':
                raw_msg = socket.recv()
                
                # Lưu dữ liệu thô vào file nếu đang bật chế độ lưu
                if save_file_handle:
                    save_file_handle.write(raw_msg)
                
                chunk_iq = np.frombuffer(raw_msg, dtype=np.complex64)
            else:
                # Đọc chunk từ file_iq_data
                if file_ptr + Nmax > len(file_iq_data):
                    file_ptr = 0  # Lặp lại file
                chunk_iq = file_iq_data[file_ptr:file_ptr+Nmax]
                file_ptr += Nmax
                time.sleep(0.001)  # Giả lập 1ms real-time

            buffer_iq = np.concatenate((buffer_iq, chunk_iq))

            if len(buffer_iq) >= Nmax:
                IQ = buffer_iq[:Nmax]
                buffer_iq = buffer_iq[Nmax:]

                start_time = time.time()

                # Bù Doppler
                IQ_shifted = IQ * np.exp(1j * 2 * np.pi * ft * n_arr / fs)

                # Tương quan nhanh (FFT size 20,000 = 10ms ở 2MHz)
                IQ_padded = np.zeros(20000, dtype=np.complex64)
                IQ_padded[:Nmax] = IQ_shifted
                f_IQ = np.fft.fft(IQ_padded)

                mag1 = np.abs(np.fft.ifft(f_local1 * np.conj(f_IQ)))
                mag2 = np.abs(np.fft.ifft(f_local2 * np.conj(f_IQ)))

                taue1 = np.argmax(mag1); max_val1 = mag1[taue1]
                taue2 = np.argmax(mag2); max_val2 = mag2[taue2]

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

                # Tính toán khoảng cách (TDOA)
                Delta_T = (taue1 - taue2) - T0
                Delta_T_m = Delta_T / fs * speedOfLight

                process_time = time.time() - start_time
                print(f"Peaks: {taue1:5d}, {taue2:5d} | Mags: {max_val1:6.1f}, {max_val2:6.1f} | Diff: {Delta_T_m:8.2f}m")

                if len(buffer_iq) > Nmax * 100:
                    buffer_iq = np.array([], dtype=np.complex64)

    except KeyboardInterrupt:
        print("\n>>> Đã dừng.")
        if save_file_handle:
            save_file_handle.close()
            print(f"Đã lưu dữ liệu vào {SAVE_FILENAME}")
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    thu_tin_hieu_rt_gui()
