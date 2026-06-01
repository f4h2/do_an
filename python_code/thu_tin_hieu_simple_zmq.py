import zmq
import time
import numpy as np
import matplotlib.pyplot as plt
from gnss_utils import generateCAcode

def thu_tin_hieu_simple_zmq():
    # --- Cấu hình tham số (phải khớp với bên phát) ---
    fs = 2e6           # Tần số lấy mẫu (2 MHz)
    Rc = 1.023e6       # Chip rate (1.023 MHz)
    prn = 1            # PRN vệ tinh cần tương quan (PRN 1)
    chunk_size = 2000  # 1 ms dữ liệu ở tần số 2 MHz

    print("--- THU TÍN HIỆU SIMPLE ZMQ (SỐ THỰC) ---")
    print(f"PRN: {prn}")
    print(f"Sample Rate: {fs/1e6} MHz")
    print(f"Kích thước mỗi chunk: {chunk_size} mẫu (1 ms)")

    # 1. Khởi tạo mã C/A cục bộ (PRN 1) số thực và thực hiện Resampling sang 2000 mẫu
    ca_code = generateCAcode(prn)
    n = np.arange(chunk_size)
    idx = np.floor(n / fs * Rc).astype(np.int64) % 1023
    local_replica = ca_code[idx].astype(np.float32)  # Bản sao cục bộ dạng số thực

    # Tính toán trước FFT của mã cục bộ
    f_local = np.fft.fft(local_replica)

    # 2. Thiết lập ZMQ SUB socket
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    
    # Sử dụng CONFLATE để chỉ lấy gói tin mới nhất
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.connect("tcp://127.0.0.1:5556")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")

    print("\nĐã kết nối tới tcp://127.0.0.1:5555. Đang đợi tín hiệu...")

    # 3. Khởi tạo đồ thị hiển thị thời gian thực (Matplotlib)
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    line, = ax.plot(np.arange(chunk_size), np.zeros(chunk_size), color='#1f77b4', linewidth=1.5, label='Cross-Correlation')
    peak_dot, = ax.plot([0], [0], 'ro', markersize=8, label='Correlation Peak')
    
    ax.set_title("Real-Time Cross-Correlation Map (PRN 1) - Pure Real", fontsize=12, fontweight='bold')
    ax.set_xlabel("Delay (Samples)", fontsize=10)
    ax.set_ylabel("Correlation Magnitude", fontsize=10)
    ax.set_xlim(0, chunk_size)
    ax.set_ylim(0, 2500)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show(block=False)

    print("Đang xử lý real-time. Nhấn Ctrl+C trong terminal để dừng.")

    try:
        while True:
            # Nhận byte thô số thực từ ZMQ
            raw_msg = socket.recv()
            
            # Chuyển đổi sang mảng số thực float32
            iq_chunk = np.frombuffer(raw_msg, dtype=np.float32)
            
            # Đảm bảo kích thước đúng 1ms
            if len(iq_chunk) != chunk_size:
                continue

            # 4. Tính tương quan chéo tuần hoàn số thực qua FFT
            # R[k] = real( ifft( FFT(recv) * conj(FFT(local)) ) )
            f_recv = np.fft.fft(iq_chunk)
            corr = np.real(np.fft.ifft(f_recv * np.conj(f_local)))

            # Tìm đỉnh tương quan
            peak_idx = np.argmax(corr)
            peak_val = corr[peak_idx]

            # 5. Cập nhật đồ thị Real-time
            line.set_ydata(corr)
            peak_dot.set_data([peak_idx], [peak_val])
            
            # Tự động điều chỉnh trục Y nếu đỉnh tương quan vượt ngưỡng
            current_ylim = ax.get_ylim()[1]
            if peak_val > current_ylim * 0.8:
                ax.set_ylim(0, peak_val * 1.3)
            elif peak_val < current_ylim * 0.2 and current_ylim > 200:
                ax.set_ylim(0, max(200, peak_val * 1.5))

            plt.pause(0.001)

            # In thông tin cập nhật liên tục trên 1 dòng
            print(f"\r[Real-Time] Peak Index: {peak_idx:4d} | Peak Magnitude: {peak_val:7.1f}", end="", flush=True)

    except KeyboardInterrupt:
        print("\n\nĐã dừng nhận tín hiệu.")
    finally:
        socket.close()
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    thu_tin_hieu_simple_zmq()
