import zmq
import time
import numpy as np
from gnss_utils import generateCAcode

def phat_tin_hieu_simple_zmq():
    # --- Cấu hình tham số ---
    fs = 2e6           # Tần số lấy mẫu (2 MHz)
    Rc = 1.023e6       # Chip rate của mã C/A (1.023 MHz)
    prn = 1            # Sử dụng mã C/A của vệ tinh PRN 1
    chunk_size = 2000  # 1 ms dữ liệu ở tần số 2 MHz có đúng 2000 mẫu

    print("--- PHÁT TÍN HIỆU SIMPLE ZMQ (SỐ THỰC) ---")
    print(f"PRN: {prn}")
    print(f"Sample Rate: {fs/1e6} MHz")
    print(f"Kích thước mỗi chunk: {chunk_size} mẫu (1 ms)")

    # 1. Sinh mã C/A cho PRN 1 (độ dài 1023 chips)
    ca_code = generateCAcode(prn)

    # 2. Thực hiện Resampling mã C/A thành 2000 mẫu số thực (1 ms)
    n = np.arange(chunk_size)
    idx = np.floor(n / fs * Rc).astype(np.int64) % 1023
    r = ca_code[idx].astype(np.float32)  # Tín hiệu hoàn toàn là số thực (float32)

    # 3. Thiết lập ZMQ PUB socket
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://127.0.0.1:5555")
    
    print("\nĐang phát sóng tại tcp://127.0.0.1:5555 ...")
    print("Nhấn Ctrl+C để dừng.")

    # 4. Vòng lặp phát real-time được pace (điều tốc) chính xác theo fs
    t_start = time.perf_counter()
    sent_samples = 0

    try:
        while True:
            # Gửi chunk dữ liệu số thực 1 ms dưới dạng byte thô
            socket.send(r.tobytes())
            sent_samples += chunk_size
            
            # Tính toán thời gian thực tế so với lý thuyết để điều tốc
            elapsed = time.perf_counter() - t_start
            expected_time = sent_samples / fs
            sleep_time = expected_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("\nĐã dừng phát.")
    finally:
        socket.close()

if __name__ == "__main__":
    phat_tin_hieu_simple_zmq()
