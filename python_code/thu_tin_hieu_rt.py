import numpy as np
import zmq
import time
from gnss_utils import generateCAcode, calcDistance

def thu_tin_hieu_rt():
    Rc = 1.023e6
    fs = 10e6
    ft = 10
    speedOfLight = 299792458
    
    # Số mẫu xử lý mỗi lần (1ms dữ liệu)
    Nmax = 10000
    
    cacodes1 = np.concatenate([generateCAcode(i) for i in range(11, 21)])
    cacodes2 = np.concatenate([generateCAcode(i) for i in range(21, 31)])
    
    tauMax = int(fs * 0.01)
    n_arr = np.arange(1, Nmax + 1)
    
    TX1 = [20.9896100, 105.7110745]
    TX2 = [20.9924397, 105.7106347]
    RX = [20.9911114, 105.7107914]
    
    T1 = calcDistance(TX1[0], TX1[1], RX[0], RX[1])
    T2 = calcDistance(TX2[0], TX2[1], RX[0], RX[1])
    T0 = (T1 - T2) / speedOfLight * fs
    D = calcDistance(TX1[0], TX1[1], TX2[0], TX2[1])
    
    # --- Thiết lập ZeroMQ (ZMQ) để nhận dữ liệu từ GNU Radio ---
    # Cần thêm block "ZMQ PUB Sink" trong GNU Radio Companion
    # Address: tcp://127.0.0.1:5555
    # Định dạng dữ liệu (Type): Complex (Float32x2)
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.CONFLATE, 1) # Giữ gói tin mới nhất (tránh trễ do python xử lý không kịp real-time)
    socket.connect("tcp://127.0.0.1:5555") 
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    
    print("Đang chờ dữ liệu realtime từ GNU Radio qua ZMQ (tcp://127.0.0.1:5555)...")

    # Tính trước mảng cacodes cho n_arr để tối ưu tốc độ realtime
    # (Nếu có thể, hãy dùng tương quan chéo bằng FFT để chạy nhanh hơn realtime)
    
    buffer_iq = np.array([], dtype=np.complex64)
    
    try:
        while True:
            # Nhận chunk dữ liệu thô từ GNU Radio
            raw_msg = socket.recv()
            
            # Block GNU Radio "ZMQ PUB Sink" loại Complex xuất ra định dạng Float32 x2
            chunk_iq = np.frombuffer(raw_msg, dtype=np.complex64)
            buffer_iq = np.concatenate((buffer_iq, chunk_iq))
            
            # Chỉ xử lý khi có đủ mẫu cần thiết
            if len(buffer_iq) >= Nmax:
                # Cắt ra lượng mẫu Nmax
                IQ = buffer_iq[:Nmax]
                
                # Cập nhật buffer (xoá những mẫu cũ để không đầy RAM)
                # Chú ý: Ở chế độ realtime nếu máy tính xử lý không kịp, ta có thể xóa trắng buffer 
                # để lấy các mẫu mới nhất chứ không xếp hàng chờ đợi
                buffer_iq = buffer_iq[Nmax:]
                
                start_time = time.time()
                
                IQ = IQ * np.exp(1j * 2 * np.pi * ft * n_arr / fs)
                
                Xcorr1 = np.zeros(tauMax + 1, dtype=complex)
                Xcorr2 = np.zeros(tauMax + 1, dtype=complex)
                
                # Vòng lặp tính toán tương quan (Nên cân nhắc dùng scipy.signal.correlate cho realtime)
                for tau in range(tauMax + 1):
                    idx = np.floor((n_arr + tau) / fs * Rc).astype(int) % 10230
                    
                    lcn1 = cacodes1[idx]
                    Xcorr1[tau] = np.sum(lcn1 * IQ)
                    
                    lcn2 = cacodes2[idx]
                    Xcorr2[tau] = np.sum(lcn2 * IQ)
                
                taue1 = np.argmax(np.abs(Xcorr1))
                taue2 = np.argmax(np.abs(Xcorr2))
                
                Delta_T = (taue1 - taue2) - T0
                Delta_T = Delta_T / fs * speedOfLight
                
                C_val = (taue1 - taue2)
                Delta_C_val = C_val / fs * speedOfLight
                X_val = (-Delta_C_val + D + Delta_T) / 2
                X2_val = D - X_val
                
                process_time = time.time() - start_time
                print(f"T1: {X_val:10.2f} | T2: {X2_val:10.2f} | Tgian xử lý: {process_time:.3f}s")
                
                # Clear buffer nếu xử lý quá chậm so với tốc độ lấy mẫu
                if len(buffer_iq) > Nmax * 10:
                    print("Cảnh báo: CPU xử lý không kịp, drop mẫu cũ...")
                    buffer_iq = np.array([], dtype=np.complex64)
                
    except KeyboardInterrupt:
        print("\nĐã dừng thu realtime.")

if __name__ == "__main__":
    thu_tin_hieu_rt()
