import numpy as np
from gnss_utils import generateCAcode

def phat_tin_hieu_21_30():
    # 1. Tham số hệ thống (Đồng bộ 10MHz)
    Rc = 1.023e6
    fs = 10e6
    ft = 10
    time_s = 1
    
    # --- THÊM LẠI ĐỘ TRỄ KHÁC ĐỂ SO SÁNH ---
    tau2 = 100 
    
    # 2. Tạo mã C/A cho PRNs từ 21 đến 30 (range 21-31 lấy 10 PRN)
    print(f"Đang tạo TX2 (PRN 21-30) với trễ tau2 = {tau2}...")
    cacodes = np.concatenate([generateCAcode(i) for i in range(21, 31)])
    
    N = int(fs * time_s)
    n = np.arange(N)
    
    # Áp dụng trễ tau2
    n2 = n + tau2
    idx = np.floor(n2 / fs * Rc).astype(int) % 10230
    r = cacodes[idx]
    
    # 3. Điều chế
    amplitude = 1024
    normalized_I = amplitude * r * np.cos(2 * np.pi * ft * n / fs)
    normalized_Q = amplitude * r * np.sin(2 * np.pi * ft * n / fs)
    
    IQ = np.zeros(2 * N, dtype=np.int16)
    IQ[0::2] = np.round(normalized_I).astype(np.int16)
    IQ[1::2] = np.round(normalized_Q).astype(np.int16)
    
    # 4. Lưu file
    filename = 'data_tx2_prn_21_30_10MHz.bin'
    with open(filename, 'wb') as fid:
        fid.write(IQ.tobytes())
    print(f"Đã lưu file {filename}")

if __name__ == "__main__":
    phat_tin_hieu_21_30()
