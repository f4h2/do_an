import numpy as np
from gnss_utils import generateCAcode
import os

def bpsk_demod():
    Rc = 1.023e6
    Nc = 1023
    fs = 10e6
    
    cacodes1 = generateCAcode(1)
    
    filename = 'signal.bin'
    if not os.path.exists(filename):
        print(f"File {filename} không tồn tại.")
        return
        
    data_demod = np.zeros(8 * 20)
    
    with open(filename, 'rb') as fid:
        for i in range(8 * 20):
            # Lặp lại từng ms
            N1ms = int(0.001 * fs)
            n = np.arange(N1ms)
            
            idx_c = np.floor(n / fs * Rc).astype(int) % Nc
            lcn1 = cacodes1[idx_c]
            
            # Đọc 1ms data
            buf = fid.read(N1ms * 2) # int16 = 2 bytes
            if not buf or len(buf) < N1ms * 2:
                break
            signal = np.frombuffer(buf, dtype=np.int16).astype(float)
            
            data_demod[i] = np.sum(signal * lcn1)
            
    print("Demodulated data:")
    print(data_demod)

if __name__ == "__main__":
    bpsk_demod()
