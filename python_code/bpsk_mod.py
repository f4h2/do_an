import numpy as np
from gnss_utils import generateCAcode

def bpsk_mod():
    Rc = 1.023e6
    Nc = 1023
    fs = 10e6
    
    cacodes1 = generateCAcode(1)
    
    Rd = 50 # 50 bits/s
    dataBits = np.array([0, 1, 0, 1, 1, 0, 0, 1])
    dataBits = 2 * dataBits - 1 # 0 -> -1, 1 -> 1
    
    N = int(8 * 0.02 * fs) # 8 bits, 0.02s per bit
    n = np.arange(N)
    
    idx_c = np.floor(n / fs * Rc).astype(int) % Nc
    lcn1 = cacodes1[idx_c]
    
    idx_d = np.floor(n / fs * Rd).astype(int)
    # Handle array bounds just in case due to floating point
    idx_d = np.minimum(idx_d, len(dataBits) - 1)
    data = dataBits[idx_d]
    
    signal = lcn1 * data
    
    signal_int16 = signal.astype(np.int16)
    with open('signal.bin', 'wb') as fid:
        fid.write(signal_int16.tobytes())
    print("Đã tạo file signal.bin")

if __name__ == "__main__":
    bpsk_mod()
