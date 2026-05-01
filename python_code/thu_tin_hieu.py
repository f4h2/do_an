import numpy as np
import math
from gnss_utils import generateCAcode, calcDistance
import os

def thu_tin_hieu():
    Rc = 1.023e6
    fs = 10e6
    ft = 10
    speedOfLight = 299792458
    
    filename = "data_thu_2606_thuc_nghiem_21.bin"
    if not os.path.exists(filename):
        print(f"File {filename} không tồn tại. Tạo dữ liệu giả để test.")
        # Nếu muốn test có thể tạo 1 file rỗng hoặc dữ liệu ngẫu nhiên
        # return
        
    num_samples = int(round(10e-3 * fs))
    cacodes1 = np.concatenate([generateCAcode(i) for i in range(11, 21)])
    cacodes2 = np.concatenate([generateCAcode(i) for i in range(21, 31)])
    
    tauMax = int(fs * 0.01)
    Nmax = 10000
    n = np.arange(Nmax)
    
    TX1 = [20.9896100, 105.7110745]
    TX2 = [20.9924397, 105.7106347]
    RX = [20.9911114, 105.7107914]
    
    T1 = calcDistance(TX1[0], TX1[1], RX[0], RX[1])
    T2 = calcDistance(TX2[0], TX2[1], RX[0], RX[1])
    T0 = (T1 - T2) / speedOfLight * fs
    D = calcDistance(TX1[0], TX1[1], TX2[0], TX2[1])
    
    print(f"T1: {T1}")
    print(f"T2: {T2}")
    print(f"D: {D}")
    
    try:
        with open(filename, 'rb') as fid:
            # Code from Matlab
            tmp = np.frombuffer(fid.read(2 * num_samples * 2), dtype=np.int16) # *2 for 16-bit
            tmp = tmp[:2*Nmax]
            IQ = tmp[0::2] + 1j * tmp[1::2]
            n_arr = np.arange(1, Nmax + 1)
            IQ = IQ * np.exp(1j * 2 * np.pi * ft * n_arr / fs)
            
            Xcorr1 = np.zeros(tauMax + 1, dtype=complex)
            Xcorr2 = np.zeros(tauMax + 1, dtype=complex)
            
            for tau in range(tauMax + 1):
                idx = np.floor((n_arr + tau) / fs * Rc).astype(int) % 10230
                lcn1 = cacodes1[idx]
                Xcorr1[tau] = np.sum(lcn1 * IQ[:Nmax])
                
                lcn2 = cacodes2[idx]
                Xcorr2[tau] = np.sum(lcn2 * IQ[:Nmax])
                
            vale1 = np.max(np.abs(Xcorr1))
            taue1 = np.argmax(np.abs(Xcorr1))
            
            vale2 = np.max(np.abs(Xcorr2))
            taue2 = np.argmax(np.abs(Xcorr2))
            
            Delta_T = (taue1 - taue2) - T0
            Delta_T = Delta_T / fs * speedOfLight
            
            for ii in range(1):
                # Read next block
                tmp = np.frombuffer(fid.read(2 * num_samples * 2), dtype=np.int16)
                if len(tmp) < 2 * Nmax:
                    break
                tmp = tmp[:2*Nmax]
                IQ = tmp[0::2] + 1j * tmp[1::2]
                IQ = IQ * np.exp(1j * 2 * np.pi * ft * n_arr / fs)
                
                for tau in range(tauMax + 1):
                    idx = np.floor((n_arr + tau) / fs * Rc).astype(int) % 10230
                    lcn1 = cacodes1[idx]
                    Xcorr1[tau] = np.sum(lcn1 * IQ[:Nmax])
                    
                    lcn2 = cacodes2[idx]
                    Xcorr2[tau] = np.sum(lcn2 * IQ[:Nmax])
                
                taue1 = np.argmax(np.abs(Xcorr1))
                taue2 = np.argmax(np.abs(Xcorr2))
                
                C_val = (taue1 - taue2)
                Delta_C_val = C_val / fs * speedOfLight
                X_val = (-Delta_C_val + D + Delta_T) / 2
                X2_val = D - X_val
                print(f"T1 được tính ra bằng: {X_val}")
                print(f"T2 được tính ra bằng: {X2_val}")
                
    except FileNotFoundError:
        pass

if __name__ == "__main__":
    thu_tin_hieu()
