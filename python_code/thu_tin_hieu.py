import numpy as np
from gnss_utils import generateCAcode, calcDistance
import os

def thu_tin_hieu():
    # Xóa màn hình console tương tự như lệnh 'clc' trong MATLAB
    os.system('cls' if os.name == 'nt' else 'clear')

    Rc = 1.023e6
    fs = 10e6
    ft = 10
    speedOfLight = 299792458
    
    filename = "data_thu_2606_thuc_nghiem_21.bin"
    try:
        fid = open(filename, 'rb')
    except Exception:
        raise FileNotFoundError("Không thể mở tệp tin để đọc.")

    num_samples = int(round(10e-3 * fs))
    
    # Tạo mã C/A cho PRNs từ 11 đến 20
    cacode11 = generateCAcode(11)
    cacode12 = generateCAcode(12)
    cacode13 = generateCAcode(13)
    cacode14 = generateCAcode(14)
    cacode15 = generateCAcode(15)
    cacode16 = generateCAcode(16)
    cacode17 = generateCAcode(17)
    cacode18 = generateCAcode(18)
    cacode19 = generateCAcode(19)
    cacode20 = generateCAcode(20)
    cacodes1 = np.concatenate([
        cacode11, cacode12, cacode13, cacode14, cacode15,
        cacode16, cacode17, cacode18, cacode19, cacode20
    ])

    # Tạo mã C/A cho PRNs từ 21 đến 30
    cacode21 = generateCAcode(21)
    cacode22 = generateCAcode(22)
    cacode23 = generateCAcode(23)
    cacode24 = generateCAcode(24)
    cacode25 = generateCAcode(25)
    cacode26 = generateCAcode(26)
    cacode27 = generateCAcode(27)
    cacode28 = generateCAcode(28)
    cacode29 = generateCAcode(29)
    cacode30 = generateCAcode(30)
    cacodes2 = np.concatenate([
        cacode21, cacode22, cacode23, cacode24, cacode25,
        cacode26, cacode27, cacode28, cacode29, cacode30
    ])

    tauMax = int(fs * 0.01)
    Nmax = 10000

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

    # Đọc block dữ liệu đầu tiên
    tmp = np.fromfile(fid, dtype=np.int16, count=2*num_samples)
    tmp = tmp[:2*Nmax]
    IQ = tmp[0::2] + 1j * tmp[1::2]
    
    n_arr = np.arange(1, Nmax + 1)
    IQ = IQ * np.exp(1j * 2 * np.pi * ft * n_arr / fs)

    Xcorr1 = np.zeros(tauMax + 1, dtype=complex)
    Xcorr2 = np.zeros(tauMax + 1, dtype=complex)

    for tau in range(tauMax + 1):
        idx = np.floor((n_arr + tau) / fs * Rc).astype(np.int64) % 10230
        lcn1 = cacodes1[idx]
        Xcorr1[tau] = np.sum(lcn1 * IQ[:Nmax])

        lcn2 = cacodes2[idx]
        Xcorr2[tau] = np.sum(lcn2 * IQ[:Nmax])

    vale1 = np.max(np.abs(Xcorr1))
    taue1 = np.argmax(np.abs(Xcorr1)) + 1  # Cộng 1 để khớp với chỉ số 1-based của MATLAB

    vale2 = np.max(np.abs(Xcorr2))
    taue2 = np.argmax(np.abs(Xcorr2)) + 1  # Cộng 1 để khớp với chỉ số 1-based của MATLAB

    Delta_T = (taue1 - taue2) - T0
    Delta_T = Delta_T / fs * speedOfLight

    C = np.zeros(1)
    Delta_C = np.zeros(1)
    X = np.zeros(1)

    for ii in range(1):
        tmp = np.fromfile(fid, dtype=np.int16, count=2*num_samples)
        tmp = tmp[:2*Nmax]
        IQ = tmp[0::2] + 1j * tmp[1::2]
        
        IQ = IQ * np.exp(1j * 2 * np.pi * ft * n_arr / fs)
        
        for tau in range(tauMax + 1):
            idx = np.floor((n_arr + tau) / fs * Rc).astype(np.int64) % 10230
            lcn1 = cacodes1[idx]
            Xcorr1[tau] = np.sum(lcn1 * IQ[:Nmax])

            lcn2 = cacodes2[idx]
            Xcorr2[tau] = np.sum(lcn2 * IQ[:Nmax])

        vale1 = np.max(np.abs(Xcorr1))
        taue1 = np.argmax(np.abs(Xcorr1)) + 1

        vale2 = np.max(np.abs(Xcorr2))
        taue2 = np.argmax(np.abs(Xcorr2)) + 1

        C[ii] = taue1 - taue2
        Delta_C[ii] = C[ii] / fs * speedOfLight
        X[ii] = (-Delta_C[ii] + D + Delta_T) / 2
        X2 = D - X[ii]
        
        print(f"T1 được tính ra bằng: {X[ii]}")
        print(f"T2 được tính ra bằng: {X2}")

    fid.close()

if __name__ == "__main__":
    thu_tin_hieu()
