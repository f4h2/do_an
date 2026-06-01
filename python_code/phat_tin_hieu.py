import numpy as np
from gnss_utils import generateCAcode
import os

def phat_tin_hieu():
    # Tạo các mã C/A cho PRNs từ 11 đến 20 và ghép chúng lại thành một mảng 1x10230
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
    
    cacodes = np.concatenate([
        cacode11, cacode12, cacode13, cacode14, cacode15,
        cacode16, cacode17, cacode18, cacode19, cacode20
    ])

    Rc = 1.023e6
    fs = 8e6
    ft = 2
    time = 2  # 10 giây
    N = int(fs * time)
    n = np.arange(N)
    tau1 = 40
    tau2 = 100
    n1 = n + tau1

    # MATLAB: r = cacodes(rem(floor(n1 / fs * Rc), 10230) + 1);
    idx = np.floor(n1 / fs * Rc).astype(np.int64) % 10230
    r = cacodes[idx]

    normalized_IQ = r * 1024
    normalized_I = normalized_IQ * np.cos(2 * np.pi * ft * n / fs)
    normalized_Q = normalized_IQ * np.sin(2 * np.pi * ft * n / fs)

    IQ = np.zeros(2 * len(r))
    IQ[0::2] = normalized_I
    IQ[1::2] = normalized_Q

    num_repeats = 5
    IQ_p_repeated = np.tile(IQ, num_repeats)

    filepath = r'data_2506_phat_ca_new2_8M.bin'
    
    # Tạo thư mục chứa file nếu cần (hữu ích cho cả Windows và Linux)
    dir_name = os.path.dirname(filepath)
    if dir_name:
        try:
            os.makedirs(dir_name, exist_ok=True)
        except Exception:
            pass

    try:
        fid_11_20 = open(filepath, 'wb')
    except Exception:
        # Nếu không mở được đường dẫn tuyệt đối (ví dụ trên Linux), chuyển sang file cục bộ
        local_filename = os.path.basename(filepath)
        try:
            fid_11_20 = open(local_filename, 'wb')
            print(f"Không thể mở đường dẫn tuyệt đối '{filepath}'. Đã chuyển sang ghi vào file cục bộ: '{local_filename}'")
        except Exception:
            raise OSError("Không thể mở tệp tin để ghi.")

    r_p_int = np.round(IQ_p_repeated).astype(np.int16)
    fid_11_20.write(r_p_int.tobytes())
    fid_11_20.close()

if __name__ == "__main__":
    phat_tin_hieu()
