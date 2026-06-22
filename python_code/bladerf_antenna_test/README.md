# BladeRF x40 — Two-Antenna Over-the-Air Test

Thu và phát tín hiệu GPS C/A (BPSK) **đồng thời** trên một BladeRF x40 duy nhất,
dùng **2 ăng ten vật lý** (TX1 và RX1), không cần GNU Radio UI.  
Sau khi thu xong, script tự động tính **cross-correlation** và vẽ đồ thị.

---

## Cấu trúc thư mục

```
bladerf_antenna_test/
├── bladerf_loopback_test.py   # Script chính
└── README.md                  # File này
```

---

## Yêu cầu phần cứng

| Thiết bị | Mô tả |
|---|---|
| BladeRF x40 | 1 thiết bị duy nhất |
| Ăng ten TX | Gắn vào cổng **TX1** (J61) |
| Ăng ten RX | Gắn vào cổng **RX1** (J51) |

**Bố trí ăng ten:**
- Đặt 2 ăng ten **cùng phân cực** (đứng–đứng hoặc ngang–ngang).
- Khoảng cách **10–50 cm** để có đủ mức tín hiệu.
- Không để ăng ten chạm nhau (tránh bão hoà ADC).

```
  [BladeRF]
  TX1 ──(ăng ten phát)  →  ~~~sóng~~~  →  (ăng ten thu)── RX1
```

---

## Cài đặt môi trường

Script sử dụng môi trường conda **gnss_env**.

### 1. Kích hoạt môi trường

```bash
conda activate gnss_env
```

### 2. Cài thư viện Python cần thiết

```bash
pip install bladerf numpy matplotlib scipy
```

> **Lưu ý:** Driver libbladeRF phải được cài ở cấp hệ thống trước.  
> Trên Ubuntu/Debian:
> ```bash
> sudo add-apt-repository ppa:nuandllc/bladerf
> sudo apt update
> sudo apt install bladerf libbladerf-dev
> ```

### 3. Kiểm tra BladeRF nhận diện được

```bash
bladeRF-cli -p
# Kết quả mong muốn:
#   [USB] Serial: xxxx  FW: x.x.x  FPGA: x.x.x
```

---

## Chạy script

```bash
cd bladerf_antenna_test
conda activate gnss_env
python bladerf_loopback_test.py
```

---

## Tuỳ chỉnh thông số

Mở `bladerf_loopback_test.py`, chỉnh các hằng số ở đầu file:

```python
FREQ_HZ     = 433e6      # Tần số sóng mang (Hz)
                         # Đổi sang 1575.42e6 cho GPS L1 (cần môi trường chắn sóng)
SAMPLE_RATE = 10e6       # Sample rate (sps)
BANDWIDTH   = 5e6        # Bandwidth (Hz)
TX_GAIN     = 60         # Gain TX (dB) — tăng nếu ăng ten đặt xa
RX_GAIN     = 60         # Gain RX (dB) — GIẢM nếu tín hiệu bị clipping
PRN         = 1          # C/A code PRN số (1–32)
NUM_MS      = 10         # Thời gian thu/phát (ms)
RF_LOOPBACK = False      # Luôn False khi dùng ăng ten vật lý
```

### Khắc phục sự cố thường gặp

| Triệu chứng | Nguyên nhân | Giải pháp |
|---|---|---|
| Không thấy đỉnh tương quan | Tín hiệu quá yếu | Tăng `TX_GAIN`, rút ngắn khoảng cách ăng ten |
| Đỉnh bị nhiễu / rộng | SNR thấp | Tăng `RX_GAIN`, dùng ăng ten có gain cao hơn |
| RX bị clipping (sóng vuông) | Tín hiệu quá mạnh | Giảm `TX_GAIN` về 40–50 dB |
| `BladeRF not found` | Chưa kết nối / thiếu driver | Kiểm tra USB, chạy `bladeRF-cli -p` |
| `Permission denied` (USB) | Thiếu quyền USB | `sudo usermod -aG plugdev $USER` rồi logout/login lại |

---

## Chế độ mô phỏng (không có BladeRF)

Nếu **không cắm BladeRF**, script tự động chuyển sang chế độ mô phỏng:
- Tín hiệu TX được tạo nội bộ.
- Thêm trễ giả lập **3 µs** và nhiễu Gaussian.
- Vẽ đồ thị tương quan bình thường.

```bash
python bladerf_loopback_test.py
# [WARN] Không tìm thấy module bladerf — chạy ở chế độ mô phỏng (simulation).
```

---

## Đầu ra

Sau khi chạy xong:

| File | Nội dung |
|---|---|
| `loopback_corr_prn1.png` | Đồ thị: phổ tần + tương quan toàn bộ + phóng to đỉnh |
| `tx_raw_prn1.npy` | Raw IQ TX (int16 interleaved) |
| `rx_raw_prn1.npy` | Raw IQ RX (int16 interleaved) |

### Đọc lại dữ liệu raw để phân tích sau:

```python
import numpy as np
tx = np.load('tx_raw_prn1.npy')
rx = np.load('rx_raw_prn1.npy')
# tx/rx là mảng int16 interleaved [I0, Q0, I1, Q1, ...]
I = rx[0::2].astype(float)
Q = rx[1::2].astype(float)
signal = I + 1j * Q
```

---

## Thông tin kỹ thuật

| Tham số | Giá trị mặc định |
|---|---|
| Định dạng IQ | SC16_Q11 (int16 interleaved) |
| Điều chế | BPSK với C/A code GPS |
| Chip rate | 1.023 Mcps |
| Độ dài C/A code | 1023 chips (~1 ms) |
| Tính tương quan | FFT cross-correlation (O(N log N)) |


khi có 1 thiết bị vừa thu vừa phát
python phat_thu_bladerf_gui.py --freq_hz 433e6 \
    --tx_prn_start 11 --tx_prn_end 20 \
    --prn1_start 11  --prn1_end 20 \
    --prn2_start 21  --prn2_end 30


python phat_tin_hieu_bladerf.py --prn_start 11 --prn_end 20 --freq_hz 433e6 --serial 9d1755063f78cde4c2960ba35600ceb5

python phat_tin_hieu_bladerf_2.py --prn_start 21 --prn_end 30 --freq_hz 433e6 --serial a0e5ffb5f1c28a2d57f5f5d9d13372ed

python thu_tin_hieu_bladerf_gui_copy2.py --prn1_start 11 --prn1_end 20 --prn2_start 21 --prn2_end 30 --freq_hz 433e6 --serial 6ed84115bdca40800254de285ec1d898

9d1755063f78cde4c2960ba35600ceb5


python thu_tin_hieu_bladerf_newton.py \
    --tx_prn_ranges 11:20,21:30,31:32,1:10 \
    --freq_hz 433e6 \
    --serial a0e5ffb5f1c28a2d57f5f5d9d13372ed
