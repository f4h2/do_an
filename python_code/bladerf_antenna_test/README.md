# BladeRF x40 — Two-Antenna Over-the-Air Test

Thu và phát tín hiệu GPS C/A (BPSK) **đồng thời** trên một BladeRF x40 duy nhất,
dùng **2 ăng ten vật lý** (TX1 và RX1), không cần GNU Radio UI.  
Sau khi thu xong, script tự động tính **cross-correlation** và vẽ đồ thị.

---

## Cấu trúc thư mục

```
bladerf_antenna_test/
├── bladerf_loopback_test.py
├── phat_tin_hieu_bladerf.py      # Phát qua BladeRF
├── phat_tin_hieu_bladerf_2.py    # Phát TX2 (PRN khác)
├── phat_tin_hieu_hackrf.py       # Phát qua HackRF One (SoapySDR)
├── thu_tin_hieu_bladerf_gui_copy.py
├── thu_tin_hieu_bladerf_newton.py
├── phat_thu_bladerf_gui.py
└── README.md
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

---

## Lệnh chạy (gốc)

khi có 1 thiết bị vừa thu vừa phát

```bash
python phat_thu_bladerf_gui.py --freq_hz 433e6 \
    --tx_prn_start 11 --tx_prn_end 20 \
    --prn1_start 11  --prn1_end 20 \
    --prn2_start 21  --prn2_end 30
```

```bash
python phat_tin_hieu_bladerf.py --prn_start 26 --prn_end 30 --freq_hz 433e6 --serial f444bf4f6a404a40a2ea650066b7e17c

python phat_tin_hieu_bladerf_2.py --prn_start 21 --prn_end 30 --freq_hz 433e6 --serial a0e5ffb5f1c28a2d57f5f5d9d13372ed

python thu_tin_hieu_bladerf_gui_copy2.py --prn1_start 11 --prn1_end 20 --prn2_start 21 --prn2_end 30 --freq_hz 433e6 --serial 6ed84115bdca40800254de285ec1d898
```

```
9d1755063f78cde4c2960ba35600ceb5
```

```bash
python thu_tin_hieu_bladerf_newton.py \
    --tx_prn_ranges 11:20,21:30,31:32,1:10 \
    --freq_hz 433e6 \
    --serial a0e5ffb5f1c28a2d57f5f5d9d13372ed
```

---

## HackRF One — phát tín hiệu (SoapySDR)

### Cài đặt

```bash
conda activate gnss_env

# Driver hệ thống + module Soapy cho HackRF
sudo apt install hackrf libhackrf-dev soapysdr-tools soapysdr-module-hackrf

# Python
pip install SoapySDR numpy
```

### Kiểm tra trước khi chạy

```bash
hackrf_info
```

Nếu dùng **conda `gnss_env`**, SoapySDR trong conda không tự thấy module HackRF của hệ thống.
Phải export trước (hoặc dùng script `phat_tin_hieu_hackrf.py` — script tự set):

```bash
export SOAPY_SDR_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/SoapySDR/modules0.8
SoapySDRUtil --find="driver=hackrf"
```

Kết quả mong muốn: thấy `driver=hackrf` và serial thiết bị.

### Phát realtime (tương đương `phat_tin_hieu_bladerf.py`)

```bash
cd python_code/bladerf_antenna_test
conda activate gnss_env

python phat_tin_hieu_hackrf.py \
    --prn_start 11 --prn_end 20 \
    --freq_hz 433e6 \
    --fs 2e6 \
    --ft 0 \
    --tau 40 \
    --tx_gain 30
```

Có serial cụ thể:

```bash
python phat_tin_hieu_hackrf.py \
    --prn_start 11 --prn_end 20 \
    --freq_hz 433e6 \
    --serial <HACKRF_SERIAL>
```

Chỉ tạo file IQ (không cần cắm HackRF):

```bash
python phat_tin_hieu_hackrf.py \
    --prn_start 11 --prn_end 20 \
    --fs 2e6 \
    --save_file tx_prn11_20_433.bin
```

### Lỗi `SoapySDR::Device::make() no match` / `No devices found! driver=hackrf`

| Tình huống | Cách xử lý |
|------------|------------|
| `hackrf_info` OK, `SoapySDRUtil` không thấy | Trong conda: `export SOAPY_SDR_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/SoapySDR/modules0.8` |
| Chưa cài module Soapy | `sudo apt install soapysdr-module-hackrf` |
| `hackrf_info` lỗi | `sudo apt install hackrf` + kiểm tra USB |
| Quyền USB | `sudo usermod -aG plugdev $USER` rồi logout/login |

**Giải thích:** `hackrf_info` dùng **libhackrf** trực tiếp (đã OK trên máy bạn).
`SoapySDRUtil` trong **conda** chỉ tìm module ở `~/anaconda3/envs/gnss_env/lib/SoapySDR/` (trống),
không tự quét `/usr/lib/.../libHackRFSupport.so` nếu chưa set `SOAPY_SDR_PLUGIN_PATH`.

Serial HackRF của bạn (từ `hackrf_info`):

```
000000000000000024b862dc31264fcb
```

Chạy với serial:

```bash
export SOAPY_SDR_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/SoapySDR/modules0.8
python phat_tin_hieu_hackrf.py \
    --prn_start 11 --prn_end 20 \
    --freq_hz 433e6 \
    --serial 000000000000000024b862dc31264fcb
```
