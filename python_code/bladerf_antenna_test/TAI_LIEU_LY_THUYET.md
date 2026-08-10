# Tài liệu lý thuyết & triển khai hệ thống định vị GNSS mặt đất

**Dự án:** Hệ thống phát/thu RF thật qua SDR, dùng mã C/A GPS để đo khoảng cách và định vị  
**Thư mục trọng tâm:** `python_code/bladerf_antenna_test/`  
**Phần cứng:** BladeRF x40, HackRF One — thu–phát RF thật qua ăng ten  
**Ngôn ngữ:** Python (NumPy, libbladeRF, SoapySDR, Flask)

---

## Mục lục

1. [Tổng quan hệ thống](#1-tổng-quan-hệ-thống)
2. [Lý thuyết GNSS liên quan dự án](#2-lý-thuyết-gnss-liên-quan-dự-án)
3. [Thiết bị phần cứng](#3-thiết-bị-phần-cứng)
4. [Cấu trúc tín hiệu trong dự án](#4-cấu-trúc-tín-hiệu-trong-dự-án)
5. [Phương pháp phát tín hiệu (TX)](#5-phương-pháp-phát-tín-hiệu-tx)
6. [Phương pháp thu & tương quan (RX)](#6-phương-pháp-thu--tương-quan-rx)
7. [Bù lệch tần số (Frequency Search)](#7-bù-lệch-tần-số-frequency-search)
8. [Định vị: TDOA, trilateration, Newton](#8-định-vị-tdoa-trilateration-newton)
9. [Hiệu chuẩn đồng bộ \(\Delta_m\)](#9-hiệu-chuẩn-đồng-bộ-delta_m)
10. [Kiến trúc triển khai thực tế](#10-kiến-trúc-triển-khai-thực-tế)
11. [Mô tả các script trong `bladerf_antenna_test`](#11-mô-tả-các-script)
12. [Tham số mặc định & khuyến nghị](#12-tham-số-mặc-định--khuyến-nghị)
13. [Quy trình thí nghiệm đề xuất](#13-quy-trình-thí-nghiệm-đề-xuất)
14. [Hạn chế & hướng phát triển](#14-hạn-chế--hướng-phát-triển)

---

## 1. Tổng quan hệ thống

Đây là hệ thống **định vị mặt đất (terrestrial ranging)** dựa trên nguyên lý GNSS:

- Các trạm phát (TX) phát **tín hiệu RF thật** mang mã C/A GPS (BPSK).
- Trạm thu (RX) thu sóng thật qua ăng ten / BladeRF, tương quan mã để ước lượng độ trễ.
- Từ độ trễ suy ra khoảng cách / TDOA / pseudorange, rồi giải vị trí.

Hệ thống **không** dùng vệ tinh GPS trên quỹ đạo; các “vệ tinh” được thay bằng **trạm phát SDR** đã biết toạ độ. Toàn bộ đường truyền là **OTA (over-the-air)** hoặc full-duplex trên phần cứng thật.

Luồng xử lý:

1. **Sinh mã C/A GPS** (PRN 1–32) trên máy tính.
2. **Điều chế BPSK** ở baseband, đẩy lên RF qua SDR (BladeRF / HackRF).
3. **Thu bằng SDR** (board khác, hoặc cùng BladeRF full-duplex TX1+RX1).
4. **Tương quan** với mã cục bộ để ước lượng độ trễ \(\tau\).
5. **Chuyển \(\tau\) → khoảng cách / TDOA / pseudorange**.
6. **Giải vị trí** bằng trilateration 2D hoặc **Newton** (2D + clock bias).

Mục tiêu ứng dụng: định vị trong phòng thí nghiệm / khu vực nhỏ, dùng sóng ISM **433 MHz** (hoặc GPS L1 **1575.42 MHz** khi có môi trường chắn và được phép phát).

```
  TX1 (BladeRF/HackRF) ──PRN nhóm A──┐
  TX2 (BladeRF)        ──PRN nhóm B──┼──► không gian RF ──► RX (BladeRF)
  TX3 (BladeRF)        ──PRN nhóm C──┘                      │
                                                            ▼
                                              Tương quan → τ₁,τ₂,τ₃
                                                            ▼
                                         TDOA / Pseudorange → Newton (x,y,b)
                                                            ▼
                                              GUI / Web (Chart.js, Flask)
```

---

## 2. Lý thuyết GNSS liên quan dự án

### 2.1. Mã C/A (Coarse/Acquisition)

- Mỗi vệ tinh (PRN) có một chuỗi giả ngẫu nhiên **1023 chip**.
- Tốc độ chip: \(R_c = 1{,}023\,\mathrm{Mcps}\) → chu kỳ mã \(\approx 1\,\mathrm{ms}\).
- Trong code: `gnss_utils.generateCAcode(prn)` trả về mảng \(\pm 1\) (ánh xạ bit 0→+1, 1→−1).
- PRN hợp lệ: **1–32**.

Tính chất quan trọng dùng trong dự án:

| Tính chất | Ý nghĩa thực nghiệm |
|-----------|---------------------|
| Tự tương quan cao khi \(\tau=0\) | Xuất hiện **đỉnh** khi mã khớp |
| Tương quan chéo thấp giữa PRN khác nhau | Nhiều TX phát cùng tần số RF vẫn tách được bằng PRN |
| Chu kỳ 1 ms | Cửa sổ tương quan thường lấy vài ms dữ liệu |

### 2.2. Điều chế BPSK

Tín hiệu baseband (dự án):

\[
s(n) = A \cdot c\!\left[\left\lfloor\frac{(n+\tau)\,R_c}{f_s}\right\rfloor \bmod L\right]
\cdot e^{j\,2\pi\,f_t\,n/f_s}
\]

trong đó:

- \(A\): biên độ (BladeRF thường 1024 trong miền SC16_Q11).
- \(c[\cdot]\): mã C/A (có thể **ghép nhiều PRN** liên tiếp).
- \(\tau\): độ trễ code (samples) — trễ chủ động trên tín hiệu phát (đồng bộ / thí nghiệm).
- \(f_t\): offset Doppler baseband (Hz).
- \(L = (prn\_end - prn\_start + 1)\times 1023\): độ dài chuỗi chip sau khi ghép.

Sau đó SDR lên sóng mang RF \(f_c\) (ví dụ 433 MHz).

### 2.3. Pseudorange & TDOA

**Pseudorange** (khoảng cách giả):

\[
P = \frac{\tau}{f_s}\,c,\quad c = 299\,792\,458\,\mathrm{m/s}
\]

Do TX/RX không đồng bộ tuyệt đối (không PPS chung), \(P\) chứa **sai số đồng hồ** và sai số đồng bộ giữa các TX. Vì vậy dự án dùng:

- **TDOA** (chênh lệch thời gian đến giữa 2 TX) để triệt một phần bias, hoặc
- **Newton** với biến phụ \(b\) (clock bias) khi có \(\ge 3\) TX.

### 2.4. Vì sao phải quét tần số?

Hai BladeRF độc lập dùng thạch anh riêng → lệch tần số tổng có thể hàng kHz đến chục kHz. Trong cửa sổ tương quan ngắn, pha xoay làm **triệt đỉnh** nếu chỉ bù một \(f_t\) cố định. Frequency search thử nhiều \(f_{\mathrm{candidate}}\) và chọn đỉnh lớn nhất.

---

## 3. Thiết bị phần cứng

### 3.1. BladeRF x40 (chính)

| Hạng mục | Chi tiết trong dự án |
|----------|----------------------|
| Vai trò | TX và/hoặc RX |
| API | `bladerf` / `_bladerf` (libbladeRF) |
| Kênh | `CHANNEL_TX(0)`, `CHANNEL_RX(0)` |
| Định dạng IQ | **SC16_Q11**: int16 xen kẽ `[I0,Q0,I1,Q1,...]`, clip \(\pm 2047\) |
| Streaming | 16 buffer × 8192, 8 transfers, timeout 3500 ms |
| Bandwidth | \(B \approx 0{,}8\,f_s\) |
| Phân biệt thiết bị | `--serial <hex>` |

Cổng ăng ten điển hình: **TX1**, **RX1**. Khi full-duplex trên một board, dùng cả hai cổng + hai ăng ten vật lý.

### 3.2. HackRF One (phụ — TX)

| Hạng mục | Chi tiết |
|----------|----------|
| Script | `phat_tin_hieu_hackrf.py` |
| API | SoapySDR (`driver=hackrf`) |
| Định dạng | CF32 (complex float32), biên độ sau chuẩn hoá \(\approx 0{,}85\) |
| Gain TX | khoảng 0–47 dB (mặc định ~30) |
| Lưu ý conda | Cần `SOAPY_SDR_PLUGIN_PATH` trỏ module hệ thống nếu SoapySDR cài trong conda |

### 3.3. Bố trí ăng ten

- Cùng phân cực (đứng–đứng hoặc ngang–ngang).
- Khoảng cách thực nghiệm thường **vài mét đến vài chục mét** trong lab; loopback gần: **10–50 cm**.
- Tránh ăng ten chạm nhau (bão hòa ADC).
- Gain TX/RX: tăng dần; nếu clipping → giảm TX gain.

### 3.4. Topology triển khai

| Topology | Script tiêu biểu |
|----------|------------------|
| 1 board TX+RX | `phat_thu_bladerf_gui.py`, `bladerf_loopback_test.py` |
| 1 TX + 1 RX (2 board) | `phat_tin_hieu_bladerf.py` + `thu_tin_hieu_bladerf_gui*.py` |
| Nhiều TX + 1 RX | Nhiều tiến trình TX + `web_newton_server.py` / `thu_tin_hieu_bladerf_newton.py` |
| HackRF TX + BladeRF RX | `phat_tin_hieu_hackrf.py` + script thu BladeRF |

Mỗi TX dùng **nhóm PRN không chồng** để máy thu tách kênh bằng tương quan.

---

## 4. Cấu trúc tín hiệu trong dự án

### 4.1. Tham số chuẩn thực nghiệm (README)

| Tham số | Giá trị thường dùng |
|---------|---------------------|
| \(f_c\) | **433 MHz** (lab); mặc định argparse nhiều file là L1 1575.42 MHz |
| \(f_s\) | **2 MHz** (BladeRF realtime) |
| \(R_c\) | 1.023 MHz |
| \(f_t\) | 0 Hz (có thể bù / quét ở RX) |
| \(\tau\) TX | 40 samples (mặc định) |
| Biên độ BladeRF | 1024 (SC16_Q11) |
| Chunk RX | 5 ms → \(N = 10\,000\) mẫu @ 2 MHz |
| FFT length | \(N_{\mathrm{fft}} = 2N\) (zero-pad) |

### 4.2. Phân nhóm PRN (ví dụ thực tế)

| Trạm | Dải PRN (ví dụ) |
|------|-----------------|
| TX1 | 11–15 hoặc 11–20 |
| TX2 | 16–20 hoặc 21–30 |
| TX3 | 26–30 |
| Web Newton (3 TX) | `11:15,16:20,26:30` |
| Newton desktop (4 TX) | `11:20,21:30,31:32,1:10` |

### 4.3. So sánh với bản MATLAB tham chiếu (`transmit.m` / `receiver.m`)

Các file MATLAB trong repo dùng để **kiểm chứng thuật toán** trên file IQ offline (`test.bin`). Nhánh `bladerf_antenna_test` là triển khai **phần cứng thật**:

| | MATLAB | `bladerf_antenna_test` |
|--|--------|------------------------|
| \(f_s\) | 10 MHz | 2 MHz (mặc định realtime) |
| Số TX | 3 (PRN 1,2,3) | 2 / 3 / 4 tùy script |
| IQ | Chủ yếu trục I | Complex IQ đầy đủ |
| Tương quan | Time-domain \(\lvert\sum\rvert^2\) | Chủ yếu **FFT** |
| Định vị | Newton 3 TX | TDOA 2 TX hoặc Newton 3–4 TX |

---

## 5. Phương pháp phát tín hiệu (TX)

### 5.1. Luồng xử lý

```
generateCAcode(PRN_i)  →  ghép PRN_start…PRN_end
        ↓
resample theo fs, Rc + áp τ
        ↓
nhân e^{j2π ft n/fs}  (BPSK + Doppler baseband)
        ↓
complex64 → SC16_Q11 (BladeRF) hoặc CF32 (HackRF)
        ↓
sync_tx / writeStream  (lặp vòng)
```

### 5.2. File chính

- `phat_tin_hieu_bladerf.py` / `phat_tin_hieu_bladerf_2.py`: phát BladeRF.
- `phat_tin_hieu_hackrf.py`: phát HackRF.
- `--save_file`: xuất IQ ra file để phát lại bằng GNU Radio hoặc đọc offline.

### 5.3. Lệnh điển hình

```bash
python phat_tin_hieu_bladerf.py \
    --prn_start 26 --prn_end 30 \
    --freq_hz 433e6 \
    --serial <TX_SERIAL>
```

---

## 6. Phương pháp thu & tương quan (RX)

### 6.1. Thu realtime BladeRF

1. Cấu hình \(f_c, f_s\), gain RX.
2. `sync_rx` từng chunk int16 SC16_Q11.
3. Chuyển sang `complex64`.
4. (Tuỳ chọn) bù \(f_t\) hoặc frequency search.
5. Zero-pad → FFT → nhân với FFT mã cục bộ → IFFT → lấy \(\lvert\cdot\rvert\).
6. \(\tau = \arg\max\) của biên độ tương quan.

### 6.2. Tương quan FFT (công thức dùng trong code)

\[
\begin{align}
s'(n) &= s(n)\,e^{j 2\pi f_t n / f_s} \\
R(\tau) &= \left|\mathcal{F}^{-1}\!\big\{\,F_{\mathrm{local}}(\omega)\cdot F_s^*(\omega)\,\big\}\right| \\
\hat\tau &= \arg\max_\tau R(\tau)
\end{align}
\]

Mã cục bộ được resample một lần lên độ dài \(N_{\mathrm{fft}}\) rồi lưu FFT sẵn để realtime.

### 6.3. Tương quan time-domain (tuỳ chọn, giống `receiver.m`)

Dùng khi `--corr_mode time` (script Newton):

\[
X(\tau)=\left(\sum_{n} c(n+\tau)\,I(n)\right)^2
\]

chỉ trên thành phần I; chậm hơn FFT khi \(N\) lớn.

### 6.4. Độ phân giải khoảng cách

\[
\Delta r = \frac{c}{f_s}
\]

Với \(f_s=2\,\mathrm{MHz}\): \(\Delta r \approx 149{,}9\,\mathrm{m}\) **mỗi mẫu** — thô cho định vị mét. Trong thực tế dự án:

- dùng **chênh lệch** \(\tau\) giữa các TX + hình học đã biết,
- và/hoặc hiệu chuẩn \(\Delta_m\),
- chấp nhận sai số lớn nếu chỉ dựa vào một mẫu; cải thiện cần \(f_s\) cao hơn hoặc nội suy đỉnh.

> **Lưu ý quan trọng khi viết báo cáo:** độ phân giải mẫu ở 2 Msps không đủ để đo mét tuyệt đối chỉ bằng \(\tau\); kết quả thực nghiệm phụ thuộc mạnh vào hiệu chuẩn đồng bộ và hình học.

---

## 7. Bù lệch tần số (Frequency Search)

Script: `thu_bladerf_freqsearch_gui.py` (và logic tương tự trong các bản freqsearch khác).

### Ý tưởng

Với mỗi ứng viên \(f_c\):

1. Bù pha \(e^{j 2\pi f_c n/f_s}\) (hoặc dịch bin FFT).
2. Tính tương quan, lấy \(\max |R|\).
3. Chọn \(f^*\) cho đỉnh lớn nhất.
4. Theo dõi bằng EMA / khóa sau vài chunk để giảm tải CPU.

Tham số điển hình: dải \(\pm 20\,\mathrm{kHz}\), bước \(200\,\mathrm{Hz}\).

Không có frequency search, hai BladeRF độc lập thường **không thấy đỉnh** dù SNR RF đủ.

---

## 8. Định vị: TDOA, trilateration, Newton

### 8.1. Hình học địa lý → local XY

Gốc thường lấy tại TX1. Chuyển (lat, lon) sang mét:

\[
\begin{align}
x &= \mathrm{rad}(\Delta\lambda)\,\cos\!\big(\mathrm{rad}\tfrac{\varphi+\varphi_0}{2}\big)\,R \\
y &= \mathrm{rad}(\Delta\varphi)\,R,\quad R=6\,371\,000\,\mathrm{m}
\end{align}
\]

Khoảng cách thực (để kiểm chứng / \(T_0\)): Haversine (`calcDistance`).

### 8.2. TDOA hai trạm phát (GUI 2 kênh)

Với \(\tau_1,\tau_2\) và khoảng cách hình học \(T_1,T_2,D\):

\[
\begin{align}
T_0 &= \frac{T_1-T_2}{c}\,f_s \\
\Delta T &= (\tau_1-\tau_2)-T_0 \\
\Delta_m &= \frac{\Delta T}{f_s}\,c \\
\Delta_C &= \frac{\tau_1-\tau_2}{f_s}\,c \\
X &= \frac{-\Delta_C + D + \Delta_m}{2}\quad\text{(ước lượng RX–TX1)} \\
X_2 &= D - X
\end{align}
\]

Sau đó **trilateration 2D** với bán kính \(X,X_2\) quanh TX1, TX2; chọn nghiệm gần vị trí gợi ý (RX thật).

Biến thể `thu_tin_hieu_bladerf_gui_copy2.py`: **khóa** \(\Delta T\) lần đầu làm bias đồng bộ cố định.

### 8.3. Phương pháp Newton (giống `receiver.m`)

Trạng thái: \(\mathbf{p}=(x,y,b)^\top\) với \(b\) = bias đồng hồ (mét).

Quan sát: \(obs_i\) = pseudorange tới TX \(i\).

\[
\begin{align}
F_i &= obs_i - \sqrt{(X_i-x)^2+(Y_i-y)^2} - b \\
J_{i,:} &= \left[
  -\frac{X_i-x}{obs_i},\;
  -\frac{Y_i-y}{obs_i},\;
  1
\right] \\
\Delta\mathbf{p} &= J^{+}\,F,\quad
\mathbf{p} \leftarrow \mathbf{p}+\Delta\mathbf{p}
\end{align}
\]

- Số ẩn = 3 → cần **ít nhất 3 TX**.
- `thu_tin_hieu_bladerf_newton.py`: **4 TX** (thừa phương trình, `lstsq`).
- `web_newton_server.py`: **3 TX** + hiệu chuẩn \(\Delta_m\).
- Số vòng lặp mặc định: **10**.
- Khởi tạo: gần vị trí RX gợi ý, \(b=0\).

Công thức Jacobian dùng \(obs_i\) ở mẫu số (giống MATLAB gốc); khi \(obs_i\approx 0\) code gán mẫu số = 1 để tránh chia 0.

### 8.4. Pseudorange từ delay (hai cách trong repo)

**Cách A — tuyệt đối (BladeRF Newton / Web):**

\[
obs_i = \frac{\tau_i}{f_s}\,c \;\;(+\;\Delta_{m,i}\;\text{nếu đã hiệu chuẩn})
\]

**Cách B — tương đối (MATLAB `receiver.m`):**

\[
T_i=\max_j\tau_j-\tau_i,\quad
P_i=\frac{5+T_i}{f_s}\,c
\]

(hằng số `5` là bias mẫu trong bản MATLAB tham chiếu).

---

## 9. Hiệu chuẩn đồng bộ \(\Delta_m\)

Áp dụng trong **`web_newton_server.py`**.

### Vấn đề

Các TX không dùng chung PPS/clock → đỉnh \(\tau\) mang **offset thời gian cố định** giữa các máy phát, không chỉ do khoảng cách hình học.

### Giả thiết hiệu chuẩn

Người dùng biết khoảng cách thật RX→TX (thước đo / bố trí lab). Với mỗi cặp \((a,b)\):

\[
\begin{align}
T_0 &= \frac{d_{RX\to a}-d_{RX\to b}}{c}\,f_s \\
\Delta T_{\mathrm{fixed}} &= (\tau_a-\tau_b)-T_0 \\
\Delta_m &= \frac{\Delta T_{\mathrm{fixed}}}{f_s}\,c
\end{align}
\]

Gán \(\Delta_{m,a}=0\), \(\Delta_{m,b}=\Delta_m\) (thường neo theo TX1).

### Định vị sau hiệu chuẩn

\[
obs_i = \frac{\tau_i}{f_s}\,c + \Delta_{m,i}
\]

rồi chạy Newton.

### Luồng web

1. `/` — trang **calibration**: nhập khoảng cách → **Khóa \(\Delta_m\)**.
2. Chuyển **positioning**.
3. `/positioning` — cập nhật hình học TX/RX, xem tương quan + \((x,y,b)\).

API chính: `/api/status`, `/api/calibration/distances`, `/api/calibration/lock`, `/api/geometry`, `/api/next`.

---

## 10. Kiến trúc triển khai thực tế

### 10.1. Môi trường phần mềm

- Conda env: `gnss_env` (Python 3.10 khuyến nghị).
- Python: `numpy`, `matplotlib`, `scipy`, `flask`, `bladerf`, (tuỳ chọn) `SoapySDR`, `pyzmq`.
- Hệ thống: `bladerf`, `libbladerf-dev`; HackRF: `hackrf`, `soapysdr-module-hackrf`.
- User thuộc group `plugdev`.

Chi tiết lệnh: xem `README.md` cùng thư mục.

### 10.2. Luồng realtime điển hình (3 TX + Web Newton)

```
Terminal TX1:  phat_tin_hieu_bladerf.py   --prn_start … --serial <TX1>
Terminal TX2:  phat_tin_hieu_bladerf_2.py --prn_start … --serial <TX2>
Terminal TX3:  phat_tin_hieu_bladerf.py   --prn_start … --serial <TX3>
Terminal RX:   web_newton_server.py --serial <RX> --freq_hz 433e6 \
                 --tx_prn_ranges 11:15,16:20,26:30
Trình duyệt:   http://127.0.0.1:5000
```

Tất cả TX/RX phải cùng \(f_c\) và \(f_s\).

### 10.3. Threading

- RX: thread đọc BladeRF → queue chunk.
- Xử lý: thread/animation loop tương quan + Newton.
- Web: Flask `threaded=True` + background `processing_loop`.

---

## 11. Mô tả các script

| File | Vai trò |
|------|---------|
| `phat_tin_hieu_bladerf.py` | Phát C/A BPSK qua BladeRF |
| `phat_tin_hieu_bladerf_2.py` | Bản copy cho TX thứ hai (serial khác) |
| `phat_tin_hieu_hackrf.py` | Phát qua HackRF (SoapySDR) |
| `phat_thu_bladerf_gui.py` | 1 board vừa TX vừa RX + GUI tương quan |
| `thu_tin_hieu_bladerf_gui.py` | Thu 2 nhóm PRN, in TDOA |
| `thu_tin_hieu_bladerf_gui_copy.py` | Như trên + nhập toạ độ + trilateration |
| `thu_tin_hieu_bladerf_gui_copy2.py` | Khóa bias \(\Delta T\) lần đầu |
| `thu_bladerf_freqsearch_gui.py` | Thu + quét lệch tần số |
| `thu_tin_hieu_bladerf_newton.py` | 4 TX / 1 RX + Newton GUI |
| `web_newton_server.py` | Web: hiệu chuẩn \(\Delta_m\) + Newton 3 TX |
| `bladerf_loopback_test.py` | Loopback TX–RX một board |
| `debug_correlation.py` | Debug phổ + bản đồ freq×delay |
| `templates/*.html` | Giao diện calibration / positioning |
| `gnss_utils.py` (thư mục cha) | `generateCAcode`, `calcDistance` |

---

## 12. Tham số mặc định & khuyến nghị

### Argparse thường gặp

| Tham số | Mặc định code | Khuyến nghị lab |
|---------|---------------|-----------------|
| `--freq_hz` | 1575.42e6 (nhiều script) | **433e6** |
| `--fs` | 2e6 | 2e6 |
| `--Rc` | 1.023e6 | giữ nguyên |
| `--ft` | 0 | 0 + frequency search nếu cần |
| `--rx_gain` | 60 | chỉnh theo clipping |
| `--tx_gain` (BladeRF) | ~70 | 40–70 |
| `--chunk_ms` | 5 | 5–10 |
| `--newton_iters` | 10 | 10–20 |
| `--tau` (TX) | 40 | tuỳ thí nghiệm |

### Serial ví dụ (README — thay bằng thiết bị thực)

| Vai trò | Serial (ví dụ) |
|---------|----------------|
| TX | `f444bf4f6a404a40a2ea650066b7e17c`, `a0e5ffb5…` |
| RX | `a0e5ffb5f1c28a2d57f5f5d9d13372ed`, `6ed84115…` |
| HackRF | `000000000000000024b862dc31264fcb` |

---

## 13. Quy trình thí nghiệm đề xuất

### Bước 1 — Kiểm tra phần cứng

```bash
bladeRF-cli -p
# (nếu HackRF) hackrf_info
```

### Bước 2 — Loopback / 1 TX–1 RX

- Chạy `bladerf_loopback_test.py` hoặc `phat_thu_bladerf_gui.py`.
- Xác nhận có **đỉnh tương quan** rõ.

### Bước 3 — Hai board, có lệch tần

- TX: `phat_tin_hieu_bladerf.py`
- RX: `thu_bladerf_freqsearch_gui.py`
- Ghi nhận \(f^*\) tìm được.

### Bước 4 — Hai TX, TDOA

- Hai tiến trình TX PRN khác nhau.
- RX: `thu_tin_hieu_bladerf_gui_copy.py`
- So sánh \(X,X_2\) với khoảng cách thước đo.

### Bước 5 — Ba TX, Web Newton

- Ba TX + `web_newton_server.py`
- Hiệu chuẩn \(\Delta_m\) với khoảng cách đo tay.
- Chuyển positioning, ghi \((x,y)\), sai số so với RX thật.

### Bước 6 — Báo cáo

Ghi rõ: \(f_c,f_s\), PRN từng TX, serial, gain, khoảng cách hình học, có/không frequency search & \(\Delta_m\), sai số (m).

---

## 14. Hạn chế & hướng phát triển

| Hạn chế hiện tại | Hướng xử lý |
|------------------|-------------|
| \(f_s=2\,\mathrm{MHz}\) → phân giải khoảng ~150 m/mẫu | Tăng \(f_s\) (5–10 MHz) hoặc nội suy đỉnh tương quan |
| Không dùng PPS / clock chung giữa TX | Hiệu chuẩn \(\Delta_m\); dài hạn: PPS / ref clock |
| Lệch TCXO giữa board | Frequency search / PLL mềm |
| Không dùng vệ tinh GPS trên quỹ đạo | Đây là hệ **GNSS mặt đất**: TX SDR phát mã C/A qua RF thật |
| Jacobian Newton dùng \(obs_i\) như MATLAB | Có thể chuẩn hoá lại theo khoảng cách hình học ước lượng |
| Đa đường / phản xạ trong phòng | Chọn PRN, bố trí LOS, lọc đỉnh phụ |

---

## Phụ lục A — Hằng số vật lý & công thức nhanh

\[
c = 299\,792\,458\,\mathrm{m/s},\quad
R_c = 1{,}023\times 10^6,\quad
N_{\mathrm{CA}}=1023
\]

\[
\text{Thời gian 1 chip}=\frac{1}{R_c}\approx 977\,\mathrm{ns}
\approx 293\,\mathrm{m}
\]

\[
\text{Thời gian 1 mẫu}=\frac{1}{f_s}\;\Rightarrow\;
\text{mét/mẫu}=\frac{c}{f_s}
\]

---

## Phụ lục B — Quan hệ với file MATLAB gốc

| MATLAB | Python tương đương |
|--------|--------------------|
| `transmit.m` | Logic trong `phat_tin_hieu_bladerf.py` (nhưng complex + SDR) |
| `receiver.m` | `python_code/receiver.py`, mở rộng trong `thu_tin_hieu_bladerf_newton.py` |
| `generateCAcode` | `gnss_utils.generateCAcode` |
| `calcDistance.m` | `gnss_utils.calcDistance` |

---

## Phụ lục C — Tài liệu kèm theo

- `README.md` — lệnh chạy, cài env, serial.
- `thu_bladerf_freqsearch_gui.md` — ghi chú frequency search (nếu có).
- Code nguồn trong thư mục này là nguồn chân lý cho tham số triển khai.

---

*Tài liệu được biên soạn theo hiện trạng mã nguồn trong `python_code/bladerf_antenna_test/` và các file MATLAB/Python liên quan của đồ án.*
