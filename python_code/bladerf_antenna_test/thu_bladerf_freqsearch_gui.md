# thu_bladerf_freqsearch_gui.py

Thu tín hiệu GNSS realtime từ BladeRF RX, **tự động tìm và bù lệch tần số (frequency search)** giữa hai thiết bị BladeRF có clock độc lập.

---

## Vấn đề giải quyết

Khi TX và RX là hai BladeRF riêng biệt, mỗi thiết bị có TCXO (Oscillator) riêng với sai số từ **5–20 ppm**. Ở tần số 433 MHz:

$$\Delta f = f_c \times \text{ppm\_error} \times 10^{-6}$$

Ví dụ 10 ppm → lệch **4.33 kHz**. Lệch tần này làm tín hiệu thu bị xoay pha liên tục:

$$s_{rx}(n) = s_{tx}(n) \cdot e^{j2\pi \Delta f \cdot n / f_s}$$

dẫn đến kết quả tương quan gần bằng 0, không ra đỉnh.

File này giải quyết bằng cách **quét nhiều giá trị ft**, chọn giá trị làm đỉnh tương quan cao nhất, rồi **bám theo** lệch tần đó theo thời gian thực.

---

## Thuật toán Frequency Search

### Bước 1 — Coarse Search (chunk đầu tiên và mỗi 20 chunk)

Quét toàn bộ dải `[-ft_range, +ft_range]` Hz theo bước `ft_step` Hz:

```
ft_candidates = [-20000, -19800, ..., 0, ..., +19800, +20000]  Hz
```

Với mỗi `ft` ứng viên, bù tần số cho tín hiệu thu bằng cách **dịch phổ trong miền tần số**:

$$F_{IQ\_shifted}[k] = F_{IQ}\left[k - \text{round}\left(\frac{f_t}{f_s/N_{fft}}\right)\right]$$

Đây là phép `np.roll` trên mảng FFT — **không cần tính FFT lại**, chỉ dịch bin — nên rất nhanh.

Tính tương quan chéo với mã tham chiếu nội bộ:

$$R(\tau) = \left| \text{IFFT}\left( F_{local}^* \cdot F_{IQ\_shifted} \right) \right|$$

Chọn `ft_best = argmax over ft_candidates of max(R(τ))`.

### Bước 2 — Refine Search

Quét chi tiết trong `[ft_best - ft_step, ft_best + ft_step]` với bước `ft_step / 10` để xác định chính xác hơn.

### Bước 3 — EMA Tracking

Cập nhật ước lượng lệch tần bằng bộ lọc EMA (Exponential Moving Average):

$$f_{t,est}[n] = \alpha \cdot f_{t,best} + (1 - \alpha) \cdot f_{t,est}[n-1]$$

- `α` nhỏ (0.05–0.15): hội tụ chậm, ổn định, ít nhiễu
- `α` lớn (0.3–0.5): hội tụ nhanh, nhạy hơn với thay đổi

### Bước 4 — Fine Search (các chunk tiếp theo)

Chỉ quét trong `[ft_est - 2×ft_step, ft_est + 2×ft_step]` với bước `ft_step / 5` → tốc độ cao hơn nhiều so với coarse search.

### Bước 5 — Tương quan nhóm 2

Sau khi có `ft_est`, áp dụng ngay cho nhóm PRN thứ hai mà không cần search thêm.

---

## Sơ đồ luồng xử lý

```
BladeRF RX (thread riêng)
        │
        ▼ chunk IQ (Nchunk mẫu)
    FFT(IQ)  →  F_IQ_base
        │
        ├──[coarse/fine]── np.roll(F_IQ_base, k_ft) cho mỗi ft ứng viên
        │                        │
        │              IFFT(F_local1* × F_IQ_shifted) → mag1
        │                        │
        │                  argmax(mag1) → ft_best
        │
        ├── EMA update → ft_est
        │
        ├── np.roll(F_IQ_base, k_est) → F_IQ_shift
        │        │
        │   IFFT(F_local2* × F_IQ_shift) → mag2
        │
        └── Vẽ GUI: mag1, mag2, ft_history, TDOA
```

---

## GUI

Gồm 3 subplot:

| Subplot | Nội dung |
|---|---|
| Trên | Đỉnh tương quan nhóm 1 (PRN 11–20), hiển thị tau, magnitude, ft tìm được |
| Giữa | Đỉnh tương quan nhóm 2 (PRN 21–30), hiển thị tau, magnitude, ft_est |
| Dưới | Lịch sử `ft_est` theo thời gian — quan sát quá trình hội tụ |

Tiêu đề cửa sổ cập nhật realtime: `ft_est` và giá trị TDOA tính được.

---

## Tham số

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--freq_hz` | 1575.42e6 | Tần số RF sóng mang (Hz) |
| `--fs` | 2e6 | Sample rate (Hz) |
| `--Rc` | 1.023e6 | Chip rate (Hz) |
| `--rx_gain` | 60 | Gain RX (dB) |
| `--prn1_start` | 11 | PRN bắt đầu nhóm 1 |
| `--prn1_end` | 20 | PRN kết thúc nhóm 1 |
| `--prn2_start` | 21 | PRN bắt đầu nhóm 2 |
| `--prn2_end` | 30 | PRN kết thúc nhóm 2 |
| `--chunk_ms` | 10.0 | Kích thước chunk (ms). Lớn hơn → phân giải tần tốt hơn |
| `--ft_range` | 20000 | Bán kính tìm kiếm tần số ban đầu (Hz) |
| `--ft_step` | 200 | Bước tìm kiếm tần số thô (Hz) |
| `--ft_refine` | True | Bật tinh chỉnh sau coarse search |
| `--ft_ema_alpha` | 0.15 | Hệ số EMA bám ft |
| `--serial` | a0e5ffb5... | Serial BladeRF RX |
| `--save` | _(trống)_ | Lưu IQ thu ra file .bin |
| `--file` | _(trống)_ | Đọc từ file .bin thay vì BladeRF (debug) |

> **Lưu ý `--chunk_ms`**: Phân giải tần số là $\Delta f = f_s / N_{fft} = f_s / (2 \times N_{chunk})$.  
> Ở `fs=2e6`, `chunk_ms=10` → $N_{chunk}=20000$, $N_{fft}=40000$ → $\Delta f = 50$ Hz/bin.  
> `--ft_step` nên ≥ $\Delta f$ để có ý nghĩa.

---

## Lệnh chạy

### Trường hợp thông thường — 433 MHz

```bash
# Terminal 1: Phát
python phat_tin_hieu_bladerf.py \
    --freq_hz 433e6 \
    --prn_start 11 --prn_end 20 \
    --serial 6ed84115bdca40800254de285ec1d898

# Terminal 2: Thu với frequency search
python thu_bladerf_freqsearch_gui.py \
    --freq_hz 433e6 \
    --prn1_start 11 --prn1_end 20 \
    --prn2_start 21 --prn2_end 30 \
    --serial a0e5ffb5f1c28a2d57f5f5d9d13372ed
```

### GPS L1 — 1575.42 MHz

```bash
# Terminal 1
python phat_tin_hieu_bladerf.py \
    --freq_hz 1575.42e6 \
    --prn_start 11 --prn_end 20 \
    --serial 6ed84115bdca40800254de285ec1d898

# Terminal 2
python thu_bladerf_freqsearch_gui.py \
    --freq_hz 1575.42e6 \
    --serial a0e5ffb5f1c28a2d57f5f5d9d13372ed
```

### Tìm kiếm dải rộng hơn (TCXO sai số lớn)

```bash
python thu_bladerf_freqsearch_gui.py \
    --freq_hz 433e6 \
    --ft_range 50000 \
    --ft_step 500 \
    --chunk_ms 20 \
    --ft_ema_alpha 0.2 \
    --serial a0e5ffb5f1c28a2d57f5f5d9d13372ed
```

### Debug từ file .bin đã thu

```bash
python thu_bladerf_freqsearch_gui.py \
    --file captured_iq.bin \
    --freq_hz 433e6
```

### Lưu IQ đồng thời với hiển thị

```bash
python thu_bladerf_freqsearch_gui.py \
    --freq_hz 433e6 \
    --save rx_iq_log.bin \
    --serial a0e5ffb5f1c28a2d57f5f5d9d13372ed
```

---

## Điều chỉnh khi không ra tương quan

1. **Tăng `--ft_range`** nếu lệch tần lớn hơn 20 kHz (TCXO kém chất lượng).
2. **Tăng `--chunk_ms`** (ví dụ 20–50 ms) để tăng SNR tương quan.
3. **Tăng `--rx_gain`** nếu tín hiệu yếu.
4. **Giảm `--ft_step`** nếu đỉnh tương quan xuất hiện nhưng không ổn định.
5. **Giảm `--ft_ema_alpha`** (ví dụ 0.05) nếu ft_est dao động quá mạnh.

---

## Yêu cầu

- Python ≥ 3.10
- `numpy`, `matplotlib`
- `bladerf` Python bindings (libbladeRF)
- `gnss_utils.py` trong thư mục cha (`../gnss_utils.py`)
