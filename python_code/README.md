# Hướng dẫn Cài đặt và Chạy code Python GNSS & BladeRF

Thư mục này chứa toàn bộ mã nguồn đã được chuyển đổi từ MATLAB sang Python nhằm mô phỏng hệ thống định vị GNSS và xử lý tín hiệu theo thời gian thực với **GNU Radio** và **BladeRF x40**.

## 1. Hướng dẫn thiết lập môi trường (Conda)

Để chạy mã nguồn, chúng ta cần tạo một môi trường Python ảo riêng biệt bằng Anaconda (hoặc Miniconda) để cài đặt các thư viện cần thiết.

**Bước 1:** Mở terminal (dòng lệnh) và tạo môi trường conda mới tên là `gnss_env` (với Python 3.10):
```bash
conda create -n gnss_env python=3.10 -y
```

**Bước 2:** Kích hoạt môi trường vừa tạo:
```bash
conda activate gnss_env
```

**Bước 3:** Cài đặt các thư viện cần thiết (Numpy, SciPy, ZeroMQ):
```bash
pip install numpy scipy pyzmq
```

**Bước 4 (Tuỳ chọn nếu chạy file `bladerf_rt.py` độc lập không qua GNU Radio):**
Cài đặt thư viện python binding cho BladeRF:
```bash
pip install bladerf
```

---

## 2. Danh sách các file code và Cách chạy

Đảm bảo bạn đã kích hoạt môi trường conda (`conda activate gnss_env`) và đang đứng ở thư mục chứa code `python_code` trước khi chạy các lệnh sau.

### Các file Mô phỏng tĩnh (Offline/File-based)
Các file này dùng để sinh dữ liệu thô và lưu ra file `.bin`, hoặc đọc từ file `.bin` để xử lý. Nó tương đương với thao tác chạy trên MATLAB.

- **Sinh tín hiệu phát:**
  Chạy mô phỏng tạo mã C/A và lưu file tín hiệu `data_2506_phat_ca_new2_8M.bin`.
  ```bash
  python phat_tin_hieu.py
  ```

- **Mô phỏng thu tín hiệu:**
  Đọc file bin giả định và tính toán thời gian trễ T1, T2.
  ```bash
  python thu_tin_hieu.py
  ```

- **Truyền - Nhận nhiều trạm (Receiver/Transmit):**
  Mô phỏng tính toán vị trí theo thuật toán bình phương tối thiểu.
  ```bash
  python transmit.py
  python receiver.py
  ```

- **Điều chế và Giải điều chế BPSK:**
  ```bash
  python bpsk_mod.py
  python bpsk_demod.py
  ```

---

### Các file Xử lý Thời gian thực (Real-time) với SDR

1. **`thu_tin_hieu_rt.py` (Khuyên dùng)**
   Đây là script nhận luồng dữ liệu liên tục từ **GNU Radio** thông qua ZeroMQ (ZMQ).
   - **Cách setup:** Trong GNU Radio Companion, bạn mở sơ đồ khối, gán đầu ra (màu xanh nước biển - Complex) vào khối `ZMQ PUB Sink` với địa chỉ `tcp://127.0.0.1:5555`. 
   - **Cách chạy:** Bấm Play trên GNU Radio để bắt đầu thu từ BladeRF x40, sau đó chạy lệnh sau:
   ```bash
   python thu_tin_hieu_rt.py
   ```

2. **`bladerf_rt.py`**
   Đây là file mẫu kết nối thẳng Python với BladeRF x40 (không cần dùng qua GNU Radio).
   - Chỉ dùng file này nếu bạn muốn kiểm soát hoàn toàn việc ghi/đọc phần cứng BladeRF từ Python.
   - **Cách chạy:**
   ```bash
   python bladerf_rt.py
   ```

---

## 3. Cấu trúc mã dùng chung
- **`gnss_utils.py`**: File thư viện tự viết. Chứa hàm `generateCAcode(prn)` để tạo chuỗi mã GPS C/A 1023 bit và hàm `calcDistance(lat1, lon1, lat2, lon2)` để tính khoảng cách địa lý (Haversine). Các file khác đều gọi hàm từ file này.


python thu_tin_hieu_rt_gui_freqsearch.py --mode ZMQ --address tcp://127.0.0.1:5555 --f-min -5000 --f-max 5000


python3 phat_tin_hieu_rt_zmq.py --mode ZMQ --address tcp://127.0.0.1:5556 --pace --repeat
INPUT_MODE=ZMQ python3 thu_tin_hieu_rt_gui.py