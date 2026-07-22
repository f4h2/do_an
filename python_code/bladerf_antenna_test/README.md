
## Cài đặt môi trường

Script sử dụng môi trường conda **gnss_env**.

### 0. Tạo môi trường conda (lần đầu)

```bash
# Tạo env Python 3.10
conda create -n gnss_env python=3.10 -y
conda activate gnss_env

# Thư viện Python cơ bản
pip install numpy matplotlib scipy flask pyzmq

# BladeRF Python binding (cần libbladeRF hệ thống trước)
pip install bladerf

# SoapySDR (HackRF) — tùy chọn nếu dùng HackRF
pip install SoapySDR
```

Driver hệ thống (Ubuntu/Debian):

```bash
# BladeRF
sudo add-apt-repository ppa:nuandllc/bladerf
sudo apt update
sudo apt install bladerf libbladerf-dev

# HackRF + Soapy module (nếu dùng HackRF)
sudo apt install hackrf libhackrf-dev soapysdr-tools soapysdr-module-hackrf
```

Quyền USB (một lần):

```bash
sudo usermod -aG plugdev $USER
# logout / login lại
```

### 1. Kích hoạt môi trường

```bash
conda activate gnss_env
```

### 2. Cài thư viện Python cần thiết

```bash
pip install bladerf numpy matplotlib scipy flask
```

### 3. Kiểm tra BladeRF nhận diện được

```bash
bladeRF-cli -p
# Kết quả mong muốn:
#   [USB] Serial: xxxx  FW: x.x.x  FPGA: x.x.x
```




```bash
#python phat_thu_bladerf_gui.py --freq_hz 433e6 \
#    --tx_prn_start 11 --tx_prn_end 20 \
#    --prn1_start 11  --prn1_end 20 \
#    --prn2_start 21  --prn2_end 30
```

```PHát tín hiệu
python phat_tin_hieu_bladerf.py --prn_start 26 --prn_end 30 --freq_hz 433e6 --serial f444bf4f6a404a40a2ea650066b7e17c

python phat_tin_hieu_bladerf_2.py --prn_start 21 --prn_end 30 --freq_hz 433e6 --serial a0e5ffb5f1c28a2d57f5f5d9d13372ed

# python thu_tin_hieu_bladerf_gui_copy2.py --prn1_start 11 --prn1_end 20 --prn2_start 21 --prn2_end 30 --freq_hz 433e6 --serial 6ed84115bdca40800254de285ec1d898
```


## Web Newton — `web_newton_server.py`

```bash
conda activate gnss_env
pip install flask
```

``` THu tín hiệu
cd python_code/bladerf_antenna_test
conda activate gnss_env

python web_newton_server.py \
    --serial a0e5ffb5f1c28a2d57f5f5d9d13372ed \
    --freq_hz 433e6 \
    --tx_prn_ranges 11:15,16:20,26:30
```

Mở trình duyệt:

```
http://127.0.0.1:5000
```

