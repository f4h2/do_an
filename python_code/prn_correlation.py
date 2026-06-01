import numpy as np
import matplotlib.pyplot as plt
import os
from gnss_utils import generateCAcode

def analyze_captured_signal():
    # --- Tham số cấu hình ---
    fs = 2e6          # Tần số lấy mẫu (Hz)
    Rc = 1.023e6      # Chip rate (Hz)
    filename = 'tin_hieu_thu.bin'
    prn_range = range(11, 21) # PRN từ 11 đến 20
    duration_ms = 10  # Độ dài mỗi khối tương quan (ms)
    
    if not os.path.exists(filename):
        print(f"LỖI: Không tìm thấy file {filename}")
        return

    # Tính toán số mẫu cho 10ms
    samples_per_10ms = int(fs * duration_ms / 1000) # 20,000 mẫu
    file_size = os.path.getsize(filename)
    total_samples = file_size // 8 # complex64 = 8 bytes
    
    print(f"--- PHÂN TÍCH TÍN HIỆU THU ĐƯỢC ---")
    print(f"File: {filename}")
    print(f"Tổng số mẫu trong file: {total_samples} ({total_samples/fs*1000:.2f} ms)")
    
    # 1. Tạo mã Local Reference (Kết hợp PRN 11-20)
    print(f"Đang tạo mã CA local cho PRN {prn_range.start}-{prn_range.stop-1}...")
    combined_ca = np.concatenate([generateCAcode(i) for i in prn_range])
    total_chips = len(combined_ca) # 10230 chips
    
    # Resample mã local sang fs
    n_local = np.arange(samples_per_10ms)
    idx_local = np.floor(n_local / fs * Rc).astype(int) % total_chips
    local_ref = combined_ca[idx_local].astype(np.complex64)
    
    # FFT của mã local (conjugate để dùng cho cross-correlation)
    f_local = np.fft.fft(local_ref)

    # 2. Đọc và xử lý dữ liệu
    # Chúng ta sẽ lấy 10ms đầu tiên để phân tích
    try:
        captured_data = np.fromfile(filename, dtype=np.complex64, count=samples_per_10ms)
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")
        return

    if len(captured_data) < samples_per_10ms:
        print(f"Cảnh báo: File quá ngắn, chỉ có {len(captured_data)} mẫu.")
        samples_per_10ms = len(captured_data)
        local_ref = local_ref[:samples_per_10ms]
        f_local = np.fft.fft(local_ref)

    # 3. Tính toán tương quan (Combined)
    print("Đang tính toán tương quan FFT (Combined PRN 11-20)...")
    f_captured = np.fft.fft(captured_data)
    # corr = IFFT( FFT(local) * conj(FFT(captured)) )
    correlation = np.abs(np.fft.ifft(f_local * np.conj(f_captured)))
    
    peak_idx = np.argmax(correlation)
    peak_val = correlation[peak_idx]
    
    # 4. Tính toán tương quan cho từng PRN riêng lẻ (để kiểm tra độ mạnh yếu)
    print("Đang phân tích từng PRN riêng lẻ...")
    individual_peaks = []
    for prn in prn_range:
        ca = generateCAcode(prn)
        # Resample single PRN (1ms)
        samples_per_1ms = int(fs * 1 / 1000)
        idx_1ms = np.floor(np.arange(samples_per_1ms) / fs * Rc).astype(int) % 1023
        local_1ms = ca[idx_1ms].astype(np.complex64)
        
        # Tương quan với 1ms đầu của captured_data
        f_l1 = np.fft.fft(local_1ms, n=samples_per_10ms) # zero pad to match
        # Hoặc đơn giản là dùng correlate của numpy cho chính xác hơn ở đoạn nhỏ
        # Nhưng dùng FFT cho nhanh
        c_val = np.abs(np.fft.ifft(f_l1 * np.conj(f_captured)))
        individual_peaks.append(np.max(c_val))

    print(f"\n--- KẾT QUẢ ---")
    print(f"Đỉnh tương quan (Combined): {peak_val:.2f} tại mẫu {peak_idx}")
    print(f"Độ trễ thời gian: {(peak_idx/fs)*1000:.4f} ms")
    
    # 5. Vẽ đồ thị
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 10))
    
    # Subplot 1: Tương quan Combined
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    ax1.plot(correlation, color='#00d2ff', linewidth=0.8)
    ax1.plot(peak_idx, peak_val, 'ro', markersize=8, label=f'Peak at {peak_idx}')
    ax1.set_title(f"Tương quan Combined PRN {prn_range.start}-{prn_range.stop-1}", fontsize=14, color='#00d2ff')
    ax1.set_ylabel("Biên độ")
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend()

    # Subplot 2: Phóng to đỉnh
    ax2 = plt.subplot2grid((3, 2), (1, 0), colspan=2)
    zoom = 200
    s_zoom = max(0, peak_idx - zoom)
    e_zoom = min(len(correlation), peak_idx + zoom)
    ax2.plot(range(s_zoom, e_zoom), correlation[s_zoom:e_zoom], color='#ff007f', linewidth=1.5)
    ax2.axvline(x=peak_idx, color='white', linestyle=':', alpha=0.5)
    ax2.set_title(f"Chi tiết khu vực đỉnh (±{zoom} mẫu)", fontsize=12)
    ax2.set_xlabel("Sample Index")
    ax2.grid(True, linestyle='--', alpha=0.3)

    # Subplot 3: Độ mạnh từng PRN
    ax3 = plt.subplot2grid((3, 2), (2, 0))
    ax3.bar(prn_range, individual_peaks, color='#00ff99')
    ax3.set_title("Độ mạnh tương quan từng PRN (1ms reference)")
    ax3.set_xlabel("PRN ID")
    ax3.set_ylabel("Peak Magnitude")
    ax3.set_xticks(list(prn_range))

    # Subplot 4: Phổ tần số (FFT của tín hiệu thu)
    ax4 = plt.subplot2grid((3, 2), (2, 1))
    freqs = np.fft.fftfreq(len(captured_data), 1/fs)
    psd = 10 * np.log10(np.abs(f_captured)**2 + 1e-9)
    ax4.plot(np.fft.fftshift(freqs)/1e6, np.fft.fftshift(psd), color='#ffcc00', linewidth=0.5)
    ax4.set_title("Phổ mật độ công suất (PSD)")
    ax4.set_xlabel("Frequency (MHz)")
    ax4.set_ylabel("dB")
    ax4.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig('correlation_result.png')
    print("\nĐã lưu đồ thị kết quả vào 'correlation_result.png'")
    plt.show()

if __name__ == "__main__":
    analyze_captured_signal()
