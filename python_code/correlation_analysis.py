# import numpy as np
# import matplotlib.pyplot as plt
# import os
#
# def correlate_signals(file1, file2, sample_rate=2e6, duration_ms=10, offset_samples=0):
#     """
#     Tính toán tương quan giữa hai file tín hiệu GNSS (định dạng complex64).
#     """
#     num_samples = int(sample_rate * duration_ms / 1000)
#
#     if not os.path.exists(file1):
#         print(f"Lỗi: Không tìm thấy file {file1}")
#         return
#
#     # Kích thước file (bytes)
#     file_size = os.path.getsize(file1)
#     # Một mẫu complex64 = 8 bytes
#     max_samples = file_size // 8
#
#     if num_samples > max_samples:
#         num_samples = max_samples
#         print(f"Cảnh báo: File chỉ có {max_samples} mẫu. Đang dùng toàn bộ file.")
#
#     print(f"--- THÔNG SỐ ---")
#     print(f"File 1: {file1}")
#     print(f"File 2: {file2}")
#     print(f"Sampling Rate: {sample_rate/1e6:.1f} MHz")
#     print(f"Số mẫu xử lý: {num_samples}")
#     print(f"----------------")
#
#     # Đọc dữ liệu
#     data1 = np.fromfile(file1, dtype=np.complex64, count=num_samples)
#
#     # Nếu là cùng 1 file và muốn tìm độ trễ nội tại (autocorrelation)
#     # hoặc so sánh 2 đoạn khác nhau.
#     if file1 == file2 and offset_samples == 0:
#         print("Đang thực hiện Tự tương quan (Autocorrelation)...")
#         data2 = data1
#     else:
#         # Đọc file 2 với một offset nếu cần
#         with open(file2, 'rb') as f:
#             f.seek(offset_samples * 8) # skip offset_samples * 8 bytes
#             data2 = np.fromfile(f, dtype=np.complex64, count=num_samples)
#
#     if len(data2) < num_samples:
#         num_samples = len(data2)
#         data1 = data1[:num_samples]
#
#     # Thực hiện tương quan nhanh dùng FFT
#     f_data1 = np.fft.fft(data1)
#     f_data2 = np.fft.fft(data2)
#
#     # Cross-correlation theorem: corr = IFFT(FFT(x) * conj(FFT(y)))
#     correlation = np.abs(np.fft.ifft(f_data1 * np.conj(f_data2)))
#
#     # Tìm đỉnh
#     peak_idx = np.argmax(correlation)
#     peak_val = correlation[peak_idx]
#
#     # Tính toán độ trễ thời gian
#     delay_ms = (peak_idx / sample_rate) * 1000
#
#     print(f"Kết quả:")
#     print(f" - Đỉnh tại index: {peak_idx}")
#     print(f" - Giá trị đỉnh: {peak_val:.2f}")
#     print(f" - Độ trễ tương ứng: {delay_ms:.4f} ms")
#
#     # Hiển thị kết quả
#     plt.style.use('dark_background')
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
#
#     # Đồ thị tổng quát
#     ax1.plot(correlation, color='#00d2ff', linewidth=0.8, alpha=0.9)
#     ax1.set_title(f"Tương quan: {os.path.basename(file1)} vs {os.path.basename(file2)}", fontsize=14, color='#00d2ff')
#     ax1.set_ylabel("Biên độ (Magnitude)")
#     ax1.grid(True, linestyle='--', alpha=0.3)
#     ax1.plot(peak_idx, peak_val, 'ro', markersize=8, label=f'Peak at {peak_idx}')
#     ax1.legend()
#
#     # Đồ thị phóng to quanh đỉnh
#     zoom_range = 100 # số mẫu quanh đỉnh
#     start_zoom = max(0, peak_idx - zoom_range)
#     end_zoom = min(len(correlation), peak_idx + zoom_range)
#
#     ax2.plot(range(start_zoom, end_zoom), correlation[start_zoom:end_zoom], color='#ff007f', linewidth=2)
#     ax2.set_title(f"Phóng to khu vực đỉnh (±{zoom_range} mẫu)", fontsize=12)
#     ax2.set_xlabel("Sample Index")
#     ax2.set_ylabel("Magnitude")
#     ax2.grid(True, linestyle='--', alpha=0.3)
#     ax2.axvline(x=peak_idx, color='white', linestyle=':', alpha=0.5)
#
#     plt.tight_layout()
#     plt.show()
#
# if __name__ == "__main__":
#     # Bạn có thể thay đổi các tham số ở đây
#     FILE = 'test-1.bin'
#
#     # Ví dụ: Tương quan file với chính nó
#     correlate_signals(FILE, "data_tx1_prn_11_20_10MHz.bin", sample_rate=2e6, duration_ms=20)


import numpy as np
import matplotlib.pyplot as plt
import os


def correlate_signals(file1, file2, sample_rate=2e6, duration_ms=20, offset_samples=0):
    """
    Tính toán tương quan giữa hai file tín hiệu GNSS (định dạng complex64).
    """
    num_samples = int(sample_rate * duration_ms / 1000)

    if not os.path.exists(file1):
        print(f"Lỗi: Không tìm thấy file {file1}")
        return

    # Kích thước file
    file_size = os.path.getsize(file1)
    max_samples = file_size // 8  # complex64 = 8 bytes

    if num_samples > max_samples:
        num_samples = max_samples
        print(f"Cảnh báo: File chỉ có {max_samples} mẫu. Đang dùng toàn bộ file.")

    print(f"--- THÔNG SỐ ---")
    print(f"File 1 (thu): {file1}")
    print(f"File 2 (local): {file2}")
    print(f"Sampling Rate: {sample_rate / 1e6:.1f} MHz")
    print(f"Số mẫu xử lý: {num_samples}")
    print(f"----------------")

    # Đọc data1
    data1 = np.fromfile(file1, dtype=np.complex64, count=num_samples)

    # Đọc data2
    if file1 == file2 and offset_samples == 0:
        print("Đang thực hiện Tự tương quan (Autocorrelation)...")
        data2 = data1.copy()
    else:
        with open(file2, 'rb') as f:
            f.seek(offset_samples * 8)
            data2 = np.fromfile(f, dtype=np.complex64, count=num_samples)

    # Cắt nếu file 2 ngắn hơn
    if len(data2) < num_samples:
        num_samples = len(data2)
        data1 = data1[:num_samples]
        data2 = data2[:num_samples]

    # Tính tương quan bằng FFT
    f_data1 = np.fft.fft(data1)
    f_data2 = np.fft.fft(data2)
    correlation = np.abs(np.fft.ifft(f_data1 * np.conj(f_data2)))

    # Tìm peak
    peak_idx = np.argmax(correlation)
    peak_val = correlation[peak_idx]
    delay_ms = (peak_idx / sample_rate) * 1000

    print(f"\nKẾT QUẢ:")
    print(f" - Đỉnh tại index : {peak_idx}")
    print(f" - Giá trị đỉnh   : {peak_val:.2f}")
    print(f" - Độ trễ         : {delay_ms:.4f} ms")

    # ====================== VẼ ĐỒ THỊ ======================
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Plot toàn bộ
    ax1.plot(correlation, color='#00d2ff', linewidth=0.8, alpha=0.9)
    ax1.set_title(f"Tương quan: {os.path.basename(file1)}  vs  {os.path.basename(file2)}",
                  fontsize=15, color='#00d2ff')
    ax1.set_ylabel("Magnitude")
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.plot(peak_idx, peak_val, 'ro', markersize=10,
             label=f'Peak = {peak_val:.2f} @ index {peak_idx}')
    ax1.legend(fontsize=12)

    # Plot zoom
    zoom_range = 300  # Tăng zoom để xem rõ peak
    start_zoom = max(0, peak_idx - zoom_range)
    end_zoom = min(len(correlation), peak_idx + zoom_range)

    ax2.plot(range(start_zoom, end_zoom), correlation[start_zoom:end_zoom],
             color='#ff007f', linewidth=2.2)
    ax2.set_title(f"Phóng to vùng đỉnh (±{zoom_range} samples)", fontsize=14, color='#ffcc00')
    ax2.set_xlabel("Sample Index")
    ax2.set_ylabel("Magnitude")
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.axvline(x=peak_idx, color='white', linestyle=':', alpha=0.7, linewidth=1.5)

    plt.tight_layout()

    # Lưu ảnh
    save_name = f"correlation_{os.path.basename(file1)}_{peak_idx}.png"
    plt.savefig(save_name, dpi=300, bbox_inches='tight', facecolor='#0a0a0a')
    print(f"Đã lưu đồ thị: {save_name}")

    plt.show()


if __name__ == "__main__":
    FILE = 'test-1.bin'

    correlate_signals(
        # file1=FILE,
        file1="data_tx1_prn_11_20_10MHz.bin",
        file2="data_tx1_prn_11_20_10MHz.bin",
        # file2=FILE,
        sample_rate=2e6,
        duration_ms=20
    )
