import numpy as np
import time
import sys

# Yêu cầu cài đặt: pip install bladerf
try:
    from bladerf import _bladerf
except ImportError:
    print("Vui lòng cài đặt module bladerf: pip install bladerf")
    _bladerf = None

def setup_bladerf():
    if not _bladerf:
        return None
    
    # Mở thiết bị BladeRF
    try:
        b = _bladerf.BladeRF()
        print(f"Đã mở BladeRF: {b.get_board_name()}")
    except Exception as e:
        print(f"Không thể kết nối BladeRF: {e}")
        return None

    # Cấu hình TX và RX
    # Thông số mẫu theo code MATLAB
    fs = 10e6
    freq = 1575.42e6 # Tần số GPS L1 (có thể thay đổi theo ý muốn)
    bw = 10e6
    
    # Kênh RX
    b.set_sample_rate(_bladerf.CHANNEL_RX(0), fs)
    b.set_frequency(_bladerf.CHANNEL_RX(0), freq)
    b.set_bandwidth(_bladerf.CHANNEL_RX(0), bw)
    b.set_gain(_bladerf.CHANNEL_RX(0), 60) # Cấu hình Gain RX
    
    # Kênh TX
    b.set_sample_rate(_bladerf.CHANNEL_TX(0), fs)
    b.set_frequency(_bladerf.CHANNEL_TX(0), freq)
    b.set_bandwidth(_bladerf.CHANNEL_TX(0), bw)
    b.set_gain(_bladerf.CHANNEL_TX(0), 50) # Cấu hình Gain TX

    return b

def realtime_transmit(b, iq_samples):
    """
    Phát tín hiệu liên tục ra BladeRF
    iq_samples: mảng interleaved IQ (I, Q, I, Q...) dạng int16
    """
    channel = _bladerf.CHANNEL_TX(0)
    
    # Cấu hình Sync TX (Buffer)
    b.sync_config(
        layout=_bladerf.ChannelLayout.TX_X1,
        fmt=_bladerf.Format.SC16_Q11,
        num_buffers=16,
        buffer_size=8192,
        num_transfers=8,
        stream_timeout=3500
    )
    
    b.enable_module(channel, True)
    
    print("Bắt đầu phát tín hiệu TX...")
    try:
        while True:
            # Phát buffer mẫu ra BladeRF
            b.sync_tx(iq_samples, len(iq_samples) // 2)
    except KeyboardInterrupt:
        print("Dừng phát.")
    finally:
        b.enable_module(channel, False)


def realtime_receive(b, num_samples):
    """
    Thu tín hiệu từ BladeRF
    Trả về mảng interleaved IQ dạng int16
    """
    channel = _bladerf.CHANNEL_RX(0)
    
    b.sync_config(
        layout=_bladerf.ChannelLayout.RX_X1,
        fmt=_bladerf.Format.SC16_Q11,
        num_buffers=16,
        buffer_size=8192,
        num_transfers=8,
        stream_timeout=3500
    )
    
    b.enable_module(channel, True)
    
    # Mảng để chứa dữ liệu nhận về
    recv_data = np.zeros(num_samples * 2, dtype=np.int16)
    
    print("Bắt đầu thu tín hiệu RX...")
    try:
        b.sync_rx(recv_data, num_samples)
    except Exception as e:
        print(f"Lỗi khi thu tín hiệu: {e}")
    finally:
        b.enable_module(channel, False)
        
    return recv_data

if __name__ == "__main__":
    b = setup_bladerf()
    if b:
        # Ví dụ: Thu tín hiệu 10ms ở fs = 10MHz
        samples_to_recv = int(10e6 * 0.01)
        data = realtime_receive(b, samples_to_recv)
        print(f"Đã thu được {len(data)/2} mẫu.")
        
        # Ở đây bạn có thể gọi các hàm xử lý từ thu_tin_hieu.py để xử lý 'data'
        
        b.close()
