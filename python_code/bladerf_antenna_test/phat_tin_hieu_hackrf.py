"""
phat_tin_hieu_hackrf.py
=======================
Phát tín hiệu GNSS giả lập qua HackRF One bằng SoapySDR (Python).

Tương đương phat_tin_hieu_bladerf.py:
  - Ghép C/A code nhiều PRN (mặc định 11–20)
  - BPSK baseband: s(n) = code * exp(j2πft·n/fs)
  - Phát lặp liên tục đến khi Ctrl+C

Cài đặt (Ubuntu/Debian):
    sudo apt install hackrf libhackrf-dev soapysdr-tools soapysdr-module-hackrf
    pip install SoapySDR numpy

Kiểm tra thiết bị:
    SoapySDRUtil --find="driver=hackrf"
    hackrf_info

Chạy:
    python phat_tin_hieu_hackrf.py --prn_start 11 --prn_end 20 --freq_hz 433e6
    python phat_tin_hieu_hackrf.py --prn_start 11 --prn_end 20 --freq_hz 433e6 --tx_gain 30
    python phat_tin_hieu_hackrf.py --save_file tx_prn11_20.bin   # chỉ xuất file, không cần HackRF
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gnss_utils import generateCAcode


def _bootstrap_soapy_plugin_path() -> None:
    """
    Conda/pip SoapySDR thường không tự tìm module hệ thống (libHackRFSupport.so).
    Trỏ SOAPY_SDR_PLUGIN_PATH về thư mục apt nếu chưa được đặt.
    """
    if os.environ.get("SOAPY_SDR_PLUGIN_PATH"):
        return
    for path in (
        "/usr/lib/x86_64-linux-gnu/SoapySDR/modules0.8",
        "/usr/lib/aarch64-linux-gnu/SoapySDR/modules0.8",
        "/usr/local/lib/SoapySDR/modules0.8",
    ):
        if os.path.isdir(path) and any(name.endswith(".so") for name in os.listdir(path)):
            os.environ["SOAPY_SDR_PLUGIN_PATH"] = path
            print(f"[INFO] SOAPY_SDR_PLUGIN_PATH={path}")
            return


_bootstrap_soapy_plugin_path()

try:
    import SoapySDR
    from SoapySDR import SOAPY_SDR_CF32, SOAPY_SDR_TX

    HAS_SOAPY = True
except ImportError:
    HAS_SOAPY = False
    print("[WARN] Chưa cài SoapySDR. Cài: pip install SoapySDR")


def make_ca_code(prn_start: int, prn_end: int) -> np.ndarray:
    return np.concatenate([generateCAcode(i) for i in range(prn_start, prn_end + 1)]).astype(np.float32)


def synthesize_iq_complex64(
    prn_start: int,
    prn_end: int,
    tau: int,
    rc: float,
    fs: float,
    ft: float,
    n_samples: int,
    amplitude: float,
) -> np.ndarray:
    total_chips = (prn_end - prn_start + 1) * 1023
    cacodes = make_ca_code(prn_start, prn_end)

    n = np.arange(n_samples, dtype=np.float64)
    idx = (np.floor((n + tau) / fs * rc).astype(np.int64)) % total_chips
    r = cacodes[idx].astype(np.float32)

    phase = (2 * np.pi * ft * n / fs).astype(np.float32)
    i = amplitude * r * np.cos(phase)
    q = amplitude * r * np.sin(phase)
    return (i + 1j * q).astype(np.complex64)


def normalize_for_hackrf(iq: np.ndarray, peak: float) -> np.ndarray:
    """HackRF + SoapySDR CF32 nên dùng biên độ gần ±1."""
    mx = float(np.max(np.abs(iq)))
    if mx <= 0:
        return iq.astype(np.complex64, copy=True)
    return (iq / mx * peak).astype(np.complex64)


def _kwargs_to_dict(obj) -> dict:
    """SoapySDR trả về SoapySDRKwargs, không phải dict Python thuần."""
    if isinstance(obj, dict):
        return obj
    try:
        return dict(obj)
    except (TypeError, ValueError):
        pass
    try:
        return {k: obj[k] for k in obj.keys()}
    except Exception:
        return {}


def _enumerate_hackrf_devices() -> list[dict]:
    devices = []
    for info in SoapySDR.Device.enumerate():
        d = _kwargs_to_dict(info)
        if d.get("driver") == "hackrf":
            devices.append(d)
    return devices


def _build_device_args(serial: str = "") -> list[str]:
    """
    SoapySDR (conda) mở HackRF ổn định bằng chuỗi 'driver=hackrf', không phải dict.
    Ví dụ: SoapySDR.Device('driver=hackrf,serial=24b862dc31264fcb')
    """
    candidates: list[str] = []
    if serial:
        candidates.append(f"driver=hackrf,serial={serial}")
        short = serial.lstrip("0")
        if short and short != serial:
            candidates.append(f"driver=hackrf,serial={short}")
    candidates.append("driver=hackrf")
    # Loại trùng, giữ thứ tự
    seen = set()
    out = []
    for item in candidates:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def open_hackrf_tx(freq_hz: float, fs: float, tx_gain: float, serial: str = ""):
    found = _enumerate_hackrf_devices()
    candidates = _build_device_args(serial)

    last_err = None
    sdr = None
    opened_with = ""
    for args in candidates:
        try:
            print(f"[INFO] Mở HackRF: {args}")
            sdr = SoapySDR.Device(args)
            opened_with = args
            break
        except Exception as exc:
            last_err = exc
            print(f"[WARN] Không mở được với '{args}': {exc}")

    if sdr is None:
        msg = (
            "Không mở được HackRF qua SoapySDR.\n"
            "  1) Cắm USB và chạy: hackrf_info\n"
            "  2) export SOAPY_SDR_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/SoapySDR/modules0.8\n"
            "  3) Thử bỏ --serial nếu chỉ có 1 HackRF: SoapySDR.Device('driver=hackrf')\n"
            "  4) Quyền USB: sudo usermod -aG plugdev $USER (logout/login)\n"
        )
        if found:
            msg += f"  Thiết bị enumerate được: {found}\n"
        if last_err:
            msg += f"  Lỗi cuối: {last_err}"
        raise RuntimeError(msg)

    sdr.setSampleRate(SOAPY_SDR_TX, 0, fs)
    sdr.setFrequency(SOAPY_SDR_TX, 0, freq_hz)
    sdr.setBandwidth(SOAPY_SDR_TX, 0, fs)
    sdr.setGain(SOAPY_SDR_TX, 0, tx_gain)

    hw = _kwargs_to_dict(sdr.getHardwareInfo())
    label = hw.get("label", "HackRF")
    opened_serial = hw.get("serial", serial or "auto")
    print(f"[INFO] Thiết bị: {label}  serial={opened_serial}  args={opened_with}")
    print(f"[INFO] TX  fc={freq_hz/1e6:.3f} MHz  fs={fs/1e6:.2f} MHz  gain={tx_gain:.1f} dB")
    return sdr


def transmit_loop(sdr, iq: np.ndarray) -> None:
    """
    Phát lặp buffer IQ.
    HackRF qua Soapy thường có MTU lớn (~131072 mẫu CF32); gửi theo chunk MTU.
    """
    stream = sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [0])
    mtu = int(sdr.getStreamMTU(stream))
    mtu = max(mtu, 4096)

    sdr.activateStream(stream)
    print(f"[TX] MTU={mtu} mẫu | vòng tín hiệu={len(iq)} mẫu | Ctrl+C để dừng")

    try:
        while True:
            pos = 0
            while pos < len(iq):
                n = min(mtu, len(iq) - pos)
                chunk = np.zeros(mtu, dtype=np.complex64)
                chunk[:n] = iq[pos : pos + n]
                pos += n

                status = sdr.writeStream(stream, [chunk], mtu, timeoutUs=int(1e6))
                if status.ret < 0:
                    raise RuntimeError(f"writeStream lỗi: {status.ret}")
    except KeyboardInterrupt:
        print("\n[TX] Nhận Ctrl+C — dừng phát.")
    finally:
        sdr.deactivateStream(stream)
        sdr.closeStream(stream)
        del sdr
        print("[TX] Đã đóng HackRF.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Phát tín hiệu GNSS qua HackRF One (SoapySDR).")
    p.add_argument("--freq_hz", type=float, default=433e6, help="Tần số RF (Hz). Mặc định 433 MHz.")
    p.add_argument("--fs", type=float, default=2e6, help="Sample rate (Hz).")
    p.add_argument("--Rc", type=float, default=1.023e6, help="Chip rate (Hz).")
    p.add_argument("--ft", type=float, default=0.0, help="Doppler offset baseband (Hz).")
    p.add_argument("--time_s", type=float, default=1.0, help="Thời lượng 1 vòng tín hiệu (giây).")
    p.add_argument("--tau", type=int, default=40, help="Độ trễ code (samples).")
    p.add_argument("--prn_start", type=int, default=11, help="PRN bắt đầu.")
    p.add_argument("--prn_end", type=int, default=20, help="PRN kết thúc.")
    p.add_argument("--amplitude", type=float, default=1.0, help="Biên độ tổng hợp trước chuẩn hoá.")
    p.add_argument("--tx_scale", type=float, default=0.85, help="Biên độ sau chuẩn hoá (0–1), gửi lên HackRF.")
    p.add_argument("--tx_gain", type=float, default=30.0, help="Gain TX (dB), HackRF ~0–47.")
    p.add_argument("--serial", default="", help="Serial HackRF (tùy chọn).")
    p.add_argument("--save_file", default="", help="Lưu file complex64 rồi thoát.")
    args = p.parse_args(argv)

    n_samples = int(args.fs * args.time_s)

    print("══════════════════════════════════════════════════════")
    print("  phat_tin_hieu_hackrf.py  —  HackRF TX (SoapySDR)")
    print(f"  PRN {args.prn_start}–{args.prn_end}  |  fs={args.fs/1e6:.2f} MHz"
          f"  |  fc={args.freq_hz/1e6:.3f} MHz")
    print(f"  ft={args.ft} Hz  |  tau={args.tau}  |  N={n_samples} samples/vòng")
    print("══════════════════════════════════════════════════════\n")

    print("[INFO] Đang tổng hợp tín hiệu IQ...")
    iq = synthesize_iq_complex64(
        prn_start=args.prn_start,
        prn_end=args.prn_end,
        tau=args.tau,
        rc=args.Rc,
        fs=args.fs,
        ft=args.ft,
        n_samples=n_samples,
        amplitude=args.amplitude,
    )
    iq_tx = normalize_for_hackrf(iq, args.tx_scale)
    print(f"[INFO] IQ: {n_samples} mẫu | peak sau chuẩn hoá={np.max(np.abs(iq_tx)):.3f}")

    if args.save_file:
        iq_tx.tofile(args.save_file)
        print(f"[INFO] Đã lưu file complex64 (đã chuẩn hoá): {args.save_file}")
        return 0

    if not HAS_SOAPY:
        print("[ERROR] Chưa cài SoapySDR. Dùng --save_file hoặc: pip install SoapySDR")
        return 1

    try:
        sdr = open_hackrf_tx(args.freq_hz, args.fs, args.tx_gain, args.serial)
    except Exception as exc:
        print(f"[ERROR] Không mở được HackRF:\n{exc}")
        return 1

    time.sleep(0.2)
    transmit_loop(sdr, iq_tx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
