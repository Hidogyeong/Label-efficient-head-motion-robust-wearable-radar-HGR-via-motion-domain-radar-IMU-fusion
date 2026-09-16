##########################################
# FMCW Radar Data Processing (NTT+CRT spectrogram with 16-bit fixed-point quantization)
##########################################
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

from helpers.DopplerAlgo import *
from helpers.DigitalBeamForming import *

# ---- NTT / CRT Utilities ----
def ntt(a, p, root):
    """
    Simple O(N^2) NTT implementation.
    a    : input integer sequence (length N)
    p    : prime with (p-1) % N == 0
    root : primitive N-th root of unity modulo p
    """
    N = len(a)
    A = []
    for k in range(N):
        s = 0
        for n in range(N):
            s = (s + a[n] * pow(root, k*n, p)) % p
        A.append(s)
    return A


def crt_combine(x1, x2, p1, p2):
    """
    Combine two modular results (x1 mod p1, x2 mod p2) via CRT.
    """
    inv_p1 = pow(p1, -1, p2)
    t = ((x2 - x1) * inv_p1) % p2
    return x1 + t * p1


def ntt_spectrogram(x, fs, N, hop, p_list, root_list):
    # Debug: input parameters
    print(f"[DEBUG] ntt_spectrogram: len(x)={len(x)}, N={N}, hop={hop}, fs={fs}")
    # 16-bit Q1.15 fixed-point scale
    FIXED_SCALE = 2**15 - 1  # 32767

    # Compute inverse of N modulo each prime
    inv_list = [pow(N, -1, p) for p in p_list]

    win = np.hanning(N)
    # Precompute twiddle tables for each prime to accelerate NTT
    twiddle_tables = []
    for (p, root) in zip(p_list, root_list):
        table = [[pow(root, k*n, p) for n in range(N)] for k in range(N)]
        twiddle_tables.append(table)

    frames = []
    M = p_list[0] * p_list[1]

    for idx, start in enumerate(range(0, len(x) - N + 1, hop)):
        # 1) Window & fixed-point quantization to 16-bit
        seg = x[start:start+N] * win
        seg_q = np.round(seg * FIXED_SCALE).astype(np.int32)
        seg_q = np.clip(seg_q, -FIXED_SCALE-1, FIXED_SCALE)
        seg_int = seg_q.astype(np.int16)

        if idx == 0:
            print(f"[DEBUG] First seg (float): {seg[:5]}")
            print(f"[DEBUG] First seg_int (Q1.15): {seg_int[:5]}")

        # 2) Accelerated NTT per prime with normalization by inv(N)
        Xs = []
        for idx_prime, (p, root, invN) in enumerate(zip(p_list, root_list, inv_list)):
            # Promote to Python ints and perform NTT using precomputed twiddles
            a_mod = [int(v) % p for v in seg_int]
            raw_ntt = []
            table = twiddle_tables[idx_prime]
            for k in range(N):
                s = 0
                row = table[k]
                for n, val in enumerate(a_mod):
                    s = (s + val * row[n]) % p
                raw_ntt.append(s)
            # Normalize in field: multiply by inv(N)
            norm_ntt = [(val * invN) % p for val in raw_ntt]
            Xs.append(norm_ntt)

        # 3) CRT combine and signed mapping
        X = []
        for k in range(N):
            x_crt = crt_combine(Xs[0][k], Xs[1][k], p_list[0], p_list[1])
            if x_crt >= M // 2:
                x_crt -= M
            X.append(x_crt)

        # Note: direct NTT produces natural order; bit-reversal removed
        # 4) Magnitude spectrum: scale back to float by FIXED_SCALE and normalize by N: scale back to float by FIXED_SCALE and normalize by N
        mag = np.abs(np.array(X)) / (FIXED_SCALE * N)

        if idx == 0:
            print(f"[DEBUG] First frame mag[:5]: {mag[:5]}")
        frames.append(mag)

    print(f"[DEBUG] Total frames: {len(frames)}")

    Sxx = np.array(frames).T
    # Shift zero-frequency component to center for display
    Sxx = np.fft.fftshift(Sxx, axes=0)
        # Shift zero-frequency component to center for display
    Sxx = np.fft.fftshift(Sxx, axes=0)
    f = np.fft.fftshift(np.fft.fftfreq(N, d=1/fs))
    t = np.arange(len(frames)) * hop / fs
    return f, t, Sxx

# ============================
# ============================
# Load and Process Radar Data
# ============================
def load_csv_files(directory):
    csv_files = glob.glob(os.path.join(directory, '*.csv'))
    data = []
    for file in csv_files:
        try:
            arr = np.loadtxt(file, delimiter=',')
            data.append(arr)
        except Exception as e:
            print(f"Warning: could not load {file}: {e}")
    return np.array(data)


def process_radar_data(data,
                       num_chirps=32,
                       num_rx_antennas=3,
                       num_samples=64,
                       sample_rate_Hz=2e6,
                       N=256,
                       hop=None):
    try:
        data = data.reshape(-1, num_rx_antennas, num_chirps, num_samples)
    except ValueError as e:
        print(f"Error reshaping data: {e}")
        return {}

    antenna_data = [data[:, i, :, :].reshape(-1, num_chirps * num_samples)
                    for i in range(num_rx_antennas)]
    data = np.stack(antenna_data, axis=0)
    rawData = []
    for ant in range(num_rx_antennas):
        rawData.append(np.zeros((data.shape[1], num_chirps * num_samples - num_samples - 1)))

    for frame_idx in range(data.shape[1]):
        for ant in range(num_rx_antennas):
            cur = data[ant, frame_idx, :]
            rawData[ant][frame_idx] = [
                cur[i + num_samples] - cur[i]
                for i in range(len(cur) - num_samples - 1)
            ]

    for ant in range(num_rx_antennas):
        rd = rawData[ant].flatten()
        print(f"[DEBUG] Antenna {ant}: len={len(rd)}, min={rd.min():.3e}, max={rd.max():.3e}, mean={rd.mean():.3e}")
    rawData = [d.flatten() for d in rawData]

    # NTT+CRT parameters
    # Primes must satisfy (p-1)%N==0 and p > N * max(input range)
    # For 16-bit Q1.15, max input ~32767, N=256 => N*32767 ~8.4e6
    p_list = [167772161, 469762049]
    root_list = [3, 3]
    if hop is None:
        hop = N // 2

    results = {}
    for ant in range(num_rx_antennas):
        try:
            f, t, Sxx = ntt_spectrogram(rawData[ant],
                                        sample_rate_Hz,
                                        N, hop,
                                        p_list, root_list)
            results[ant] = (f, t, Sxx)
        except Exception as e:
            print(f"NTT spectrogram error ant {ant}: {e}")
    return results

# ============================
# Main
# ============================
if __name__ == "__main__":
    radar_dir = "/media/dokyeong/MINJI/dokyeong/Radar_rawdata/"
    output_dir = "/media/dokyeong/MINJI/dokyeong/Radar_STFT_NTT/"

    files = glob.glob(os.path.join(radar_dir, '**', '*.csv'), recursive=True)
    for file in files:
        print(f"\n[INFO] Processing file: {file}")
        data = np.loadtxt(file, delimiter=',')
        sats = process_radar_data(data)
        base = os.path.splitext(os.path.basename(file))[0]
        rel = os.path.relpath(file, radar_dir)
        outd = os.path.join(output_dir, os.path.dirname(rel))
        os.makedirs(outd, exist_ok=True)

        for ant, (f, t, Sxx) in sats.items():
            save_path = os.path.join(outd, f"{base}_ant{ant}.png")
            # Normalize Sxx for display (0 to 1)
            Sxx_disp = (Sxx - Sxx.min()) / (Sxx.max() - Sxx.min() + 1e-12)
            plt.pcolormesh(t, f, Sxx_disp, cmap='jet')
            plt.axis('off')
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()

            if os.path.isfile(save_path):
                print(f"Verified saved image: {save_path}")
            else:
                print(f"Error: image not found at {save_path}")
