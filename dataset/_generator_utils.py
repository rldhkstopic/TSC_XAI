import os
import numpy as np
from scipy.signal import stft
import matplotlib.pyplot as plt
from PIL import Image

def add_awgn(signal, snr_db):
    signal_power = np.mean(np.abs(signal) ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
    return signal + noise, noise

fs = 100e6
def deployment(wav, snr, waveform, fps_idx, dir= '/data/kiwan/LPI_KIWAN', only_stft=False):
    attenuationset = np.arange(-20, 2, 2)
    delayset = np.arange(1, 1001, 50) * 1e-9
    
    delays = np.random.choice(delayset, 5, replace=False)
    gains = np.random.choice(attenuationset, 5, replace=False)
    
    signal = multipath_channel(wav, fs, delays, gains)
    noisy_signal, noise = add_awgn(signal, snr)
    
    I = signal.real
    Q = signal.imag
    power = np.mean(I ** 2 + Q ** 2)
    pnorm_signal = signal / (np.sqrt(power) + 1e-6)
    
    pnorm_signal, _ = add_awgn(pnorm_signal, snr)
    
    if only_stft==False:
        waveform_folder = os.path.join(dir, waveform)
        np.save(os.path.join(waveform_folder, f'{waveform}_snr{snr}_Signal_{fps_idx}.npy'), wav)
        np.save(os.path.join(waveform_folder, f'{waveform}_snr{snr}_Noise_{fps_idx}.npy'), noise)
        np.save(os.path.join(waveform_folder, f'{waveform}_snr{snr}_Noisy_{fps_idx}.npy'), noisy_signal)
        np.save(os.path.join(waveform_folder, f'{waveform}_snr{snr}_pwnNoisy_{fps_idx}.npy'), pnorm_signal)

    elif only_stft==True:
        stft_folder = dir + '_STFT'
        stft_result(pnorm_signal, fs, stft_folder, waveform, snr, fps_idx, img_size=(128, 128))
    
def stft_result(signal, fs, out_dir, wf, snr, idx, img_size=(128, 128)):
    f, t, Zxx = stft(signal, fs, nperseg=256)
    Zxx_mag = np.abs(Zxx[:len(f)//2])

    fig, ax = plt.subplots(figsize=(img_size[0] / 100, img_size[1] / 100), dpi=100)
    ax.pcolormesh(t, f[:len(f)//2], Zxx_mag, shading='gouraud')
    ax.axis('off')
    
    img_dir = os.path.join(out_dir, f'{wf}')
    os.makedirs(img_dir, exist_ok=True)
    img_path = os.path.join(img_dir, f'{wf}_snr{snr}_STFT_{idx}.png')
    fig.savefig(img_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def multipath_channel(signal, fs, delay_set, gain_set, doppler_shift_range=(10, 1000)):
    """
    다중 경로 채널을 시뮬레이션
    - signal: 입력 신호
    - fs: 샘플링 주파수
    - delay_set: 가능한 지연 값 배열 (초 단위)
    - gain_set: 가능한 경로 감쇠 값 배열 (dB)
    - doppler_shift_range: 도플러 주파수 범위 (Hz)
    """
    
    selected_delays = [0] + list(np.random.choice(delay_set, 5, replace=False))
    selected_gains = [0] + list(np.random.choice(gain_set, 5, replace=False))

    doppler_shift = np.random.randint(doppler_shift_range[0], doppler_shift_range[1])

    filtered_signal = np.zeros_like(signal)
    
    
    for delay, gain in zip(selected_delays, selected_gains):
        sample_delay = int(delay * fs)
        
        delayed_signal = np.zeros_like(signal)
        if sample_delay < len(signal):
            delayed_signal[sample_delay:] = signal[:len(signal) - sample_delay]
        
        # dB 단위의 이득을 선형 이득으로 변환
        linear_gain = 10 ** (gain / 20.0)
        
        filtered_signal += linear_gain * delayed_signal
    
    # 도플러 효과 적용 (선택 사항)
    # 도플러 효과는 여기서는 생략했지만 필요 시 추가할 수 있음

    return filtered_signal


def bpsk(cpp, fs, A, fc, phase_code):
    Ts = 1 / fs
    t = np.arange(0, cpp / fc, Ts)
    s = np.zeros(len(phase_code) * len(t), dtype=complex)
    
    for k, phase in enumerate(phase_code):
        s[k * len(t):(k + 1) * len(t)] = A * np.exp(1j * (2 * np.pi * fc * t + phase))
    
    return s

def type_barker(cpp, fs, A, fc, code):
    Ts = 1 / fs                 
    Tc = cpp / fc               
    t = np.arange(0, Tc, Ts)    

    modulated_signal = np.array([])
    for bit in code:
        phase = 0 if bit == 1 else np.pi
        symbol = A * np.exp(1j * (2 * np.pi * fc * t + phase))
        modulated_signal = np.concatenate((modulated_signal, symbol))
    
    return modulated_signal

def type_costas(N, fs, A, fc, Lc):
    t = np.linspace(0, N/fs, N)
    hop_indices = np.random.permutation(np.arange(Lc))
    hop_freqs = hop_indices * (fs / (2 * Lc))           
    signal = np.zeros(N)
    hop_length = N // Lc
    for i in range(Lc):
        start_idx = i * hop_length
        end_idx = (i + 1) * hop_length
        signal[start_idx:end_idx] = A * np.cos(2 * np.pi * (fc + hop_freqs[i]) * t[start_idx:end_idx])
    return signal


def type_frank(cpp, fs, A, fc, M):
    phaseCode = np.zeros((M, M))
    for ii in range(M):
        for jj in range(M):
            phaseCode[ii, jj] = 2 * np.pi / M * ii * jj
    phaseCode = phaseCode.flatten()
    
    s = bpsk(cpp, fs, A, fc, phaseCode)
    return s

def type_lfm(num_samples, fs, A, fc, Df, updown):
    pw = num_samples / fs
    t = np.arange(1, num_samples + 1) / fs
    phi0 = 2 * np.pi * np.random.rand() - np.pi

    if updown.lower() == "up":
        f = fc + Df / pw * t
    elif updown.lower() == "down":
        f = fc - Df / pw * t
    else:
        print('Default up direction!')
        f = fc + Df / pw * t

    s = A * np.exp(1j * (2 * np.pi * f * t + phi0))
    return s

def type_P1(cpp, fs, A, fc, M):
    phase_code = np.zeros((M, M))
    
    for ii in range(M):
        for jj in range(M):
            phase_code[ii, jj] = -np.pi / M * ((M - (2 * jj + 1)) * ((jj) * M + (ii)))
    
    phase_code = phase_code.flatten()
    s = bpsk(cpp, fs, A, fc, phase_code)
    return s

def type_P2(cpp, fs, A, fc, M):
    phase_code = np.zeros((M, M))
    
    for ii in range(M):
        for jj in range(M):
            phase_code[ii, jj] = -np.pi / (2 * M) * (2 * ii - 1 - M) * (2 * jj - 1 - M)
    
    phase_code = phase_code.flatten()
    s = bpsk(cpp, fs, A, fc, phase_code)
    return s

def type_P3(cpp, fs, A, fc, p):
    phase_code = np.zeros(p)
    
    for ii in range(p):
        phase_code[ii] = np.pi / p * (ii ** 2)
    
    s = bpsk(cpp, fs, A, fc, phase_code)
    return s

def type_P4(cpp, fs, A, fc, p):
    phase_code = np.zeros(p)
    
    for ii in range(p):
        phase_code[ii] = np.pi / p * ((ii) ** 2) - np.pi * (ii)
    
    s = bpsk(cpp, fs, A, fc, phase_code)
    return s

def type_T1(fs, A, fc, Nps, Ng):
    Ts = 1 / fs
    Tc = 1 / fc
    t = np.arange(0, Tc, Ts)
    pw = Tc * Ng

    phase_code = np.zeros((Ng - 1, len(t)))
    for jj in range(Ng - 1):
        phase_code[jj, :] = np.mod(2 * np.pi / Nps * np.floor((Ng * t - jj * pw) * jj * Nps / pw), 2 * np.pi)

    phase_code = phase_code.T.flatten()
    s = bpsk(2, fs, A, fc, phase_code)
    return s

def type_T2(fs, A, fc, Nps, Ng):
    Ts = 1 / fs
    Tc = 1 / fc
    t = np.arange(0, Tc, Ts)
    pw = Tc * Ng
    
    phase_code = np.zeros((Ng - 1, len(t)))
    for jj in range(Ng - 1):
        phase_code[jj, :] = np.mod(2 * np.pi / Nps * np.floor((Ng * t - jj * pw) * (2 * jj - Ng + 1) / pw * Nps / 2), 2 * np.pi)

    phase_code = phase_code.T.flatten()
    s = bpsk(2, fs, A, fc, phase_code)
    return s

def type_T3(NumberSamples, fs, A, fc, Nps, B):
    Ts = 1 / fs
    pw = NumberSamples / fs
    t = np.arange(0, pw, Ts)

    phase_code = np.mod(2 * np.pi / Nps * np.floor(Nps * B * t ** 2 / (2 * pw)), 2 * np.pi)
    s = A * np.exp(1j * (2 * np.pi * fc * t + phase_code))
    return s

def type_T4(NumberSamples, fs, A, fc, Nps, B):
    Ts = 1 / fs
    pw = NumberSamples / fs
    t = np.arange(0, pw, Ts)

    # Phase code 계산
    phase_code = np.mod(2 * np.pi / Nps * np.floor(Nps * B * t ** 2 / (2 * pw) - Nps * B * t / 2), 2 * np.pi)

    # 변조된 신호 생성
    s = A * np.exp(1j * (2 * np.pi * fc * t + phase_code))
    return s


