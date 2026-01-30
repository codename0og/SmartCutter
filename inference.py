import torch
import torchaudio
import os
import glob
import numpy as np
import scipy.signal
import scipy.ndimage

from model import SmartCutterUNet

# --- CONFIG ---
IN_DIR = "infer_input"
OUT_DIR = "infer_output"
CKPT_DIR = "ckpts"
N_MELS = 160
CUTTING_PROBABILITY = 0.5           # Threshold for mask binarization

DEBUG_SAVE_NOISE_INJECTED = True

SILENCE_TARGET_DURATION = 0.100     # Duration of injected silence ( seconds )
REJECT_MASKS_BELOW_LENGTH = 0.150   # Minimum duration to keep a mask ( seconds )
GAP_FILL_THRESHOLD = 0.100 # 0.050          # Bridge gaps in masks smaller than 50ms
SEARCH_WINDOW_MS = 5                # Window size for Zero-Crossing search


def inject_targeted_noise(waveform, sr, min_silence_len=0.100, buffer_ms=10):
    is_zero = (waveform.abs() < 1e-6).float()

    min_samples = int(sr * min_silence_len)
    buffer_samples = int(sr * (buffer_ms / 1000))

    eroded = -torch.nn.functional.max_pool1d(-is_zero.unsqueeze(0), kernel_size=min_samples, stride=1, padding=min_samples//2).squeeze(0)
    silence_mask = torch.nn.functional.max_pool1d(eroded.unsqueeze(0), kernel_size=min_samples, stride=1, padding=min_samples//2).squeeze(0)

    shrink_samples = buffer_samples * 2
    targeted_mask = -torch.nn.functional.max_pool1d(-silence_mask.unsqueeze(0), kernel_size=shrink_samples, stride=1, padding=shrink_samples//2).squeeze(0)

    targeted_mask = targeted_mask[:, :waveform.shape[1]]

    # Inject Noise (-65dB RMS approx)
    noise = torch.randn_like(waveform) * 0.00055
    return waveform + (targeted_mask * noise)

def SmartCutter(waveform, mask, sr=48000):
    device = mask.device
    target_size = waveform.shape[1]

    # Ensure device consistency
    if waveform.device != device:
        waveform = waveform.to(device)

    # gpu interpolation
    mask_full = torch.nn.functional.interpolate(
        mask, size=target_size, mode='linear', align_corners=True
    )

    # Binarization
    binary_mask = (mask_full > CUTTING_PROBABILITY).float()

    # Safety; Bridge tiny flickers ( Binary Closing )
    fill_samples = int(sr * GAP_FILL_THRESHOLD)
    pad_f = fill_samples // 2
    dilated = torch.nn.functional.max_pool1d(binary_mask, fill_samples, 1, pad_f)
    binary_mask = -torch.nn.functional.max_pool1d(-dilated, fill_samples, 1, pad_f)

    # Safety;  Reject too short masks ( Binary Opening )
    min_samples = int(sr * REJECT_MASKS_BELOW_LENGTH)
    pad_s = min_samples // 2

    # Erosion (MinPool)
    eroded = -torch.nn.functional.max_pool1d(
        -binary_mask, kernel_size=min_samples, stride=1, padding=pad_s
    )
    # Dilation (MaxPool)
    opened_mask = torch.nn.functional.max_pool1d(
        eroded, kernel_size=min_samples, stride=1, padding=pad_s
    )

    # Identify points & Zero-Crossing Reconstruction
    mask_cpu = opened_mask.squeeze().cpu().numpy() > 0.5
    diff = np.diff(mask_cpu.astype(int), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    # Zero-Crossing reconstruction
    target_silence_samples = int(sr * SILENCE_TARGET_DURATION)
    search_win = int(sr * (SEARCH_WINDOW_MS / 1000))
    pieces = []
    last_idx = 0

    for start, end in zip(starts, ends):
        # Search for Zero-Crossing to prevent clicks
        win_start = max(0, start - search_win)
        win_end = min(target_size, start + search_win)
        segment = waveform[0, win_start:win_end]

        # Find index where waveform crosses or touches 0
        zero_cross = torch.argmin(torch.abs(segment)).item()
        safe_start = win_start + zero_cross

        # Append Speech with silence bits
        pieces.append(waveform[:, last_idx:safe_start])

        # Generate pure silence
        pieces.append(torch.zeros((waveform.shape[0], target_silence_samples), device=device))

        last_idx = end

    pieces.append(waveform[:, last_idx:])
    return torch.cat(pieces, dim=1).cpu()

def processing():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Dirs
    os.makedirs(IN_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    files = glob.glob(os.path.join(IN_DIR, "*.wav"))
    print(f"Found {len(files)} files.")

    # Cache models and transforms to avoid reloading
    loaded_models = {}

    for f_path in files:
        fname = os.path.basename(f_path)
        print(f"Processing: {fname}...")

        wav, sr = torchaudio.load(f_path)

        # Ensure consistent hop scaling with rvc standards.
        current_hop = sr // 100

        # Ensure an appropriate model is loaded
        if sr not in loaded_models:
            model_path = os.path.join(CKPT_DIR, f"model_{sr}.pth")
            if not os.path.exists(model_path):
                print(f"Skipping {fname}: No model found for {sr}Hz at {model_path}")
                continue

            print(f"Switching to {sr}Hz model (Hop: {current_hop})...")
            model = SmartCutterUNet(n_channels=N_MELS).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sr, 
                n_mels=N_MELS, 
                n_fft=2048, 
                hop_length=current_hop
            ).to(device)

            loaded_models[sr] = (model, mel_transform)
        curr_model, curr_mel_transform = loaded_models[sr]

        # Inference pipeline
        with torch.no_grad():
            wav_dev = wav.to(device)

            # Inject -65~ db gaussian noise in pure digital silence segments, that are 100ms+
            wav_for_model = inject_targeted_noise(wav_dev, sr)

            if DEBUG_SAVE_NOISE_INJECTED:
                debug_fname = fname.replace(".wav", "_noise_injected.wav")
                debug_path = os.path.join(OUT_DIR, debug_fname)
                torchaudio.save(debug_path, wav_for_model.cpu(), sr)
                print(f"Debug file saved: {debug_path}")

            # Pad to match standard mel windowing
            pad_len = 2048
            wav_padded = torch.nn.functional.pad(wav_for_model, (0, pad_len))

            mel = curr_mel_transform(wav_padded)
            mel = torchaudio.transforms.AmplitudeToDB()(mel)

            # Normalization logic
            m_std = mel.std()
            if m_std > 0.1:
                mel = (mel - mel.mean()) / (m_std + 1e-6)
            else:
                mel = torch.full_like(mel, -5.0)

            # Predict
            mask_pred = curr_model(mel)

        # Cut
        cleaned = SmartCutter(wav, mask_pred, sr=sr)

        # Save
        out_path = os.path.join(OUT_DIR, fname)
        torchaudio.save(out_path, cleaned, sr)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    processing()