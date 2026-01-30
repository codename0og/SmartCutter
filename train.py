import os
import glob
import torch
import torchaudio
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model import SmartCutterUNet

# Config
INTENDED_SR_FOR_TRAIN = 40000  # Options: 32000, 40000, 48000
SAMPLE_RATE = INTENDED_SR_FOR_TRAIN
HOP_LENGTH = SAMPLE_RATE // 100

N_MELS = 160
N_FFT = 2048
SEGMENT_LEN = SAMPLE_RATE * 4 # ( 192,000 samples = 4 secs. )
TRAIN_DIR = "training_files"
CKPT_DIR = "ckpts"

# Misc
smoothing_on_mask = True
enable_validation = True

# Hparams
BATCH_SIZE = 8
EPOCHS = 60
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01

##################### Augmentation config
# Random Gain
rand_gain_max = 1.0
rand_gain_min = 0.2

# Random Noise
rand_noise_max = 0.0 #0.02
rand_noise_min = 0.0 #0.001


class DiceLoss(nn.Module):
    """Focuses on the overlap between mask and prediction."""
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)  
        
        return 1 - dice

class SlicedDataset(Dataset):
    def __init__(self, folder, segment_len=SEGMENT_LEN, overlap=0.20):
        self.samples = [] # Holds (raw_slice, mask_slice)

        # Mel Spectrogram Config
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, 
            n_mels=N_MELS, 
            n_fft=N_FFT, 
            hop_length=HOP_LENGTH
        )
        raw_files = sorted(glob.glob(os.path.join(folder, "*_raw.wav")))
        marked_files = sorted(glob.glob(os.path.join(folder, "*_marked.wav")))

        if len(raw_files) == 0:
            raise FileNotFoundError(f"No *_raw.wav files found in {folder}")

        print(f"--- Pre-processing {len(raw_files)} pairs... ---")

        for r_file, m_file in zip(raw_files, marked_files):
            # Load full pair
            raw_wav, sr_r = torchaudio.load(r_file)
            marked_wav, sr_m = torchaudio.load(m_file)

            # Resample if necessary 
            if sr_r != SAMPLE_RATE:
                print(f"[Resampling] Currently training for sr: {SAMPLE_RATE}")
                print(f"[Resampling] {os.path.basename(r_file)} from {sr_r}Hz to {SAMPLE_RATE}Hz ---")
                raw_wav = torchaudio.transforms.Resample(sr_r, SAMPLE_RATE)(raw_wav)
            if sr_m != SAMPLE_RATE:
                marked_wav = torchaudio.transforms.Resample(sr_m, SAMPLE_RATE)(marked_wav)

            # Match lengths (Trimming the excess of the longer file)
            min_len = min(raw_wav.shape[1], marked_wav.shape[1])
            raw_wav = raw_wav[:, :min_len]
            marked_wav = marked_wav[:, :min_len]

            # Global Normalization
            max_val = torch.max(torch.abs(marked_wav))
            if max_val > 0.95:
                scale = 0.95 / max_val
                marked_wav *= scale
                raw_wav *= scale

            # Generate Mask Globally
            diff = torch.abs(marked_wav - raw_wav)
            if smoothing_on_mask: 
                # Smoothing helps remove micro-gaps in the marking
                diff = torch.nn.functional.avg_pool1d(diff.unsqueeze(0), kernel_size=101, stride=1, padding=50).squeeze(0)

            mask = (diff > 0.1).float()

            mask = torch.nn.functional.max_pool1d(mask.unsqueeze(0), kernel_size=21, stride=1, padding=10).squeeze(0)

            # Slicing
            # Calculate stride based on overlap
            stride = int(segment_len * (1 - overlap))
            length = raw_wav.shape[1]

            # Loop through the file ( extracting every valid segment )
            count_for_this_file = 0
            for i in range(0, length - segment_len + 1, stride):
                r_slice = raw_wav[:, i : i + segment_len]
                m_slice = mask[:, i : i + segment_len]

                # Check for silence (Optional: skip if slice is purely empty)
                if r_slice.abs().max() > 0.001: 
                    self.samples.append((r_slice, m_slice))
                    count_for_this_file += 1

            # print(f"File {os.path.basename(r_file)}: Extracted {count_for_this_file} segments.")

        print(f"--- Dataset Prepared: {len(self.samples)} total segments from {len(raw_files)} speakers ---")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw_crop, mask_crop = self.samples[idx]
        raw_crop = raw_crop.clone()
        mask_crop = mask_crop.clone()

        # ////////////////      DYNAMIC AUGMENTATION START      ////////////////

        # Random Gain
        gain = torch.empty(1).uniform_(rand_gain_min, rand_gain_max)
        raw_crop = raw_crop * gain

        # Random noise injection
        if (rand_noise_min != 0.0 and rand_noise_max != 0.0):
            if torch.rand(1).item() > 0.5:
                noise_level = torch.empty(1).uniform_(rand_noise_min, rand_noise_max)
                noise = torch.randn_like(raw_crop) * noise_level
                raw_crop = raw_crop + noise

        # ////////////////      PROCESSING      ////////////////

        # Transform to mel
        mel = self.mel_transform(raw_crop).squeeze(0)
        mel = torchaudio.transforms.AmplitudeToDB()(mel)

        # Stability Normalization
        mean = mel.mean()
        std = mel.std()
        if std > 0.1:
            mel = (mel - mean) / (std + 1e-6)
        else:
            mel = torch.full_like(mel, -5.0)

        # Resize mask to match Mel dimensions (1D -> 1D subsampled)
        mask_final = torch.nn.functional.interpolate(
            mask_crop.unsqueeze(0),
            size=mel.shape[1],
            mode='nearest'
        ).squeeze(0)

        return mel, mask_final


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(CKPT_DIR):
        os.makedirs(CKPT_DIR)

    print(f"--- Training on {device} ---")

    if not os.path.exists(TRAIN_DIR):
        os.makedirs(TRAIN_DIR)
        print(f"Created {TRAIN_DIR}. Please put your files there and restart.")
        return

    # Training loader
    dataset = SlicedDataset(TRAIN_DIR, segment_len=SEGMENT_LEN, overlap=0.50)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SmartCutterUNet(n_channels=N_MELS, n_classes=1).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Binary Cross Entropy + Dice for better boundary precision
    bce_crit = nn.BCELoss()
    dice_crit = DiceLoss() 

    # Trackers
    best_avg_loss = float('inf')
    best_stability_score = 0.0
    best_balanced_score = 0.0

    # dict for summary table
    summary_data = {
        "loss": {"epoch": 0, "loss": 0, "stab": 0},
        "stab": {"epoch": 0, "loss": 0, "stab": 0},
        "bal":  {"epoch": 0, "loss": 0, "stab": 0}
    }


    for epoch in range(EPOCHS):
        # TRAINING PHASE
        model.train()
        epoch_step_losses = []

        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{EPOCHS} [Train]")
        
        for mels, masks in pbar:
            mels, masks = mels.to(device), masks.to(device)

            preds = model(mels)

            # Combined Loss
            loss = bce_crit(preds, masks) + dice_crit(preds, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_step_losses.append(loss.item())
            pbar.set_postfix({'step_loss': f"{loss.item():.4f}"})

        # Calculate metrics
        avg_train_loss = np.mean(epoch_step_losses)
        stability = 1.0 / (1.0 + np.std(epoch_step_losses))
        balanced_score = (1.0 / (avg_train_loss + 1e-6)) * stability
        current_lr = optimizer.param_groups[0]['lr']

        # console stats logging
        print(f"Epoch {epoch+1} | AVG Loss: {avg_train_loss:.4f} | Stability: {stability:.2%} | LR: {current_lr:.8f}")

        # Save every epoch
        torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"model_epoch_{epoch+1}_{SAMPLE_RATE}.pth"))

        # Best Avg Loss
        if avg_train_loss < best_avg_loss:
            best_avg_loss = avg_train_loss
            summary_data["loss"] = {"epoch": epoch+1, "loss": avg_train_loss, "stab": stability}
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"best_avg_loss_{SAMPLE_RATE}.pth"))
            
        # Best Stability
        if stability > best_stability_score:
            best_stability_score = stability
            summary_data["stab"] = {"epoch": epoch+1, "loss": avg_train_loss, "stab": stability}
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"best_stability_{SAMPLE_RATE}.pth"))
            
        # Best Balanced
        if balanced_score > best_balanced_score:
            best_balanced_score = balanced_score
            summary_data["bal"] = {"epoch": epoch+1, "loss": avg_train_loss, "stab": stability}
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"best_balanced_{SAMPLE_RATE}.pth"))

        scheduler.step()

    # FINAL SUMMARY
    print("\n" + "="*65)
    print(" TRAINING COMPLETE - BEST MODEL SUMMARY")
    print("="*65)
    print(f"Best Avg Loss  | Epoch: {summary_data['loss']['epoch']:<3} | Loss: {summary_data['loss']['loss']:.4f} | Stab: {summary_data['loss']['stab']:.2%}")
    print(f"Best Stability | Epoch: {summary_data['stab']['epoch']:<3} | Loss: {summary_data['stab']['loss']:.4f} | Stab: {summary_data['stab']['stab']:.2%}")
    print(f"Best Balanced  | Epoch: {summary_data['bal']['epoch']:<3} | Loss: {summary_data['bal']['loss']:.4f} | Stab: {summary_data['bal']['stab']:.2%}")
    print("="*65)

if __name__ == "__main__":
    train()