import os
import glob
import torch
import torchaudio
import torchaudio.functional as F_audio
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model_v3 import DSCA_ResUNet_v3

# Config
INTENDED_SR_FOR_TRAIN = 48000  # Options: 32000, 40000, 48000
SAMPLE_RATE = INTENDED_SR_FOR_TRAIN
HOP_LENGTH = SAMPLE_RATE // 100

N_MELS = 160
N_FFT = 2048
SEGMENT_LEN = SAMPLE_RATE * 4 # ( 192,000 samples = 4 secs. )
TRAIN_DIR = "training_files"
VAL_DIR = "validation_files"
CKPT_DIR = "ckpts"

# Hparams
BATCH_SIZE = 8
MAX_EPOCHS = 1000
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
VALIDATE_INTERVAL = 1
PATIENCE_LIMIT = 10

# Random Gain
rand_gain_max = 1.0
rand_gain_min = 0.2

class DiceLoss(nn.Module):
    """Focuses on the overlap between mask and prediction."""
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)  

        return 1 - dice

class SlicedDataset(Dataset):
    def __init__(self, file_pairs, segment_len=SEGMENT_LEN, overlap=0.20):
        self.samples = []

        # Mel Spectrogram Config
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, 
            n_mels=N_MELS, 
            n_fft=N_FFT, 
            hop_length=HOP_LENGTH
        )
        print(f"[DATASET] Processing {len(file_pairs)} file pairs...")

        for r_file, m_file in file_pairs:
            # Load
            raw_wav, sr_r = torchaudio.load(r_file)
            marked_wav, sr_m = torchaudio.load(m_file)

            # Resample if needed
            if sr_r != SAMPLE_RATE: raw_wav = torchaudio.transforms.Resample(sr_r, SAMPLE_RATE)(raw_wav)
            if sr_m != SAMPLE_RATE: marked_wav = torchaudio.transforms.Resample(sr_m, SAMPLE_RATE)(marked_wav)

            # Match lengths
            min_len = min(raw_wav.shape[1], marked_wav.shape[1])
            raw_wav, marked_wav = raw_wav[:, :min_len], marked_wav[:, :min_len]

            # Normalize
            max_val = torch.max(torch.abs(marked_wav))
            if max_val > 0.95:
                scale = 0.95 / max_val
                marked_wav *= scale
                raw_wav *= scale

            # Generate Mask
            diff = torch.abs(marked_wav - raw_wav)

            # Smoothing (removes micro-gaps)
            diff = torch.nn.functional.avg_pool1d(diff.unsqueeze(0), kernel_size=101, stride=1, padding=50).squeeze(0)

            # Creation of binary mask
            mask = (diff > 0.05).float()

            # Slight mask dilation to ensure coverage
            mask = torch.nn.functional.max_pool1d(mask.unsqueeze(0), kernel_size=21, stride=1, padding=10).squeeze(0)

            # Slicing
            stride = int(segment_len * (1 - overlap))
            length = raw_wav.shape[1]

            for i in range(0, length - segment_len + 1, stride):
                r_slice = raw_wav[:, i : i + segment_len]
                m_slice = mask[:, i : i + segment_len]

                # meaningless silence filtering
                has_audio = r_slice.abs().max() > 0.001 # audio content presence check
                has_error = m_slice.max() > 0.5 # mask presence check

                if has_audio or has_error: 
                    self.samples.append((r_slice, m_slice))

        print(f"[DATASET] Loaded {len(self.samples)} segments.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw_crop, mask_crop = self.samples[idx]
        raw_crop = raw_crop.clone()
        mask_crop = mask_crop.clone()

        # --------------------   Dynamic Augmentation

        # Random Gain (Applied to input only)
        if raw_crop.abs().max() > 0.001: 
            gain = torch.empty(1).uniform_(rand_gain_min, rand_gain_max)
            raw_crop = raw_crop * gain

        # Decoy (Random noise injection in non-mask areas to prevent false positives)
        if torch.rand(1).item() < 0.3:
             # Create low level noise
            decoy = torch.randn_like(raw_crop) * torch.empty(1).uniform_(0.001, 0.01)
            # Decoy only where there is no mask
            raw_crop = raw_crop + (decoy * (1 - mask_crop))



        # --------------------   Processing
        # Transform to mel
        mel = self.mel_transform(raw_crop).squeeze(0) 

        # log normalization
        mel = torchaudio.transforms.AmplitudeToDB()(mel)
        min_db = -80.0
        max_db = 0.0
        mel = torch.clamp(mel, min=min_db, max=max_db)
        mel = (mel - min_db) / (max_db - min_db) # Scaled 0.0 to 1.0

        # Deltas
        delta = F_audio.compute_deltas(mel.unsqueeze(0)).squeeze(0)
        combined_input = torch.stack([mel, delta], dim=0)

        # Resize mask to match Mel dimensions
        mask_final = torch.nn.functional.interpolate(
            mask_crop.unsqueeze(0).unsqueeze(0),
            size=(1, combined_input.shape[2]), # (1, Time)
            mode='nearest'
        ).squeeze(0).squeeze(0)

        return combined_input, mask_final


def get_file_pairs(folder):
    if not os.path.exists(folder):
        return []

    raw_files = sorted(glob.glob(os.path.join(folder, "*_raw.wav")) + glob.glob(os.path.join(folder, "*_raw.flac")))
    marked_files = sorted(glob.glob(os.path.join(folder, "*_marked.wav")) + glob.glob(os.path.join(folder, "*_marked.flac")))

    # Check to ensure pairs match up
    if len(raw_files) != len(marked_files):
        print(f"[WARNING] Mismatch in {folder}: {len(raw_files)} raw vs {len(marked_files)} marked.")

    return list(zip(raw_files, marked_files))

def validate(model, loader, device, bce_crit, dice_crit):
    model.eval()
    total_loss = 0
    step_losses = []
    
    with torch.no_grad():
        for mels, masks in loader:
            mels, masks = mels.to(device), masks.to(device)
            masks_expanded = masks.unsqueeze(2).expand(-1, -1, 160, -1)

            output = model(mels)
            if isinstance(output, dict):
                output = output["main"]

            loss = bce_crit(output, masks_expanded) + dice_crit(output, masks_expanded)
            total_loss += loss.item()
            step_losses.append(loss.item())

    avg_loss = total_loss / len(loader)
    stability = 1.0 / (1.0 + np.std(step_losses)) # Val Stability
    return avg_loss, stability

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    if not os.path.exists(CKPT_DIR): os.makedirs(CKPT_DIR)
    if not os.path.exists(TRAIN_DIR): os.makedirs(TRAIN_DIR)
    if not os.path.exists(VAL_DIR): os.makedirs(VAL_DIR)

    print(f"[DEVICE] {device}")

    # Loading data
    train_pairs = get_file_pairs(TRAIN_DIR)
    val_pairs = get_file_pairs(VAL_DIR)

    if not train_pairs:
        print(f"[ERROR] No files in {TRAIN_DIR}. Please add *_raw and *_marked files.")
        return
    if not val_pairs:
        print(f"[WARNING] No files in {VAL_DIR}. Training without validation is risky!")
        # uncomment below to force split
        #split_idx = int(len(train_pairs) * 0.9)
        #val_pairs = train_pairs[split_idx:]
        #train_pairs = train_pairs[:split_idx]

    # Datasets preparation
    # Train
    train_ds = SlicedDataset(train_pairs, overlap=0.50)
    # Va
    val_ds = SlicedDataset(val_pairs, overlap=0.0) 

    # Dataloaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    # Model setup
    model = DSCA_ResUNet_v3(n_channels=2, n_classes=1).to(device)
    #model = torch.compile(model)

    # optimizer init
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # scheduler init
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Binary Cross Entropy + Dice for better boundary precision
    bce_crit = nn.BCELoss()
    dice_crit = DiceLoss()

    # Trackers
    best_val_loss = float('inf')
    early_stop_counter = 0

    aux_weight = 1.0
    prev_train_stability = 0.0

    # Loop
    for epoch in range(MAX_EPOCHS):
        model.train()
        train_losses = []

        if prev_train_stability > 0.90:
            aux_weight = 0.0
        elif prev_train_stability > 0.75:
            # Linear fade: 1.0 -> 0.0
            aux_weight = 1.0 - ((prev_train_stability - 0.75) / 0.15)
        else:
            aux_weight = 1.0
        aux_weight = max(0.0, min(1.0, aux_weight))

        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch: {epoch+1} [Train] (Aux: {aux_weight:.2f})")

        for mels, masks in pbar:
            mels, masks = mels.to(device), masks.to(device)

            # zero grad
            optimizer.zero_grad(set_to_none=True)

            # expand
            masks_expanded = masks.unsqueeze(2).expand(-1, -1, 160, -1) # [Batch, 160, Time] -> [Batch, 1, 160, Time]

            # get model outputs
            outputs = model(mels)

            if isinstance(outputs, dict):
                preds_main = outputs["main"]

                # Main Loss
                loss_main = bce_crit(outputs["main"], masks_expanded) + dice_crit(outputs["main"], masks_expanded)

                # Dynamic Aux Loss
                if aux_weight > 0:
                    loss_aux2 = bce_crit(outputs["aux2"], masks_expanded) + dice_crit(outputs["aux2"], masks_expanded)
                    loss_aux3 = bce_crit(outputs["aux3"], masks_expanded) + dice_crit(outputs["aux3"], masks_expanded)
                    loss = loss_main + (aux_weight * (0.5 * loss_aux2 + 0.4 * loss_aux3))
                else:
                    loss = loss_main
            else:
                loss = bce_crit(outputs, masks_expanded) + dice_crit(outputs, masks_expanded)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Calculate metrics
        avg_train_loss = np.mean(train_losses)
        train_stability = 1.0 / (1.0 + np.std(train_losses))
        prev_train_stability = train_stability

        print(f"      ->  TRAIN AVG LOSS: {avg_train_loss:.4f} | Stability: {train_stability:.2%}")

        # Validation & Scheduler
        if val_loader and (epoch + 1) % VALIDATE_INTERVAL == 0:
            val_loss, val_stab = validate(model, val_loader, device, bce_crit, dice_crit)
            curr_lr = optimizer.param_groups[0]['lr']
            
            print(f"      ->  VAL AVG LOSS:   {val_loss:.4f} | Stability: {val_stab:.2%} | LR: {curr_lr:.2e}")
            
            # Scheduler watches VAL loss now (Robustness check)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"best_model_{SAMPLE_RATE}.pth"))
                print("      ->  New Best Model Saved!")
            else:
                early_stop_counter += 1
                print(f"      ->  No improvement ({early_stop_counter}/{PATIENCE_LIMIT})")

            if early_stop_counter >= PATIENCE_LIMIT:
                print("\n[!!!] Early Stopping Triggered (Validation stagnated).")
                break

        torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"latest_model_{SAMPLE_RATE}.pth"))

if __name__ == "__main__":
    train()