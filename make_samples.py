"""
Demo için sentetik EEG örnekleri üretir.
Gerçek dataset (DS004504, DS004584) ile değiştirilebilir.
14 kanal, 256 örnek (~1 saniye @ 256Hz).
"""
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent / "dataset"
OUT.mkdir(exist_ok=True)

np.random.seed(42)

def make_signal(profile: str, n_channels=14, n_samples=256, fs=256):
    """profile: 'healthy' / 'alzheimer' / 'parkinson'"""
    t = np.arange(n_samples) / fs
    sig = np.zeros((n_channels, n_samples), dtype=np.float32)

    for ch in range(n_channels):
        # Alfa, Beta, Teta, Delta, Gamma bant karışımları profile'a göre değişir
        if profile == "healthy":
            alpha_amp, beta_amp, theta_amp = 1.0, 0.6, 0.3
        elif profile == "alzheimer":
            # AD: alfa/beta düşer, teta/delta artar
            alpha_amp, beta_amp, theta_amp = 0.4, 0.3, 1.0
        elif profile == "parkinson":
            # PD: beta artışı, tremor 4-6 Hz
            alpha_amp, beta_amp, theta_amp = 0.7, 1.2, 0.6
        else:
            alpha_amp, beta_amp, theta_amp = 1, 1, 1

        sig[ch] += alpha_amp * np.sin(2 * np.pi * 10 * t + ch * 0.1)
        sig[ch] += beta_amp  * np.sin(2 * np.pi * 20 * t + ch * 0.2)
        sig[ch] += theta_amp * np.sin(2 * np.pi * 6  * t + ch * 0.3)
        sig[ch] += 0.4 * np.random.randn(n_samples)

    # Normalize
    sig = (sig - sig.mean()) / (sig.std() + 1e-6)
    return sig.astype(np.float32)


# 6 örnek üret (her sınıftan 2)
samples = [
    ("healthy_01.npy",   "healthy"),
    ("healthy_02.npy",   "healthy"),
    ("alzheimer_01.npy", "alzheimer"),
    ("alzheimer_02.npy", "alzheimer"),
    ("parkinson_01.npy", "parkinson"),
    ("parkinson_02.npy", "parkinson"),
]

for fname, prof in samples:
    arr = make_signal(prof)
    np.save(OUT / fname, arr)
    print(f"  ✓ {fname}  shape={arr.shape}  profile={prof}")

print(f"\n✅ {len(samples)} örnek oluşturuldu: {OUT}")
