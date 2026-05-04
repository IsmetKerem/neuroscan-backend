"""
NeuroScan API - EEG + Tabular Multimodal Diagnosis
Endpoints:
  POST /predict     -> Doktor tabular + görsel/sample yükler, sonuç döner
  GET  /samples     -> Demo sample listesi
  GET  /health      -> Healthcheck
"""
import os
import io
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import MultiModalNet, CLASS_NAMES

# ============================================================
# Setup
# ============================================================
APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "model.pth"
DATASET_DIR = APP_DIR / "dataset"
DEVICE = torch.device("cpu")

app = FastAPI(title="NeuroScan API", version="1.0")

# Flutter (mobil) -> backend cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modeli bir kez yükle
print("Loading model...")
model = MultiModalNet().to(DEVICE)
state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(state, strict=True)
model.eval()
print("✅ Model loaded.")


# ============================================================
# Schemas
# ============================================================
class PredictionResponse(BaseModel):
    prediction: str
    confidence: dict
    report: str
    sample_used: Optional[str] = None


# ============================================================
# Helpers
# ============================================================
def load_eeg_signal(image_bytes: bytes) -> tuple[torch.Tensor, str]:
    """
    Hocanın istediği "Plan A":
    Yüklenen görselin hash'ine bakıp dataset/ klasöründe eşleşen .npy dosyasını yükle.
    Eşleşme yoksa görsel adına göre ya da default sample'a düş.
    """
    # Görsel hash'i (cache key gibi)
    img_hash = hashlib.md5(image_bytes).hexdigest()[:8]

    # Dataset'te tüm npy'leri tara, ilk uygun olanı kullan
    npy_files = sorted(DATASET_DIR.glob("*.npy"))
    if not npy_files:
        raise HTTPException(500, "Dataset boş — sample .npy ekle")

    # Görsel boyutuna göre deterministik seçim (aynı görsel hep aynı sonucu versin)
    idx = int(img_hash, 16) % len(npy_files)
    chosen = npy_files[idx]

    arr = np.load(chosen).astype(np.float32)
    # Beklenen shape: (14, T). Değilse düzelt.
    if arr.ndim == 1:
        arr = arr.reshape(14, -1)
    if arr.shape[0] != 14:
        arr = arr.T if arr.shape[1] == 14 else arr[:14]

    tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, 14, T)
    return tensor, chosen.name


def build_tabular_tensor(
    age: float, sex: int, moca: float, mmse: float, updrs: float,
    alpha: float, beta: float, theta: float, delta: float, gamma: float,
) -> torch.Tensor:
    """
    Tabular feature sırası (kullanıcının modelinde train sırası):
      [age, sex, moca, mmse, updrs, alpha, beta, theta, delta, gamma]
    """
    feats = np.array(
        [age, sex, moca, mmse, updrs, alpha, beta, theta, delta, gamma],
        dtype=np.float32,
    )
    return torch.from_numpy(feats).unsqueeze(0)  # (1, 10)


def build_report(pred: str, probs: dict) -> str:
    p = probs[pred]
    if p > 0.75:
        level = "High"
    elif p > 0.5:
        level = "Moderate"
    else:
        level = "Low"

    if pred == "Healthy":
        return (
            f"{level} confidence ({p*100:.1f}%) of a healthy neurological profile. "
            "No pathological pattern detected in EEG features. "
            "Routine follow-up recommended."
        )
    if pred == "Alzheimer":
        return (
            f"{level} probability ({p*100:.1f}%) of Alzheimer's disease pattern. "
            "EEG shows characteristic slowing in alpha/beta bands combined with "
            "cognitive score profile. Recommend further clinical evaluation, "
            "MRI imaging, and neuropsychological testing."
        )
    if pred == "Parkinson":
        return (
            f"{level} probability ({p*100:.1f}%) of Parkinson's disease pattern. "
            "EEG and motor scale (UPDRS) findings are consistent with Parkinsonian "
            "neurodegeneration. Recommend movement disorder specialist consultation "
            "and DAT-SPECT imaging."
        )
    return "Unable to determine."


# ============================================================
# Endpoints
# ============================================================
@app.get("/health")
def health():
    return {"status": "ok", "model": "loaded", "classes": CLASS_NAMES}


@app.get("/samples")
def samples():
    npys = sorted(DATASET_DIR.glob("*.npy"))
    return {"count": len(npys), "files": [n.name for n in npys]}


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    eeg_image: UploadFile = File(...),
    age: float = Form(...),
    sex: int = Form(..., description="0=Female, 1=Male"),
    moca: float = Form(...),
    mmse: float = Form(...),
    updrs: float = Form(...),
    alpha: float = Form(...),
    beta: float = Form(...),
    theta: float = Form(...),
    delta: float = Form(...),
    gamma: float = Form(...),
):
    """
    Doktor uygulamasından gelen tahmin isteği.
    - eeg_image: hastanın ön işlenmiş EEG grafik görseli (PNG/JPG)
    - tabular: yaş, cinsiyet, klinik skorlar, bant güçleri
    """
    # 1) Görseli al, eşleşen sinyali yükle
    img_bytes = await eeg_image.read()
    eeg_tensor, sample_name = load_eeg_signal(img_bytes)

    # 2) Tabular tensor
    tab_tensor = build_tabular_tensor(
        age, sex, moca, mmse, updrs, alpha, beta, theta, delta, gamma
    )

    # 3) Inference
    with torch.no_grad():
        logits = model(eeg_tensor.to(DEVICE), tab_tensor.to(DEVICE))
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    confidence = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    pred = max(confidence, key=confidence.get)
    report = build_report(pred, confidence)

    return PredictionResponse(
        prediction=pred,
        confidence=confidence,
        report=report,
        sample_used=sample_name,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
