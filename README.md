# NeuroScan Backend

EEG + Tabular multimodal diagnosis API (Healthy / Alzheimer / Parkinson).

## Çalıştırma

```bash
pip install -r requirements.txt
python make_samples.py        # demo .npy'leri üret (1 kere yeter)
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Telefondan test (Flutter ile)

1. Bilgisayarın yerel IP'sini öğren: `ipconfig` (Windows) / `ifconfig` (Mac/Linux)
2. Telefon ve bilgisayar **aynı Wi-Fi'da** olmalı.
3. Flutter'da base URL: `http://<bilgisayar_IP>:8000`

## Endpoints

### `GET /health`
```json
{"status": "ok", "model": "loaded", "classes": ["Healthy","Alzheimer","Parkinson"]}
```

### `POST /predict` (multipart/form-data)
**Fields:**
- `eeg_image`: file (PNG/JPG)
- `age`, `sex` (0=F, 1=M), `moca`, `mmse`, `updrs`
- `alpha`, `beta`, `theta`, `delta`, `gamma` (band powers)

**Response:**
```json
{
  "prediction": "Alzheimer",
  "confidence": {"Healthy": 0.1, "Alzheimer": 0.7, "Parkinson": 0.2},
  "report": "High probability (70.0%) of Alzheimer's disease pattern...",
  "sample_used": "alzheimer_01.npy"
}
```

## Model

- `MultiModalNet` (PyTorch)
- EEG branch: Conv1D(14→32→64) + Attention → 128-d
- Tabular branch: Linear(10→32→64)
- Fusion classifier: 192 → 64 → 3
- Strict-loaded `model.pth` ✅
