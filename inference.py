#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/env python
# inference_group3.py

import os
import numpy as np
import joblib
import librosa

from tensorflow.keras.models import load_model

# ─── 1) Rutas a artefactos ──────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
MODEL3_DIR     = os.path.join(BASE_DIR, "models_group3")
RF_PATH        = os.path.join(MODEL3_DIR, "rf_group3_tuned.pkl")        # Modelo tunado
LE_PATH        = os.path.join(MODEL3_DIR, "le_group3.pkl")
FEATCOLS3_PATH = os.path.join(MODEL3_DIR, "feature_cols_group3.pkl")
CNN_PATH       = os.path.join(BASE_DIR, "models", "cnn_model.h5")

# ─── 2) Cargo modelos y utilidades ──────────────────────────────────────────────
clf3         = joblib.load(RF_PATH)
le3          = joblib.load(LE_PATH)
feat_cols3   = joblib.load(FEATCOLS3_PATH)
cnn_extractor = load_model(CNN_PATH)

# ─── 3) Función mel-spectrum fijo ───────────────────────────────────────────────
def extract_mel_spec_fixed(y, sr,
                           n_mels=50, n_fft=512, hop_length=256,
                           target_frames=245):
    S      = librosa.feature.melspectrogram(y=y, sr=sr,
                  n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, power=2.0)
    mel_db = librosa.power_to_db(S, ref=np.max)
    mn, mx = mel_db.min(), mel_db.max()
    if mx > mn:
        mel = (mel_db - mn) / (mx - mn)
    else:
        mel = np.zeros_like(mel_db)
    if mel.shape[1] < target_frames:
        pad = target_frames - mel.shape[1]
        mel = np.pad(mel, ((0,0),(0,pad)), 'constant')
    else:
        mel = mel[:, :target_frames]
    return mel

# ─── 4) Extraer “pct-features” usando el extractor CNN ─────────────────────────
def extract_pct_features_from_signal(y, sr, window_sec=5.0, step_sec=1.0):
    wlen  = int(window_sec * sr)
    step  = int(step_sec   * sr)
    probs = []
    for start in range(0, len(y) - wlen + 1, step):
        seg = y[start:start+wlen]
        if len(seg) < wlen:
            seg = np.pad(seg, (0, wlen - len(seg)), 'constant')
        mel = extract_mel_spec_fixed(seg, sr)
        x   = mel[np.newaxis, ..., np.newaxis]
        p   = cnn_extractor.predict(x, verbose=0)[0]
        probs.append(p)
    if not probs:
        # en caso de audio muy corto, rellenamos con ceros de la longitud correcta
        return np.zeros(len(feat_cols3))
    return np.mean(np.vstack(probs), axis=0)

# ─── 5) Función principal de inferencia ─────────────────────────────────────────
def predict_group3_from_audio(y: np.ndarray, sr: int):
    """
    Devuelve (grupo_str, {grupo:prob}) donde grupo ∈ le3.classes_.
    """
    # 1) extraer pct-features
    pct = extract_pct_features_from_signal(y, sr)
    # 2) predecir con RF
    proba = clf3.predict_proba(pct.reshape(1, -1))[0]
    idx   = np.argmax(proba)
    grp   = le3.inverse_transform([idx])[0]
    # 3) devolver dct de probabilidades
    prob_dict = {cl: float(f"{p:.3f}") for cl, p in zip(le3.classes_, proba)}
    return grp, prob_dict

# ─── 6) CLI de prueba rápida ────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Uso: python inference_group3.py ruta/al/audio.wav")
        sys.exit(1)

    wav_path = sys.argv[1]
    y, sr = librosa.load(wav_path, sr=None)
    grp, probs = predict_group3_from_audio(y, sr)
    print("Predicción 3-grupos:", grp)
    print("Probabilidades:")
    for k, v in probs.items():
        print(f"  {k}: {v}")
