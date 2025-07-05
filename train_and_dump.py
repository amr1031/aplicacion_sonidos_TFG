#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/env python
# train_and_dump_bueno_tuned.py

import os, glob
import numpy as np
import pandas as pd
import joblib
import librosa

from sklearn.ensemble       import RandomForestClassifier
from sklearn.preprocessing  import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics        import accuracy_score, f1_score, classification_report, confusion_matrix

from tensorflow.keras.models import load_model

# 1) Rutas
AUDIO_DIR = 'input/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files/'
CSV_PATH  = 'input/Respiratory_Sound_Database/Respiratory_Sound_Database/patient_diagnosis.csv'
OUT_DIR   = 'models_group3'
os.makedirs(OUT_DIR, exist_ok=True)

# 2) Mapeo a 3 grupos
GROUP_MAP = {
    'Healthy':        'Sano',
    'URTI':           'Infecciosa',
    'Pneumonia':      'Infecciosa',
    'Bronchiolitis':  'Infecciosa',
    'LRTI':           'Infecciosa',
    'Bronchiectasis': 'Obstructiva',
    'COPD':           'Obstructiva',
    'Asthma':         'Obstructiva',
}

# 3) Carga diagnósticos y filtrado
df = pd.read_csv(CSV_PATH, header=None, names=['patient_id','diagnosis'])
df['group3'] = df['diagnosis'].map(GROUP_MAP)
df = df.dropna(subset=['group3']).reset_index(drop=True)

# 4.1) Función para mel-espec fijo (50×245)
def extract_mel_spec_fixed(y, sr,
                           n_mels=50, n_fft=512, hop_length=256,
                           target_frames=245):
    S      = librosa.feature.melspectrogram(y=y, sr=sr,
                  n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, power=2.0)
    mel_db = librosa.power_to_db(S, ref=np.max)
    mn, mx = mel_db.min(), mel_db.max()
    if mx>mn:
        mel = (mel_db-mn)/(mx-mn)
    else:
        mel = np.zeros_like(mel_db)
    if mel.shape[1] < target_frames:
        pad = target_frames-mel.shape[1]
        mel = np.pad(mel, ((0,0),(0,pad)), 'constant')
    else:
        mel = mel[:, :target_frames]
    return mel

# 4.2) Carga tu extractor original de CW
CNN_PATH   = os.path.join('models', 'cnn_model.h5')
cnn_model  = load_model(CNN_PATH)

# 4.3) Función para extraer pct-features de un señal completo
def extract_pct_features_from_signal(y, sr,
                                     window_sec=5.0, step_sec=1.0):
    wlen  = int(window_sec * sr)
    step  = int(step_sec   * sr)
    probs = []
    for start in range(0, len(y)-wlen+1, step):
        seg = y[start:start+wlen]
        if len(seg)<wlen:
            seg = np.pad(seg, (0, wlen-len(seg)), 'constant')
        mel = extract_mel_spec_fixed(seg, sr)
        x   = mel[np.newaxis, ..., np.newaxis]
        p   = cnn_model.predict(x, verbose=0)[0]
        probs.append(p)
    if not probs:
        return [0.,0.,0.,0.]
    return np.mean(np.vstack(probs), axis=0).tolist()

# 5) Construir DataFrame de features
records = []
for pid in df['patient_id']:
    wavs = glob.glob(os.path.join(AUDIO_DIR, f"{pid}_*.wav"))
    feats = []
    for w in wavs:
        y,sr = librosa.load(w, sr=None)
        feats.append(extract_pct_features_from_signal(y, sr))
    feats = feats or [[0.,0.,0.,0.]]
    mean_feats = np.mean(feats, axis=0)
    records.append([pid, *mean_feats])

feat_cols = ['patient_id','pct_none','pct_crackle','pct_wheeze','pct_both']
df_feat = pd.DataFrame(records, columns=feat_cols)

# 6) Merge con etiquetas y preparo X, y
df_full = df_feat.merge(df[['patient_id','group3']], on='patient_id')
X = df_full[feat_cols[1:]].values
le = LabelEncoder().fit(df_full['group3'])
y  = le.transform(df_full['group3'])

# 7) Hold-out split + RF tunado
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

clf = RandomForestClassifier(
    n_estimators=300,                   # más árboles para mayor estabilidad
    criterion='entropy',                # split por ganancia de información
    max_depth=None,                     # deja crecer hasta tope
    min_samples_split=5,                # evita divisiones con pocas muestras
    min_samples_leaf=2,                 # hojas con al menos 2 muestras
    class_weight='balanced_subsample',  # penaliza automáticamente clases minoritarias
    oob_score=True,                     # puntuación “out-of-bag” interna
    n_jobs=-1,                          # usa todos los núcleos CPU
    random_state=42
)
clf.fit(X_tr, y_tr)

# 8) Métricas
y_pr = clf.predict(X_te)
print("\n=== Hold-out Metrics ===")
print(f"Accuracy   = {accuracy_score(y_te, y_pr):.4f}")
print(f"Macro-F1   = {f1_score(y_te, y_pr, average='macro'):.4f}\n")
print(classification_report(
    le.inverse_transform(y_te),
    le.inverse_transform(y_pr),
    zero_division=0
))
print("Confusion matrix:")
print(confusion_matrix(y_te, y_pr))

# 9) Serializo RF, encoder y columnas
joblib.dump(clf,               os.path.join(OUT_DIR, 'rf_group3_tuned.pkl'))
joblib.dump(le,                os.path.join(OUT_DIR, 'le_group3.pkl'))
joblib.dump(feat_cols[1:],     os.path.join(OUT_DIR, 'feature_cols_group3.pkl'))
print(f"\n✓ Modelo tunado guardado en '{OUT_DIR}/rf_group3_tuned.pkl'")
