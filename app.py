#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pandas as pd
from tempfile import NamedTemporaryFile
from cnn_utils import read_wav_file, sample2MelSpectrum
from io import BytesIO
import wave
from fpdf import FPDF
from datetime import datetime
import sqlite3
import json
import os
import streamlit as st

# Solo en local intentamos importar sounddevice
if os.getenv("STREAMLIT_SERVER", "") == "":
    try:
        import sounddevice as sd
    except OSError:
        sd = None
else:
    sd = None  # en Cloud sd no estará disponible


# Importamos la función de inferencia que usa tu RandomForest

from inference import predict_group3_from_audio

# Mapeo diagnóstico → gran familia
GROUP_MAP = {
    'Asthma':         'Obstructiva',
    'COPD':           'Obstructiva',
    'Bronchiectasis': 'Obstructiva',
    # … otros obstructivos …
    'URTI':           'Infecciosa',
    'Pneumonia':      'Infecciosa',
    'Bronchiolitis':  'Infecciosa',
    # … otros infecciosos …
    'Healthy':        'Sano'
}

# ─── AQUÍ LA FUNCIÓN QUE FALTABA ───
def audio_segment_to_wav_bytes(segment: np.ndarray, sr: int) -> bytes:
    """
    Convierte un array float32 [-1,1] en bytes WAV (PCM int16).
    """
    # Clip & escalado a int16
    int16 = np.int16(np.clip(segment, -1, 1) * 32767)
    buf = BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)       # mono
        wf.setsampwidth(2)       # 2 bytes = 16 bits
        wf.setframerate(sr)
        wf.writeframes(int16.tobytes())
    return buf.getvalue()

# ─── CONSTANTES ───
AUDIO_DIR = '../input/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files/'

@st.cache_resource
def load_cnn_model():
    return load_model('models/cnn_model.h5')
model = load_cnn_model()
class_names = ['None', 'Crackles', 'Wheezes', 'Both']

st.title("Detección de Wheezes, Crackles y Enfermedad Respiratoria")


# ─── 1) Carga Audio ───
source = st.radio("Fuente de entrada:", ["Subir archivo", "Grabar micrófono"])
sr = 22000

if source == "Subir archivo":
    uploaded = st.file_uploader("Selecciona un WAV:", type="wav")
    if not uploaded:
        st.stop()
    data_bytes = uploaded.read()
    with NamedTemporaryFile(delete=False, suffix=".wav") as tmpf:
        tmpf.write(data_bytes)
        path = tmpf.name
    sr, data = read_wav_file(path, sr)
    st.audio(data_bytes, format="audio/wav")

else:  # Grabar micrófono
    # Si sd es None (p.ej. en Streamlit Cloud), avisamos y salimos
    if sd is None:
        st.warning("🔴 Grabación en vivo no disponible en este entorno.")
        st.stop()

    rec_dur = st.slider("Duración grabación (s)", 1, 30, 5)
    if not st.button("Grabar"):
        st.stop()
    # Grabar desde el micrófono
    rec = sd.rec(int(rec_dur * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    data = rec.flatten()
    # Convertir a bytes WAV
    data_bytes = audio_segment_to_wav_bytes(data, sr)
    filename = f"grab_{int(datetime.now().timestamp())}.wav"
    st.audio(data_bytes, format="audio/wav")
# --- Parámetros fijos ---
window_sec = 5.0
step_sec = 1.0
w_len = int(window_sec * sr)
duration = len(data) / sr

# --- 1) Mejor ventana de 5 s ---
best_score = -1
best_window = None
for start in np.arange(0, duration - window_sec + 1e-6, step_sec):
    idx = int(start * sr)
    seg = data[idx:idx + w_len]
    if len(seg) < w_len:
        seg = np.pad(seg, (0, w_len-len(seg)), mode='constant')
    mel, _ = sample2MelSpectrum((seg, False, False), sr, 50, None)
    m2 = mel.squeeze()
    pred = model.predict(m2.reshape(1, *m2.shape, 1), verbose=0)[0]
    score = pred[1] + pred[2] + pred[3]
    if score > best_score:
        best_score = score
        best_window = {'start': start, 'end': start+window_sec, 'probs': pred.tolist(), 'pred': class_names[int(np.argmax(pred))], 'mel': m2}

st.subheader("Mejor ventana (0-5s)")
fig, ax = plt.subplots(figsize=(6,3))
ax.imshow(best_window['mel'], aspect='auto'); ax.axis('off')
st.pyplot(fig)
for name,val in zip(class_names, best_window['probs']):
    st.write(f"{name}: {val:.3f}")
st.bar_chart(pd.DataFrame([best_window['probs']], columns=class_names))
st.success(f"Clase predicha: {best_window['pred']}")
st.markdown("---")

# --- 2) Análisis por segmentos de 5s ---
n_segments = int(np.ceil(duration / window_sec))
results = []
for i in range(n_segments):
    start = i * window_sec
    idx0 = int(start * sr)
    seg = data[idx0: idx0 + w_len]
    if len(seg) < w_len:
        seg = np.pad(seg, (0, w_len-len(seg)), mode='constant')
    mel, _ = sample2MelSpectrum((seg, False, False), sr, 50, None)
    m2 = mel.squeeze()
    pred = model.predict(m2.reshape(1, *m2.shape, 1), verbose=0)[0]
    label = class_names[int(np.argmax(pred))]
    entry = {'start': round(start,1), 'end': round(min(start+window_sec, duration),1), 'predicted': label}
    for j,name in enumerate(class_names): entry[name] = float(pred[j])
    results.append(entry)
    with st.expander(f"Segmento {entry['start']}-{entry['end']}s: {label}"):
        st.audio(audio_segment_to_wav_bytes(seg, sr), format='audio/wav')
        st.image(m2, use_container_width=True)
        st.write({name: f"{entry[name]:.3f}" for name in class_names})

# --- 3) Resumen global ---
df_res = pd.DataFrame(results)
pred_cls = df_res['predicted']
duration_per = pred_cls.value_counts().reindex(class_names, fill_value=0) * window_sec
percent_per = (duration_per / duration * 100).round(1)
summary = pd.DataFrame({'Duration (s)': duration_per, 'Percentage (%)': percent_per})
st.subheader("Resumen global de clases")
st.table(summary)
st.bar_chart(summary['Percentage (%)'])

# --- Resumen basado en probabilidades ---
prob_sum = {
    'None':     sum(r['None']     * window_sec for r in results),
    'Crackles': sum(r['Crackles'] * window_sec for r in results),
    'Wheezes':  sum(r['Wheezes']  * window_sec for r in results),
    'Both':     sum(r['Both']     * window_sec for r in results)
}
df_prob = pd.DataFrame.from_dict(prob_sum, orient='index', columns=['Duration prob (s)'])
st.markdown('---')
st.subheader('Resumen basado en probabilidades')
st.table(df_prob)
st.markdown("---")

# --- 4) Índice de severidad ---
st.subheader("Índice de severidad")
severity = sum(max(r['Crackles'], r['Wheezes'], r['Both'])*window_sec for r in results)
st.write(f"Severity score: {round(severity,3)}")

# ─── 7) Clasificación en 3-Grupos (Sano / Infecciosa / Obstructiva) ─────────
st.markdown("---")
st.subheader("Clasificación 3-Grupos (Sano / Infecciosa / Obstructiva)")

# Aquí usamos EXACTAMENTE la función de inference.py
grp3, probs3 = predict_group3_from_audio(data, sr)

st.write("**Grupo:**", grp3)
st.write("**Probabilidades:**")
st.json(probs3)

# --- 5) Descargar Informe PDF ---
st.subheader("Descargar Informe PDF")
patient_id = st.text_input("ID paciente")
notes = st.text_area("Observaciones (opcional)")
def generate_pdf():
    pdf = FPDF(); pdf.add_page(); pdf.set_font('Arial','B',16)
    pdf.cell(0,10,'Informe Wheezes y Crackles', ln=1, align='C')
    pdf.set_font('Arial','',12)
    pdf.cell(0,8,f'Paciente: {patient_id or "N/A"}', ln=1)
    pdf.cell(0,8,f'Fecha: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', ln=1)
    pdf.ln(5)

    pdf.set_font('Arial','B',14)
    pdf.cell(0,8,'Mejor ventana (0-5s)', ln=1)
    pdf.set_font('Arial','',12)
    pdf.cell(0,6,f"{best_window['start']:.1f}-{best_window['end']:.1f}s -> {best_window['pred']}", ln=1)
    pdf.ln(5)

    pdf.set_font('Arial','B',14)
    pdf.cell(0,8,'Resumen global', ln=1)
    pdf.set_font('Arial','',12)
    for cls in class_names:
        pdf.cell(0,6,f"{cls}: {duration_per[cls]}s ({percent_per[cls]}%)", ln=1)
    pdf.ln(5)

    # NUEVA SECCIÓN: Enfermedad detectada
    pdf.set_font('Arial','B',14)
    pdf.cell(0,8,'Clasificación 3-Grupos', ln=1)
    pdf.set_font('Arial','',12)
    pdf.cell(0,6,f"Enfermedad detectada: {grp3}", ln=1)
    pdf.ln(5)

    pdf.set_font('Arial','B',14)
    pdf.cell(0,8,'Observaciones', ln=1)
    pdf.set_font('Arial','',12)
    pdf.multi_cell(0,6, notes or 'Ninguna')

    pdf_str = pdf.output(dest='S')
    return pdf_str.encode('utf-8')



pdf_bytes = generate_pdf()
st.download_button("Descargar Informe PDF", pdf_bytes, file_name=f"informe_{patient_id}.pdf", mime="application/pdf")

# --- 6) Guardar en SQLite y ver historial ---
st.subheader("Guardar/Historial BBDD")
patient_db = st.text_input("ID paciente para BBDD")

if st.button("Guardar en BBDD"):
    conn = sqlite3.connect('resultados.db')
    cur = conn.cursor()

    # 1) Crear tabla si no existe, incluyendo la columna 'disease'
    cur.execute('''
        CREATE TABLE IF NOT EXISTS resultados (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            source TEXT,
            patient_id TEXT UNIQUE,
            filename TEXT,
            best_start REAL,
            best_end REAL,
            best_label TEXT,
            summary TEXT,
            disease TEXT
        )
    ''')

    # 2) Insertar o avisar si ya existe ese paciente
    cur.execute("SELECT COUNT(*) FROM resultados WHERE patient_id=?", (patient_db,))
    if cur.fetchone()[0]:
        st.error("El ID de paciente ya existe.")
    else:
        cur.execute(
            "INSERT INTO resultados "
            "(timestamp, source, patient_id, filename, best_start, best_end, best_label, summary, disease) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (
                datetime.now().isoformat(),
                source,
                patient_db,
                filename,
                best_window['start'],
                best_window['end'],
                best_window['pred'],
                json.dumps(results),
                grp3           # aquí guardamos la enfermedad detectada
            )
        )
        conn.commit()
        st.success("Guardado OK")

    conn.close()

if st.checkbox("Ver historial"):
    conn = sqlite3.connect('resultados.db')
    dfh = pd.read_sql_query('SELECT * FROM resultados', conn)
    # Exportar historial a CSV
    csv = dfh.to_csv(index=False).encode('utf-8')
    st.download_button("Exportar historial CSV", csv,
                       file_name="historial.csv", mime="text/csv")
    st.dataframe(dfh)
    conn.close()

# In[ ]:




