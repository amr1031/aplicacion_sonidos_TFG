#!/usr/bin/env python
# coding: utf-8

# In[1]:
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
import sounddevice as sd
import os

# --- Utilidad para convertir segmento a WAV bytes ---
def audio_segment_to_wav_bytes(segment: np.ndarray, sr: int) -> bytes:
    int16 = np.int16(np.clip(segment, -1, 1) * 32767)
    buf = BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(int16.tobytes())
    return buf.getvalue()

# --- Carga del modelo con caché ---
@st.cache_resource
def load_cnn_model():
    return load_model('models/cnn_model.h5')
model = load_cnn_model()
class_names = ['None', 'Crackles', 'Wheezes', 'Both']

st.title("Detección de Wheezes y Crackles")
# Mostrar versión del modelo
model_path = 'models/cnn_model.h5'
mod_time = datetime.fromtimestamp(os.path.getmtime(model_path)).strftime('%Y-%m-%d %H:%M:%S')
st.text(f"Modelo: {model_path} (modificado: {mod_time})")

# --- Selección de fuente ---
source = st.radio("Fuente de entrada:", ["Subir archivo", "Grabar micrófono"])

sr = 22000
if source == "Subir archivo":
    uploaded = st.file_uploader("Selecciona un archivo WAV:", type=['wav'])
    if not uploaded:
        st.info("Por favor, sube un archivo WAV.")
        st.stop()
    data_bytes = uploaded.read()
    filename = uploaded.name
    with NamedTemporaryFile(delete=False, suffix='.wav') as tmpf:
        tmpf.write(data_bytes)
        path = tmpf.name
    sr, data = read_wav_file(path, sr)
    st.audio(data_bytes, format='audio/wav')
else:
    rec_duration = st.slider("Duración de grabación (s)", min_value=1, max_value=30, value=5)
    if st.button("📢 Grabar desde micrófono"):
        rec = sd.rec(int(rec_duration * sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()
        data = rec.flatten()
        data_bytes = audio_segment_to_wav_bytes(data, sr)
        filename = f"grabacion_{int(datetime.now().timestamp())}.wav"
        st.audio(data_bytes, format='audio/wav')
    else:
        st.stop()

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
    pdf.set_font('Arial','B',14); pdf.cell(0,8,'Mejor ventana (0-5s)', ln=1)
    pdf.set_font('Arial','',12)
    pdf.cell(0,6,f"{best_window['start']:.1f}-{best_window['end']:.1f}s -> {best_window['pred']}", ln=1)
    pdf.ln(5)
    pdf.set_font('Arial','B',14); pdf.cell(0,8,'Resumen global', ln=1)
    pdf.set_font('Arial','',12)
    for cls in class_names:
        pdf.cell(0,6,f"{cls}: {duration_per[cls]}s ({percent_per[cls]}%)", ln=1)
    pdf.ln(5)
    pdf.set_font('Arial','B',14); pdf.cell(0,8,'Observaciones', ln=1)
    pdf.set_font('Arial','',12); pdf.multi_cell(0,6, notes or 'Ninguna')
    return bytes(pdf.output(dest='S'))

pdf_bytes = generate_pdf()
st.download_button("Descargar Informe PDF", pdf_bytes, file_name="informe.pdf", mime="application/pdf")

# --- 6) Guardar en SQLite y ver historial ---
st.subheader("Guardar/Historial BBDD")
patient_db = st.text_input("ID paciente para BBDD")
if st.button("Guardar en BBDD"):
    conn = sqlite3.connect('results.db'); cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY, timestamp TEXT, source TEXT, patient_id TEXT UNIQUE,
        filename TEXT, best_start REAL, best_end REAL, best_label TEXT, summary TEXT
    )''')
    cur.execute("SELECT COUNT(*) FROM results WHERE patient_id=?", (patient_db,))
    if cur.fetchone()[0]:
        st.error("El ID de paciente ya existe.")
    else:
        cur.execute(
            "INSERT INTO results (timestamp, source, patient_id, filename, best_start, best_end, best_label, summary) VALUES (?,?,?,?,?,?,?,?)",
            (datetime.now().isoformat(), source, patient_db, filename,
             best_window['start'], best_window['end'], best_window['pred'], json.dumps(results))
        )
        conn.commit(); st.success("Guardado OK")
    conn.close()
if st.checkbox("Ver historial"):
    conn = sqlite3.connect('results.db')
    dfh = pd.read_sql_query('SELECT * FROM results', conn)
    # Exportar historial a CSV
    csv = dfh.to_csv(index=False).encode('utf-8')
    st.download_button("Exportar historial CSV", csv, file_name="historial.csv", mime="text/csv")
    st.dataframe(dfh)
    conn.close()


# In[ ]:




