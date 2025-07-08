# Aplicación_StreamlitCloud
<img src='INPUT/IMAGENES/escudoUBU.jpeg' align="right" height="120" />

> Repositorio del Trabajo Fin de Grado para el grado de Ingeniería de la Salud, Universidad de Burgos (UBU), 2024-2025;

> **ALVARO MARTIN** ([amr1031\@alu.ubu.es](mailto:amr1031@alu.ubu.es))[📩](https://emojipedia.org/shortcodes)
> - 4º INGENERIA DE LA SALUD / [TRABAJO FIN DE GRADO.](https://ubuvirtual.ubu.es/course/view.php?id=15233)[🎓](https://emojipedia.org/shortcodes) 


Este repositorio reúne todo lo necesario para ejecutar la aplicación web de detección de crepitaciones, sibilancias y diagnóstico de enfermedad respiratoria en Streamlit Cloud.

---

## Despliegue en Streamlit Cloud

1. **Repositorio público**  
   URL: `https://github.com/amr1031/aplicacion_sonidos_TFG`

2. **Conectar con Streamlit Cloud**  
   - Accede a https://streamlit.io/cloud y haz login con tu cuenta de GitHub.  
   - Pulsa **New app**, selecciona:
     - **Repository**: `amr1031/aplicacion_sonidos_TFG`  
     - **Branch**: `main`  
     - **File path**: `app.py`  
   - Haz clic en **Deploy**.  
   - En segundos tendrás una URL pública donde probar la app.

3. **Estructura del repositorio**  
   ```text
   ├── audios_muestra/          # WAV de ejemplo para testing
   ├── models/                  # Modelo CNN preentrenado
   ├── models_group3/           # RF entrenado, encoder y columnas
   ├── app.py                   # Script principal de Streamlit
   ├── inference.py             # Lógica de inferencia CNN→RF
   ├── cnn_utils.py             # Funciones de preprocesado (lectura WAV, mel)
   ├── requirements.txt         # Dependencias Python
   └── resultados.db            # SQLite para historial (opcional)
