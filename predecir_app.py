import tkinter as tk
from tkinter import messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.tooltip import ToolTip
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os
import io
import contextlib
import datetime

# --- CONFIGURACIÓN ---
THEME_NAME = "superhero" 

# --- 1. Cargar Modelo, Scaler y Opciones ---
MODEL_FILE = "modelo_casas.keras"
SCALER_FILE = "scaler_casas.pkl"
CSV_FILE = "casas_limpias.csv"

files_exist = all(os.path.exists(f) for f in [MODEL_FILE, SCALER_FILE, CSV_FILE])
model, scaler, df_opciones = None, None, None

NUMERIC_COLS = [
    'lotSize', 'age', 'landValue', 'livingArea', 'pctCollege',
    'bedrooms', 'fireplaces', 'bathrooms', 'rooms'
]
CATEGORICAL_COLS_KEYS = [
    'heating', 'fuel', 'sewer', 'waterfront', 'newConstruction', 'centralAir'
]

# Diccionario de Traducción para la Interfaz (Vista Usuario)
UI_LABELS = {
    'lotSize': 'Tamaño del Lote (sqft)',
    'livingArea': 'Área Habitable (sqft)',
    'landValue': 'Valor del Terreno ($)',
    'age': 'Antigüedad (Años)',
    'pctCollege': '% Educación Univ.',
    'rooms': 'Habitaciones Totales',
    'bedrooms': 'Dormitorios',
    'bathrooms': 'Baños Completos',
    'fireplaces': 'Chimeneas',
    'heating': 'Tipo de Calefacción',
    'fuel': 'Fuente de Combustible',
    'sewer': 'Sistema de Desagüe',
    'waterfront': 'Frente al Agua',
    'newConstruction': 'Nueva Construcción',
    'centralAir': 'Aire Acondicionado'
}

# Descripciones para los Tooltips (Ayuda al usuario)
FIELD_HELP = {
    'lotSize': 'Tamaño total del terreno en pies cuadrados.',
    'livingArea': 'Área habitable construida en pies cuadrados.',
    'landValue': 'Valor estimado del terreno (sin la casa).',
    'age': 'Antigüedad de la propiedad en años.',
    'pctCollege': 'Porcentaje del vecindario con educación universitaria.',
    'rooms': 'Número total de habitaciones (incluyendo cocina, sala, etc).',
    'bedrooms': 'Número de dormitorios.',
    'bathrooms': 'Número de baños completos.',
    'fireplaces': 'Cantidad de chimeneas en la casa.'
}

if files_exist:
    try:
        model = tf.keras.models.load_model(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        df_opciones = pd.read_csv(CSV_FILE)
        FINAL_FEATURES_ORDER = scaler.feature_names_in_
        CATEGORICAL_COLS = {key: sorted(df_opciones[key].unique().tolist()) for key in CATEGORICAL_COLS_KEYS}
    except Exception as e:
        print(f"Error: {e}")
        CATEGORICAL_COLS = {key: ["Error"] for key in CATEGORICAL_COLS_KEYS}
else:
    CATEGORICAL_COLS = {key: ["Demo"] for key in CATEGORICAL_COLS_KEYS}

# Historial de la sesión actual
prediction_history = []

# --- 2. Funciones Auxiliares ---

def validar_numero(texto_nuevo):
    """Permite solo la entrada de números y puntos decimales"""
    if texto_nuevo == "": return True
    try:
        float(texto_nuevo)
        return True
    except ValueError:
        return False

def mostrar_info_modelo():
    """Muestra una ventana emergente con el summary() del modelo de Keras"""
    if model is None:
        messagebox.showerror("Error", "El modelo no está cargado.")
        return

    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        model.summary()
    summary_string = stream.getvalue()

    info_window = ttk.Toplevel(app)
    info_window.title("Arquitectura de la Red Neuronal")
    info_window.geometry("700x500")
    
    ttk.Label(info_window, text="Detalles del Modelo TensorFlow/Keras", font=("Helvetica", 14, "bold"), bootstyle="info").pack(pady=10)
    text_area = ttk.Text(info_window, wrap="none", font=("Consolas", 9))
    text_area.insert("1.0", summary_string)
    text_area.config(state="disabled")
    text_area.pack(fill=BOTH, expand=True, padx=10, pady=10)

def mostrar_historial():
    """Muestra el historial de predicciones"""
    hist_window = ttk.Toplevel(app)
    hist_window.title("Historial de Sesión")
    hist_window.geometry("500x400")
    
    ttk.Label(hist_window, text="Historial de Predicciones", font=("Helvetica", 14, "bold"), bootstyle="primary").pack(pady=10)
    
    # Lista con scroll
    frame_list = ttk.Frame(hist_window)
    frame_list.pack(fill=BOTH, expand=True, padx=10, pady=10)
    
    scroll = ttk.Scrollbar(frame_list)
    scroll.pack(side=RIGHT, fill=Y)
    
    listbox = tk.Listbox(frame_list, yscrollcommand=scroll.set, font=("Consolas", 10), bg="#2b3e50", fg="white")
    listbox.pack(side=LEFT, fill=BOTH, expand=True)
    scroll.config(command=listbox.yview)
    
    if not prediction_history:
        listbox.insert(END, "No hay predicciones todavía.")
    else:
        for idx, item in enumerate(reversed(prediction_history), 1):
            listbox.insert(END, f"{idx}. {item['time']} -> ${item['price']:,.2f} ({item['desc']})")

def limpiar_formulario():
    """Resetea todos los campos a su estado inicial"""
    # Limpiar campos numéricos
    for col in NUMERIC_COLS:
        widgets[col].delete(0, tk.END)
        widgets[col].insert(0, "0")
    
    # Resetear desplegables al primer elemento
    for col in CATEGORICAL_COLS:
        widgets[col].current(0)
    
    # Resetear resultados visuales
    lbl_result_value.config(text="$ 0.00", bootstyle="inverse-secondary")
    lbl_status.config(text="Formulario reiniciado.", bootstyle="secondary")
    progress_bar['value'] = 0

def iniciar_prediccion():
    """Inicia la secuencia de animación y cálculo"""
    try:
        if float(widgets['livingArea'].get()) == 0 or float(widgets['lotSize'].get()) == 0:
             messagebox.showwarning("Datos Incompletos", "Por favor ingresa 'LotSize' y 'LivingArea' mayores a 0.")
             return
    except ValueError:
        messagebox.showwarning("Error", "Revisa los números ingresados.")
        return

    btn_predict.config(state="disabled")
    progress_bar['value'] = 0
    lbl_result_value.config(text="Calculando...", bootstyle="secondary")
    actualizar_paso(1)

def actualizar_paso(paso):
    """Simula los pasos internos del pipeline de ML"""
    if paso == 1:
        lbl_status.config(text="Paso 1/4: Validando y recolectando inputs...", bootstyle="info")
        progress_bar['value'] = 25
        app.after(300, lambda: actualizar_paso(2))
    elif paso == 2:
        lbl_status.config(text="Paso 2/4: Aplicando One-Hot Encoding...", bootstyle="info")
        progress_bar['value'] = 50
        app.after(300, lambda: actualizar_paso(3))
    elif paso == 3:
        lbl_status.config(text="Paso 3/4: Escalando datos (StandardScaler)...", bootstyle="info")
        progress_bar['value'] = 75
        app.after(300, lambda: actualizar_paso(4))
    elif paso == 4:
        lbl_status.config(text="Paso 4/4: Inferencia en Red Neuronal...", bootstyle="warning")
        progress_bar['value'] = 90
        app.after(300, ejecutar_calculo_real)

def ejecutar_calculo_real():
    """Realiza la predicción real"""
    try:
        input_data = {}
        for col in NUMERIC_COLS:
            val = widgets[col].get()
            input_data[col] = float(val) if val else 0.0
        for col in CATEGORICAL_COLS:
            input_data[col] = widgets[col].get()

        input_df = pd.DataFrame([input_data])
        for col, options in CATEGORICAL_COLS.items():
            input_df[col] = pd.Categorical(input_df[col], categories=options)
        
        input_encoded = pd.get_dummies(input_df, columns=CATEGORICAL_COLS.keys(), drop_first=True)
        
        input_final = pd.DataFrame(columns=FINAL_FEATURES_ORDER)
        input_final = input_final.reindex(columns=FINAL_FEATURES_ORDER, fill_value=0)
        
        for col in input_encoded.columns:
            if col in input_final.columns:
                input_final[col] = input_encoded[col]

        input_scaled = scaler.transform(input_final)
        prediction = model.predict(input_scaled)
        price = prediction[0][0]

        # Guardar en historial
        ahora = datetime.datetime.now().strftime("%H:%M:%S")
        desc_corta = f"Area: {input_data['livingArea']}, Edad: {input_data['age']}"
        prediction_history.append({'time': ahora, 'price': price, 'desc': desc_corta})

        progress_bar['value'] = 100
        lbl_status.config(text="✔ Proceso completado exitosamente.", bootstyle="success")
        lbl_result_value.config(text=f"$ {price:,.2f}", bootstyle="inverse-success")

    except Exception as e:
        lbl_status.config(text="Error en el cálculo", bootstyle="danger")
        messagebox.showerror("Error Crítico", str(e))
    
    finally:
        btn_predict.config(state="normal")

# --- 3. Interfaz Gráfica ---

app = ttk.Window(themename=THEME_NAME)
app.title("AI Real Estate Predictor v4.0")
app.geometry("950x750")

# Registro de validación numérica para Inputs
vcmd = (app.register(validar_numero), '%P')

main_container = ttk.Frame(app, padding=20)
main_container.pack(fill=BOTH, expand=True)

# --- HEADER ---
header_frame = ttk.Frame(main_container)
header_frame.pack(fill=X, pady=(0, 15))

title_frame = ttk.Frame(header_frame)
title_frame.pack(side=LEFT)
ttk.Label(title_frame, text="PREDICTOR INMOBILIARIO AI", font=("Helvetica", 22, "bold"), bootstyle="primary").pack(anchor=W)
ttk.Label(title_frame, text="Red Neuronal Profunda • TensorFlow", font=("Helvetica", 10), bootstyle="secondary").pack(anchor=W)

# Botones Header
btn_frame_header = ttk.Frame(header_frame)
btn_frame_header.pack(side=RIGHT)
ttk.Button(btn_frame_header, text="📜 Historial", bootstyle="outline-secondary", command=mostrar_historial).pack(side=LEFT, padx=5)
ttk.Button(btn_frame_header, text="ℹ Info Modelo", bootstyle="outline-info", command=mostrar_info_modelo).pack(side=LEFT, padx=5)

# --- BODY ---
body_frame = ttk.Frame(main_container)
body_frame.pack(fill=BOTH, expand=True)

widgets = {}
def create_input_group(parent, title, items, is_combo=False):
    frame = ttk.Labelframe(parent, text=title, padding=15, bootstyle="light")
    frame.pack(fill=X, pady=8)
    for i, item in enumerate(items):
        # Usamos el diccionario para mostrar el nombre en español, 
        # pero usamos 'item' (la clave en inglés) para la lógica interna.
        label_text = UI_LABELS.get(item, item.capitalize())
        lbl = ttk.Label(frame, text=label_text)
        lbl.grid(row=i, column=0, sticky=W, pady=5)
        
        if is_combo:
            inp = ttk.Combobox(frame, values=CATEGORICAL_COLS[item], state="readonly")
            if CATEGORICAL_COLS[item]: inp.current(0)
        else:
            # Input numérico con validación
            inp = ttk.Entry(frame, validate='key', validatecommand=vcmd)
            inp.insert(0, "0")
            # Agregar Tooltip si existe descripción
            if item in FIELD_HELP:
                ToolTip(inp, text=FIELD_HELP[item], bootstyle="info.inverse")
                ToolTip(lbl, text=FIELD_HELP[item], bootstyle="info.inverse")

        inp.grid(row=i, column=1, sticky=EW, padx=(10, 0))
        frame.columnconfigure(1, weight=1)
        widgets[item] = inp

col1 = ttk.Frame(body_frame); col1.pack(side=LEFT, fill=BOTH, expand=True, padx=5)
col2 = ttk.Frame(body_frame); col2.pack(side=LEFT, fill=BOTH, expand=True, padx=5)
col3 = ttk.Frame(body_frame); col3.pack(side=LEFT, fill=BOTH, expand=True, padx=5)

create_input_group(col1, "Dimensiones", ['lotSize', 'livingArea', 'landValue'])
create_input_group(col1, "Distribución", ['rooms', 'bedrooms', 'bathrooms'])
create_input_group(col2, "Detalles", ['age', 'fireplaces', 'pctCollege'])
create_input_group(col2, "Sistemas", ['heating', 'fuel', 'sewer'], is_combo=True)
create_input_group(col3, "Extras", ['centralAir', 'waterfront', 'newConstruction'], is_combo=True)

# --- RESULT PANEL ---
result_panel = ttk.Labelframe(main_container, text="Panel de Control e Inferencia", padding=20, bootstyle="primary")
result_panel.pack(fill=X, side=BOTTOM, pady=10)

ctrl_frame = ttk.Frame(result_panel)
ctrl_frame.pack(fill=X)

# Botones de Acción
btn_predict = ttk.Button(ctrl_frame, text="CALCULAR VALOR ➔", command=iniciar_prediccion, style="success.TButton", width=20)
btn_predict.pack(side=LEFT)

# --- BOTÓN DE LIMPIAR ---
btn_clean = ttk.Button(ctrl_frame, text="🧹 Limpiar", command=limpiar_formulario, bootstyle="secondary", width=10)
btn_clean.pack(side=LEFT, padx=10)
# ------------------------

lbl_status = ttk.Label(ctrl_frame, text="Sistema listo. Esperando datos.", font=("Consolas", 10), bootstyle="secondary")
lbl_status.pack(side=LEFT, padx=10)

progress_bar = ttk.Progressbar(result_panel, value=0, maximum=100, bootstyle="success-striped")
progress_bar.pack(fill=X, pady=15)

lbl_result_value = ttk.Label(result_panel, text="$ 0.00", font=("Helvetica", 36, "bold"), bootstyle="inverse-success", anchor="center")
lbl_result_value.pack(fill=X)

if not files_exist:
    messagebox.showwarning("Modo Demo", "Archivos no encontrados. Modo diseño activado.")

app.mainloop()