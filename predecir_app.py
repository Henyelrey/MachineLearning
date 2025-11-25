import tkinter as tk
from tkinter import messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os

# --- CONFIGURACIÓN ---
THEME_NAME = "superhero"  # Prueba: "journal", "flatly", "darkly", "superhero", "cyborg"

# --- 1. Cargar Modelo, Scaler y Opciones ---
MODEL_FILE = "modelo_casas.keras"
SCALER_FILE = "scaler_casas.pkl"
CSV_FILE = "casas_limpias.csv"

# Validaciones iniciales
files_exist = all(os.path.exists(f) for f in [MODEL_FILE, SCALER_FILE, CSV_FILE])
model, scaler, df_opciones = None, None, None

# Definiciones de columnas (Se mantienen igual que tu lógica)
NUMERIC_COLS = [
    'lotSize', 'age', 'landValue', 'livingArea', 'pctCollege',
    'bedrooms', 'fireplaces', 'bathrooms', 'rooms'
]
CATEGORICAL_COLS_KEYS = [
    'heating', 'fuel', 'sewer', 'waterfront', 'newConstruction', 'centralAir'
]

if files_exist:
    try:
        model = tf.keras.models.load_model(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        df_opciones = pd.read_csv(CSV_FILE)
        FINAL_FEATURES_ORDER = scaler.feature_names_in_
        
        # Cargar opciones reales del CSV
        CATEGORICAL_COLS = {
            key: sorted(df_opciones[key].unique().tolist()) for key in CATEGORICAL_COLS_KEYS
        }
    except Exception as e:
        print(f"Error cargando archivos ML: {e}")
        # Valores dummy por si falla la carga para que veas la interfaz
        CATEGORICAL_COLS = {key: ["Opcion A", "Opcion B"] for key in CATEGORICAL_COLS_KEYS}
        FINAL_FEATURES_ORDER = []
else:
    # Valores dummy para diseño si no hay archivos
    CATEGORICAL_COLS = {key: ["Opcion A", "Opcion B"] for key in CATEGORICAL_COLS_KEYS}
    FINAL_FEATURES_ORDER = []

# --- 2. Lógica de Predicción ---
def predecir_precio():
    if not files_exist or model is None:
        messagebox.showerror("Error", "No se cargaron los modelos de IA.")
        return

    # 1. Recolectar datos
    try:
        input_data = {}
        for col in NUMERIC_COLS:
            val = widgets[col].get()
            if not val: val = 0
            input_data[col] = float(val)
            
        for col in CATEGORICAL_COLS:
            input_data[col] = widgets[col].get()
            
    except ValueError:
        messagebox.showwarning("Error", "Revisa los campos numéricos.")
        return

    # --- VALIDACIÓN ESTRICTA CON MENSAJE ---
    # Verificamos si los campos críticos son 0.
    if input_data['livingArea'] == 0 or input_data['lotSize'] == 0:
        # 1. Mostrar mensaje emergente
        messagebox.showwarning(
            "Datos Incompletos", 
            "Para realizar una predicción válida, debes ingresar valores mayores a 0 en:\n\n- LotSize (Tamaño Lote)\n- LivingArea (Área Habitable)"
        )
        # 2. Resetear el resultado visual para no confundir
        lbl_result_value.config(text="$ 0.00", bootstyle="secondary")
        lbl_result_msg.config(text="Faltan datos", bootstyle="warning")
        return
    # ---------------------------------------------

    # 2. Procesamiento (Igual a tu script original)
    input_df = pd.DataFrame([input_data])
    
    for col, options in CATEGORICAL_COLS.items():
        input_df[col] = pd.Categorical(input_df[col], categories=options)

    input_encoded = pd.get_dummies(input_df, columns=CATEGORICAL_COLS.keys(), drop_first=True)

    input_final = pd.DataFrame(columns=FINAL_FEATURES_ORDER)
    input_final = input_final.reindex(columns=FINAL_FEATURES_ORDER, fill_value=0)
    
    for col in input_encoded.columns:
        if col in input_final.columns:
            input_final[col] = input_encoded[col]

    try:
        input_scaled = scaler.transform(input_final)
        prediction = model.predict(input_scaled)
        price = prediction[0][0]
        
        # Animación simple del resultado
        lbl_result_value.config(text=f"$ {price:,.2f}", bootstyle="inverse-success")
        lbl_result_msg.config(text="Valor Estimado", bootstyle="success")
        
    except Exception as e:
        messagebox.showerror("Error", f"Fallo en predicción: {e}")

# --- 3. Interfaz Gráfica (MEJORADA) ---

# Usamos ttkbootstrap Window en lugar de tk.Tk
app = ttk.Window(themename=THEME_NAME)
app.title("AI Real Estate Predictor")
app.geometry("900x650")

# Contenedor principal con scroll (por si la pantalla es chica)
main_container = ttk.Frame(app, padding=20)
main_container.pack(fill=BOTH, expand=True)

# --- CABECERA ---
header_frame = ttk.Frame(main_container)
header_frame.pack(fill=X, pady=(0, 20))

ttk.Label(
    header_frame, 
    text="PREDICTOR DE VALOR INMOBILIARIO", 
    font=("Helvetica", 20, "bold"),
    bootstyle="primary"
).pack(side=LEFT)

ttk.Label(
    header_frame, 
    text="v2.0 • Powered by TensorFlow", 
    font=("Helvetica", 10),
    bootstyle="secondary"
).pack(side=LEFT, padx=10, pady=8)

# --- CUERPO (Grid de Inputs) ---
# Dividiremos los inputs en 3 columnas lógicas para que se vea ordenado

body_frame = ttk.Frame(main_container)
body_frame.pack(fill=BOTH, expand=True)

widgets = {}

def create_input_group(parent, title, items, is_combo=False):
    """Ayuda a crear grupos de inputs con estilo"""
    frame = ttk.Labelframe(parent, text=title, padding=15, bootstyle="info")
    frame.pack(fill=X, pady=10)
    
    for i, item in enumerate(items):
        ttk.Label(frame, text=item.capitalize()).grid(row=i, column=0, sticky=W, pady=5)
        
        if is_combo:
            # Dropdown
            inp = ttk.Combobox(frame, values=CATEGORICAL_COLS[item], state="readonly")
            if CATEGORICAL_COLS[item]: inp.current(0)
        else:
            # Input numérico
            inp = ttk.Entry(frame)
            inp.insert(0, "0")
            
        inp.grid(row=i, column=1, sticky=EW, padx=(10, 0))
        frame.columnconfigure(1, weight=1)
        widgets[item] = inp

# Columnas de diseño
col1 = ttk.Frame(body_frame); col1.pack(side=LEFT, fill=BOTH, expand=True, padx=5)
col2 = ttk.Frame(body_frame); col2.pack(side=LEFT, fill=BOTH, expand=True, padx=5)
col3 = ttk.Frame(body_frame); col3.pack(side=LEFT, fill=BOTH, expand=True, padx=5)

# Grupo 1: Dimensiones y Habitaciones (Columna 1)
create_input_group(col1, "Dimensiones", ['lotSize', 'livingArea', 'landValue'])
create_input_group(col1, "Distribución", ['rooms', 'bedrooms', 'bathrooms'])

# Grupo 2: Características Técnicas (Columna 2)
create_input_group(col2, "Detalles", ['age', 'fireplaces', 'pctCollege'])
create_input_group(col2, "Sistemas", ['heating', 'fuel', 'sewer'], is_combo=True)

# Grupo 3: Extras y Estatus (Columna 3)
create_input_group(col3, "Características Extra", ['centralAir', 'waterfront', 'newConstruction'], is_combo=True)

# --- PANEL DE RESULTADOS Y BOTÓN ---
# Un panel lateral o inferior muy visible
result_panel = ttk.Frame(main_container, padding=(0, 20, 0, 0))
result_panel.pack(fill=X, side=BOTTOM)

# Separador
ttk.Separator(result_panel, orient=HORIZONTAL).pack(fill=X, pady=10)

btn_predict = ttk.Button(
    result_panel, 
    text="CALCULAR PRECIO ➔", 
    command=predecir_precio,
    style="success.TButton",
    width=25
)
btn_predict.pack(pady=10)

lbl_result_msg = ttk.Label(result_panel, text="Esperando datos...", font=("Helvetica", 12))
lbl_result_msg.pack()

lbl_result_value = ttk.Label(
    result_panel, 
    text="$ 0.00", 
    font=("Helvetica", 32, "bold"), 
    bootstyle="inverse-success"
)
lbl_result_value.pack(pady=5)

# Footer
ttk.Label(result_panel, text="El cálculo puede variar según condiciones del mercado", font=("Arial", 8), bootstyle="secondary").pack(pady=10)

if not files_exist:
    messagebox.showwarning("Modo Demo", "No se encontraron los archivos .keras o .pkl.\nLa interfaz se muestra en modo diseño, pero no predecirá.")

app.mainloop()