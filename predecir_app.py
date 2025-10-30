import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os

# --- 1. Cargar Modelo, Scaler y Opciones ---

# Nombres de archivos
MODEL_FILE = "modelo_casas.keras"
SCALER_FILE = "scaler_casas.pkl"
CSV_FILE = "casas_limpias.csv"

# Validar que los archivos existan
if not all(os.path.exists(f) for f in [MODEL_FILE, SCALER_FILE, CSV_FILE]):
    messagebox.showerror(
        "Error de Archivos",
        f"No se encontraron uno o más archivos necesarios:\n"
        f"- {MODEL_FILE}\n- {SCALER_FILE}\n- {CSV_FILE}\n\n"
        "Asegúrate de que estén en la misma carpeta que esta aplicación."
    )
    exit()

try:
    # Cargar el modelo y el escalador al inicio
    model = tf.keras.models.load_model(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    
    # Cargar el CSV solo para obtener las opciones de los menús desplegables
    df_opciones = pd.read_csv(CSV_FILE)
except Exception as e:
    messagebox.showerror(
        "Error al Cargar",
        f"No se pudieron cargar los archivos del modelo o el escalador: {e}"
    )
    exit()

# --- 2. Definir las Características (Features) ---

# Columnas numéricas (basadas en tu script de entrenamiento)
NUMERIC_COLS = [
    'lotSize', 'age', 'landValue', 'livingArea', 'pctCollege',
    'bedrooms', 'fireplaces', 'bathrooms', 'rooms'
]

# Columnas categóricas y sus opciones (leídas del CSV)
CATEGORICAL_COLS = {
    'heating': sorted(df_opciones['heating'].unique().tolist()),
    'fuel': sorted(df_opciones['fuel'].unique().tolist()),
    'sewer': sorted(df_opciones['sewer'].unique().tolist()),
    'waterfront': sorted(df_opciones['waterfront'].unique().tolist()),
    'newConstruction': sorted(df_opciones['newConstruction'].unique().tolist()),
    'centralAir': sorted(df_opciones['centralAir'].unique().tolist()),
}

# Obtenemos la lista exacta de 18 características que el modelo espera
# de los archivos del escalador
FINAL_FEATURES_ORDER = scaler.feature_names_in_

# --- 3. Lógica de Predicción ---

def predecir_precio():
    """
    Toma los valores de la GUI, los procesa y muestra la predicción.
    """
    
    # 1. Recolectar datos de la GUI
    try:
        input_data = {}
        # Recolectar numéricos
        for col in NUMERIC_COLS:
            input_data[col] = float(widgets[col].get())
            
        # Recolectar categóricos
        for col in CATEGORICAL_COLS:
            input_data[col] = widgets[col].get()
            
    except ValueError:
        messagebox.showwarning(
            "Entrada Inválida",
            "Por favor, ingrese números válidos en todos los campos numéricos."
        )
        return

    # 2. Crear un DataFrame de una fila
    input_df = pd.DataFrame([input_data])

    # 3. Aplicar One-Hot Encoding (como en el script de entrenamiento)
    # Convertimos las columnas categóricas a tipo "category" con todas las
    # opciones posibles, para asegurar que pd.get_dummies cree todas
    # las columnas necesarias, incluso si no están en esta fila.
    for col, options in CATEGORICAL_COLS.items():
        input_df[col] = pd.Categorical(input_df[col], categories=options)

    # Aplicamos get_dummies
    input_encoded = pd.get_dummies(
        input_df, 
        columns=CATEGORICAL_COLS.keys(), 
        drop_first=True
    )

    # 4. Reordenar y rellenar columnas
    # Aseguramos que el DataFrame tenga exactamente las mismas 18 columnas
    # en el orden correcto que el modelo espera.
    # Rellenamos con 0 las columnas que no se generaron (si aplica)
    input_final = pd.DataFrame(columns=FINAL_FEATURES_ORDER)
    input_final = input_final.reindex(columns=FINAL_FEATURES_ORDER, fill_value=0)
    
    # Copiamos los valores que sí tenemos
    for col in input_encoded.columns:
        if col in input_final.columns:
            input_final[col] = input_encoded[col]

    # 5. Escalar los datos
    # Usamos el escalador cargado
    try:
        input_scaled = scaler.transform(input_final)
    except Exception as e:
        messagebox.showerror("Error de Escalado", f"Error al transformar los datos: {e}\n\nDatos:\n{input_final.to_string()}")
        return

    # 6. Realizar la predicción
    try:
        prediction = model.predict(input_scaled)
        predicted_price = prediction[0][0]
        
        # 7. Mostrar el resultado
        result_var.set(f"Precio Predicho: ${predicted_price:,.2f}")
        
    except Exception as e:
        messagebox.showerror("Error de Predicción", f"Error al predecir: {e}")
        result_var.set("Error al predecir.")


# --- 4. Configuración de la Interfaz (GUI) ---
app = tk.Tk()
app.title("Predictor de Precios de Casas")
app.geometry("600x550")

# Estilo
style = ttk.Style(app)
style.theme_use("clam")
style.configure("TLabel", padding=5, font=("Arial", 10))
style.configure("TEntry", padding=5)
style.configure("TCombobox", padding=5)
style.configure("Accent.TButton", font=("Arial", 12, "bold"), padding=10)
style.configure("Result.TLabel", font=("Arial", 14, "bold"), padding=10, foreground="#0078D4")

# Frame principal
main_frame = ttk.Frame(app, padding="15")
main_frame.pack(fill=tk.BOTH, expand=True)

# --- Frame de Entradas (Inputs) ---
inputs_frame = ttk.LabelFrame(main_frame, text="Especificaciones de la Casa", padding="10")
inputs_frame.pack(fill=tk.X, expand=True)

# Crear la cuadrícula de widgets (2 columnas)
cols = 2
row_num = 0
col_num = 0
widgets = {}

# Crear campos numéricos
for col in NUMERIC_COLS:
    ttk.Label(inputs_frame, text=f"{col}:").grid(row=row_num, column=col_num*2, sticky=tk.W, padx=5)
    entry = ttk.Entry(inputs_frame, width=15)
    entry.grid(row=row_num, column=col_num*2 + 1, sticky=tk.EW, padx=5, pady=2)
    entry.insert(0, "0") # Valor por defecto
    widgets[col] = entry
    
    col_num = (col_num + 1) % cols
    if col_num == 0:
        row_num += 1

# Asegurar que los categóricos empiecen en una nueva fila
if col_num != 0:
    row_num += 1
    col_num = 0

# Crear campos categóricos (desplegables)
for col, options in CATEGORICAL_COLS.items():
    ttk.Label(inputs_frame, text=f"{col}:").grid(row=row_num, column=col_num*2, sticky=tk.W, padx=5)
    combo = ttk.Combobox(inputs_frame, values=options, state="readonly", width=18)
    combo.grid(row=row_num, column=col_num*2 + 1, sticky=tk.EW, padx=5, pady=2)
    combo.current(0) # Seleccionar el primer ítem
    widgets[col] = combo
    
    col_num = (col_num + 1) % cols
    if col_num == 0:
        row_num += 1

# Configurar expansión de columnas en el grid
inputs_frame.grid_columnconfigure(1, weight=1)
inputs_frame.grid_columnconfigure(3, weight=1)

# --- Frame de Resultados ---
result_frame = ttk.Frame(main_frame, padding="10")
result_frame.pack(fill=tk.X, expand=True)

# Botón de predicción
btn_predict = ttk.Button(
    result_frame,
    text="Predecir Precio",
    command=predecir_precio,
    style="Accent.TButton"
)
btn_predict.pack(pady=15)

# Etiqueta de resultado
result_var = tk.StringVar()
result_var.set("Ingrese las especificaciones y presione 'Predecir'")
lbl_result = ttk.Label(
    result_frame,
    textvariable=result_var,
    style="Result.TLabel",
    anchor=tk.CENTER
)
lbl_result.pack(pady=10)

# Iniciar la aplicación
app.mainloop()