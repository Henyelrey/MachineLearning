import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ==============================================================================
# 1️⃣ Cargar el dataset limpio
# ==============================================================================
df = pd.read_csv("casas_limpias.csv")

# Comprobamos que exista la columna 'price'
if 'price' not in df.columns:
    print("❌ No se encontró la columna 'price'. Renómbrala en tu CSV.")
    exit()

# ==============================================================================
# 2️⃣ Separar variables numéricas y categóricas
# ==============================================================================
# Separamos las variables numéricas y categóricas automáticamente
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print(f"📊 Columnas numéricas: {numeric_cols}")
print(f"🔤 Columnas categóricas: {categorical_cols}")

# ==============================================================================
# 3️⃣ Convertir variables categóricas a numéricas (One-Hot Encoding)
# ==============================================================================
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# ==============================================================================
# 4️⃣ Separar variables independientes (X) y dependiente (y)
# ==============================================================================
X = df_encoded.drop(columns=['price'])
y = df_encoded['price']

# ==============================================================================
# 5️⃣ Escalar características numéricas
# ==============================================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==============================================================================
# 6️⃣ Dividir en entrenamiento y prueba
# ==============================================================================
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ==============================================================================
# 7️⃣ Crear el modelo neuronal
# ==============================================================================
model = Sequential([
    Dense(64, input_dim=X.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse')

# ==============================================================================
# 8️⃣ Entrenar el modelo
# ==============================================================================
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=1)

# ==============================================================================
# 9️⃣ Evaluar el modelo
# ==============================================================================
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"\n✅ MAE (Error Absoluto Medio): {mae:.2f}")

# ==============================================================================
# 🔟 Guardar modelo y escalador
# ==============================================================================
model.save("modelo_casas.keras")
joblib.dump(scaler, "scaler_casas.pkl")
print("\n💾 Modelo y escalador guardados correctamente.")

# ==============================================================================
# 11️⃣ Graficar la pérdida del entrenamiento
# ==============================================================================
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Entrenamiento', linewidth=2)
plt.plot(history.history['val_loss'], label='Validación', linewidth=2)
plt.legend()
plt.title("Evolución del Error durante el Entrenamiento")
plt.xlabel("Épocas")
plt.ylabel("Error cuadrático medio (MSE)")
plt.show()

