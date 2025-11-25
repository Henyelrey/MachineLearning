🏠 AI Real Estate Predictor

Una aplicación de escritorio moderna que utiliza Deep Learning para estimar el valor de mercado de propiedades inmobiliarias. La interfaz gráfica permite ingresar características de la vivienda y obtener una predicción en tiempo real basada en un modelo de Red Neuronal Artificial.

✨ Características

Predicción con IA: Utiliza un modelo entrenado con TensorFlow/Keras (.keras).

Interfaz Moderna: Diseño oscuro estilo "Superhero" usando ttkbootstrap.

Preprocesamiento Automático: Maneja la normalización de datos (Scaling) y codificación de variables categóricas (One-Hot Encoding) internamente.

Validación Lógica: Incluye sistemas de seguridad para evitar predicciones erróneas cuando los campos están vacíos (ej. Área = 0).

🛠️ Tecnologías Utilizadas

Lenguaje: Python

Machine Learning: TensorFlow, Keras, Scikit-Learn

Manejo de Datos: Pandas, Numpy, Joblib

Interfaz Gráfica (GUI): Tkinter, Ttkbootstrap

📂 Estructura del Proyecto

Para que la aplicación funcione, asegúrate de tener los siguientes archivos en la misma carpeta:

predecir_app.py: El código fuente principal de la aplicación.

modelo_casas.keras: El modelo de red neuronal entrenado.

scaler_casas.pkl: El objeto escalador (StandardScaler) guardado.

casas_limpias.csv: Dataset auxiliar para cargar las opciones de los menús desplegables.

🚀 Instalación y Ejecución

Sigue estos pasos para ejecutar el proyecto en tu máquina local.

1. Prerrequisitos

Necesitas tener instalado Python (versión recomendada 3.10 o 3.11).

2. Preparar el Entorno

Es recomendable crear un entorno virtual para no afectar tu instalación global de Python. Abre tu terminal (PowerShell o CMD) en la carpeta del proyecto:

# Crear el entorno virtual llamado "venv"
python -m venv venv

# Activar el entorno (Windows)
.\venv\Scripts\activate
# En Mac/Linux usa: source venv/bin/activate


3. Instalar Dependencias

Una vez activado el entorno, instala las librerías necesarias ejecutando:

pip install tensorflow pandas numpy joblib scikit-learn ttkbootstrap


4. Ejecutar la Aplicación

Con todo instalado, lanza el programa con el siguiente comando:

python predecir_app.py


🧠 ¿Cómo funciona internamente?

Entrada de Datos: El usuario ingresa datos numéricos (pies cuadrados, edad, habitaciones) y selecciona categorías (tipo de calefacción, aire central, etc.).

Validación: El sistema verifica que las dimensiones (Area y Lote) no sean 0. Si lo son, bloquea la predicción.

Codificación (Encoding): Las variables de texto se convierten en números binarios (One-Hot Encoding) para que la IA las entienda, igualando la estructura usada durante el entrenamiento.

Escalado (Scaling): Los números se transforman usando el archivo scaler_casas.pkl para ponerlos en la misma escala matemática que el modelo espera.

Inferencia: Los datos procesados entran al modelo modelo_casas.keras, el cual calcula y devuelve el precio estimado.

📸 Capturas

<img width="1908" height="1078" alt="image" src="https://github.com/user-attachments/assets/beea16ff-71a4-4c58-abe5-ba7adea528c4" />

Autor: Henyelrey Lucio Garcia Chura










