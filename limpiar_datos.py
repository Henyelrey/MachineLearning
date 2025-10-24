# limpiar_datos.py
import pandas as pd
import glob
import os

# Carpeta donde tienes tus CSV
carpeta = "datasets/"  # cambia si tus archivos están en otro lugar
archivos = glob.glob(os.path.join(carpeta, "*.csv"))

if not archivos:
    print("⚠️ No se encontraron archivos CSV en la carpeta especificada.")
    exit()

# Combinar todos los CSV en un solo DataFrame
df_list = [pd.read_csv(f) for f in archivos]
datos = pd.concat(df_list, ignore_index=True)

print(f"✅ Se cargaron {len(archivos)} archivos con un total de {len(datos)} registros.")

# Limpieza básica
datos = datos.drop_duplicates()
datos = datos.dropna()

# Verifica las columnas
print("\nColumnas encontradas:")
print(datos.columns.tolist())

# Guarda el dataset limpio
datos.to_csv("casas_limpias.csv", index=False)
print("\n✅ Archivo 'casas_limpias.csv' guardado correctamente.")
