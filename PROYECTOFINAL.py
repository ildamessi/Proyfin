import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Título de la aplicación
st.title("Predicción de Resolución Correcta de Problemas")

# Cargar datos con cache
@st.cache_data

def cargar_datos():
    df = pd.read_csv("D:/ilda/dataset_calculo_problemas.csv")
    return df

df = cargar_datos()

# Previsualizar los datos
st.subheader("Vista previa de los datos")
st.dataframe(df.head())

# Preprocesamiento
df = df.drop(columns=["ID_Estudiante"])
df = df.dropna(subset=["Dificultad_Percibida"])  # Eliminar filas con dificultad vacía

# Codificar columnas categóricas
le_tema = LabelEncoder()
le_dificultad = LabelEncoder()
df["Tipo_Problema"] = le_tema.fit_transform(df["Tipo_Problema"])
df["Dificultad_Percibida"] = le_dificultad.fit_transform(df["Dificultad_Percibida"])
df["Resuelto_Correctamente"] = df["Resuelto_Correctamente"].map({"Sí":1, "No":0})

# Variables predictoras y objetivo
x = df.drop("Resuelto_Correctamente", axis=1)
y = df["Resuelto_Correctamente"]

# División train/test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Entrenamiento
modelo = RandomForestClassifier(n_estimators=100, random_state=0)
modelo.fit(x_train, y_train)
score = modelo.score(x_test, y_test)

st.subheader(f"Precisión del modelo: {score:.2f}")

# Matriz de Confusión
y_pred = modelo.predict(x_test)
mc = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(mc, annot=True, fmt='d', cmap='Blues', ax=ax)
st.subheader("Matriz de Confusión")
st.pyplot(fig)

# Importancia de características
importancias = modelo.feature_importances_
importancia_df = pd.DataFrame({"Característica": x.columns, "Importancia": importancias})
st.subheader("Importancia de las características")
st.bar_chart(importancia_df.set_index("Característica"))

# Formulario de predicción
st.subheader("Formulario de Predicción")
with st.form("formulario"):
    errores = st.slider("Errores cometidos", 0, 10, 2)
    tiempo = st.number_input("Tiempo en segundos", min_value=0.0, max_value=2000.0)
    intentos = st.slider("Número de intentos", 1, 5, 2)
    tema = st.selectbox("Tipo de problema", le_tema.classes_)
    dificultad = st.selectbox("Dificultad percibida", le_dificultad.classes_)

    enviar = st.form_submit_button("Predecir")

    if enviar:
        tema_cod = le_tema.transform([tema])[0]
        dificultad_cod = le_dificultad.transform([dificultad])[0]

        entrada = pd.DataFrame([[tema_cod, errores, tiempo, intentos, dificultad_cod]], columns=x.columns)
        pred = modelo.predict(entrada)[0]
        resultado = "Sí" if pred == 1 else "No"
        st.success(f"¿Resolverá correctamente el problema? → {resultado}")

