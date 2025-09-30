import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# --- Cargar modelo ---
@st.cache_resource
def load_cnn_model():
    return load_model("modelo.keras")

model = load_cnn_model()

# --- Título de la App ---
st.title("🔎 Clasificador Visual con CNN")
st.markdown("### Sube una imagen y descubre a qué clase pertenece según el modelo entrenado.")

# Subir imagen
uploaded_file = st.file_uploader("📂 Selecciona una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostrar imagen subida en dos columnas
    col1, col2 = st.columns(2)

    with col1:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Imagen cargada", use_container_width=True)

    with col2:
        # Preprocesamiento
        img_resized = img.resize((32, 32))   # Ajusta según tu modelo
        img_array = image.img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predicción
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)

        # Diccionario de clases (CIFAR-10)
        class_names = ['✈️ avión', '🚗 automóvil', '🐦 pájaro', '🐱 gato', '🦌 ciervo',
                       '🐶 perro', '🐸 rana', '🐴 caballo', '⛵ barco', '🚛 camión/bus']

        # Mostrar resultado
        st.subheader("✅ Resultado")
        st.write(f"**Clase predicha:** {class_names[predicted_class]}")
        st.write(f"**Confianza:** {confidence:.2%}")

    # --- Mostrar todas las probabilidades en gráfico ---
    st.markdown("### 📊 Distribución de probabilidades")

    probs = pd.DataFrame({
        "Clase": class_names,
        "Probabilidad": prediction[0]
    }).sort_values(by="Probabilidad", ascending=False)

    st.dataframe(probs, use_container_width=True)

    # Gráfico de barras
    fig, ax = plt.subplots()
    ax.barh(probs["Clase"], probs["Probabilidad"], color="skyblue")
    ax.set_xlabel("Probabilidad")
    ax.set_title("Confianza del modelo por clase")
    st.pyplot(fig)
