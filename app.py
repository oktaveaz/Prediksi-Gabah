import streamlit as st
import numpy as np
import joblib


# Load model
model = joblib.load('last_model_random_forest.pkl')

# Judul halaman
st.title("Prediksi Hasil Produksi Gabah Basah")

# Input dari pengguna
st.header("Masukkan Data Pertanian Anda")

luas_lahan = st.number_input("Luas Lahan (dalam m\u00b2) (min. 500)", min_value=500.0, max_value=5000, step=0.1)
ph_tanah = st.slider("pH Tanah", min_value=5.0, max_value=8.0, step=0.1)
varietas = st.selectbox("Varietas", ["inpari_32", "srinuk", "C4", "sunggal"])

df_varietas = np.array([
    1 if varietas == "c4" else 0,
    1 if varietas == "inpari_32" else 0,
    1 if varietas == "srinuk" else 0,
    1 if varietas == "sunggal" else 0
])

# Prediksi
if st.button("Prediksi Produksi Gabah Basah"):
    # Format input untuk model
    variable = np.array([luas_lahan, ph_tanah])
    input_data = np.concatenate((variable, df_varietas))
    
    # Prediksi
    hasil_prediksi = model.predict(input_data.reshape(1, -1))
    hasil_gabah = hasil_prediksi[0]

    st.success(f"Prediksi hasil produksi gabah basah: {hasil_gabah:.2f} gram")
