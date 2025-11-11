import streamlit as st
import numpy as np
import pickle

# Judul Aplikasi
st.set_page_config(page_title="Prediksi Penyakit", layout="centered")
st.title("🤖 Prediksi Penyakit Berdasarkan Gejala")
st.write("Pilih gejala-gejala yang Anda alami untuk memprediksi kemungkinan penyakit.")

# --- Fungsi untuk Memuat Model dan Metadata ---
@st.cache_resource
def load_resources():
    """Memuat model, nama kelas, dan nama fitur dari file pickle."""
    try:
        model = pickle.load(open('prediksi_penyakit_model.sav', 'rb'))
        class_names = pickle.load(open('class_names.sav', 'rb'))
        feature_names = pickle.load(open('feature_names.sav', 'rb'))
        return model, class_names, feature_names
    except FileNotFoundError:
        st.error("Error: File model (.sav) tidak ditemukan. Jalankan skrip `buat_model.py` terlebih dahulu.")
        return None, None, None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
        return None, None, None

# Muat model dan metadata
model, class_names, feature_names = load_resources()

if model is not None:
    # --- Buat Form Input Gejala ---
    with st.form("gejala_form"):
        st.header("Pilih Gejala Anda (1 = Ya, 0 = Tidak)")
        
        col1, col2 = st.columns(2)
        user_input_list = []
        half_len = len(feature_names) // 2
        
        with col1:
            for i, feature in enumerate(feature_names[:half_len]):
                label = feature.replace("_", " ").title()
                user_input = st.selectbox(
                    label,
                    options=[0, 1],
                    format_func=lambda x: "Ya" if x == 1 else "Tidak",
                    key=f"feat_{i}"
                )
                user_input_list.append(user_input)

        with col2:
            for i, feature in enumerate(feature_names[half_len:], start=half_len):
                label = feature.replace("_", " ").title()
                user_input = st.selectbox(
                    label,
                    options=[0, 1],
                    format_func=lambda x: "Ya" if x == 1 else "Tidak",
                    key=f"feat_{i}"
                )
                user_input_list.append(user_input)

        submitted = st.form_submit_button("Prediksi Penyakit")

    # --- Logika Prediksi ---
    if submitted:
        if sum(user_input_list) == 0:
            st.warning("Anda tidak memilih gejala apapun. Silakan pilih minimal satu gejala.")
        else:
            input_array = np.array(user_input_list)
            input_reshaped = input_array.reshape(1, -1)
            
            try:
                prediction_index = model.predict(input_reshaped)
                disease_name = class_names[prediction_index[0]]
                
                st.success(f"Hasil Prediksi: **{disease_name}**")
                st.info("Catatan: Ini adalah prediksi berdasarkan model Naive Bayes dan tidak menggantikan diagnosis medis profesional.")
                
            except Exception as e:
                st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
