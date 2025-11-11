import streamlit as st
import numpy as np
import pickle

# --- 1. Konfigurasi Halaman & Judul ---
# Ini harus menjadi perintah Streamlit pertama
st.set_page_config(
    page_title="Form Prediksi Penyakit",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- 2. Kamus Penyakit ---
# Di sinilah kita mendefinisikan warna, ikon, dan deskripsi
# Tambahkan/ubah sesuai dengan 'TARGET' di data Anda
DISEASE_INFO = {
    "ALLERGY": {
        "color": "blue",
        "icon": "🤧",
        "description": (
            "Alergi terjadi ketika sistem kekebalan tubuh Anda bereaksi "
            "terhadap zat asing (seperti serbuk sari, bulu, atau tungau) "
            "yang biasanya tidak berbahaya. Gejala utamanya seringkali gatal."
        )
    },
    "FLU": {
        "color": "orange",
        "icon": "🤒",
        "description": (
            "Flu (Influenza) adalah penyakit pernapasan menular yang disebabkan "
            "oleh virus influenza. Gejalanya cenderung muncul tiba-tiba dan "
            "melibatkan demam tinggi, nyeri otot, dan kelelahan ekstrem."
        )
    },
    "COVID": {
        "color": "red",
        "icon": "🦠",
        "description": (
            "COVID-19 adalah penyakit menular yang disebabkan oleh virus SARS-CoV-2. "
            "Gejalanya sangat bervariasi, tetapi seringkali mencakup demam, "
            "batuk kering, dan kehilangan indra penciuman atau perasa."
        )
    },
    "COLD": {
        "color": "gray",
        "icon": "🥶",
        "description": (
            "Pilek biasa (Common Cold) adalah infeksi virus ringan pada hidung "
            "dan tenggorokan. Gejalanya biasanya berkembang perlahan dan "
            "lebih ringan daripada flu, seperti hidung meler dan bersin."
        )
    },
    # Default jika penyakit tidak ada di kamus
    "DEFAULT": {
        "color": "black",
        "icon": "❓",
        "description": "Tidak ada deskripsi untuk penyakit ini."
    }
}


# --- 3. Fungsi Helper (Memuat Model) ---
@st.cache_resource
def load_resources():
    """Memuat model, nama kelas, dan nama fitur."""
    try:
        model = pickle.load(open('prediksi_penyakit_model.sav', 'rb'))
        class_names = pickle.load(open('class_names.sav', 'rb'))
        feature_names = pickle.load(open('feature_names.sav', 'rb'))
        return model, class_names, feature_names
    except FileNotFoundError:
        st.error("Error: File model (.sav) tidak ditemukan.")
        st.stop()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
        st.stop()

# --- 4. Fungsi Tampilan Hasil ---
def display_result(disease_name):
    """Menampilkan hasil diagnosa"""
    
    # Ambil info dari kamus, gunakan DEFAULT jika tidak ketemu
    info = DISEASE_INFO.get(disease_name, DISEASE_INFO["DEFAULT"])
    
    icon = info["icon"]
    color = info["color"]
    description = info["description"]
    
    # Gunakan st.error/warning/info/success berdasarkan warna
    if color == "red":
        st.error(f"**{icon} Hasil Prediksi: {disease_name}**", icon=icon)
    elif color == "orange":
        st.warning(f"**{icon} Hasil Prediksi: {disease_name}**", icon=icon)
    elif color == "blue":
        st.info(f"**{icon} Hasil Prediksi: {disease_name}**", icon=icon)
    else:
        st.success(f"**{icon} Hasil Prediksi: {disease_name}**", icon=icon)

    # Tampilkan deskripsi
    st.subheader("Deskripsi Singkat")
    st.write(description)
    st.markdown("---")
    st.caption("**Peringatan:** Ini adalah prediksi berdasarkan model AI dan **bukan** diagnosis medis profesional. Silakan berkonsultasi dengan dokter.")


# --- 5. Fungsi Utama Aplikasi ---
def main():
    
    # Muat model dan nama fitur
    model, class_names, feature_names = load_resources()

    # --- SIDEBAR: Input Profil Pasien ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/4997/4997426.png", width=150)
        st.title("Profil Pasien")
        nama_pasien = st.text_input("Nama Pasien", placeholder="contoh: Zulkidin")
        usia_pasien = st.number_input("Usia Pasien", min_value=0, max_value=120, value=None, placeholder="0")

    # --- Halaman Utama ---
    st.title("🩺 Form Prediksi Penyakit")
    st.write(f"Harap isi gejala untuk pasien: **{nama_pasien if nama_pasien else '...'}**")

    # Inisialisasi session state untuk menyimpan hasil
    if 'prediction' not in st.session_state:
        st.session_state.prediction = None
    if 'patient_name_display' not in st.session_state:
        st.session_state.patient_name_display = ""

    # --- FORMULIR GEJALA ---
    with st.form("gejala_form"):
        st.subheader("Pilih Gejala yang Dialami")
        
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
        
        st.markdown("---") # Garis pemisah
        
        # --- Tombol Submit dan Reset ---
        submit_col, reset_col = st.columns([3, 1]) # Tombol submit lebih besar
        
        submitted = submit_col.form_submit_button("🚀 Prediksi Penyakit", use_container_width=True, type="primary")
        reset_pressed = reset_col.form_submit_button("Reset", use_container_width=True)

    # --- Logika Setelah Form disubmit ---
    if submitted:
        # Validasi input
        if not nama_pasien or usia_pasien == 0:
            st.warning("Mohon isi Nama dan Usia Pasien di sidebar terlebih dahulu.")
        elif sum(user_input_list) == 0:
            st.warning("Anda tidak memilih gejala apapun. Silakan pilih minimal satu gejala.")
        else:
            # Lakukan prediksi
            input_array = np.array(user_input_list).reshape(1, -1)
            prediction_index = model.predict(input_array)
            disease_name = class_names[prediction_index[0]]
            
            # Simpan hasil ke session state
            st.session_state.prediction = disease_name
            st.session_state.patient_name_display = nama_pasien

    # Jika tombol reset ditekan, bersihkan hasil
    if reset_pressed:
        st.session_state.prediction = None
        st.session_state.patient_name_display = ""
        st.info("Formulir telah di-reset.")

    # --- Area Tampilan Hasil (di luar form) ---
    st.markdown("## Hasil Diagnosa")
    
    if st.session_state.prediction is not None:
        st.subheader(f"Pasien: {st.session_state.patient_name_display} ({usia_pasien} tahun)")
        display_result(st.session_state.prediction)
    else:
        st.info("Hasil prediksi akan muncul di sini setelah Anda mengisi form dan menekan 'Prediksi'.")


# --- Jalankan Aplikasi ---
if __name__ == "__main__":
    main()





