import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image  

# Load model dan scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model", "random_forest_model.pkl")
scaler_path = os.path.join(BASE_DIR, "model", "scaler.pkl")
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Diabetes", layout="wide")

# ====== Sidebar logo ======
logo_path = os.path.join(BASE_DIR, "..", "assets", "logo_kucing_bulat.png")
img = Image.open(logo_path)
st.sidebar.image(img, width=120)
st.sidebar.markdown("### 😺 Prediksi oleh Kucing AI")


# ====== Judul halaman utama ======
st.markdown("""
# 🧠 Aplikasi Prediksi Diabetes  
Prediksi cepat risiko diabetes berdasarkan data pasien menggunakan **Random Forest Classifier**.  
Silakan masukkan data manual atau unggah file CSV untuk prediksi batch.
""", unsafe_allow_html=True)

st.markdown("---")

# Tabs
tab1, tab2 = st.tabs(["🔍 Prediksi Manual", "📁 Prediksi dari File CSV"])

with tab1:
    st.subheader("Masukkan Data Pasien")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        preg = st.number_input("Pregnancies", 0)
    with col2:
        gluc = st.number_input("Glucose", 0)
    with col3:
        bp = st.number_input("Blood Pressure", 0)
    with col4:
        skin = st.number_input("Skin Thickness", 0)
    with col1:
        insulin = st.number_input("Insulin", 0)
    with col2:
        bmi = st.number_input("BMI", 0.0)
    with col3:
        dpf = st.number_input("Diabetes Pedigree Function", 0.0)
    with col4:
        age = st.number_input("Age", 0)

    if st.button("🔮 Prediksi"):
        data = np.array([[preg, gluc, bp, skin, insulin, bmi, dpf, age]])
        scaled = scaler.transform(data)
        pred = model.predict(scaled)[0]
        prob = model.predict_proba(scaled)[0][1]

        st.write("---")
        if pred == 1:
            st.markdown(f"""
            <div style="background-color:#ffe6e6;padding:15px;border-radius:10px">
                <h4 style="color:#cc0000;">🚨 Hasil: <strong>POSITIF Diabetes</strong></h4>
                <p>Probabilitas: <strong>{prob:.2%}</strong></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background-color:#e6ffe6;padding:15px;border-radius:10px">
                <h4 style="color:#006600;">✅ Hasil: <strong>NEGATIF Diabetes</strong></h4>
                <p>Probabilitas: <strong>{prob:.2%}</strong></p>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.subheader("📄 Upload File CSV")

    file = st.file_uploader("Unggah file CSV", type=["csv"])
    if file:
        try:
            df = pd.read_csv(file)
            st.write("📋 Data Awal:", df.head())

            scaled = scaler.transform(df)
            predictions = model.predict(scaled)
            df['Prediksi'] = predictions
            df['Probabilitas Diabetes'] = model.predict_proba(scaled)[:, 1]
            st.success("✅ Prediksi selesai!")
            st.write(df)

            # Grafik
            st.subheader("📊 Distribusi Hasil Prediksi")
            fig1, ax1 = plt.subplots()
            df['Prediksi'].value_counts().sort_index().plot.pie(
                autopct="%1.1f%%",
                labels=["Negatif", "Positif"],
                colors=["green", "red"],
                ax=ax1
            )
            ax1.set_ylabel("")
            st.pyplot(fig1)

            # Download
            csv_download = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Hasil", csv_download, "hasil_prediksi.csv", "text/csv")

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")

# Footer
st.markdown("""
<hr style="margin-top:50px;">
<div style='text-align: center;'>
    Dibuat dengan ❤️ ig:malikimayzar oleh <a href='https://github.com/malikimayzar' target='_blank'>Mayzar Maliki</a>  
    | Powered by Streamlit & scikit-learn  
</div>
""", unsafe_allow_html=True)
