import streamlit as st
import joblib
import matplotlib.pyplot as plt

# Memuat model dan vectorizer
model = joblib.load('text_classification_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Mengatur tampilan aplikasi
st.set_page_config(page_title="Text Classification App", layout="centered")

# Menampilkan header dengan judul dan deskripsi
st.title("Text Classification with Machine Learning")
st.markdown("""
    Selamat datang di aplikasi klasifikasi teks menggunakan model **LinearSVC**. 
    Anda dapat memasukkan teks di bawah ini, dan sistem akan memprediksi labelnya berdasarkan data yang telah dilatih.
""")

# Membuat dua kolom untuk tata letak yang lebih rapi
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("Masukkan Teks untuk Klasifikasi")
    # Input teks dari pengguna
    user_input = st.text_area("Tulis teks Anda di sini:")

with col2:
    st.subheader("Hasil Prediksi")
    # Menampilkan hasil prediksi jika ada input
    if user_input:
        # Transformasi teks input menggunakan TF-IDF
        user_input_tfidf = vectorizer.transform([user_input])

        # Memprediksi label
        predicted_label = model.predict(user_input_tfidf)

        # Menampilkan hasil prediksi dengan ikon
        st.markdown(f"### Prediksi: :label: {predicted_label[0]}")

# Menambahkan tombol untuk membersihkan input
if st.button('Clear Input'):
    st.text_area("Tulis teks Anda di sini:", value="", key="clear_input")


