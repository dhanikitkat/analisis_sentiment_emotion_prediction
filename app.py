import os
import streamlit as st
import pandas as pd
from transformers import pipeline
import base64

# Set to use CPU only
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Load pipelines
sentiment_pipe = pipeline("text-classification", model="ayameRushia/bert-base-indonesian-1.5G-sentiment-analysis-smsa")
emotion_pipe = pipeline("text-classification", model="azizp128/prediksi-emosi-indobert")


def direct_sentiment_analysis(text):
    texts = text.split('\n')  # Memisahkan teks berdasarkan baris
    
    # Hasil analisis sentiment
    results = []
    for text in texts:
        if text.strip():
            result = sentiment_pipe(text)[0]  # Melakukan analisis sentiment pada setiap teks
            results.append((text, result['label'].lower(), result['score']))
    
    # Ubah ke DataFrame untuk tampilan tabel
    df = pd.DataFrame(results, columns=['Content', 'Sentiment', 'Score'])
    return df


def direct_emotion_analysis(text):
    texts = text.split('\n')  # Memisahkan teks berdasarkan baris
    
    # Hasil analisis sentiment
    results = []
    for text in texts:
        if text.strip():
            result = emotion_pipe(text)[0]  # Melakukan analisis sentiment pada setiap teks
            results.append((text, result['label'].lower(), result['score']))
    
    # Ubah ke DataFrame untuk tampilan tabel
    df = pd.DataFrame(results, columns=['Content', 'Emotion', 'Score'])
    return df

def process_file_sentiment(file):
    if file.name.endswith('.xlsx'):
        df = pd.read_excel(file)  # Baca file XLSX
    elif file.name.endswith('.csv'):
        df = pd.read_csv(file)  # Baca file CSV
    else:
        st.error("Format file tidak didukung. Harap unggah file CSV atau XLSX.")
        return None
    
    # Analisis sentimen dan tambahkan hasil ke DataFrame
    results = []
    for index, row in df.iterrows():
        if pd.notna(row['content']) and isinstance(row['content'], str):
            sentiment, score = analyze_sentiment(row['content'])
            results.append((row['content'], sentiment, score))
        else:
            results.append((row['content'], None, None))  # Menambahkan nilai None jika kosong
    
    df['Sentimen'] = [r[1] for r in results]
    df['Skor Sentimen'] = [r[2] for r in results]
    
    return df

def process_file_emotion(file):
    if file.name.endswith('.xlsx'):
        df = pd.read_excel(file)  # Baca file XLSX
    elif file.name.endswith('.csv'):
        df = pd.read_csv(file)  # Baca file CSV
    else:
        st.error("Format file tidak didukung. Harap unggah file CSV atau XLSX.")
        return None
    
    # Prediksi emosi dan tambahkan hasil ke DataFrame
    results = []
    for index, row in df.iterrows():
        if pd.notna(row['content']) and isinstance(row['content'], str):
            emotion, score = emotion_prediction(row['content'])
            results.append((row['content'], emotion, score))
        else:
            results.append((row['content'], None, None))  # Menambahkan nilai None jika kosong
    
    df['Emosi'] = [r[1] for r in results]
    df['Skor Emosi'] = [r[2] for r in results]
    
    return df

def analyze_sentiment(text):
    result = sentiment_pipe(text)[0]
    return result['label'].lower(), result['score']

def emotion_prediction(text):
    result = emotion_pipe(text)[0]
    return result['label'].lower(), result['score']

def get_download_link_sentiment(df):
    # Generate a link to download the dataframe with Sentimen and Skor Sentimen as CSV
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Encode CSV to base64
    href = f'<a href="data:file/csv;base64,{b64}" download="analisis_sentimen.csv">Download CSV</a>'
    return href

def get_download_link_emotion(df):
    # Generate a link to download the dataframe with Emosi and Skor Emosi as CSV
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Encode CSV to base64
    href = f'<a href="data:file/csv;base64,{b64}" download="prediksi_emosi.csv">Download CSV</a>'
    return href

def main():
    st.title("Aplikasi Analisis Sentimen dan Prediksi Emosi by Ramdhani")

    # Pilihan Program
    program = st.sidebar.selectbox("Pilih Program", ["Analisis Sentiment", "Prediksi Emosi"])

    if program == "Analisis Sentiment":
        # Menu untuk analisis sentimen
        st.header("Analisis Sentiment")
        menu_sentiment = st.sidebar.selectbox("Pilih Metode", ["Analisis Langsung", "Import dari File"])

        if menu_sentiment == "Analisis Langsung":
        # Masukan teks untuk analisis sentimen
            user_input = st.text_area("Masukkan teks yang ingin dianalisis (pisahkan dengan enter):")

            if st.button("Analisis Sentimen"):
                df = direct_sentiment_analysis(user_input)
                st.write("Hasil Analisis Sentimen:")
                st.write(df)

                # Tambahkan tombol download CSV
                st.markdown(get_download_link_sentiment(df), unsafe_allow_html=True)
                
        elif menu_sentiment == "Import dari File":
            st.subheader("Import dari File")
            uploaded_file = st.file_uploader("Upload file CSV atau XLSX", type=["csv", "xlsx"])

            if uploaded_file is not None:
                df = process_file_sentiment(uploaded_file)

                # Tampilkan hasil analisis sentimen
                st.write("Hasil Analisis Sentimen:")
                st.write(df)

                # Tambahkan tombol download CSV
                st.markdown(get_download_link_sentiment(df), unsafe_allow_html=True)

    elif program == "Prediksi Emosi":
        # Menu untuk prediksi emosi
        st.header("Prediksi Emosi")
        menu_emot = st.sidebar.selectbox("Pilih Metode", ["Prediksi Langsung", "Import dari File"])

        if menu_emot == "Prediksi Langsung":
            user_input = st.text_area("Masukkan teks yang ingin dianalisis (pisahkan dengan enter):")

            if st.button("Analisis Sentimen"):
                df = direct_emotion_analysis(user_input)
                st.write("Hasil Analisis Sentimen:")
                st.write(df)

                # Tambahkan tombol download CSV
                st.markdown(get_download_link_emotion(df), unsafe_allow_html=True)

        elif menu_emot == "Import dari File":
            st.subheader("Import dari File")
            uploaded_file = st.file_uploader("Upload file CSV atau XLSX", type=["csv", "xlsx"])

            if uploaded_file is not None:
                df = process_file_emotion(uploaded_file)

                # Tampilkan hasil prediksi emosi
                st.write("Hasil Prediksi Emosi:")
                st.write(df)

                # Tambahkan tombol download CSV
                st.markdown(get_download_link_emotion(df), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
