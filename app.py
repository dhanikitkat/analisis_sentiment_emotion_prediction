import streamlit as st
import pandas as pd
from transformers import pipeline
import base64
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from PIL import ImageFont
import os


nltk.download('punkt')


# Load pipelines
sentiment_pipe = pipeline("text-classification", model="ayameRushia/bert-base-indonesian-1.5G-sentiment-analysis-smsa")
emotion_pipe = pipeline("text-classification", model="azizp128/prediksi-emosi-indobert")

def load_slank_formal(file):
    if file.name.endswith('.txt'):
        df = pd.read_csv(file, sep=';', header=None, names=['Slank', 'Formal'])
    else:
        st.error("Format file tidak didukung. Harap unggah file TXT.")
        return None
    df.columns = ['Slank', 'Formal']
    return df

def replace_slank_to_formal(sentence, slank_formal_df):
    words = re.findall(r'[\w\',./:-]+|[.,]+|[^\x00-\x7F]+', sentence)
    for i, word in enumerate(words):
        replacement = slank_formal_df.loc[slank_formal_df['Slank'] == word.lower(), 'Formal'].values
        if replacement.size > 0:
            words[i] = str(replacement[0])
    return ' '.join(words)

def preprocess_text(text, slank_formal_df):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = replace_slank_to_formal(text, slank_formal_df)
    tokens = word_tokenize(text)
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

def generate_wordcloud(text, font_path, title, colormap):
    wordcloud = WordCloud(
        width=600,
        height=600,
        background_color='white',
        font_path=font_path,
        prefer_horizontal=1.0,
        colormap=colormap,
        max_words=100
    ).generate(text)
    
    plt.figure(figsize=(10, 10))
    plt.title(title, fontsize=20)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
    
    # Save word cloud to file
    wordcloud.to_file(f"{title}.png")

    # Add download link for word cloud
    st.markdown(get_image_download_link(f"{title}.png"), unsafe_allow_html=True)

def get_image_download_link(image_path):
    with open(image_path, "rb") as image_file:
        b64 = base64.b64encode(image_file.read()).decode()
    href = f'<a href="data:file/png;base64,{b64}" download="{image_path}">Download {image_path}</a>'
    return href


def combined_analysis(text, slank_formal_df):
    texts = text.split('\n')
    results = []
    for text in texts:
        if text.strip():
            cleaned_text = preprocess_text(text, slank_formal_df)
            sentiment_result = sentiment_pipe(cleaned_text)[0]
            emotion_result = emotion_pipe(cleaned_text)[0]
            results.append((text, cleaned_text, sentiment_result['label'].lower(), sentiment_result['score'], emotion_result['label'].lower(), emotion_result['score']))
    df = pd.DataFrame(results, columns=['Content', 'Cleaned Content', 'Sentiment', 'Score Sentiment', 'Emotion', 'Score Emotion'])
    
    # Sentiment pie chart
    sentiment_counts = df['Sentiment'].value_counts()
    fig_sentiment = px.pie(sentiment_counts, values=sentiment_counts.values, names=sentiment_counts.index, title='Sentiment Distribution')
    st.plotly_chart(fig_sentiment, use_container_width=True)

    # Emotion pie chart
    emotion_counts = df['Emotion'].value_counts()
    fig_emotion = px.pie(emotion_counts, values=emotion_counts.values, names=emotion_counts.index, title='Emotion Distribution')
    st.plotly_chart(fig_emotion, use_container_width=True)

    # Generate word clouds
    font_path = os.path.join('assets', 'Poppins-Regular.ttf')
    
    # Overall word cloud
    overall_text = ' '.join(df['Cleaned Content'].dropna())
    generate_wordcloud(overall_text, font_path, 'Overall Word Cloud', 'viridis')
    
    # Positive sentiment and happy emotion word cloud
    positive_happy_text = ' '.join(df[(df['Sentiment'] == 'positive') & (df['Emotion'] == 'senang')]['Cleaned Content'].dropna())
    generate_wordcloud(positive_happy_text, font_path, 'Positive Sentiment & Happy Emotion Word Cloud', 'Greens')

    # Negative sentiment and angry or sad emotion word cloud
    negative_angry_sad_text = ' '.join(df[(df['Sentiment'] == 'negative') & (df['Emotion'].isin(['marah', 'sedih']))]['Cleaned Content'].dropna())
    generate_wordcloud(negative_angry_sad_text, font_path, 'Negative Sentiment & Angry or Sad Emotion Word Cloud', 'Reds')

    # Word frequency
    word_freq = pd.Series(' '.join(df['Cleaned Content'].dropna()).split()).value_counts()
    st.write("Word Frequency:")
    st.write(word_freq)

    # Download link for word frequency
    word_freq_df = word_freq.reset_index()
    word_freq_df.columns = ['Word', 'Frequency']
    st.markdown(get_word_freq_download_link(word_freq_df), unsafe_allow_html=True)

    return df



def process_file(file, slank_formal_df):
    if file.name.endswith('.xlsx'):
        df = pd.read_excel(file)
    elif file.name.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        st.error("Format file tidak didukung. Harap unggah file CSV atau XLSX.")
        return None

    results = []
    for index, row in df.iterrows():
        if pd.notna(row['content']) and isinstance(row['content'], str):
            cleaned_text = preprocess_text(row['content'], slank_formal_df)
            sentiment, score_sentiment = analyze_sentiment(cleaned_text)
            emotion, score_emotion = analyze_emotion(cleaned_text)
            results.append((row['content'], cleaned_text, sentiment, score_sentiment, emotion, score_emotion))
        else:
            results.append((row['content'], None, None, None, None, None))
    
    df['Cleaned Content'] = [r[1] for r in results]
    df['Sentiment'] = [r[2] for r in results]
    df['Score Sentiment'] = [r[3] for r in results]
    df['Emotion'] = [r[4] for r in results]
    df['Score Emotion'] = [r[5] for r in results]

    # Sentiment pie chart
    sentiment_counts = df['Sentiment'].value_counts()
    fig_sentiment = px.pie(sentiment_counts, values=sentiment_counts.values, names=sentiment_counts.index, title='Sentiment Distribution')
    st.plotly_chart(fig_sentiment, use_container_width=True)

    # Emotion pie chart
    emotion_counts = df['Emotion'].value_counts()
    fig_emotion = px.pie(emotion_counts, values=emotion_counts.values, names=emotion_counts.index, title='Emotion Distribution')
    st.plotly_chart(fig_emotion, use_container_width=True)

    # Generate word clouds
    font_path = os.path.join('assets', 'Poppins-Regular.ttf')
    
    # Overall word cloud
    overall_text = ' '.join(df['Cleaned Content'].dropna())
    generate_wordcloud(overall_text, font_path, 'Overall Word Cloud', 'viridis')
    
    # Positive sentiment and happy emotion word cloud
    positive_happy_text = ' '.join(df[(df['Sentiment'] == 'positive') & (df['Emotion'] == 'senang')]['Cleaned Content'].dropna())
    generate_wordcloud(positive_happy_text, font_path, 'Positive Sentiment & Happy Emotion Word Cloud', 'Greens')

    # Negative sentiment and angry or sad emotion word cloud
    negative_angry_sad_text = ' '.join(df[(df['Sentiment'] == 'negative') & (df['Emotion'].isin(['marah', 'sedih']))]['Cleaned Content'].dropna())
    generate_wordcloud(negative_angry_sad_text, font_path, 'Negative Sentiment & Angry or stSad Emotion Word Cloud', 'Reds')

    # Word frequency
    word_freq = pd.Series(' '.join(df['Cleaned Content'].dropna()).split()).value_counts()
    st.write("Word Frequency:")
    st.write(word_freq)

    # Download link for word frequency
    word_freq_df = word_freq.reset_index()
    word_freq_df.columns = ['Word', 'Frequency']
    st.markdown(get_word_freq_download_link(word_freq_df), unsafe_allow_html=True)

    return df


def analyze_sentiment(text):
    result = sentiment_pipe(text)[0]
    return result['label'].lower(), result['score']

def analyze_emotion(text):
    result = emotion_pipe(text)[0]
    return result['label'].lower(), result['score']

def get_download_link(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download CSV</a>'
    return href

def get_word_freq_download_link(word_freq_df):
    csv = word_freq_df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="word_frequency.csv">Download Word Frequency CSV</a>'
    return href

def main():
    st.title("Aplikasi Analisis Sentimen dan Prediksi Emosi")

    slank_file = st.file_uploader("Upload file slank (CSV atau TXT)", type=["csv", "txt"])
    if slank_file is not None:
        df_slank_formal = load_slank_formal(slank_file)
        if df_slank_formal is None:
            st.stop()
    else:
        st.warning("Harap upload file slank terlebih dahulu.")
        st.stop()

    menu = st.sidebar.selectbox("Pilih Metode", ["Analisis Langsung", "Import dari File"])

    if menu == "Analisis Langsung":
        user_input = st.text_area("Masukkan teks yang ingin dianalisis (pisahkan dengan enter):")
        if st.button("Analisis"):
            df = combined_analysis(user_input, df_slank_formal)
            st.write("Hasil Analisis:")
            st.write(df)
            st.markdown(get_download_link(df, "analisis_sentimen_emosi"), unsafe_allow_html=True)
            
    elif menu == "Import dari File":
        uploaded_file = st.file_uploader("Upload file CSV atau XLSX", type=["csv", "xlsx"])
        if uploaded_file is not None:
            df = process_file(uploaded_file, df_slank_formal)
            st.write("Hasil Analisis:")
            st.write(df)
            st.markdown(get_download_link(df, "analisis_sentimen_emosi"), unsafe_allow_html=True)

if __name__ == '__main__':
    main()
