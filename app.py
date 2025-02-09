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
nltk.download('stopwords')

# Load pipelines
sentiment_pipe = pipeline("text-classification", model="dhanikitkat/indo_smsa-1.5G_sentiment_analysis") 
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
    text = re.sub(r'([^\w\s\U0001F000-\U0001F9FF])\1+', r'\1', text)
    text = re.sub(r'([\U0001F600-\U0001F64F\U0001F900-\U0001F9FF\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F])', r' \1 ', text)
    text = re.sub(r'([.,])', r' \1 ', text)
    text = re.sub(r'[&%]', lambda x: f' {x.group()} ', text)
    text = re.sub(r'(\w)\1{1,}', r'\1\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\b(\w+)\b\s*-\s*\b\1\b', r'\1-\1', text)
    text = re.sub(r'(?<=\d)\s*\.\s*(?=\d)', '.', text)
    text = re.sub(r'(?<=\d)\s*,\s*(?=\d)', ',', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = replace_slank_to_formal(text, slank_formal_df)
    tokens = word_tokenize(text)
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

def generate_wordcloud(text, font_path, colormap, title):
    # Create a circular mask for Full HD resolution
    x, y = np.ogrid[:1400, :1400]  # Adjusted for 1400x1400 resolution
    mask = (x - 700) ** 2 + (y - 700) ** 2 > 630 ** 2  # Adjusted mask size for 1400x1400 resolution
    mask = 255 * mask.astype(int)

    # Remove Indonesian stopwords
    indo_stopwords = set(stopwords.words('indonesian'))
    words = text.split()
    words = [word for word in words if word.lower() not in indo_stopwords]
    text = ' '.join(words)

    wordcloud = WordCloud(
        width=1400,
        height=1400,
        background_color='white',
        font_path=font_path,
        prefer_horizontal=1.0,
        colormap=colormap,
        max_words=100,
        mask=mask
    ).generate(text)
    
    # Configure plot settings for high-quality output
    plt.figure(figsize=(14, 14))  # Adjusted figure size for 1400x1400 resolution
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=20, pad=20)  # Title directly in matplotlib plot

    # Save word cloud to file with high DPI for better quality
    plt.savefig(f"{title}.png", dpi=300, bbox_inches='tight', pad_inches=0.1)

    # Display word cloud in Streamlit
    st.image(f"{title}.png", use_column_width=True)

    # Add download link for word cloud
    st.markdown(get_image_download_link(f"{title}.png"), unsafe_allow_html=True)

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

def get_example_download_link(file_path, link_text):
    with open(file_path, "rb") as file:
        b64 = base64.b64encode(file.read()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{os.path.basename(file_path)}">{link_text}</a>'

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
    
    # Define custom CSS to adjust the height
    st.markdown(
        """
        <style>
        .chart-container {
            display: flex;
            justify-content: center;
        }
        .user-select-none.svg-container {
            height: 360px !important;
        }
        .average-score {
            text-align: center;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )

        # Sentiment pie chart
    sentiment_counts = df['Sentiment'].value_counts()
    sentiment_colors = {
        'positive': px.colors.qualitative.Set3[0],
        'negative': px.colors.qualitative.Set3[3],
        'neutral': px.colors.qualitative.Set3[1]
    }

    fig_sentiment = px.pie(
        sentiment_counts,
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title='Sentiment Distribution',
        width=400,
        height=400,
        color=sentiment_counts.index,
        color_discrete_map=sentiment_colors
    )

    # Calculate sentiment average
    sentiment_average = df['Score Sentiment'].mean()

    # Add average sentiment score as an annotation
    fig_sentiment.add_annotation(
        text=f"Average Sentiment Score: {sentiment_average:.4f}",
        xref="paper", yref="paper",
        x=0.5, y=-0.2,
        showarrow=False,
        font=dict(size=18)
    )

    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.plotly_chart(fig_sentiment, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Emotion pie chart
     # Sentiment pie chart
    emotion_counts = df['Emotion'].value_counts()
    emotion_colors = {
        'marah': px.colors.qualitative.Safe[9],
        'sedih': px.colors.qualitative.Safe[1],
        'senang': px.colors.qualitative.Safe[0],
        'cinta': px.colors.qualitative.Safe[2],
        'jijik': px.colors.qualitative.Safe[6],
        'takut': px.colors.qualitative.Safe[7],
    }
    fig_emotion = px.pie(
        emotion_counts, 
        values=emotion_counts.values, 
        names=emotion_counts.index, 
        title='Emotion Distribution', 
        width=400, 
        height=400, 
        color=emotion_counts.index, 
        color_discrete_map=emotion_colors
    )

    # Calculate emotion average
    emotion_average = df['Score Emotion'].mean()

    # Add average emotion score as an annotation
    fig_emotion.add_annotation(
        text=f"Average Emotion Score: {emotion_average:.4f}",
        xref="paper", yref="paper",
        x=0.5, y=-0.2,
        showarrow=False,
        font=dict(size=18)
    )

    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.plotly_chart(fig_emotion, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Generate word clouds
    font_path = os.path.join('assets', 'Poppins-Regular.ttf')
    
    # Ensure `df` is your DataFrame and 'Cleaned Content', 'Sentiment', and 'Emotion' columns exist
    overall_text = ' '.join(df['Cleaned Content'].dropna())
    generate_wordcloud(overall_text, font_path, 'hsv_r', 'Overall Word Cloud')

    positive_happy_text = ' '.join(df[(df['Sentiment'] == 'positive') & (df['Emotion'] == 'senang')]['Cleaned Content'].dropna())
    generate_wordcloud(positive_happy_text, font_path, 'gist_rainbow_r', 'Positive Sentiment & Happy Emotion Word Cloud')

    negative_angry_sad_text = ' '.join(df[(df['Sentiment'] == 'negative') & (df['Emotion'].isin(['marah', 'sedih']))]['Cleaned Content'].dropna())
    generate_wordcloud(negative_angry_sad_text, font_path, 'inferno', 'Negative Sentiment & Angry or Sad Emotion Word Cloud')

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

    # Define custom CSS to adjust the height
    st.markdown(
        """
        <style>
        .chart-container {
            display: flex;
            justify-content: center;
        }
        .user-select-none.svg-container {
            height: 360px !important;
        }
        .average-score {
            text-align: center;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )

        # Sentiment pie chart
    sentiment_counts = df['Sentiment'].value_counts()
    sentiment_colors = {
        'positive': px.colors.qualitative.Set3[0],
        'negative': px.colors.qualitative.Set3[3],
        'neutral': px.colors.qualitative.Set3[1]
    }

    fig_sentiment = px.pie(
        sentiment_counts,
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title='Sentiment Distribution',
        width=400,
        height=400,
        color=sentiment_counts.index,
        color_discrete_map=sentiment_colors
    )

    # Calculate sentiment average
    sentiment_average = df['Score Sentiment'].mean()

    # Add average sentiment score as an annotation
    fig_sentiment.add_annotation(
        text=f"Average Sentiment Score: {sentiment_average:.4f}",
        xref="paper", yref="paper",
        x=0.5, y=-0.2,
        showarrow=False,
        font=dict(size=18)
    )

    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.plotly_chart(fig_sentiment, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Emotion pie chart
     # Sentiment pie chart
    emotion_counts = df['Emotion'].value_counts()
    emotion_colors = {
        'marah': px.colors.qualitative.Safe[9],
        'sedih': px.colors.qualitative.Safe[1],
        'senang': px.colors.qualitative.Safe[0],
        'cinta': px.colors.qualitative.Safe[2],
        'jijik': px.colors.qualitative.Safe[6],
        'takut': px.colors.qualitative.Safe[7],
    }
    fig_emotion = px.pie(
        emotion_counts, 
        values=emotion_counts.values, 
        names=emotion_counts.index, 
        title='Emotion Distribution', 
        width=400, 
        height=400, 
        color=emotion_counts.index, 
        color_discrete_map=emotion_colors
    )

    # Calculate emotion average
    emotion_average = df['Score Emotion'].mean()

    # Add average emotion score as an annotation
    fig_emotion.add_annotation(
        text=f"Average Emotion Score: {emotion_average:.4f}",
        xref="paper", yref="paper",
        x=0.5, y=-0.2,
        showarrow=False,
        font=dict(size=18)
    )

    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.plotly_chart(fig_emotion, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Generate word clouds
    font_path = os.path.join('assets', 'Poppins-Regular.ttf')
    
    # Ensure `df` is your DataFrame and 'Cleaned Content', 'Sentiment', and 'Emotion' columns exist
    overall_text = ' '.join(df['Cleaned Content'].dropna())
    generate_wordcloud(overall_text, font_path, 'hsv_r', 'Overall Word Cloud')

    positive_happy_text = ' '.join(df[(df['Sentiment'] == 'positive') & (df['Emotion'] == 'senang')]['Cleaned Content'].dropna())
    generate_wordcloud(positive_happy_text, font_path, 'gist_rainbow_r', 'Positive Sentiment & Happy Emotion Word Cloud')

    negative_angry_sad_text = ' '.join(df[(df['Sentiment'] == 'negative') & (df['Emotion'].isin(['marah', 'sedih']))]['Cleaned Content'].dropna())
    generate_wordcloud(negative_angry_sad_text, font_path, 'inferno', 'Negative Sentiment & Angry or Sad Emotion Word Cloud')

    # Word frequency
    word_freq = pd.Series(' '.join(df['Cleaned Content'].dropna()).split()).value_counts()
    st.write("Word Frequency:")
    st.write(word_freq)

    # Download link for word frequency
    word_freq_df = word_freq.reset_index()
    word_freq_df.columns = ['Word', 'Frequency']
    st.markdown(get_word_freq_download_link(word_freq_df), unsafe_allow_html=True)

    return df

def main():
    st.title("Aplikasi Analisis Sentimen dan Prediksi Emosi")

    # Add download link for example slank template
    slank_template_path = "assets/contoh template data slank.txt"
    st.markdown(get_example_download_link(slank_template_path, "Download Contoh Template Data Slank (TXT)"), unsafe_allow_html=True)

    slank_file = st.file_uploader("Upload file slank dengan baris pertama Slank;Formal (TXT)", type=["txt"])
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
        # Add download link for example content template
        content_template_path = "assets/contoh template data content.xlsx"
        st.markdown(get_example_download_link(content_template_path, "Download Contoh Template Data Content (XLSX)"), unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload file CSV atau XLSX", type=["csv", "xlsx"])
        if uploaded_file is not None:
            df = process_file(uploaded_file, df_slank_formal)
            st.write("Hasil Analisis:")
            st.write(df)
            st.markdown(get_download_link(df, "analisis_sentimen_emosi"), unsafe_allow_html=True)

if __name__ == '__main__':
    main()
