import os
import subprocess
import sys

# Otomatik kütüphane yükleyici
required = ["pandas", "numpy", "streamlit", "seaborn", "matplotlib", 
            "scikit-learn", "statsmodels", "plotly", "openpyxl"]
for package in required:
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Yüklemeler tamam, analiz başlasın
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

# Sayfa ayarları
st.set_page_config(page_title="Otel Müşteri Analizi", layout="wide")
st.title("\U0001F3E8 Otel Müşteri Verileri Analiz Arayüzü")

uploaded_file = st.file_uploader("\U0001F4C2 Lütfen bir .xlsx dosyası yükleyin", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.success("Veri başarıyla yüklendi!")

    section = st.selectbox("Bir bölüm seçin:", [
        "CRISP-DM", "Veri Önizleme", "Keşifsel Veri Analizi (EDA)", 
        "Makine Öğrenmesi", "Zaman Serisi (ARIMA)", "Öneri Sistemi (KNN)"
    ])

    if section == "CRISP-DM":
        st.header("\U0001F4CA CRISP-DM Metodolojisi")
        st.markdown("""
        **1. İş Hedefi**: Otel yöneticilerinin müşterileri daha iyi analiz ederek strateji belirlemesini sağlamak.\n
        **2. Veri Anlama**: Rezervasyonlar, gelirler, müşteri demografileri.\n
        **3. Veri Hazırlama**: Eksik verilerin temizlenmesi, özellik seçimi.\n
        **4. Modelleştirme**: Lojistik Regresyon, Random Forest, KMeans, ARIMA, KNN\n
        **5. Değerlendirme**: Model performansları, doğruluk skorları, grafiksel analizler.\n
        **6. Dağıtım**: Bu Streamlit uygulaması ile sunum.
        """)

    elif section == "Veri Önizleme":
        st.header("\U0001F4C4 Veri Önizleme")
        st.dataframe(df.head())
        st.write("Veri şekli:", df.shape)
        st.write("Veri türleri:")
        st.write(df.dtypes)

    elif section == "Keşifsel Veri Analizi (EDA)":
        st.header("\U0001F4C8 Keşifsel Veri Analizi")

        st.subheader("Sayısal Özelliklerin Dağılımı")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        selected_col = st.selectbox("Bir sütun seçin", num_cols)
        fig = px.histogram(df, x=selected_col, marginal="box", nbins=30, title=f"{selected_col} Dağılımı")
        st.plotly_chart(fig)

        st.subheader("Korelasyon Matrisi")
        fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax_corr)
        st.pyplot(fig_corr)

        st.subheader("Eksik Değerler")
        missing = df.isnull().sum()
        st.write(missing[missing > 0])

        st.subheader("Kategorik Değişkenlerin Dağılımı")
        cat_cols = df.select_dtypes(include='object').columns
        for col in cat_cols:
            st.write(f"**{col}**")
            st.bar_chart(df[col].value_counts())

    elif section == "Makine Öğrenmesi":
        st.header("\U0001F916 Makine Öğrenmesi Modelleri")
        model_choice = st.selectbox("Bir model seçin:", [
            "Lojistik Regresyon", "Random Forest Regresyon", "K-Means"
        ])

        if model_choice == "Lojistik Regresyon":
            df_model = df.dropna(subset=['BookingsCanceled'])
            X = df_model[["Age", "DaysSinceLastStay", "DaysSinceFirstStay"]].fillna(0)
            y = df_model['BookingsCanceled']
            model = LogisticRegression()
            model.fit(X, y)
            st.write("**Model Skoru:**", model.score(X, y))

        elif model_choice == "Random Forest Regresyon":
            df_model = df.dropna(subset=['LodgingRevenue'])
            X = df_model[["Age", "DaysSinceLastStay", "DaysSinceFirstStay"]].fillna(0)
            y = df_model['LodgingRevenue']
            model = RandomForestRegressor()
            model.fit(X, y)
            st.write("**R^2 Skoru:**", model.score(X, y))

        elif model_choice == "K-Means":
            df_k = df[["Age", "DaysSinceCreation"]].dropna()
            scaler = StandardScaler()
            scaled = scaler.fit_transform(df_k)
            kmeans = KMeans(n_clusters=3)
            df_k['Cluster'] = kmeans.fit_predict(scaled)
            fig_k = px.scatter(df_k, x="Age", y="DaysSinceCreation", color="Cluster", title="K-Means Kümelemesi")
            st.plotly_chart(fig_k)

    elif section == "Zaman Serisi (ARIMA)":
        st.header("\u23F1\ufe0f Zaman Serisi - ARIMA")
        ts = df[['DaysSinceCreation', 'LodgingRevenue']].dropna()
        ts = ts.groupby('DaysSinceCreation').sum()
        ts_series = ts['LodgingRevenue']
        model = ARIMA(ts_series, order=(2,1,1))
        result = model.fit()
        forecast = result.predict(start=0, end=len(ts_series)+20, typ='levels')
        fig_arima = px.line(x=range(len(forecast)), y=forecast, title="ARIMA Tahmini")
        st.plotly_chart(fig_arima)

    elif section == "Öneri Sistemi (KNN)":
        st.header("\U0001F91D KNN Tabanlı Öneri Sistemi")
        age = st.slider("Yaş", 18, 100, 35)
        days = st.slider("DaysSinceCreation", 0, 1000, 100)
        lead = st.slider("AvgLeadTime", 0, 300, 50)

        df_knn = df[["Age", "DaysSinceCreation", "AvgLeadTime"]].dropna()
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df_knn)
        knn = NearestNeighbors(n_neighbors=3)
        knn.fit(scaled)
        distances, indices = knn.kneighbors([[age, days, lead]])
        st.write("**Benzer Müşteriler:**")
        st.dataframe(df_knn.iloc[indices[0]])

else:
    st.warning("Lütfen bir Excel (.xlsx) veri dosyası yükleyin.")
