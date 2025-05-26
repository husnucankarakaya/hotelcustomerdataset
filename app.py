# Streamlit App: Hotel Customer Analysis & ML Platform

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import NearestNeighbors
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Otel Analizi", layout="wide")
st.title("\U0001F3E8 Otel Müşteri Verileri: Tümleşik Analiz ve ML Arayüzü")

# Dosya yükleme
uploaded_file = st.sidebar.file_uploader("\U0001F4C2 .xlsx dosyanızı yükleyin", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.success("Veri başarıyla yüklendi!")

    # Sidebar Menü
    menu = st.sidebar.radio("Menü Seçimi", [
        "CRISP-DM Süreci", "EDA (Keşifsel Veri Analizi)",
        "Makine Öğrenmesi", "Hiperparametre Optimizasyonu",
        "Zaman Serisi (ARIMA)", "KNN Öneri Sistemi"
    ])

    # 1. CRISP-DM
    if menu == "CRISP-DM Süreci":
        st.header("\U0001F4C8 CRISP-DM Süreci")
        st.markdown("""
        - **İş Hedefi:** Otel yöneticilerinin müşteri davranışlarını analiz ederek karar destek sunmak
        - **Veri Anlama:** Yaş, gelir, harcama, iptal durumu vb. veriler
        - **Veri Hazırlığı:** Eksik verilerin giderilmesi, tür dönüşümleri
        - **Modelleme:** Lojistik Regresyon, Random Forest, K-Means, ARIMA, KNN
        - **Değerlendirme:** Model skorları, görseller
        - **Dağıtım:** Etkileşimli Streamlit arayüzü
        """)

    # 2. EDA
    elif menu == "EDA (Keşifsel Veri Analizi)":
        st.header("\U0001F4CA Keşifsel Veri Analizi")

        st.subheader("Veri Önizleme")
        st.dataframe(df.head())
        st.write(f"Veri Boyutu: {df.shape}")

        st.subheader("İstatistiksel Özet")
        st.dataframe(df.describe())

        st.subheader("Eksik Veri Görselleştirme")
        fig, ax = plt.subplots()
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis", ax=ax)
        st.pyplot(fig)

        st.subheader("Korelasyon Matrisi")
        num_df = df.select_dtypes(include=np.number)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        st.subheader("Dağılım Grafiği")
        col = st.selectbox("Kolon seçin:", num_df.columns)
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)

    # 3. ML Modelleri
    elif menu == "Makine Öğrenmesi":
        st.header("\U0001F9EA Makine Öğrenmesi Modelleri")
        model = st.selectbox("Model Seçin", ["Lojistik Regresyon", "Rastgele Orman Regresyonu", "K-Means"])

        if model == "Lojistik Regresyon":
            df_model = df.dropna(subset=['BookingsCanceled'])
            X = df_model[['Age', 'DaysSinceLastStay', 'DaysSinceFirstStay']].fillna(0)
            y = df_model['BookingsCanceled']
            m = LogisticRegression()
            m.fit(X, y)
            st.write("Doğruluk:", m.score(X, y))

        elif model == "Rastgele Orman Regresyonu":
            df_model = df.dropna(subset=['LodgingRevenue'])
            X = df_model[['Age', 'DaysSinceLastStay', 'DaysSinceFirstStay']].fillna(0)
            y = df_model['LodgingRevenue']
            m = RandomForestRegressor()
            m.fit(X, y)
            st.write("R^2 Skoru:", m.score(X, y))

        elif model == "K-Means":
            df_cluster = df[['Age', 'DaysSinceCreation']].dropna()
            X_scaled = StandardScaler().fit_transform(df_cluster)
            kmeans = KMeans(n_clusters=3, random_state=42)
            df_cluster['Cluster'] = kmeans.fit_predict(X_scaled)
            fig, ax = plt.subplots()
            sns.scatterplot(data=df_cluster, x='Age', y='DaysSinceCreation', hue='Cluster', palette='Set2')
            st.pyplot(fig)

    # 4. Hiperparametre Optimizasyonu
    elif menu == "Hiperparametre Optimizasyonu":
        st.header("\U0001F527 Hiperparametre Tuning")

        df_model = df.dropna(subset=['LodgingRevenue'])
        X = df_model[['Age', 'DaysSinceLastStay', 'DaysSinceFirstStay']].fillna(0)
        y = df_model['LodgingRevenue']

        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 20]
        }

        rf = RandomForestRegressor(random_state=42)
        grid = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1)
        grid.fit(X, y)

        st.write("En iyi parametreler:", grid.best_params_)
        st.write("En iyi skor:", grid.best_score_)

    # 5. ARIMA Zaman Serisi
    elif menu == "Zaman Serisi (ARIMA)":
        st.header("\u23F3 Zaman Serisi - ARIMA")
        ts = df[['DaysSinceCreation', 'LodgingRevenue']].dropna()
        ts = ts.groupby('DaysSinceCreation').sum()
        ts_series = ts['LodgingRevenue']

        model = ARIMA(ts_series, order=(2, 1, 2))
        result = model.fit()
        forecast = result.predict(start=0, end=len(ts_series)+20, typ='levels')

        fig, ax = plt.subplots()
        ts_series.plot(ax=ax, label='Gerçek')
        forecast.plot(ax=ax, label='Tahmin', color='red')
        ax.legend()
        st.pyplot(fig)

    # 6. KNN Öneri Sistemi
    elif menu == "KNN Öneri Sistemi":
        st.header("\U0001F91D KNN Tabanlı Öneri Sistemi")
        age = st.slider("Yaş", 18, 100, 35)
        days = st.slider("DaysSinceCreation", 0, 1000, 100)
        lead = st.slider("AvgLeadTime", 0, 300, 50)

        df_knn = df[['Age', 'DaysSinceCreation', 'AvgLeadTime']].dropna()
        data_scaled = StandardScaler().fit_transform(df_knn)
        knn = NearestNeighbors(n_neighbors=3)
        knn.fit(data_scaled)
        distances, indices = knn.kneighbors([[age, days, lead]])
        st.write("Benzer Müşteri Profilleri:")
        st.dataframe(df_knn.iloc[indices[0]])

else:
    st.warning("Lütfen bir Excel dosyası yükleyin.")
