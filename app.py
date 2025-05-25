import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Otel Veri Analizi", layout="wide")

st.title("🏨 Otel Müşteri Verileri Analiz Arayüzü")

uploaded_file = st.file_uploader("📂 .xlsx veri dosyanızı yükleyin", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.success("Veri başarıyla yüklendi!")

    st.sidebar.title("📊 Menü")
    menu = st.sidebar.selectbox("Bir bölüm seçin:", [
        "CRISP-DM Süreci", "Veri Önizleme", "Keşifsel Veri Analizi (EDA)",
        "Makine Öğrenmesi", "Zaman Serisi (ARIMA)", "Öneri Sistemi (KNN)"
    ])

    if menu == "CRISP-DM Süreci":
        st.header("📌 CRISP-DM Metodolojisi")
        st.markdown('''**1. İş Hedefi:** Otel yöneticilerinin müşterileri daha iyi anlaması  
**2. Veri Anlama:** Müşteri profili, gelir, rezervasyon durumu  
**3. Veri Hazırlığı:** Eksik verilerin temizlenmesi  
**4. Modelleştirme:** Lojistik Regresyon, Rastgele Orman, K-Means, ARIMA, KNN  
**5. Değerlendirme:** Doğruluk skorları ve grafiklerle inceleme  
**6. Dağıtım:** Bu Streamlit uygulaması''')

    elif menu == "Veri Önizleme":
        st.header("📄 Veri Önizleme")
        st.dataframe(df.head())
        st.write("Veri boyutu:", df.shape)

    elif menu == "Keşifsel Veri Analizi (EDA)":
        st.header("📈 EDA")
        st.subheader("İstatistiksel Özet")
        st.write(df.describe())

        st.subheader("Korelasyon Matrisi")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        st.subheader("Dağılım Grafiği")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        col = st.selectbox("Bir değişken seçin", num_cols)
        fig2, ax2 = plt.subplots()
        sns.histplot(df[col].dropna(), kde=True, ax=ax2)
        st.pyplot(fig2)

    elif menu == "Makine Öğrenmesi":
        st.header("🤖 Makine Öğrenmesi")

        model = st.selectbox("Model Seçin:", ["Lojistik Regresyon", "Rastgele Orman", "K-Means"])

        if model == "Lojistik Regresyon":
            df_model = df.dropna(subset=['BookingsCanceled'])
            X = df_model[['Age', 'DaysSinceLastStay', 'DaysSinceFirstStay']].fillna(0)
            y = df_model['BookingsCanceled']
            m = LogisticRegression()
            m.fit(X, y)
            st.write("Doğruluk Skoru:", m.score(X, y))

        elif model == "Rastgele Orman":
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
            labels = kmeans.fit_predict(X_scaled)
            df_cluster['Cluster'] = labels
            fig, ax = plt.subplots()
            sns.scatterplot(data=df_cluster, x='Age', y='DaysSinceCreation', hue='Cluster', palette='Set2', ax=ax)
            st.pyplot(fig)

    elif menu == "Zaman Serisi (ARIMA)":
        st.header("⏳ Zaman Serisi - ARIMA")
        ts = df[['DaysSinceCreation', 'LodgingRevenue']].dropna()
        ts = ts.groupby('DaysSinceCreation').sum().sort_index()
        ts_series = ts['LodgingRevenue']

        try:
            model = ARIMA(ts_series, order=(2,1,1))
            result = model.fit()
            forecast = result.predict(start=len(ts_series), end=len(ts_series)+20, typ='levels')
            fig, ax = plt.subplots()
            ts_series.plot(label='Gerçek', ax=ax)
            forecast.plot(label='Tahmin', ax=ax, color='red')
            ax.legend()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"ARIMA modeli başarısız: {e}")

    elif menu == "Öneri Sistemi (KNN)":
        st.header("🤝 KNN Öneri Sistemi")
        age = st.slider("Yaş", 18, 100, 35)
        days = st.slider("DaysSinceCreation", 0, 1000, 100)
        lead = st.slider("AvgLeadTime", 0, 300, 50)

        df_knn = df[['Age', 'DaysSinceCreation', 'AvgLeadTime']].dropna()
        data_scaled = StandardScaler().fit_transform(df_knn)
        knn = NearestNeighbors(n_neighbors=3)
        knn.fit(data_scaled)
        distances, indices = knn.kneighbors([[age, days, lead]])
        st.write("Benzer Müşteriler:")
        st.dataframe(df_knn.iloc[indices[0]])
else:
    st.warning("Lütfen bir .xlsx dosyası yükleyin.")
