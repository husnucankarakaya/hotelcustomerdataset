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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, silhouette_score
import plotly.express as px
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore")

# Sayfa ayarı
st.set_page_config(page_title="Otel Analitik Arayüzü", layout="wide")
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .stSidebar {
            background-color: #e9ecef;
        }
        .block-container {
            padding-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🏨 Otel Müşteri Verisi Analitiği")
st.markdown("Kapsamlı analiz, makine öğrenmesi ve tahmin modelleri")

# Dosya yükleme
uploaded_file = st.file_uploader("📂 .xlsx veri dosyanızı yükleyin", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.sidebar.title("🔍 Menüler")
    menu = st.sidebar.selectbox("Bir modül seçin:", [
        "CRISP-DM Süreci", "Veri Önizleme", "Keşifsel Veri Analizi (EDA)",
        "Makine Öğrenmesi Modelleri", "Zaman Serisi Tahmini (ARIMA)", "KNN Öneri Sistemi"
    ])

    if menu == "CRISP-DM Süreci":
        st.header("📌 CRISP-DM Metodolojisi")
        st.markdown('''
        **1. İş Hedefi:** Otel yöneticilerine müşteri verisiyle karar destek sağlamak.  
        **2. Veri Anlama:** Yaş, rezervasyonlar, gelir, tarihsel bilgiler vs.  
        **3. Veri Hazırlığı:** Eksik değerlerin temizlenmesi, dönüşümler  
        **4. Modelleştirme:** 5 farklı model uygulandı  
        **5. Değerlendirme:** Performans metrikleri, görselleştirme  
        **6. Dağıtım:** Streamlit arayüzü olarak sunum
        ''')

    elif menu == "Veri Önizleme":
        st.header("📄 Veri Önizleme")
        st.dataframe(df.head(10))
        st.write("Veri boyutu:", df.shape)
        st.write("Sütunlar:", df.columns.tolist())

    elif menu == "Keşifsel Veri Analizi (EDA)":
        st.header("🔍 Keşifsel Veri Analizi")

        st.subheader("İstatistiksel Özet")
        st.dataframe(df.describe().T)

        st.subheader("Eksik Değerler")
        st.dataframe(df.isnull().sum())

        st.subheader("Sayısal Değişken Dağılımları")
        numeric_cols = df.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            fig = px.histogram(df, x=col, marginal="box", nbins=30, title=f"{col} Dağılımı")
            st.plotly_chart(fig)

        st.subheader("Korelasyon Matrisi")
        corr = df.corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='Viridis')
        st.plotly_chart(fig)

    elif menu == "Makine Öğrenmesi Modelleri":
        st.header("🤖 Makine Öğrenmesi")
        model = st.selectbox("Model Seçin:", ["Lojistik Regresyon", "Rastgele Orman Regresyonu", "K-Means Kümeleme"])

        if model == "Lojistik Regresyon":
            df_model = df.dropna(subset=['BookingsCanceled'])
            X = df_model[['Age', 'DaysSinceLastStay', 'DaysSinceFirstStay']].fillna(0)
            y = df_model['BookingsCanceled']
            logreg = LogisticRegression()
            logreg.fit(X, y)
            y_pred = logreg.predict(X)
            st.write("Doğruluk:", logreg.score(X, y))
            st.text("Sınıflandırma Raporu:")
            st.text(classification_report(y, y_pred))
            cm = confusion_matrix(y, y_pred)
            fig = px.imshow(cm, text_auto=True, title="Confusion Matrix")
            st.plotly_chart(fig)

        elif model == "Rastgele Orman Regresyonu":
            df_model = df.dropna(subset=['LodgingRevenue'])
            X = df_model[['Age', 'DaysSinceLastStay', 'DaysSinceFirstStay']].fillna(0)
            y = df_model['LodgingRevenue']
            rf = RandomForestRegressor()
            rf.fit(X, y)
            preds = rf.predict(X)
            st.write("R² Skoru:", rf.score(X, y))
            st.write("MSE:", mean_squared_error(y, preds))
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=y, name="Gerçek"))
            fig.add_trace(go.Scatter(y=preds, name="Tahmin"))
            st.plotly_chart(fig)

        elif model == "K-Means Kümeleme":
            df_cluster = df[['Age', 'DaysSinceCreation']].dropna()
            X = StandardScaler().fit_transform(df_cluster)
            kmeans = KMeans(n_clusters=3, random_state=42)
            labels = kmeans.fit_predict(X)
            df_cluster['Cluster'] = labels
            st.write("Silhouette Skoru:", silhouette_score(X, labels))
            fig = px.scatter(df_cluster, x='Age', y='DaysSinceCreation', color='Cluster', title="Kümeleme Sonuçları")
            st.plotly_chart(fig)

    elif menu == "Zaman Serisi Tahmini (ARIMA)":
        st.header("⏳ Zaman Serisi Analizi - ARIMA")
        ts = df[['DaysSinceCreation', 'LodgingRevenue']].dropna()
        ts = ts.groupby('DaysSinceCreation').sum()
        series = ts['LodgingRevenue']
        model = ARIMA(series, order=(2,1,2))
        fit = model.fit()
        forecast = fit.forecast(steps=30)
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=series, name="Gerçek"))
        fig.add_trace(go.Scatter(y=forecast, name="Tahmin"))
        st.plotly_chart(fig)

    elif menu == "KNN Öneri Sistemi":
        st.header("🤝 KNN Tabanlı Müşteri Öneri Sistemi")
        age = st.slider("Yaş", 18, 100, 35)
        days = st.slider("DaysSinceCreation", 0, 1000, 100)
        lead = st.slider("AvgLeadTime", 0, 300, 50)
        df_knn = df[['Age', 'DaysSinceCreation', 'AvgLeadTime']].dropna()
        scaled = StandardScaler().fit_transform(df_knn)
        knn = NearestNeighbors(n_neighbors=3)
        knn.fit(scaled)
        dist, ind = knn.kneighbors([[age, days, lead]])
        st.dataframe(df_knn.iloc[ind[0]])

else:
    st.warning("Lütfen bir .xlsx dosyası yükleyin.")
