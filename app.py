import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="🏨 Otel Müşteri Analiz Paneli", layout="wide")

# Stil
st.markdown("""
    <style>
        body {
            color: white;
            background-color: #1e1e2f;
        }
        .block-container {
            padding: 2rem 2rem 2rem 2rem;
        }
        h1, h2, h3 {
            color: #ffcc00;
        }
        .stSelectbox > div {
            color: black !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🏨 Otel Müşteri Verisi Analizi ve Karar Destek Sistemi")

uploaded_file = st.sidebar.file_uploader("📂 .xlsx dosyanızı yükleyin", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)

    section = st.sidebar.selectbox("Bölüm Seçin", [
        "CRISP-DM Süreci", "Veri Önizleme", "Keşifsel Veri Analizi", 
        "Makine Öğrenmesi Modelleri", "Hipermetre Optimizasyonu", 
        "Zaman Serisi (ARIMA)", "KNN Öneri Sistemi"
    ])

    if section == "CRISP-DM Süreci":
        st.header("🔍 CRISP-DM Süreci")
        st.markdown("""
        - **1. İş Hedefi**: Otel yöneticilerinin müşteri eğilimlerini anlaması  
        - **2. Veri Anlama**: Gelir, iptal durumu, kalış süresi gibi veriler  
        - **3. Veri Hazırlığı**: Eksik verilerin temizlenmesi, tip dönüşümleri  
        - **4. Modelleştirme**: Lojistik Regresyon, Random Forest, KMeans, KNN, ARIMA  
        - **5. Değerlendirme**: Başarı skorları, grafiklerle analiz  
        - **6. Dağıtım**: Bu Streamlit uygulaması
        """)

    elif section == "Veri Önizleme":
        st.subheader("📋 İlk 5 Satır")
        st.dataframe(df.head())
        st.write("🔢 Veri seti boyutu:", df.shape)

    elif section == "Keşifsel Veri Analizi":
        st.header("📊 Keşifsel Veri Analizi (EDA)")
        st.subheader("İstatistiksel Özet")
        st.dataframe(df.describe())

        st.subheader("Eksik Değerler")
        st.dataframe(df.isnull().sum().reset_index().rename(columns={0: "Eksik Sayısı"}))

        st.subheader("Korelasyon Matrisi")
        numeric_df = df.select_dtypes(include=np.number)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        st.subheader("Sayısal Değişken Dağılımı")
        selected_col = st.selectbox("Görselleştirilecek Sütun", numeric_df.columns)
        fig = px.histogram(df, x=selected_col, nbins=30, title=f"{selected_col} Dağılımı")
        st.plotly_chart(fig)

    elif section == "Makine Öğrenmesi Modelleri":
        st.header("🤖 Makine Öğrenmesi Modelleri")

        model_option = st.selectbox("Model Seçin", [
            "Lojistik Regresyon", "Rastgele Orman", "K-Means Kümeleme"
        ])

        if model_option == "Lojistik Regresyon":
            df_model = df.dropna(subset=['BookingsCanceled'])
            X = df_model[['Age', 'DaysSinceFirstStay', 'DaysSinceLastStay']].fillna(0)
            y = df_model['BookingsCanceled']
            model = LogisticRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            st.subheader("Model Değerlendirme")
            st.text(classification_report(y, y_pred))

        elif model_option == "Rastgele Orman":
            df_model = df.dropna(subset=['LodgingRevenue'])
            X = df_model[['Age', 'DaysSinceCreation']].fillna(0)
            y = df_model['LodgingRevenue']
            model = RandomForestRegressor()
            model.fit(X, y)
            y_pred = model.predict(X)
            st.subheader("Model Değerlendirme")
            st.write("MSE:", mean_squared_error(y, y_pred))
            st.write("R2 Skoru:", r2_score(y, y_pred))

        elif model_option == "K-Means Kümeleme":
            df_cluster = df[['Age', 'DaysSinceCreation']].dropna()
            X = StandardScaler().fit_transform(df_cluster)
            kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
            df_cluster['Cluster'] = kmeans.labels_
            fig = px.scatter(df_cluster, x='Age', y='DaysSinceCreation', color='Cluster',
                             title="K-Means Kümeleme Sonucu", color_continuous_scale='Viridis')
            st.plotly_chart(fig)

    elif section == "Hipermetre Optimizasyonu":
        st.header("⚙️ Hipermetre Optimizasyonu")

        model_choice = st.selectbox("Model Seçin", ["Lojistik Regresyon", "Random Forest"])
        if model_choice == "Lojistik Regresyon":
            X = df[['Age', 'DaysSinceFirstStay']].fillna(0)
            y = df['BookingsCanceled'].fillna(0)
            param_grid = {'C': [0.1, 1, 10]}
            grid = GridSearchCV(LogisticRegression(), param_grid, cv=3)
            grid.fit(X, y)
            st.write("En iyi parametreler:", grid.best_params_)
        else:
            X = df[['Age', 'DaysSinceCreation']].fillna(0)
            y = df['LodgingRevenue'].fillna(0)
            param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
            grid = GridSearchCV(RandomForestRegressor(), param_grid, cv=3)
            grid.fit(X, y)
            st.write("En iyi parametreler:", grid.best_params_)

    elif section == "Zaman Serisi (ARIMA)":
        st.header("⏱️ Zaman Serisi Tahmini (ARIMA)")
        ts_df = df[['DaysSinceCreation', 'LodgingRevenue']].dropna()
        ts_df = ts_df.groupby('DaysSinceCreation').sum()
        series = ts_df['LodgingRevenue']
        model = ARIMA(series, order=(2, 1, 2))
        result = model.fit()
        forecast = result.forecast(steps=20)
        fig = px.line(x=series.index, y=series.values, labels={'x': 'Gün', 'y': 'Gelir'}, title="ARIMA Zaman Serisi")
        st.plotly_chart(fig)

    elif section == "KNN Öneri Sistemi":
        st.header("🧠 KNN Öneri Sistemi")
        age = st.slider("Yaş", 18, 90, 30)
        days = st.slider("DaysSinceCreation", 0, 1000, 300)
        lead = st.slider("AvgLeadTime", 0, 300, 100)
        df_knn = df[['Age', 'DaysSinceCreation', 'AvgLeadTime']].dropna()
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_knn)
        knn = NearestNeighbors(n_neighbors=3)
        knn.fit(df_scaled)
        distances, indices = knn.kneighbors(scaler.transform([[age, days, lead]]))
        st.write("Benzer Müşteriler")
        st.dataframe(df_knn.iloc[indices[0]])

else:
    st.warning("Lütfen analiz için bir Excel dosyası yükleyin.")
