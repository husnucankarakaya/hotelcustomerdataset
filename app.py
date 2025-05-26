import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import classification_report, mean_squared_error, r2_score, confusion_matrix, roc_curve, auc, silhouette_score
from sklearn.impute import KNNImputer
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import pmdarima as pm
from prophet import Prophet
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# SHAP opsiyonel
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Uygulama konfigürasyonu
st.set_page_config(page_title="🏨 Otel Müşteri Analiz Paneli", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
    <style>
    .main {background-color: #f8fafc;}
    .stButton>button {background-color: #10b981; color: white; border-radius: 8px; padding: 12px; font-weight: bold;}
    .stSelectbox, .stMultiselect, .stSlider {background-color: #ffffff; border-radius: 8px; padding: 8px; border: 1px solid #e2e8f0;}
    .stSidebar {background-color: #1e293b; color: white;}
    h1, h2, h3 {color: #1e293b; font-family: 'Inter', sans-serif;}
    .stMarkdown {font-family: 'Inter', sans-serif; font-size: 16px;}
    .footer {text-align: center; color: #1e293b; margin-top: 50px; font-size: 14px;}
    .card {background-color: #ffffff; border-radius: 8px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
    .stAlert {border-radius: 8px;}
    .metric-card {background-color: #e0f2fe; border-radius: 8px; padding: 10px; text-align: center;}
    </style>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
""", unsafe_allow_html=True)

# Tema seçimi
theme = st.sidebar.selectbox("Tema Seçin", ["Açık", "Koyu"])
if theme == "Koyu":
    st.markdown("""
        <style>
        .main {background-color: #1e293b; color: white;}
        h1, h2, h3 {color: #ffcc00;}
        .card {background-color: #334155; color: white;}
        .metric-card {background-color: #4b5563; color: white;}
        </style>
    """, unsafe_allow_html=True)

# Veri yükleme
@st.cache_data
def load_data(file=None):
    try:
        if file:
            if file.name.endswith('.xlsx'):
                return pd.read_excel(file)
            else:
                raise ValueError("Yalnızca XLSX dosyaları destekleniyor.")
        return pd.read_excel("HotelCustomersDataset.xlsx")
    except Exception as e:
        st.error(f"Veri yükleme hatası: {e}")
        return None

# Dosya yükleyici
st.sidebar.header("📤 Veri Yükleme")
uploaded_file = st.sidebar.file_uploader("📂 .xlsx dosyanızı yükleyin", type=["xlsx"])
if uploaded_file:
    df = load_data(uploaded_file)
    if df is not None:
        st.success("✅ Dosya başarıyla yüklendi!")
else:
    st.warning("ℹ️ Lütfen analiz için bir Excel dosyası yükleyin.")
    df = load_data()

if df is None:
    st.stop()

# Eksik veri doldurma
imputer = KNNImputer(n_neighbors=5)
df['Age'] = imputer.fit_transform(df[['Age']])

# Sidebar navigasyon
st.sidebar.header("📊 Menü")
section = st.sidebar.selectbox("Bölüm Seçin", [
    "CRISP-DM Süreci",
    "Veri Önizleme",
    "Keşifsel Veri Analizi (EDA)",
    "Makine Öğrenmesi Modelleri",
    "Hiperparametre Optimizasyonu",
    "Zaman Serisi Analizi",
    "KNN Öneri Sistemi",
    "RFM Analizi",
    "Özel Talepler Analizi",
    "İptal/No-Show Analizi",
    "Gelir Optimizasyonu"
])

# Filtreleme
st.sidebar.header("🔍 Filtreleme")
nationality = st.sidebar.multiselect("Milliyet Seçin", df['Nationality'].unique(), default=['PRT', 'FRA', 'DEU'])
age_range = st.sidebar.slider("Yaş Aralığı", 0, 100, (0, 100))
market_segment = st.sidebar.multiselect("Pazar Segmenti", df['MarketSegment'].unique(), default=df['MarketSegment'].unique())
distribution_channel = st.sidebar.multiselect("Dağıtım Kanalı", df['DistributionChannel'].unique(), default=df['DistributionChannel'].unique())
filtered_df = df[
    (df['Nationality'].isin(nationality)) &
    (df['Age'].between(age_range[0], age_range[1], inclusive='both')) &
    (df['MarketSegment'].isin(market_segment)) &
    (df['DistributionChannel'].isin(distribution_channel))
]
if st.sidebar.button("Filtreleri Sıfırla"):
    filtered_df = df

# Görselleştirme fonksiyonları
def plot_histogram(df, column, title):
    fig = px.histogram(df, x=column, title=title, color_discrete_sequence=['#10b981'], nbins=50, marginal="box")
    fig.update_layout(xaxis_title=column, yaxis_title="Frekans", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

def plot_violin(df, column, title):
    fig = px.violin(df, y=column, title=title, color_discrete_sequence=['#3b82f6'], box=True, points="outliers")
    fig.update_layout(yaxis_title=column, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

def plot_pie(df, column, title):
    counts = df[column].value_counts().head(10)
    fig = px.pie(values=counts.values, names=counts.index, title=title, color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_traces(textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

def plot_correlation_matrix(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='Viridis', text=corr.values.round(2), texttemplate="%{text}"))
    fig.update_layout(title="Korelasyon Matrisi", height=700, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

def plot_elbow_method(X_scaled):
    inertias = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    fig = px.line(x=range(1, 11), y=inertias, title="Elbow Yöntemi ile Optimal Küme Sayısı", markers=True)
    fig.update_layout(xaxis_title="Küme Sayısı", yaxis_title="Inertia", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

def plot_silhouette_score(X_scaled, max_clusters=10):
    scores = []
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores.append(score)
    fig = px.line(x=range(2, max_clusters + 1), y=scores, title="Silhouette Skoru ile Küme Kalitesi", markers=True)
    fig.update_layout(xaxis_title="Küme Sayısı", yaxis_title="Silhouette Skoru", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# CRISP-DM Süreci
if section == "CRISP-DM Süreci":
    st.header("📌 CRISP-DM Metodolojisi")
    st.markdown("""
    <div class='card'>
        <h3>1. İş Hedefi</h3>
        <p>Otel yöneticilerinin müşteri davranışlarını anlaması, gelir optimizasyonu, müşteri memnuniyetini artırma ve iptal/no-show oranlarını azaltma.</p>
        <h3>2. Veri Anlama</h3>
        <p>83,590 satırlık veri seti; müşteri profili (yaş, milliyet), rezervasyon alışkanlıkları, gelir ve özel talepler içerir.</p>
        <h3>3. Veri Hazırlığı</h3>
        <p>Eksik verilerin KNN Imputer ile doldurulması, aykırı değerlerin yönetimi, kategorik değişkenlerin kodlanması.</p>
        <h3>4. Modelleştirme</h3>
        <p>Lojistik Regresyon, Random Forest, XGBoost, KMeans, KNN, ARIMA, Prophet ve RFM analizi.</p>
        <h3>5. Değerlendirme</h3>
        <p>Doğruluk, R², Silhouette skoru, ROC eğrisi ve SHAP analizleriyle model performansı.</p>
        <h3>6. Dağıtım</h3>
        <p>Streamlit ile interaktif arayüz üzerinden sonuçların sunulması.</p>
    </div>
    """, unsafe_allow_html=True)

# Veri Önizleme
elif section == "Veri Önizleme":
    st.header("📄 Veri Önizleme")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='metric-card'><h4>Toplam Satır</h4><p>{}</p></div>".format(filtered_df.shape[0]), unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='metric-card'><h4>Sütun Sayısı</h4><p>{}</p></div>".format(filtered_df.shape[1]), unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='metric-card'><h4>Eksik Değerler</h4><p>{}</p></div>".format(filtered_df.isnull().sum().sum()), unsafe_allow_html=True)

    st.subheader("Veri Seti Örneği")
    st.dataframe(filtered_df.head(10), use_container_width=True)

    st.subheader("Eksik Değerler")
    st.dataframe(filtered_df.isnull().sum().reset_index().rename(columns={0: "Eksik Sayısı"}), use_container_width=True)

    st.subheader("Temel İstatistikler")
    st.dataframe(filtered_df.describe(), use_container_width=True)

# Keşifsel Veri Analizi
elif section == "Keşifsel Veri Analizi (EDA)":
    st.header("📊 Keşifsel Veri Analizi (EDA)")
    st.markdown("<div class='card'>Veri setinin detaylı görselleştirmeleri ve analizleri.</div>", unsafe_allow_html=True)

    st.subheader("İstatistiksel Özet")
    st.dataframe(filtered_df.describe(), use_container_width=True)

    st.subheader("Kategorik Değişken Dağılımları")
    col1, col2 = st.columns(2)
    with col1:
        plot_pie(filtered_df, 'Nationality', 'Milliyet Dağılımı (İlk 10)')
    with col2:
        plot_pie(filtered_df, 'MarketSegment', 'Pazar Segmenti Dağılımı')

    st.subheader("Sayısal Değişken Analizi")
    num_cols = filtered_df.select_dtypes(include=np.number).columns.tolist()
    selected_col = st.selectbox("Bir değişken seçin", num_cols)
    col1, col2 = st.columns(2)
    with col1:
        plot_histogram(filtered_df, selected_col, f"{selected_col} Dağılımı")
    with col2:
        plot_violin(filtered_df, selected_col, f"{selected_col} Violin Grafiği")

    st.subheader("Milliyet Bazında Gelir Analizi")
    nationality_revenue = filtered_df.groupby('Nationality')[['LodgingRevenue', 'OtherRevenue']].sum().reset_index()
    fig = px.bar(nationality_revenue, x='Nationality', y=['LodgingRevenue', 'OtherRevenue'], 
                 title="Milliyet Bazında Toplam Gelir", barmode='group', color_discrete_sequence=['#10b981', '#3b82f6'])
    fig.update_layout(xaxis_title="Milliyet", yaxis_title="Gelir", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Korelasyon Analizi")
    plot_correlation_matrix(filtered_df)

    st.subheader("İkili Değişken Analizi")
    pair_cols = st.multiselect("Değişkenler seçin", num_cols, default=['Age', 'LodgingRevenue', 'OtherRevenue'])
    if pair_cols:
        fig = px.scatter_matrix(filtered_df, dimensions=pair_cols, title="Pair Plot", color='MarketSegment', 
                                color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=700, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sankey Diyagramı - Rezervasyon Akışı")
    sankey_data = filtered_df.groupby(['DistributionChannel', 'MarketSegment']).size().reset_index(name='Count')
    labels = list(sankey_data['DistributionChannel'].unique()) + list(sankey_data['MarketSegment'].unique())
    source = [labels.index(x) for x in sankey_data['DistributionChannel']]
    target = [labels.index(x) + len(sankey_data['DistributionChannel'].unique()) for x in sankey_data['MarketSegment']]
    fig = go.Figure(data=[go.Sankey(
        node=dict(label=labels),
        link=dict(source=source, target=target, value=sankey_data['Count'])
    )])
    fig.update_layout(title="Dağıtım Kanalı ve Pazar Segmenti Akışı", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# Makine Öğrenmesi Modelleri
elif section == "Makine Öğrenmesi Modelleri":
    st.header("🤖 Makine Öğrenmesi Modelleri")
    st.markdown("<div class='card'>Lojistik Regresyon, Random Forest, XGBoost ve KMeans modelleri.</div>", unsafe_allow_html=True)

    model_option = st.selectbox("Model Seçin", ["Lojistik Regresyon", "Random Forest", "XGBoost", "K-Means Kümeleme"])
    df_model = filtered_df.dropna()

    if model_option == "Lojistik Regresyon":
        st.subheader("Lojistik Regresyon - İptal Tahmini")
        X = df_model[['Age', 'AverageLeadTime', 'LodgingRevenue', 'OtherRevenue', 'RoomNights']].fillna(0)
        y = df_model['BookingsCanceled'].apply(lambda x: 1 if x > 0 else 0)
        X_scaled = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.markdown(f"""
        <div class='card'>
            <h4>Model Değerlendirme</h4>
            <pre>{classification_report(y_test, y_pred)}</pre>
        </div>
        """, unsafe_allow_html=True)

        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, text_auto=True, title="Confusion Matrix", color_continuous_scale='Blues')
        fig.update_layout(xaxis_title="Tahmin", yaxis_title="Gerçek")
        st.plotly_chart(fig, use_container_width=True)

        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        roc_auc = auc(fpr, tpr)
        fig = px.line(x=fpr, y=tpr, title=f"ROC Eğrisi (AUC = {roc_auc:.2f})")
        fig.add_scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Rastgele Tahmin')
        fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    elif model_option == "Random Forest":
        st.subheader("Random Forest - Gelir Tahmini")
        X = df_model[['Age', 'AverageLeadTime', 'RoomNights', 'PersonsNights']].fillna(0)
        y = df_model['LodgingRevenue']
        X_scaled = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.markdown(f"""
        <div class='card'>
            <p><strong>MSE:</strong> {mean_squared_error(y_test, y_pred):.2f}</p>
            <p><strong>R² Skoru:</strong> {r2_score(y_test, y_pred):.2f}</p>
        </div>
        """, unsafe_allow_html=True)

        importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
        fig = px.bar(importance, x='Feature', y='Importance', title="Özellik Önem Sıralaması", color='Importance', 
                     color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)

        if SHAP_AVAILABLE:
            st.subheader("SHAP Analizi")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test, feature_names=X.columns, show=False)
            st.pyplot(plt)

    elif model_option == "XGBoost":
        st.subheader("XGBoost - Gelir Tahmini")
        X = df_model[['Age', 'AverageLeadTime', 'RoomNights', 'PersonsNights']].fillna(0)
        y = df_model['LodgingRevenue']
        X_scaled = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        model = xgb.XGBRegressor(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.markdown(f"""
        <div class='card'>
            <p><strong>MSE:</strong> {mean_squared_error(y_test, y_pred):.2f}</p>
            <p><strong>R² Skoru:</strong> {r2_score(y_test, y_pred):.2f}</p>
        </div>
        """, unsafe_allow_html=True)

        importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
        fig = px.bar(importance, x='Feature', y='Importance', title="Özellik Önem Sıralaması", color='Importance', 
                     color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)

    elif model_option == "K-Means Kümeleme":
        st.subheader("K-Means Kümeleme")
        X = df_model[['Age', 'LodgingRevenue', 'OtherRevenue']].dropna()
        X_scaled = StandardScaler().fit_transform(X)

        col1, col2 = st.columns(2)
        with col1:
            plot_elbow_method(X_scaled)
        with col2:
            plot_silhouette_score(X_scaled)

        n_clusters = st.slider("Küme Sayısı", 2, 10, 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df_model['Cluster'] = kmeans.fit_predict(X_scaled)

        fig = px.scatter_3d(df_model, x='Age', y='LodgingRevenue', z='OtherRevenue', color='Cluster', 
                            title="KMeans Kümeleri (3D)", color_continuous_scale=px.colors.qualitative.Set2)
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Küme Özetleri")
        cluster_summary = df_model.groupby('Cluster').agg({
            'Age': ['mean', 'count'],
            'LodgingRevenue': 'mean',
            'OtherRevenue': 'mean'
        }).round(2)
        st.dataframe(cluster_summary, use_container_width=True)

# Hiperparametre Optimizasyonu
elif section == "Hiperparametre Optimizasyonu":
    st.header("⚙️ Hiperparametre Optimizasyonu")
    model_choice = st.selectbox("Model Seçin", ["Lojistik Regresyon", "Random Forest", "XGBoost"])

    if model_choice == "Lojistik Regresyon":
        X = df[['Age', 'AverageLeadTime', 'DaysSinceFirstStay']].fillna(0)
        y = df['BookingsCanceled'].apply(lambda x: 1 if x > 0 else 0)
        param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'solver': ['lbfgs', 'liblinear'], 'max_iter': [1000]}
        grid = GridSearchCV(LogisticRegression(), param_grid, cv=5, n_jobs=-1)
        grid.fit(X, y)
        st.markdown(f"""
        <div class='card'>
            <p><strong>En iyi parametreler:</strong> {grid.best_params_}</p>
            <p><strong>En iyi skor:</strong> {grid.best_score_:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

    elif model_choice == "Random Forest":
        X = df[['Age', 'DaysSinceCreation', 'AverageLeadTime']].fillna(0)
        y = df['LodgingRevenue'].fillna(0)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
        grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, n_jobs=-1)
        grid.fit(X, y)
        st.markdown(f"""
        <div class='card'>
            <p><strong>En iyi parametreler:</strong> {grid.best_params_}</p>
            <p><strong>En iyi skor:</strong> {grid.best_score_:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

    elif model_choice == "XGBoost":
        X = df[['Age', 'DaysSinceCreation', 'AverageLeadTime']].fillna(0)
        y = df['LodgingRevenue'].fillna(0)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.3]
        }
        grid = GridSearchCV(xgb.XGBRegressor(random_state=42), param_grid, cv=5, n_jobs=-1)
        grid.fit(X, y)
        st.markdown(f"""
        <div class='card'>
            <p><strong>En iyi parametreler:</strong> {grid.best_params_}</p>
            <p><strong>En iyi skor:</strong> {grid.best_score_:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

# Zaman Serisi Analizi
elif section == "Zaman Serisi Analizi":
    st.header("⏱️ Zaman Serisi Analizi")
    st.markdown("<div class='card'>ARIMA ve Prophet ile gelir tahmini.</div>", unsafe_allow_html=True)

    ts_df = filtered_df[['DaysSinceCreation', 'LodgingRevenue']].dropna()
    ts_df = ts_df.groupby('DaysSinceCreation').sum().reset_index()
    series = ts_df['LodgingRevenue']

    st.subheader("Mevsimsellik ve Trend Analizi")
    try:
        decomposition = seasonal_decompose(series, model='additive', period=30)
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=["Gerçek", "Trend", "Mevsimsellik"])
        fig.add_trace(go.Scatter(x=ts_df['DaysSinceCreation'], y=decomposition.observed, mode='lines', name='Gerçek'), row=1, col=1)
        fig.add_trace(go.Scatter(x=ts_df['DaysSinceCreation'], y=decomposition.trend, mode='lines', name='Trend'), row=2, col=1)
        fig.add_trace(go.Scatter(x=ts_df['DaysSinceCreation'], y=decomposition.seasonal, mode='lines', name='Mevsimsellik'), row=3, col=1)
        fig.update_layout(title="Mevsimsellik Ayrıştırma", height=600, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Mevsimsellik analizi başarısız: {e}")

    st.subheader("ARIMA Tahmini")
    try:
        model = pm.auto_arima(series, seasonal=True, m=30, stepwise=True, trace=False)
        forecast, conf_int = model.predict(n_periods=20, return_conf_int=True)
        forecast_index = range(ts_df['DaysSinceCreation'].max() + 1, ts_df['DaysSinceCreation'].max() + 21)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts_df['DaysSinceCreation'], y=series, mode='lines', name='Gerçek', line=dict(color='#10b981')))
        fig.add_trace(go.Scatter(x=forecast_index, y=forecast, mode='lines', name='Tahmin', line=dict(color='#f43f5e')))
        fig.add_trace(go.Scatter(x=forecast_index, y=conf_int[:, 0], mode='lines', name='Alt CI', line=dict(color='#93c5fd', dash='dash')))
        fig.add_trace(go.Scatter(x=forecast_index, y=conf_int[:, 1], mode='lines', name='Üst CI', line=dict(color='#93c5fd', dash='dash'), fill='tonexty'))
        fig.update_layout(title="ARIMA Gelir Tahmini", xaxis_title="Gün", yaxis_title="Gelir", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        <div class='card'>
            <p><strong>En iyi ARIMA parametreleri:</strong> {model.order}</p>
            <p><strong>AIC:</strong> {model.aic():.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"ARIMA modeli başarısız: {e}")

    st.subheader("Prophet Tahmini")
    try:
        prophet_df = ts_df.rename(columns={'DaysSinceCreation': 'ds', 'LodgingRevenue': 'y'})
        prophet_df['ds'] = pd.date_range(start='2020-01-01', periods=len(prophet_df), freq='D')
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=20)
        forecast = model.predict(future)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prophet_df['ds'], y=prophet_df['y'], mode='lines', name='Gerçek', line=dict(color='#10b981')))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Tahmin', line=dict(color='#f43f5e')))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Alt CI', line=dict(color='#93c5fd', dash='dash')))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Üst CI', line=dict(color='#93c5fd', dash='dash'), fill='tonexty'))
        fig.update_layout(title="Prophet Gelir Tahmini", xaxis_title="Tarih", yaxis_title="Gelir", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Prophet modeli başarısız: {e}")

# KNN Öneri Sistemi
elif section == "KNN Öneri Sistemi":
    st.header("🧠 KNN Öneri Sistemi")
    st.markdown("<div class='card'>Benzer müşteri profillerini bulma.</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Yaş", 18, 90, 30)
        days = st.slider("DaysSinceCreation", 0, 1000, 300)
    with col2:
        lead = st.slider("AverageLeadTime", 0, 300, 100)
        revenue = st.slider("LodgingRevenue", 0, 5000, 500)

    df_knn = filtered_df[['Age', 'DaysSinceCreation', 'AverageLeadTime', 'LodgingRevenue']].dropna()
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_knn)
    knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
    knn.fit(df_scaled)
    distances, indices = knn.kneighbors(scaler.transform([[age, days, lead, revenue]]))

    st.subheader("Benzer Müşteriler")
    st.dataframe(df_knn.iloc[indices[0]], use_container_width=True)

    df_knn['Distance'] = np.nan
    df_knn.iloc[indices[0], df_knn.columns.get_loc('Distance')] = distances[0]
    fig = px.scatter(df_knn, x='Age', y='LodgingRevenue', size='AverageLeadTime', color='Distance', 
                     title="Benzer Müşteriler (KNN)", color_continuous_scale='Blues', hover_data=['DaysSinceCreation'])
    fig.add_scatter(x=[age], y=[revenue], mode='markers', marker=dict(size=20, color='red'), name='Seçilen Profil')
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# RFM Analizi
elif section == "RFM Analizi":
    st.header("📊 RFM Analizi")
    st.markdown("<div class='card'>Recency, Frequency ve Monetary bazında müşteri segmentasyonu.</div>", unsafe_allow_html=True)

    rfm = filtered_df[['ID', 'DaysSinceLastStay', 'BookingsCheckedIn', 'LodgingRevenue', 'OtherRevenue']].copy()
    rfm['Monetary'] = rfm['LodgingRevenue'] + rfm['OtherRevenue']
    rfm = rfm.rename(columns={'DaysSinceLastStay': 'Recency', 'BookingsCheckedIn': 'Frequency'})
    rfm = rfm[['ID', 'Recency', 'Frequency', 'Monetary']]

    rfm['R_Score'] = pd.qcut(rfm['Recency'].rank(method='first'), 4, labels=[4, 3, 2, 1])
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), 4, labels=[1, 2, 3, 4])
    rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

    st.subheader("RFM Özeti")
    st.dataframe(rfm.head(), use_container_width=True)

    rfm_summary = rfm.groupby('RFM_Score').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'ID': 'count'
    }).rename(columns={'ID': 'Count'}).reset_index()
    fig = px.scatter_3d(rfm_summary, x='Recency', y='Frequency', z='Monetary', size='Count', color='RFM_Score',
                        title="RFM Segmentleri (3D)", color_continuous_scale='Viridis')
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# Özel Talepler Analizi
elif section == "Özel Talepler Analizi":
    st.header("🛏️ Özel Talepler Analizi")
    st.markdown("<div class='card'>Müşterilerin özel oda talepleri.</div>", unsafe_allow_html=True)

    sr_cols = [col for col in filtered_df.columns if col.startswith('SR')]
    sr_counts = filtered_df[sr_cols].sum().sort_values(ascending=False)

    fig = px.bar(x=sr_counts.index, y=sr_counts.values, title="Özel Talep Sıklıkları", 
                 color=sr_counts.values, color_continuous_scale='Viridis', text_auto=True)
    fig.update_layout(xaxis_title="Talep", yaxis_title="Sayı", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Milliyet Bazında Talepler")
    selected_nationality = st.selectbox("Milliyet Seçin", filtered_df['Nationality'].unique())
    nationality_sr = filtered_df[filtered_df['Nationality'] == selected_nationality][sr_cols].sum()
    fig = px.bar(x=sr_cols, y=nationality_sr.values, title=f"{selected_nationality} için Özel Talepler",
                 color=nationality_sr.values, color_continuous_scale='Blues', text_auto=True)
    fig.update_layout(xaxis_title="Talep", yaxis_title="Sayı", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# İptal/No-Show Analizi
elif section == "İptal/No-Show Analizi":
    st.header("🚫 İptal ve No-Show Analizi")
    st.markdown("<div class='card'>İptal ve no-show davranışlarının analizi.</div>", unsafe_allow_html=True)

    st.subheader("İptal Oranları")
    cancel_rate = filtered_df.groupby('MarketSegment')['BookingsCanceled'].mean().reset_index()
    fig = px.bar(cancel_rate, x='MarketSegment', y='BookingsCanceled', title="Pazar Segmentine Göre İptal Oranı",
                 color='BookingsCanceled', color_continuous_scale='Reds')
    fig.update_layout(xaxis_title="Pazar Segmenti", yaxis_title="Ortalama İptal Sayısı", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Yaş ve İptal İlişkisi")
    fig = px.scatter(filtered_df, x='Age', y='BookingsCanceled', color='Nationality', size='LodgingRevenue',
                     title="Yaş ve İptal İlişkisi", hover_data=['MarketSegment'])
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# Gelir Optimizasyonu
elif section == "Gelir Optimizasyonu":
    st.header("💰 Gelir Optimizasyonu")
    st.markdown("<div class='card'>Gelir maximizasyonu için öneriler.</div>", unsafe_allow_html=True)

    segment_revenue = filtered_df.groupby('MarketSegment')[['LodgingRevenue', 'OtherRevenue']].sum().reset_index()
    fig = px.bar(segment_revenue, x='MarketSegment', y=['LodgingRevenue', 'OtherRevenue'], 
                 title="Pazar Segmentine Göre Gelir", barmode='group', color_discrete_sequence=['#10b981', '#3b82f6'])
    fig.update_layout(xaxis_title="Pazar Segmenti", yaxis_title="Gelir", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Öneriler")
    high_revenue_segments = segment_revenue[segment_revenue['LodgingRevenue'] > segment_revenue['LodgingRevenue'].median()]['MarketSegment'].tolist()
    st.markdown(f"""
    <div class='card'>
        <p><strong>Yüksek Gelir Getiren Segmentler:</strong> {', '.join(high_revenue_segments)}</p>
        <p><strong>Öneri:</strong> Bu segmentlere yönelik pazarlama kampanyaları düzenleyin ve özel teklifler sunun.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class='footer'>
        <p>Otel Müşteri Analiz Paneli - Powered by Streamlit & xAI | Version 2.0.0</p>
    </div>
""", unsafe_allow_html=True)
