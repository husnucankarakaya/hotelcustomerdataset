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
    .main { 
        background-color: #f8fafc;
        color: #1e293b;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button { 
        background-color: #10b981;
        color: white; 
        border-radius: 8px; 
        padding: 12px; 
        font-weight: bold;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #059669;
    }
    .stSelectbox, .stMultiselect, .stSlider { 
        background-color: #ffffff; 
        border-radius: 8px; 
        padding: 8px; 
        border: 1px solid #e2e8f0;
        color: #1e293b;
    }
    .stSidebar { 
        background-color: #e2e8f0;
        color: #1e293b;
    }
    h1, h2, h3 { 
        color: #1e293b; 
        font-weight: 700;
    }
    .stMarkdown { 
        font-family: 'Inter', sans-serif; 
        font-size: 16px; 
        color: #1e293b;
        line-height: 1.6;
    }
    .footer { 
        text-align: center; 
        color: #475569;
        margin-top: 50px; 
        font-size: 14px;
    }
    .card { 
        background-color: #ffffff;
        border-radius: 8px; 
        padding: 20px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: #1e293b;
    }
    .stAlert { 
        border-radius: 8px;
        background-color: #fef3c7;
        color: #1e293b;
    }
    .metric-card { 
        background-color: #e0f2fe;
        border-radius: 8px; 
        padding: 10px; 
        text-align: center;
        color: #1e293b;
        font-weight: 600;
    }
    .stDataFrame {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
    }
    </style>
    <link href="[invalid url, do not cite] rel="stylesheet">
""", unsafe_allow_html=True)

# Session state başlatma
if 'filtered_df' not in st.session_state:
    st.session_state.filtered_df = None
if 'filters_applied' not in st.session_state:
    st.session_state.filters_applied = False

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
        st.error(f"Veri yükleme hatası: {e}. Lütfen geçerli bir XLSX dosyası yükleyin.")
        return None

# Eksik veri doldurma
@st.cache_data
def preprocess_data(df):
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    imputer = KNNImputer(n_neighbors=5)
    df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
    return df

# Dosya yükleyici
st.sidebar.header("📤 Veri Yükleme")
uploaded_file = st.sidebar.file_uploader("📂 .xlsx dosyanızı yükleyin", type=["xlsx"])
if uploaded_file:
    df = load_data(uploaded_file)
    if df is not None:
        df = preprocess_data(df)
        st.session_state.filtered_df = df
        st.success("✅ Dosya başarıyla yüklendi!")
else:
    st.warning("ℹ️ Lütfen analiz için bir Excel dosyası yükleyin.")
    df = load_data()
    if df is not None:
        df = preprocess_data(df)
        st.session_state.filtered_df = df

if df is None:
    st.stop()

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

# Filtreleme (Form ile)
st.sidebar.header("🔍 Filtreleme")
with st.sidebar.form(key="filter_form"):
    top_nationalities = df['Nationality'].value_counts().head(10).index.tolist()
    nationality = st.multiselect("Milliyet Seçin", top_nationalities, default=['PRT', 'FRA', 'DEU'])
    market_segment = st.selectbox("Pazar Segmenti", df['MarketSegment'].unique())
    distribution_channel = st.selectbox("Dağıtım Kanalı", df['DistributionChannel'].unique())
    age_range = st.slider("Yaş Aralığı", 0, 100, (0, 100), step=5)
    submit_button = st.form_submit_button("Filtreleri Uygula")

if submit_button:
    st.session_state.filters_applied = True
    filtered_df = df[
        (df['Nationality'].isin(nationality)) &
        (df['Age'].between(age_range[0], age_range[1], inclusive='both')) &
        (df['MarketSegment'] == market_segment) &
        (df['DistributionChannel'] == distribution_channel)
    ]
    st.session_state.filtered_df = filtered_df
elif st.session_state.filters_applied:
    filtered_df = st.session_state.filtered_df
else:
    filtered_df = df
    st.session_state.filtered_df = df

if st.sidebar.button("Filtreleri Sıfırla"):
    st.session_state.filters_applied = False
    st.session_state.filtered_df = df
    filtered_df = df

# Görselleştirme fonksiyonları
@st.cache_data
def plot_histogram(df, column, title):
    fig = px.histogram(df, x=column, title=title, color_discrete_sequence=['#10b981'], nbins=50, marginal="box")
    fig.update_layout(xaxis_title=column, yaxis_title="Frekans", template="plotly_white", font=dict(color='#1e293b'))
    return fig

@st.cache_data
def plot_violin(df, column, title):
    fig = px.violin(df, y=column, title=title, color_discrete_sequence=['#3b82f6'], box=True, points="outliers")
    fig.update_layout(yaxis_title=column, template="plotly_white", font=dict(color='#1e293b'))
    return fig

@st.cache_data
def plot_pie(df, column, title):
    counts = df[column].value_counts().head(10)
    fig = px.pie(values=counts.values, names=counts.index, title=title, color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(template="plotly_white", font=dict(color='#1e293b'))
    return fig

@st.cache_data
def plot_correlation_matrix(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='Viridis', text=corr.values.round(2), texttemplate="%{text}"))
    fig.update_layout(title="Korelasyon Matrisi", height=700, template="plotly_white", font=dict(color='#1e293b'))
    return fig

@st.cache_data
def plot_elbow_method(X_scaled):
    inertias = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    fig = px.line(x=range(1, 11), y=inertias, title="Elbow Yöntemi ile Optimal Küme Sayısı", markers=True)
    fig.update_layout(xaxis_title="Küme Sayısı", yaxis_title="Inertia", template="plotly_white", font=dict(color='#1e293b'))
    return fig

@st.cache_data
def plot_silhouette_score(X_scaled, max_clusters=10):
    scores = []
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores.append(score)
    fig = px.line(x=range(2, max_clusters + 1), y=scores, title="Silhouette Skoru ile Küme Kalitesi", markers=True)
    fig.update_layout(xaxis_title="Küme Sayısı", yaxis_title="Silhouette Skoru", template="plotly_white", font=dict(color='#1e293b'))
    return fig

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
        st.plotly_chart(plot_pie(filtered_df, 'Nationality', 'Milliyet Dağılımı (İlk 10)'), use_container_width=True)
    with col2:
        st.plotly_chart(plot_pie(filtered_df, 'MarketSegment', 'Pazar Segmenti Dağılımı'), use_container_width=True)

    st.subheader("Sayısal Değişken Analizi")
    num_cols = filtered_df.select_dtypes(include=np.number).columns.tolist()
    selected_col = st.selectbox("Bir değişken seçin", num_cols)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_histogram(filtered_df, selected_col, f"{selected_col} Dağılımı"), use_container_width=True)
    with col2:
        st.plotly_chart(plot_violin(filtered_df, selected_col, f"{selected_col} Violin Grafiği"), use_container_width=True)

    st.subheader("Milliyet Bazında Gelir Analizi")
    @st.cache_data
    def plot_nationality_revenue(df):
        nationality_revenue = df.groupby('Nationality')[['LodgingRevenue', 'OtherRevenue']].sum().reset_index()
        fig = px.bar(nationality_revenue, x='Nationality', y=['LodgingRevenue', 'OtherRevenue'], 
                     title="Milliyet Bazında Toplam Gelir", barmode='group', color_discrete_sequence=['#10b981', '#3b82f6'])
        fig.update_layout(xaxis_title="Milliyet", yaxis_title="Gelir", template="plotly_white", font=dict(color='#1e293b'))
        return fig
    st.plotly_chart(plot_nationality_revenue(filtered_df), use_container_width=True)

    st.subheader("Korelasyon Analizi")
    st.plotly_chart(plot_correlation_matrix(filtered_df), use_container_width=True)

    st.subheader("İkili Değişken Analizi")
    pair_cols = st.multiselect("Değişkenler seçin", num_cols, default=['Age', 'LodgingRevenue', 'OtherRevenue'])
    if pair_cols:
        @st.cache_data
        def plot_pair_plot(df, pair_cols):
            fig = px.scatter_matrix(df, dimensions=pair_cols, title="Pair Plot", color='MarketSegment', 
                                    color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(height=700, template="plotly_white", font=dict(color='#1e293b'))
            return fig
        st.plotly_chart(plot_pair_plot(filtered_df, pair_cols), use_container_width=True)

    st.subheader("Sankey Diyagramı - Rezervasyon Akışı")
    @st.cache_data
    def plot_sankey(df):
        sankey_data = df.groupby(['DistributionChannel', 'MarketSegment']).size().reset_index(name='Count')
        top_channels = sankey_data['DistributionChannel'].value_counts().head(5).index
        top_segments = sankey_data['MarketSegment'].value_counts().head(5).index
        sankey_data = sankey_data[
            (sankey_data['DistributionChannel'].isin(top_channels)) & 
            (sankey_data['MarketSegment'].isin(top_segments))
        ]
        labels = list(sankey_data['DistributionChannel'].unique()) + list(sankey_data['MarketSegment'].unique())
        source = [labels.index(x) for x in sankey_data['DistributionChannel']]
        target = [labels.index(x) + len(sankey_data['DistributionChannel'].unique()) for x in sankey_data['MarketSegment']]
        fig = go.Figure(data=[go.Sankey(
            node=dict(label=labels),
            link=dict(source=source, target=target, value=sankey_data['Count'])
        )])
        fig.update_layout(title="Dağıtım Kanalı ve Pazar Segmenti Akışı", template="plotly_white", font=dict(color='#1e293b'))
        return fig
    st.plotly_chart(plot_sankey(filtered_df), use_container_width=True)

# Makine Öğrenmesi Modelleri
elif section == "Makine Öğrenmesi Modelleri":
    st.header("🤖 Makine Öğrenmesi Modelleri")
    st.markdown("<div class='card'>Lojistik Regresyon, Random Forest, XGBoost ve KMeans modelleri.</div>", unsafe_allow_html=True)

    model_option = st.selectbox("Model Seçin", ["Lojistik Regresyon", "Random Forest", "XGBoost", "K-Means Kümeleme"])
    df_model = filtered_df

    if model_option == "Lojistik Regresyon":
        st.subheader("Lojistik Regresyon - İptal Tahmini")
        @st.cache_resource
        def train_logistic_regression(X, y):
            model = LogisticRegression()
            model.fit(X, y)
            return model
        X = df_model[['Age', 'AverageLeadTime', 'RoomNights', 'PersonsNights']].fillna(0)
        y = df_model['BookingsCanceled'].apply(lambda x: 1 if x > 0 else 0)
        X_scaled = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        model = train_logistic_regression(X_train, y_train)
        y_pred = model.predict(X_test)

        st.markdown(f"""
        <div class='card'>
            <h4>Model Değerlendirme</h4>
            <pre>{classification_report(y_test, y_pred)}</pre>
        </div>
        """, unsafe_allow_html=True)

        @st.cache_data
        def plot_confusion_matrix(y_test, y_pred):
            cm = confusion_matrix(y_test, y_pred)
            fig = px.imshow(cm, text_auto=True, title="Confusion Matrix", color_continuous_scale='Blues')
            fig.update_layout(xaxis_title="Tahmin", yaxis_title="Gerçek", template="plotly_white", font=dict(color='#1e293b'))
            return fig
        st.plotly_chart(plot_confusion_matrix(y_test, y_pred), use_container_width=True)

        @st.cache_data
        def plot_roc_curve(y_test, y_pred_proba):
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            fig = px.line(x=fpr, y=tpr, title=f"ROC Eğrisi (AUC = {roc_auc:.2f})")
            fig.add_scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Rastgele Tahmin')
            fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", template="plotly_white", font=dict(color='#1e293b'))
            return fig
        st.plotly_chart(plot_roc_curve(y_test, model.predict_proba(X_test)), use_container_width=True)

    elif model_option == "Random Forest":
        st.subheader("Random Forest - Gelir Tahmini")
        @st.cache_resource
        def train_random_forest(X, y):
            model = RandomForestRegressor(random_state=42)
            model.fit(X, y)
            return model
        X = df_model[['Age', 'AverageLeadTime', 'RoomNights', 'PersonsNights']].fillna(0)
        y = df_model['LodgingRevenue']
        X_scaled = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        model = train_random_forest(X_train, y_train)
        y_pred = model.predict(X_test)

        st.markdown(f"""
        <div class='card'>
            <p><strong>MSE:</strong> {mean_squared_error(y_test, y_pred):.2f}</p>
            <p><strong>R² Skoru:</strong> {r2_score(y_test, y_pred):.2f}</p>
        </div>
        """, unsafe_allow_html=True)

        @st.cache_data
        def plot_feature_importance(X, model):
            importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
            fig = px.bar(importance, x='Feature', y='Importance', title="Özellik Önem Sıralaması", color='Importance', 
                         color_continuous_scale='Viridis')
            fig.update_layout(template="plotly_white", font=dict(color='#1e293b'))
            return fig
        st.plotly_chart(plot_feature_importance(pd.DataFrame(X, columns=['Age', 'AverageLeadTime', 'RoomNights', 'PersonsNights']), model), use_container_width=True)

        if SHAP_AVAILABLE and st.checkbox("SHAP Analizi Göster"):
            st.subheader("SHAP Analizi")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test, feature_names=['Age', 'AverageLeadTime', 'RoomNights', 'PersonsNights'], show=False)
            st.pyplot(plt)

    elif model_option == "XGBoost":
        st.subheader("XGBoost - Gelir Tahmini")
        @st.cache_resource
        def train_xgboost(X, y):
            model = xgb.XGBRegressor(random_state=42)
            model.fit(X, y)
            return model
        X = df_model[['Age', 'AverageLeadTime', 'RoomNights', 'PersonsNights']].fillna(0)
        y = df_model['LodgingRevenue']
        X_scaled = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        model = train_xgboost(X_train, y_train)
        y_pred = model.predict(X_test)

        st.markdown(f"""
        <div class='card'>
            <p><strong>MSE:</strong> {mean_squared_error(y_test, y_pred):.2f}</p>
            <p><strong>R² Skoru:</strong> {r2_score(y_test, y_pred):.2f}</p>
        </div>
        """, unsafe_allow_html=True)

        st.plotly_chart(plot_feature_importance(pd.DataFrame(X, columns=['Age', 'AverageLeadTime', 'RoomNights', 'PersonsNights']), model), use_container_width=True)

    elif model_option == "K-Means Kümeleme":
        st.subheader("K-Means Kümeleme")
        X = df_model[['Age', 'LodgingRevenue', 'OtherRevenue']].dropna().head(10000)
        X_scaled = StandardScaler().fit_transform(X)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_elbow_method(X_scaled), use_container_width=True)
        with col2:
            st.plotly_chart(plot_silhouette_score(X_scaled), use_container_width=True)

        n_clusters = st.slider("Küme Sayısı", 2, 10, 3)
        @st.cache_resource
        def train_kmeans(X_scaled, n_clusters):
