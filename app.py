import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.impute import KNNImputer
from sklearn.metrics import confusion_matrix, roc_curve, auc, silhouette_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import pmdarima as pm
import matplotlib.pyplot as plt
import seaborn as sns
import uuid
import warnings
warnings.filterwarnings('ignore')

# Uygulama konfigürasyonu
st.set_page_config(page_title="Muhteşem Otel Müşteri Analizi", layout="wide", initial_sidebar_state="expanded")
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
    </style>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
""", unsafe_allow_html=True)

# Veri yükleme
@st.cache_data
def load_data(file=None):
    try:
        if file:
            if file.name.endswith('.csv'):
                return pd.read_csv(file)
            elif file.name.endswith('.xlsx'):
                return pd.read_excel(file)
            else:
                raise ValueError("Desteklenmeyen dosya formatı. Lütfen CSV veya XLSX yükleyin.")
        return pd.read_excel("HotelCustomersDataset.xlsx")
    except Exception as e:
        st.error(f"Veri yükleme hatası: {e}")
        return None

# Dosya yükleyici
st.sidebar.header("📤 Veri Yükleme")
uploaded_file = st.sidebar.file_uploader("CSV veya Excel dosyanızı yükleyin", type=["csv", "xlsx"])
if uploaded_file:
    df = load_data(uploaded_file)
    if df is not None:
        st.success("✅ Dosya başarıyla yüklendi!")
else:
    st.info("ℹ️ Öntanımlı veri seti yüklendi.")
    df = load_data()

if df is None:
    st.stop()

# Eksik veri doldurma (Age için KNN Imputer)
imputer = KNNImputer(n_neighbors=5)
df['Age'] = imputer.fit_transform(df[['Age']])

# Sidebar navigasyon
st.sidebar.header("📊 Menü")
menu = st.sidebar.selectbox("Bölüm Seçin:", [
    "CRISP-DM Süreci",
    "Veri Önizleme",
    "Keşifsel Veri Analizi (EDA)",
    "Makine Öğrenmesi Modelleri",
    "Zaman Serisi Analizi",
    "Öneri Sistemi (KNN)",
    "Özel Talepler Analizi",
    "RFM Analizi",
    "Aykırı Değer Analizi"
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
if menu == "CRISP-DM Süreci":
    st.header("📌 CRISP-DM Metodolojisi")
    st.markdown("""
    <div class='card'>
        <h3>1. İş Hedefi</h3>
        <p>Otel yöneticilerinin müşteri davranışlarını anlaması, gelir optimizasyonu, müşteri memnuniyetini artırma ve iptal/no-show oranlarını azaltma.</p>
        <h3>2. Veri Anlama</h3>
        <p>83,590 satırlık veri seti; müşteri profili (yaş, milliyet), rezervasyon alışkanlıkları, gelir ve özel talepler içerir.</p>
        <h3>3. Veri Hazırlığı</h3>
        <p>Eksik verilerin KNN Imputer ile doldurulması, aykırı değerlerin IQR ve Isolation Forest ile yönetimi, kategorik değişkenlerin kodlanması.</p>
        <h3>4. Modelleştirme</h3>
        <p>KNN, ARIMA, KMeans, Lojistik Regresyon, Random Forest ve RFM analizi ile müşteri segmentasyonu ve tahmin.</p>
        <h3>5. Değerlendirme</h3>
        <p>Doğruluk, R², Silhouette skoru, ROC eğrisi ve SHAP analizleriyle model performansı.</p>
        <h3>6. Dağıtım</h3>
        <p>Streamlit ile interaktif, kullanıcı dostu arayüz üzerinden sonuçların sunulması.</p>
    </div>
    """, unsafe_allow_html=True)

# Veri Önizleme
elif menu == "Veri Önizleme":
    st.header("📄 Veri Önizleme")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Veri Seti Örneği")
        st.dataframe(filtered_df.head(10), use_container_width=True)
    with col2:
        st.subheader("Veri Bilgileri")
        st.markdown(f"""
        <div class='card'>
            <p><strong>Satır Sayısı:</strong> {filtered_df.shape[0]}</p>
            <p><strong>Sütun Sayısı:</strong> {filtered_df.shape[1]}</p>
            <p><strong>Eksik Veriler:</strong></p>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(filtered_df.isnull().sum(), use_container_width=True)

    st.subheader("Temel İstatistikler")
    st.dataframe(filtered_df.describe(), use_container_width=True)

# Keşifsel Veri Analizi
elif menu == "Keşifsel Veri Analizi (EDA)":
    st.header("📈 Keşifsel Veri Analizi (EDA)")
    st.markdown("<div class='card'>Bu bölümde veri setinin detaylı görselleştirmeleri ve analizleri yer almaktadır.</div>", unsafe_allow_html=True)

    # İstatistiksel Özet
    st.subheader("İstatistiksel Özet")
    st.dataframe(filtered_df.describe(), use_container_width=True)

    # Kategorik Değişkenler
    st.subheader("Kategorik Değişken Dağılımları")
    col1, col2 = st.columns(2)
    with col1:
        plot_pie(filtered_df, 'Nationality', 'Milliyet Dağılımı (İlk 10)')
    with col2:
        plot_pie(filtered_df, 'MarketSegment', 'Pazar Segmenti Dağılımı')

    # Sayısal Değişkenler
    st.subheader("Sayısal Değişken Analizi")
    num_cols = filtered_df.select_dtypes(include=np.number).columns.tolist()
    selected_col = st.selectbox("Bir değişken seçin", num_cols)
    col1, col2 = st.columns(2)
    with col1:
        plot_histogram(filtered_df, selected_col, f"{selected_col} Dağılımı")
    with col2:
        plot_violin(filtered_df, selected_col, f"{selected_col} Violin Grafiği")

    # Milliyet Bazında Gelir Analizi
    st.subheader("Milliyet Bazında Gelir Analizi")
    nationality_revenue = filtered_df.groupby('Nationality')[['LodgingRevenue', 'OtherRevenue']].sum().reset_index()
    fig = px.bar(nationality_revenue, x='Nationality', y=['LodgingRevenue', 'OtherRevenue'], 
                 title="Milliyet Bazında Toplam Gelir", barmode='group', color_discrete_sequence=['#10b981', '#3b82f6'])
    fig.update_layout(xaxis_title="Milliyet", yaxis_title="Gelir", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # Korelasyon Matrisi
    st.subheader("Korelasyon Analizi")
    plot_correlation_matrix(filtered_df)

    # Pair Plot
    st.subheader("İkili Değişken Analizi")
    pair_cols = st.multiselect("Değişkenler seçin", num_cols, default=['Age', 'LodgingRevenue', 'OtherRevenue'])
    if pair_cols:
        fig = px.scatter_matrix(filtered_df, dimensions=pair_cols, title="Pair Plot", color='MarketSegment', 
                                color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=700, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

# Makine Öğrenmesi Modelleri
elif menu == "Makine Öğrenmesi Modelleri":
    st.header("🤖 Makine Öğrenmesi Modelleri")
    st.markdown("<div class='card'>Bu bölümde Lojistik Regresyon, Random Forest ve KMeans modelleri ile hiperparametre optimizasyonu.</div>", unsafe_allow_html=True)

    model = st.selectbox("Model Seçin:", ["Lojistik Regresyon", "Random Forest", "KMeans"])
    df_model = filtered_df.dropna()

    if model == "Lojistik Regresyon":
        st.subheader("Lojistik Regresyon - İptal Tahmini")
        X = df_model[['Age', 'AverageLeadTime', 'LodgingRevenue', 'OtherRevenue', 'RoomNights']].fillna(0)
        y = df_model['BookingsCanceled'].apply(lambda x: 1 if x > 0 else 0)
        X_scaled = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        param_grid = {'C': [0.01, 0.1, 1, 10], 'solver': ['lbfgs', 'liblinear'], 'max_iter': [1000]}
        lr = GridSearchCV(LogisticRegression(), param_grid, cv=5, n_jobs=-1)
        lr.fit(X_train, y_train)

        st.markdown(f"""
        <div class='card'>
            <p><strong>En iyi parametreler:</strong> {lr.best_params_}</p>
            <p><strong>Doğruluk Skoru:</strong> {lr.score(X_test, y_test):.2f}</p>
        </div>
        """, unsafe_allow_html=True)

        # Confusion Matrix
        cm = confusion_matrix(y_test, lr.predict(X_test))
        fig = px.imshow(cm, text_auto=True, title="Confusion Matrix", color_continuous_scale='Blues')
        fig.update_layout(xaxis_title="Tahmin", yaxis_title="Gerçek")
        st.plotly_chart(fig, use_container_width=True)

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, lr.predict_proba(X_test)[:, 1])
        roc_auc = auc(fpr, tpr)
        fig = px.line(x=fpr, y=tpr, title=f"ROC Eğrisi (AUC = {roc_auc:.2f})")
        fig.add_scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Rastgele Tahmin')
        fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    elif model == "Random Forest":
        st.subheader("Random Forest Regresyon - Gelir Tahmini")
        X = df_model[['Age', 'AverageLeadTime', 'RoomNights', 'PersonsNights', 'DaysSinceLastStay']].fillna(0)
        y = df_model['LodgingRevenue']
        X_scaled = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]}
        rf = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, n_jobs=-1)
        rf.fit(X_train, y_train)

        st.markdown(f"""
        <div class='card'>
            <p><strong>En iyi parametreler:</strong> {rf.best_params_}</p>
            <p><strong>R² Skoru:</strong> {rf.score(X_test, y_test):.2f}</p>
        </div>
        """, unsafe_allow_html=True)

        # Özellik Önem Sıralaması
        importance = pd.DataFrame({'Feature': X.columns, 'Importance': rf.best_estimator_.feature_importances_})
        fig = px.bar(importance, x='Feature', y='Importance', title="Özellik Önem Sıralaması", color='Importance', 
                     color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)

        # SHAP Analizi
        try:
            import shap
            explainer = shap.TreeExplainer(rf.best_estimator_)
            shap_values = explainer.shap_values(X_test)
            st.subheader("SHAP Değerleri")
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test, feature_names=X.columns, show=False)
            st.pyplot(plt)
        except:
            st.warning("SHAP analizi için 'shap' kütüphanesi gereklidir. Lütfen yükleyin: `pip install shap`")

    elif model == "KMeans":
        st.subheader("KMeans Kümeleme")
        X = df_model[['Age', 'LodgingRevenue', 'OtherRevenue', 'RoomNights']].dropna()
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

        # Küme Özetleri
        st.subheader("Küme Özetleri")
        cluster_summary = df_model.groupby('Cluster').agg({
            'Age': ['mean', 'count'],
            'LodgingRevenue': 'mean',
            'OtherRevenue': 'mean',
            'RoomNights': 'mean'
        }).round(2)
        st.dataframe(cluster_summary, use_container_width=True)

# Zaman Serisi Analizi
elif menu == "Zaman Serisi Analizi":
    st.header("⏳ Zaman Serisi Analizi")
    st.markdown("<div class='card'>Bu bölümde LodgingRevenue için ARIMA ve mevsimsellik analizi yapılmaktadır.</div>", unsafe_allow_html=True)

    ts = filtered_df[['DaysSinceCreation', 'LodgingRevenue']].dropna()
    ts = ts.groupby('DaysSinceCreation').sum().sort_index()
    ts_series = ts['LodgingRevenue']

    # Mevsimsellik Analizi
    st.subheader("Mevsimsellik ve Trend Analizi")
    try:
        decomposition = seasonal_decompose(ts_series, model='additive', period=30)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts_series.index, y=decomposition.observed, mode='lines', name='Gerçek', line=dict(color='#10b981')))
        fig.add_trace(go.Scatter(x=ts_series.index, y=decomposition.trend, mode='lines', name='Trend', line=dict(color='#3b82f6')))
        fig.add_trace(go.Scatter(x=ts_series.index, y=decomposition.seasonal, mode='lines', name='Mevsimsellik', line=dict(color='#f43f5e')))
        fig.update_layout(title="Mevsimsellik Ayrıştırma", xaxis_title="Gün", yaxis_title="Gelir", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Mevsimsellik analizi başarısız: {e}")

    # ARIMA
    st.subheader("ARIMA Tahmini")
    try:
        model = pm.auto_arima(ts_series, seasonal=True, m=30, stepwise=True, trace=False)
        forecast, conf_int = model.predict(n_periods=20, return_conf_int=True)
        forecast_index = range(ts_series.index[-1] + 1, ts_series.index[-1] + 21)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts_series.index, y=ts_series, mode='lines', name='Gerçek', line=dict(color='#10b981')))
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

# Öneri Sistemi (KNN)
elif menu == "Öneri Sistemi (KNN)":
    st.header("🤝 KNN Öneri Sistemi")
    st.markdown("<div class='card'>Benzer müşteri profillerini bulmak için KNN algoritması kullanılmaktadır.</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Yaş", 18, 100, 35)
        days = st.slider("DaysSinceCreation", 0, 1095, 100)
    with col2:
        lead = st.slider("AverageLeadTime", 0, 300, 50)
        revenue = st.slider("LodgingRevenue", 0, 5000, 500)

    df_knn = filtered_df[['Age', 'DaysSinceCreation', 'AverageLeadTime', 'LodgingRevenue']].dropna()
    X_scaled = StandardScaler().fit_transform(df_knn)
    knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
    knn.fit(X_scaled)
    distances, indices = knn.kneighbors([[age, days, lead, revenue]])

    st.subheader("Benzer Müşteriler")
    st.dataframe(df_knn.iloc[indices[0]], use_container_width=True)

    # Görselleştirme
    df_knn['Distance'] = np.nan
    df_knn.iloc[indices[0], df_knn.columns.get_loc('Distance')] = distances[0]
    fig = px.scatter(df_knn, x='Age', y='LodgingRevenue', size='AverageLeadTime', color='Distance', 
                     title="Benzer Müşteriler (KNN)", color_continuous_scale='Blues', hover_data=['DaysSinceCreation'])
    fig.add_scatter(x=[age], y=[revenue], mode='markers', marker=dict(size=20, color='red'), name='Seçilen Profil')
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# Özel Talepler Analizi
elif menu == "Özel Talepler Analizi":
    st.header("🛏️ Özel Talepler Analizi")
    st.markdown("<div class='card'>Müşterilerin özel oda talepleri milliyet, yaş ve segment bazında analiz edilmektedir.</div>", unsafe_allow_html=True)

    sr_cols = [col for col in filtered_df.columns if col.startswith('SR')]
    sr_counts = filtered_df[sr_cols].sum().sort_values(ascending=False)

    fig = px.bar(x=sr_counts.index, y=sr_counts.values, title="Özel Talep Sıklıkları", 
                 color=sr_counts.values, color_continuous_scale='Viridis', text_auto=True)
    fig.update_layout(xaxis_title="Talep", yaxis_title="Sayı", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # Milliyet Bazında Talepler
    st.subheader("Milliyet Bazında Talepler")
    selected_nationality = st.selectbox("Milliyet Seçin", filtered_df['Nationality'].unique())
    nationality_sr = filtered_df[filtered_df['Nationality'] == selected_nationality][sr_cols].sum()
    
    fig = px.bar(x=sr_cols, y=nationality_sr.values, title=f"{selected_nationality} için Özel Talepler",
                 color=nationality_sr.values, color_continuous_scale='Blues', text_auto=True)
    fig.update_layout(xaxis_title="Talep", yaxis_title="Sayı", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # Segment Bazında Talepler
    st.subheader("Pazar Segmenti Bazında Talepler")
    segment_sr = filtered_df.groupby('MarketSegment')[sr_cols].sum().reset_index()
    fig = px.bar(segment_sr, x='MarketSegment', y=sr_cols, title="Segment Bazında Özel Talepler", 
                 barmode='stack', color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_layout(xaxis_title="Pazar Segmenti", yaxis_title="Talep Sayısı", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# RFM Analizi
elif menu == "RFM Analizi":
    st.header("📊 RFM Analizi")
    st.markdown("<div class='card'>Recency, Frequency ve Monetary değerlerine göre müşteri segmentasyonu.</div>", unsafe_allow_html=True)

    rfm = filtered_df[['ID', 'DaysSinceLastStay', 'BookingsCheckedIn', 'LodgingRevenue', 'OtherRevenue']].copy()
    rfm['Monetary'] = rfm['LodgingRevenue'] + rfm['OtherRevenue']
    rfm = rfm.rename(columns={'DaysSinceLastStay': 'Recency', 'BookingsCheckedIn': 'Frequency'})
    rfm = rfm[['ID', 'Recency', 'Frequency', 'Monetary']]

    # RFM Skorları
    rfm['R_Score'] = pd.qcut(rfm['Recency'].rank(method='first'), 4, labels=[4, 3, 2, 1])
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), 4, labels=[1, 2, 3, 4])
    rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

    st.subheader("RFM Özeti")
    st.dataframe(rfm.head(), use_container_width=True)

    # RFM Segment Görselleştirme
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

# Aykırı Değer Analizi
elif menu == "Aykırı Değer Analizi":
    st.header("🔍 Aykırı Değer Analizi")
    st.markdown("<div class='card'>IQR, Z-Skor ve Isolation Forest ile aykırı değer tespiti.</div>", unsafe_allow_html=True)

    # IQR Yöntemi
    st.subheader("IQR Yöntemi")
    Q1 = filtered_df['LodgingRevenue'].quantile(0.25)
    Q3 = filtered_df['LodgingRevenue'].quantile(0.75)
    IQR = Q3 - Q1
    outliers_iqr = filtered_df[(filtered_df['LodgingRevenue'] < Q1 - 1.5 * IQR) | (filtered_df['LodgingRevenue'] > Q3 + 1.5 * IQR)]
    st.markdown(f"""
    <div class='card'>
        <p><strong>Aykırı Değer Sayısı (LodgingRevenue):</strong> {len(outliers_iqr)}</p>
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(outliers_iqr[['ID', 'LodgingRevenue', 'OtherRevenue', 'Nationality']], use_container_width=True)

    # Isolation Forest
    st.subheader("Isolation Forest")
    iso = IsolationForest(contamination=0.05, random_state=42)
    X = filtered_df[['LodgingRevenue', 'OtherRevenue', 'Age']].fillna(0)
    outliers_iso = iso.fit_predict(X)
    filtered_df['Outlier'] = outliers_iso
    outliers_df = filtered_df[filtered_df['Outlier'] == -1]
    st.markdown(f"""
    <div class='card'>
        <p><strong>Aykırı Değer Sayısı (Isolation Forest):</strong> {len(outliers_df)}</p>
    </div>
    """, unsafe_allow_html=True)
    fig = px.scatter_3d(filtered_df, x='Age', y='LodgingRevenue', z='OtherRevenue', color='Outlier', 
                        title="Aykırı Değerler (Isolation Forest)", color_continuous_scale=['#10b981', '#f43f5e'])
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("""
    <div class='footer'>
        <p>Muhteşem Otel Müşteri Analizi - Powered by Streamlit & xAI | Version 1.0.0</p>
    </div>
""", unsafe_allow_html=True)
