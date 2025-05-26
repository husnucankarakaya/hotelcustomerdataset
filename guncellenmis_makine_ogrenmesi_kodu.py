
# Makine Öğrenmesi Modelleri
elif section == "Makine Öğrenmesi Modelleri":
    st.header("🤖 Makine Öğrenmesi Modelleri")
    st.markdown("<div class='card'>Lojistik Regresyon, Random Forest ve KMeans modelleri güncellenmiştir.</div>", unsafe_allow_html=True)

    model_option = st.selectbox("Model Seçin", ["Lojistik Regresyon", "Random Forest", "K-Means Kümeleme"])
    df_model = filtered_df.dropna()
    df_model = df_model.sample(n=min(10000, len(df_model)), random_state=42)

    if model_option == "Lojistik Regresyon":
        st.subheader("Lojistik Regresyon - İptal Tahmini")
        X = df_model[['Age', 'AverageLeadTime', 'LodgingRevenue', 'OtherRevenue', 'RoomNights']]
        y = df_model['BookingsCanceled'].apply(lambda x: 1 if x > 0 else 0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.markdown(f"<pre>{classification_report(y_test, y_pred)}</pre>", unsafe_allow_html=True)
        cm = confusion_matrix(y_test, y_pred)
        st.plotly_chart(px.imshow(cm, text_auto=True, title="Confusion Matrix", color_continuous_scale='Blues'), use_container_width=True)

    elif model_option == "Random Forest":
        st.subheader("Random Forest - Gelir Tahmini")
        X = df_model[['Age', 'AverageLeadTime', 'RoomNights', 'PersonsNights']]
        y = df_model['LodgingRevenue']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.markdown(f"<p><strong>MSE:</strong> {mean_squared_error(y_test, y_pred):.2f}</p>", unsafe_allow_html=True)
        st.markdown(f"<p><strong>R²:</strong> {r2_score(y_test, y_pred):.2f}</p>", unsafe_allow_html=True)

    elif model_option == "K-Means Kümeleme":
        st.subheader("K-Means Kümeleme")
        X = df_model[['Age', 'LodgingRevenue', 'OtherRevenue']]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        n_clusters = st.slider("Küme Sayısı", 2, 10, 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, clusters)
        df_model['Cluster'] = clusters

        st.markdown(f"<p><strong>Silhouette Skoru:</strong> {score:.2f}</p>", unsafe_allow_html=True)
        fig = px.scatter_3d(df_model, x='Age', y='LodgingRevenue', z='OtherRevenue', color='Cluster',
                            title="K-Means Kümeleme (3D)", color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Küme Özetleri")
        st.dataframe(df_model.groupby('Cluster').agg({
            'Age': ['mean', 'count'],
            'LodgingRevenue': 'mean',
            'OtherRevenue': 'mean'
        }).round(2), use_container_width=True)
