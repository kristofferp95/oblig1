def process_text_file(filepath):
    # Leser og deler teksten i segmenter
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read()
    segments = [text[i:i+100] for i in range(0, len(text), 100)]  # Segmentlengde kan justeres

    # Vektorisering
    vectorizer = CountVectorizer(analyzer='char')
    text_vectors = vectorizer.fit_transform(segments).toarray()

    # Normalisering
    text_normalized = StandardScaler().fit_transform(text_vectors)

    # K-means clustering
    kmeans = KMeans(n_clusters=4)
    clusters = kmeans.fit_predict(text_normalized)

    # PCA for visualisering
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(text_normalized)

    # Plotting
    fig, ax = plt.subplots()
    ax.scatter(principal_components[:, 0], principal_components[:, 1], c=clusters, cmap='viridis')
    ax.set_title('Tekst Clustering')
    return fig


def process_audio_file(filepath):
    # Ekstrahering av MFCCs
    audio, sample_rate = librosa.load(filepath, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    mfccs_processed = np.mean(mfccs.T, axis=0)

    # Normalisering
    mfccs_normalized = StandardScaler().fit_transform([mfccs_processed])

    # K-means clustering
    kmeans = KMeans(n_clusters=3)
    clusters = kmeans.fit_predict(mfccs_normalized)

    # PCA for visualisering
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(mfccs_normalized)

    # Plotting
    fig, ax = plt.subplots()
    ax.scatter(principal_components[:, 0], principal_components[:, 1], c=clusters, cmap='viridis')
    ax.set_title('Lyd Clustering')
    return fig
