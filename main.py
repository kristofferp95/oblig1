import os
import tempfile
import zipfile
import io
import PyPDF2
import matplotlib
matplotlib.use('Agg')
import base64
import cv2
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA

app = Flask(__name__)

# Konfigurasjoner
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'zip'}

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Sjekk om POST-anmodningen har filen delen
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            return redirect(url_for('result', filename=filename))
    return render_template('upload.html')

@app.route('/result/<filename>')
def result(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        if any(file.lower().endswith(('.png', '.jpg', '.jpeg')) for file in file_list):
            fig = process_zip_file(filepath)
        elif any(file.lower().endswith(('.txt', '.pdf')) for file in file_list):
            fig = process_text_zip_file(filepath)
        else:
            return "Unsupported file type in ZIP", 400

    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template('result.html', image_plot=plot_url)



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'zip', 'mp3', 'wav'}

def process_zip_file(filepath, n_clusters=3, size=(64, 64)):
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        folder_name = os.path.splitext(os.path.basename(filepath))[0]
        folder_path = os.path.join(temp_dir, folder_name)

        image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if len(image_files) < n_clusters:
            raise ValueError(f"Antall bilder ({len(image_files)}) er mindre enn antall klynger ({n_clusters}).")

        image_data = [process_image_file(file, size) for file in image_files]

        scaler = StandardScaler()
        images_normalized = scaler.fit_transform(image_data)

        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(images_normalized)

        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        clusters = kmeans.fit_predict(principal_components)

        fig, ax = plt.subplots()
        scatter = ax.scatter(principal_components[:, 0], principal_components[:, 1], c=clusters, cmap='viridis')
        ax.set_title('Bilde Clustering')
        plt.colorbar(scatter, ax=ax)
        return fig  # Returnerer matplotlib-figuren

def process_image_file(filepath, size):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, size)
    img_flattened = img_resized.flatten()
    return img_flattened

def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfFileReader(file)
        text = ""
        for page in range(reader.numPages):
            text += reader.getPage(page).extractText()
        return text

def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def process_text_zip_file(filepath, n_clusters=3, segment_length=100):
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        text_segments = []
        for root, dirs, files in os.walk(temp_dir):
            for filename in files:
                if filename.lower().endswith('.txt'):
                    text = extract_text_from_txt(os.path.join(root, filename))
                elif filename.lower().endswith('.pdf'):
                    text = extract_text_from_pdf(os.path.join(root, filename))
                else:
                    continue
                text_segments.extend([text[i:i+segment_length] for i in range(0, len(text), segment_length)])

        if not text_segments:
            raise ValueError("Ingen tekst funnet i filene.")

        # Vektoriser tekstsegmentene
        vectorizer = CountVectorizer(analyzer='char')
        X = vectorizer.fit_transform(text_segments).toarray()

        # Normaliser vektorene
        scaler = StandardScaler()
        text_vectors_normalized = scaler.fit_transform(X)

        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        clusters = kmeans.fit_predict(text_vectors_normalized)

        # Bruk PCA for dimensjonsreduksjon for visualisering
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(text_vectors_normalized)

        # Visualisering
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(principal_components[:, 0], principal_components[:, 1], c=clusters, cmap='viridis')
        ax.set_title('Visualisering av Tekst Clustering med PCA')
        plt.colorbar(scatter, ax=ax)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.close(fig)
        return fig  # Returnerer matplotlib-figuren



if __name__ == '__main__':
    app.run(debug=True)
