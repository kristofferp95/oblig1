import os, tempfile, zipfile, io, matplotlib, base64, docx, cv2, numpy as pd
matplotlib.use('Agg')
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA

app = Flask(__name__)

# Configuration settings
app.config['UPLOAD_FOLDER'] = 'uploads/'  # Directory for storing uploaded files
app.config['ALLOWED_EXTENSIONS'] = {'zip'}  # Allowed file extensions for upload

def allowed_file(filename):
    # Check if the file has a valid extension
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def upload():
    # Handles file uploads
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # If no file is selected, redirect
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)  # Save the file
            return redirect(url_for('result', filename=filename))
    return render_template('upload.html')  # Render the upload form

@app.route('/result/<filename>')
def result(filename):
    # Displays the results after processing the uploaded file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        # Process images if found
        if any(file.lower().endswith(('.png', '.jpg', '.jpeg')) for file in file_list):
            fig = process_zip_file(filepath)
        # Process text files if found
        if any(file.lower().endswith(('.txt', '.pdf', '.docx')) for file in file_list):
            fig = process_text_zip_file(filepath)

    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return render_template('result.html', image_plot=plot_url)  # Display results

def process_zip_file(filepath, n_clusters=3, size=(64, 64)):
    # Process a zip file containing images
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract the zip file contents
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # Get the base folder name from the zip file
        folder_name = os.path.splitext(os.path.basename(filepath))[0]
        folder_path = os.path.join(temp_dir, folder_name)

        # List all image files in the extracted folder
        image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Check if the number of images is sufficient for clustering
        if len(image_files) < n_clusters:
            raise ValueError(f"Number of images ({len(image_files)}) is less than the number of clusters ({n_clusters}).")

        # Process each image and flatten it
        image_data = [process_image_file(file, size) for file in image_files]

        # Normalize the image data
        scaler = StandardScaler()
        images_normalized = scaler.fit_transform(image_data)

        # Reduce dimensions using PCA
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(images_normalized)

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        clusters = kmeans.fit_predict(principal_components)

        # Create a scatter plot of the clustered images
        fig, ax = plt.subplots()
        scatter = ax.scatter(principal_components[:, 0], principal_components[:, 1], c=clusters, cmap='viridis')
        ax.set_title('Image Clustering')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.colorbar(scatter, ax=ax)
        return fig  # Return the matplotlib figure

def process_image_file(filepath, size):
    # Process an individual image file for analysis
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # Read the image in grayscale
    img_resized = cv2.resize(img, size)  # Resize the image
    img_flattened = img_resized.flatten()  # Flatten the image data
    return img_flattened  # Return the flattened image data

def extract_text_from_txt(file_path):
    # Extract text from a .txt file
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()  # Return the file content

def extract_text_from_docx(file_path):
    # Extract text from a .docx file
    doc = docx.Document(file_path)
    text = [paragraph.text for paragraph in doc.paragraphs]  # Extract text from each paragraph
    return '\n'.join(text)  # Join and return the text

def process_text_zip_file(filepath, n_clusters=3, segment_length=100):
    # Process a zip file containing text documents
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract the zip file contents
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        text_segments = []
        # Traverse through the extracted files and extract text
        for root, dirs, files in os.walk(temp_dir):
            for filename in files:
                # Ignore hidden and system files
                if not filename.startswith('._') and not filename.startswith('.__'):
                    file_path = os.path.join(root, filename)
                    # Process text files based on their extensions
                    if filename.lower().endswith('.txt'):
                        text = extract_text_from_txt(file_path)
                    elif filename.lower().endswith('.docx'):
                        text = extract_text_from_docx(file_path)
                    else:
                        continue
                    # Break the text into segments for processing
                    text_segments.extend([text[i:i+segment_length] for i in range(0, len(text), segment_length)])

        if not text_segments:
            raise ValueError("No text found in files.")

        # Vectorize the text segments
        vectorizer = CountVectorizer(analyzer='char')
        X = vectorizer.fit_transform(text_segments).toarray()

        # Normalize the vectors
        scaler = StandardScaler()
        text_vectors_normalized = scaler.fit_transform(X)

        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        clusters = kmeans.fit_predict(text_vectors_normalized)

        # Reduce dimensions using PCA for visualization
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(text_vectors_normalized)

        # Visualize the clustering of text segments
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(principal_components[:, 0], principal_components[:, 1], c=clusters, cmap='viridis')
        ax.set_title('Text Clustering Visualization with PCA')
        plt.colorbar(scatter, ax=ax)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.close(fig)
        return fig  # Return the matplotlib figure for display


if __name__ == '__main__':
    app.run(debug=True)
