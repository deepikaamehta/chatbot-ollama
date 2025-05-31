from flask import Flask, request, render_template
import os
import easyocr
import numpy as np
from pdf2image import convert_from_path
import uuid
import requests
import json

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
reader = easyocr.Reader(['en'])

# Function to split text into chunks
def split_text_into_chunks(text, chunk_size=500, overlap=50):
    """
    Split text into chunks for embedding.
    """
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap  # overlap for context

    return chunks

# Function to generate embeddings via Ollama
def generate_ollama_embeddings(chunks, model="nomic-embed-text"):
    """
    Generate embeddings using Ollama's local API.
    """
    embedded_chunks = []

    for chunk in chunks:
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={
                "model": model,
                "prompt": chunk
            }
        )

        if response.status_code == 200:
            data = response.json()
            embedded_chunks.append({
                "file_name": str(uuid.uuid4()) + ".txt",
                "text": chunk,
                "embedding": data["embedding"]
            })
        else:
            print(f"Embedding failed for chunk: {chunk[:30]}...")

    return embedded_chunks

@app.route('/', methods=['GET', 'POST'])
def index():
    extracted_text = ""
    chunks = []
    embedded_data = []

    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            if filepath.lower().endswith('.pdf'):
                try:
                    images = convert_from_path(filepath)
                    for img in images:
                        result = reader.readtext(np.array(img), detail=0)
                        extracted_text += "\n".join(result) + "\n"
                except Exception as e:
                    extracted_text = f"Error reading PDF: {str(e)}"
            else:
                result = reader.readtext(filepath, detail=0)
                extracted_text = "\n".join(result)

            # 🔃 Create chunks
            chunks = split_text_into_chunks(extracted_text)

            # 🔁 Generate embeddings using Ollama
            embedded_data = generate_ollama_embeddings(chunks)

            # 💾 Optional: Save to file
            with open("embedded_chunks.json", "w") as f:
                json.dump(embedded_data, f, indent=2)

    return render_template('upload.html', text=extracted_text, chunks=chunks)

if __name__ == '__main__':
    app.run(debug=True)
