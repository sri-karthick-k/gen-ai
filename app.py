from flask import Flask, request, render_template
import pandas as pd
import docx
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import faiss
import numpy as np
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from textblob import TextBlob
import os

app = Flask(__name__)

# Ensure NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load Sentence Transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Initialize FAISS index
embedding_dim = 384  # Dimension should match the embeddings from the model
faiss_index = faiss.IndexFlatL2(embedding_dim)

def load_text_file(file):
    return file.read().decode('utf-8')

def load_pdf_file(file):
    text = ''
    reader = PdfReader(file)
    for page in reader.pages:
        text += page.extract_text()
    return text

def load_word_file(file):
    doc = docx.Document(file)
    return '\n'.join([paragraph.text for paragraph in doc.paragraphs])

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

def split_into_chunks(text, chunk_size):
    sentences = sent_tokenize(text)
    chunks = [' '.join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]
    return chunks

def embed_text(chunks):
    embeddings = model.encode(chunks)
    return embeddings

def process_document(file, file_type, chunk_size):
    if file_type == 'text':
        text = load_text_file(file)
    elif file_type == 'pdf':
        text = load_pdf_file(file)
    elif file_type == 'word':
        text = load_word_file(file)
    else:
        raise ValueError("Unsupported file type")

    clean_text = preprocess_text(text)
    chunks = split_into_chunks(clean_text, chunk_size)
    embeddings = embed_text(chunks)
    return embeddings, chunks

def save_embeddings(embeddings):
    global faiss_index
    faiss_index.add(np.array(embeddings).astype('float32'))


def summarize_text(text, language='english', sentences_count=2):
    parser = PlaintextParser.from_string(text, Tokenizer(language))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences_count)
    return ' '.join([str(sentence) for sentence in summary])

def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    return sentiment_score


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    file_type = request.form['file_type']
    chunk_size = int(request.form['chunk_size'])
    
    embeddings, chunks = process_document(file, file_type, chunk_size)
    save_embeddings(embeddings)
    
    summaries = []
    sentiment_scores = []
    for chunk in chunks:
        summary = summarize_text(chunk)
        sentiment_score = analyze_sentiment(summary)
        summaries.append(summary)
        sentiment_scores.append(sentiment_score)
    
    return render_template('result.html', chunks=chunks, summaries=summaries, sentiment_scores=sentiment_scores)

@app.route('/view_database')
def view_database():
    global faiss_index
    n = faiss_index.ntotal  # Get the total number of vectors in the index
    embeddings = []
    chunks = []

    # Retrieve embeddings and chunks from the index
    for i in range(n):
        embedding = faiss_index.reconstruct(i)
        embeddings.append(np.array_str(embedding))  # Convert NumPy array to string
        chunks.append(" ".join(np.array_str(embedding).split()[:10]))  # Example: take first 10 words as chunk

    return render_template('view_database.html', embeddings=embeddings, chunks=chunks, zip=zip)

if __name__ == '__main__':
    # Ensure FAISS index is persistent by loading it if it exists
    if os.path.exists('faiss_index.index'):
        faiss_index = faiss.read_index('faiss_index.index')
    app.run(debug=True)
    # Save the FAISS index before exiting
    faiss.write_index(faiss_index, 'faiss_index.index')
