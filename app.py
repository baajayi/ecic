import os
import json
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from openai import OpenAI
from llama_index.core import SimpleDirectoryReader
from dotenv import load_dotenv, find_dotenv
import tiktoken
import concurrent.futures
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load environment variables
_ = load_dotenv(find_dotenv())


client = OpenAI()
docs = []

# Function to load markdown files
def load_markdown_files(directory):
    reader = SimpleDirectoryReader(directory, recursive=True)
    return reader.load_data()

# Function to split text into chunks
def split_text(text, max_tokens, encoding_name="cl100k_base"):
    tokenizer = tiktoken.get_encoding(encoding_name)
    tokens = tokenizer.encode(text)
    
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunks.append(tokenizer.decode(chunk_tokens))
    
    return chunks

# Function to get embeddings for text chunks
def get_embeddings_for_text_chunks(text_chunks, model="text-embedding-ada-002"):
    embeddings = []
    for chunk in text_chunks:
        response = client.embeddings.create(input=chunk, model=model)
        embedding = response.data[0].embedding
        embeddings.append(embedding)
    return embeddings

# Function to save embeddings to a file
def save_embeddings_to_npz(embeddings, filename):
    sparse_embeddings = csr_matrix(embeddings)
    save_npz(filename, sparse_embeddings)

# Function to load embeddings from a file
def load_embeddings_from_npz(filename):
    return load_npz(filename).toarray()

# Function to check if embedding is valid
def is_valid_embedding(embedding):
    return np.all(np.isfinite(embedding))

# Filenames for embeddings and document texts
embeddings_file = 'ecic_embeddings.npz'
document_texts_file = 'ecic_document_texts.json'

# Load or generate embeddings
if os.path.exists(embeddings_file) and os.path.exists(document_texts_file):
    all_embeddings = load_embeddings_from_npz(embeddings_file)
    with open(document_texts_file, 'r') as f:
        document_texts = json.load(f)
else:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_docs = executor.submit(load_markdown_files, "ECIC_MDFiles").result()
    docs.extend(future_docs)

    all_embeddings = []
    document_texts = []
    for doc in docs:
        if hasattr(doc, 'text'):
            doc_text = doc.text
            text_chunks = split_text(doc_text, max_tokens=8192)
            embeddings = get_embeddings_for_text_chunks(text_chunks)
            all_embeddings.extend(embeddings)
            document_texts.extend(text_chunks)

    save_embeddings_to_npz(all_embeddings, embeddings_file)
    with open(document_texts_file, 'w') as f:
        json.dump(document_texts, f)

# Validate embeddings
valid_embeddings = []
valid_texts = []
for embedding, text in zip(all_embeddings, document_texts):
    if is_valid_embedding(embedding):
        valid_embeddings.append(embedding)
        valid_texts.append(text)

# Function to find similar documents based on embeddings
def find_similar_documents(query_embedding, embeddings, texts, top_k=5):
    similarities = cosine_similarity([query_embedding], embeddings)
    top_k_indices = np.argsort(similarities[0])[-top_k:][::-1]
    return [(texts[i], similarities[0][i]) for i in top_k_indices]

# Function to generate response
def get_retrieval_augmented_response(prompt, model="text-embedding-ada-002"):
    query_embedding_response = client.embeddings.create(input=prompt, model=model)
    query_embedding = query_embedding_response.data[0].embedding
    similar_docs = find_similar_documents(query_embedding, valid_embeddings, valid_texts)

    max_context_length = 8192
    context = ""
    total_length = 0
    tokenizer = tiktoken.get_encoding("cl100k_base")
    for doc, _ in similar_docs:
        doc_length = len(tokenizer.encode(doc))
        if total_length + doc_length <= max_context_length:
            context += doc + " "
            total_length += doc_length
        else:
            break

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """You are a smart assistant
                 You answer questions based on the context provided.
                 If you cannot find any relevant information in the context to answer the question, just say 'I do not know the answer to that question.'
                                            """},
                {"role": "system", "content": context.strip()},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    query = data.get('query', '')
    response = get_retrieval_augmented_response(query)
    print(f'{query}\n{response}')
    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
