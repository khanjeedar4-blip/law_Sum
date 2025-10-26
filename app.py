from flask import Flask, render_template, request, send_file,abort,send_from_directory
import torch
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, render_template, request, redirect, url_for, jsonify, session

from flask import Flask, request, jsonify
from pymongo import MongoClient
import pickle
from pymongo import MongoClient
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from PyPDF2 import PdfReader
from docx import Document
import re
import seaborn as sns

import matplotlib.pyplot as plt
import os
import string
import requests
from bs4 import BeautifulSoup
from flask import jsonify
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
import faiss
import fitz 
import groq
import PyPDF2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_argon2 import Argon2
from pymongo import MongoClient
import os
import pdfplumber
from groq import Groq
import logging
logging.getLogger("pdfminer").setLevel(logging.ERROR)



app = Flask(__name__,template_folder="FYP RAG/summerization-app/templates")

GROQ_API_KEY = os.environ.get("lawsumm")
cli = Groq(api_key=GROQ_API_KEY)

# Load embedding model globally
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load preprocessed data
with open("FYP RAG/summerization-app/data/legal_data.pkl", "rb") as f:
    legal_data = pickle.load(f)

# Load FAISS indices
faiss_indices = {}
for law in legal_data:
    try:
        index_path = f"FYP RAG/summerization-app/data/{law.replace(' ', '_')}_faiss.index"
        index = faiss.read_index(index_path)
        faiss_indices[law] = (index, legal_data[law])
    except Exception as e:
        print(f"Error loading FAISS index for {law}: {str(e)}")

# Helper to match section
def get_exact_section(section_number, structured_data):
    for section in structured_data:
        if section["section_id"].strip() == section_number.strip():
            return section
    return None

# Helper to find relevant section via similarity
def find_relevant_section(query, model, index, structured_data, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [structured_data[i] for i in indices[0] if i < len(structured_data)]

# Generate answer using Groq
def generate_response_with_groq(prompt, section_number, book_name, context):
    full_prompt = f"According to Section {section_number} of {book_name}, {prompt}"
    try:
        response = cli.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are a legal assistant providing detailed and comprehensive legal explanations based on Pakistani law. Always provide at least 5-6 sentences per response."
                },
                {
                    "role": "user",
                    "content": f"{full_prompt}\n\nContext: {context}"
                }
            ],
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error with Groq API: {str(e)}")
        return "Error communicating with Groq API."



# Main route
@app.route('/ask', methods=['POST'])
def ask():
    try:
        # Ensure the request is JSON
        if not request.is_json:
            return jsonify({'response': 'Request must be JSON'}), 400

        data = request.get_json()
        query = data.get('query', '').strip()

        if not query:
            return jsonify({'response': 'Please enter a valid question.'}), 400

        # Extract section number and book name using regex
        pattern = r"what\s+is\s+section\s+no\.?\s*(\d+[A-Z]?(?:\(\d+\))?)\s+of\s+(.*)"
        match = re.search(pattern, query, re.IGNORECASE)

        if match:
            section_number = match.group(1).strip()
            book_name = match.group(2).strip()

            matched_book = None
            for law in legal_data:
                if book_name.lower() in law.lower():
                    matched_book = law
                    break

            if not matched_book:
                return jsonify({'response': 'Book name not recognized. Please try again with a valid book name.'}), 404

            index, structured_data = faiss_indices.get(matched_book, (None, None))
            if index is None or structured_data is None:
                return jsonify({'response': 'Error loading FAISS index for the selected law.'}), 500

            exact_section = get_exact_section(section_number, structured_data)
            if exact_section:
                response = generate_response_with_groq(query, section_number, matched_book, exact_section['content'])
                return jsonify({'response': response}), 200
            else:
                relevant = find_relevant_section(query, embedding_model, index, structured_data)
                if relevant:
                    response = generate_response_with_groq(query, relevant[0]['section_id'], matched_book, relevant[0]['content'])
                    return jsonify({'response': response}), 200
                else:
                    return jsonify({'response': 'No relevant section found.'}), 404
        else:
            return jsonify({'response': 'Please ask your question in this format: "What is Section No. 302 of Pakistan Penal Code?"'}), 400

    except Exception as e:
        print(f"Server error at /ask: {str(e)}")
        return jsonify({'response': 'An internal error occurred. Please try again later.'}), 500



# Flask route


# Load the fine-tuned Legal LED model
MODEL_NAME = "Izza-shahzad-13/legal-LED-final"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Function to generate summary
def generate_summary(text):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    outputs = model.generate(inputs, max_length=800, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Function to calculate sentence importance scores
def calculate_sentence_importance(summary):
    sentences = summary.split(". ")
    scores = [len(sentence) for sentence in sentences]  # Score based on sentence length
    max_score = max(scores) if scores else 1
    normalized_scores = [score / max_score for score in scores]
    return sentences, normalized_scores

# Function to generate heatmap
def generate_heatmap(scores):
    plt.figure(figsize=(10, 2))
    sns.heatmap([scores], annot=True, cmap="coolwarm", xticklabels=False, yticklabels=False, cbar=True)
    plt.title("Sentence Importance Heatmap")
    os.makedirs("static", exist_ok=True)
    plt.savefig("static/heatmap.png")  # Save heatmap image
    plt.close()

# Function to highlight sentences in the summary
def highlight_summary(sentences, scores):
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    highlighted_summary = ""

    for sentence, score in zip(sentences, scores):
        color = cmap(score)
        rgb_color = f"rgb({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)})"
        highlighted_summary += f'<span style="background-color:{rgb_color};padding:2px;">{sentence}.</span> '

    return highlighted_summary

# Function to highlight legal terms
def highlight_keywords(text):
    patterns = {
        'act_with_year': r'\b([A-Za-z\s]+(?:\sAct(?:\s[\d]{4})?))\s*,\s*(\d{4})\b',
        'article': r'\bArticle\s\d{1,3}(-[A-Z])?\b',
        'section': r'\bSection\s\d{1,3}[-A-Za-z]?\(?[a-zA-Z]?\)?\b',
        'date': r'\b(?:[A-Za-z]+)\s\d{4}\b|\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',
        'persons': r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b',
        'ordinance': r'\b([A-Z][a-z\s]+Ordinance(?:,\s\d{4})?)\b',  # Example: PEMRA Ordinance, 2002
        'petition': r'\b(?:[A-Za-z\s]*Petition\sNo\.\s\d+/\d{4})\b',  # Example: Constitutional Petition No. 123/2024
        'act_with_year': r'\b([A-Za-z\s]+(?:\sAct(?:\s\d{4})?)),\s*(\d{4})\b',  # Example: Control of Narcotic Substances Act, 1997
        'article': r'\b(Article\s\d{1,3}(-[A-Z])?)\b',  # Example: Article 10-A
        'section': r'\b(Section\s\d{1,3}(\([a-zA-Z0-9]+\))?)\b',  # Example: Section 302(b), Section 9(c), Section 144-A
        'date': r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{1,2},?\s\d{4})\b',  
        # Examples: 15/07/2015, July 2015, March 5, 2021, 2023
        'person': r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)\b'  # Example: Justice Ali Raza

    }

    highlighted_text = text
    for pattern in patterns.values():
        highlighted_text = re.sub(pattern, lambda match: f'<span class="highlight">{match.group(0)}</span>', highlighted_text)

    return highlighted_text

# Function to read uploaded files
def read_file(file):
    if file.filename.endswith(".txt"):
        return file.read().decode("utf-8")
    elif file.filename.endswith(".pdf"):
        pdf_reader = PdfReader(file)
        return " ".join(page.extract_text() for page in pdf_reader.pages)
    elif file.filename.endswith(".docx"):
        doc = Document(file)
        return " ".join(paragraph.text for paragraph in doc.paragraphs)
    return None

# Function to fetch text from a URL
def fetch_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()

        # Check content type
        content_type = response.headers.get("Content-Type", "")
        if "text/html" in content_type:  # If it's a webpage
            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = soup.find_all("p")  # Extract paragraph text
            return " ".join([p.get_text() for p in paragraphs])

        elif "text/plain" in content_type:  # If it's a plain text file
            return response.text

        else:
            return None
    except Exception as e:
        print("Error fetching URL:", e)
        return None

@app.route("/", methods=["GET", "POST"])
def index():
    document_text = None
    summary = None
    heatmap_url = None

    if request.method == "POST":
        file = request.files.get("file")
        pasted_text = request.form.get("pasteText", "").strip()
        url = request.form.get("url", "").strip()

        if file and file.filename:
            document_text = read_file(file)
        elif pasted_text:
            document_text = pasted_text
        elif url:
            document_text = fetch_text_from_url(url)

        if document_text:
            summary = generate_summary(document_text)
            sentences, scores = calculate_sentence_importance(summary)

            generate_heatmap(scores)

            highlighted_summary = highlight_summary(sentences, scores)
            highlighted_summary = highlight_keywords(highlighted_summary)

            # Save the summary to a text file
            with open("summary.txt", "w", encoding="utf-8") as f:
                f.write(summary)

            return render_template("mainscreen.html", document_text=document_text, summary=highlighted_summary, heatmap_url="static/heatmap.png")

    return render_template("mainscreen.html", document_text=None, summary=None, heatmap_url=None)

@app.route("/download_summary")
def download_summary():
    file_path = os.path.join(os.getcwd(), "summary.txt")

    if not os.path.exists(file_path):
        return abort(404, description="File not found")

    return send_file(file_path, as_attachment=True, download_name="summary.txt", mimetype="text/plain")


  # Homepage 
@app.route("/home")
def home():
     return render_template("homepage.html")
@app.route("/about")
def about():
    return render_template("aboutpage.html") 
@app.route("/summarization")
def summarization():
    return render_template("mainscreen.html")  # Login Page


@app.route('/lawbooks/<filename>')
def serve_pdf(filename):
    return send_from_directory('static/lawbooks', filename)




# MongoDB connection
client = MongoClient('mongodb+srv://law:X1PNiOZtTdyIIO0m@law.urpdise.mongodb.net/?retryWrites=true&w=majority&appName=law')
db = client['chatbotDB']
users = db['users']



@app.route('/signup', methods=['GET'])
def signup():
    return render_template('signuppage.html')  # Render the HTML form

@app.route('/api/signup', methods=['POST'])
def api_signup():
    # Get JSON data from the request
    data = request.get_json()
    first_name = data.get('firstName')
    last_name = data.get('lastName')
    email = data.get('email')
    password = data.get('password')

    # Hash the password for security before storing it in the database
    hashed_pw = generate_password_hash(password)

    # Check if the user already exists
    if users.find_one({'email': email}):
        return jsonify({'message': 'Email already exists!'}), 400

    # Insert the user data into MongoDB
    users.insert_one({
        'first_name': first_name,
        'last_name': last_name,
        'email': email,
        'password': hashed_pw
    })

    # Return a success response
    return jsonify({'message': 'Signup successful!'}), 201

# Success page or login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Handle POST request for login
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')

        # Log login attempt
        print(f"Login attempt - Email: {email}")

        # Check if the user exists
        user = users.find_one({'email': email})
        if not user:
            print(f"Login failed - Email '{email}' not found.")
            return jsonify({'message': 'Invalid email or password!'}), 401

        # Check if the password is correct (compare hashed passwords)
        if not check_password_hash(user['password'], password):
            print(f"Login failed - Incorrect password for email '{email}'.")
            return jsonify({'message': 'Invalid email or password!'}), 401

        # Log successful login
        print(f"Login successful - Email: {email}")
        return jsonify({'message': 'Login successful!'}), 200

    # Handle GET request - Show login form (if needed)
    return render_template('loginpage.html')  # This would be the login form page (replace with your template)


@app.route('/reset-password', methods=['GET', 'POST'])
def reset_password():
    if request.method == 'POST':
        email = request.form['email']
        new_password = request.form['newPassword']
        confirm_password = request.form['confirmPassword']

        # Check if passwords match
        if new_password != confirm_password:
            return jsonify({'message': 'Passwords do not match!'}), 400

        # Check if user exists
        user = users.find_one({'email': email})
        if not user:
            return jsonify({'message': 'User not found!'}), 404

        # Hash the new password
        hashed_pw = generate_password_hash(new_password)

        # Update the user's password in the database
        users.update_one({'email': email}, {'$set': {'password': hashed_pw}})
        return jsonify({'message': 'Password updated successfully!'}), 200

    return render_template('forgetpasswordpage.html')


contacts_collection = db["contacts"]
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')

        print(f"Name: {name}, Email: {email}, Message: {message}")  # Debug

        if not name or not email or not message:
            return jsonify({'message': 'All fields are required!'}), 400

        contact_data = {
            'name': name,
            'email': email,
            'message': message
        }

        contacts_collection.insert_one(contact_data)
        return jsonify({'message': f'Thank you, {name}! Your message has been sent successfully.',
            'status': 'success'}), 200

    return render_template('contactpage.html')







if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)