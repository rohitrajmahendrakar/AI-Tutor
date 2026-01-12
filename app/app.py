from dotenv import load_dotenv
load_dotenv()  

import os
from pathlib import Path
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from werkzeug.utils import secure_filename
import pymysql
from werkzeug.security import generate_password_hash, check_password_hash
import json
import ast
import re

try:
    from .chatbot import get_chatbot_response, model_name_used, add_uploaded_file
    from .database import init_db, DB_CONFIG
    from .constants import TEMPLATE_DIR, STATIC_DIR, PROMPT_PREFIX, SUPPORTED_FILE_TYPES, DATASET_DIR
except ImportError:  # Allow running as a standalone script
    from chatbot import get_chatbot_response, model_name_used, add_uploaded_file
    from database import init_db, DB_CONFIG
    from constants import TEMPLATE_DIR, STATIC_DIR, PROMPT_PREFIX, SUPPORTED_FILE_TYPES, DATASET_DIR

app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR),
    static_folder=str(STATIC_DIR),
)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-me")

# Configure upload settings
UPLOAD_DIR = DATASET_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_DIR)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

init_db()

def get_connection():
    return pymysql.connect(**DB_CONFIG)

@app.route('/')
def home():
    return render_template('index.html', username=session.get('user'), logged_in='user' in session)

@app.route('/register', methods=['GET', 'POST'])
def register():

    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()

        if not name or not email or not password:
            return render_template('register.html', error="All fields are required.")

        try:
            conn = get_connection()
            cursor = conn.cursor()
            password_hash = generate_password_hash(password)
            cursor.execute(
                "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)",
                (name, email, password_hash),
            )
            conn.commit()
            conn.close()
            return redirect(url_for('login'))
        except pymysql.err.IntegrityError:
            if 'conn' in locals():
                conn.rollback()
                conn.close()
            return render_template('register.html', error="An account with that email already exists.")
        except pymysql.err.OperationalError as e:
            if 'conn' in locals():
                conn.close()
            return render_template('register.html', error="Database connection error. Please check your configuration.")
        except Exception as e:
            if 'conn' in locals():
                conn.rollback()
                conn.close()
            return render_template('register.html', error=f"Registration failed: {str(e)}")
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()

        if not email or not password:
            return render_template('login.html', error="Email and password are required.")

        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name, password FROM users WHERE email=%s", (email,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[1], password):
            session['user'] = user[0]
            return redirect(url_for('chat'))
        return render_template('login.html', error="Invalid credentials.")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/chat')
def chat():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('chat.html', username=session['user'])

def strip_prompt_prefix(text, question):
    """
    Remove injected prompt prefix and echoed question from LLM responses.
    """
    if not isinstance(text, str):
        return text

    cleaned = text.lstrip()
    if cleaned.startswith(PROMPT_PREFIX):
        cleaned = cleaned[len(PROMPT_PREFIX):].lstrip("\n :")
        question_clean = (question or "").strip()
        if question_clean:
            lower_cleaned = cleaned.lstrip()
            if lower_cleaned.lower().startswith(question_clean.lower()):
                cleaned = lower_cleaned[len(question_clean):].lstrip("\n :")
            else:
                cleaned = lower_cleaned
    return cleaned


def deduplicate_sentences_preserving_format(text):
    seen = set()
    result_lines = []
    in_code_block = False
    sentence_splitter = re.compile(r'(?<=[.!?])\s+')

    for line in text.splitlines():
        stripped_line = line.strip()
        if stripped_line.startswith("```"):
            in_code_block = not in_code_block
            result_lines.append(line)
            continue
        if in_code_block or not stripped_line:
            result_lines.append(line)
            continue

        leading_space = line[: len(line) - len(line.lstrip())]
        content = stripped_line
        bullet_prefix = ""
        if content.startswith(("* ", "- ", "+ ")):
            bullet_prefix = content[:2]
            content = content[2:].lstrip()

        sentences = sentence_splitter.split(content)
        filtered = []
        for sentence in sentences:
            normalized = re.sub(r"\s+", " ", sentence).strip().lower()
            if not normalized:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            filtered.append(sentence.strip())

        if filtered:
            rebuilt = " ".join(filtered)
            prefix = f"{leading_space}{bullet_prefix}".rstrip()
            if prefix:
                if not prefix.endswith(" "):
                    prefix += " "
                result_lines.append(f"{prefix}{rebuilt}")
            else:
                result_lines.append(f"{leading_space}{rebuilt}")

    cleaned = "\n".join(result_lines).strip()
    if cleaned and not cleaned.rstrip().endswith("```"):
        terminal = cleaned.rstrip()[-1]
        if terminal not in ".!?":
            cleaned = cleaned.rstrip() + "."
    return cleaned


def format_chatbot_response(raw_response, original_question=None):
    """
    Normalize chatbot outputs so the frontend always receives readable text.
    Handles responses stored as serialized lists and cleans LLM artifacts.
    """
    if raw_response is None:
        return ''

    if isinstance(raw_response, list):
        # Join list items into readable format
        cleaned_items = []
        for item in raw_response:
            cleaned = str(item).strip()
            if cleaned:
                cleaned_items.append(cleaned)
        return '\n'.join(cleaned_items)

    if isinstance(raw_response, str):
        # Remove prompt prefix and echoed question
        stripped = strip_prompt_prefix(raw_response, original_question)
        stripped = stripped.strip()
        
        # Handle list-like string representations
        if stripped.startswith('[') and stripped.endswith(']'):
            for loader in (json.loads, ast.literal_eval):
                try:
                    parsed = loader(stripped)
                    if isinstance(parsed, list):
                        # Format list items properly
                        formatted_items = []
                        for item in parsed:
                            item_str = str(item).strip()
                            if item_str:
                                formatted_items.append(item_str)
                        return '\n'.join(formatted_items)
                except Exception:
                    continue
        
        # Remove context markers that might have leaked through
        import re
        stripped = re.sub(r'Context\s+\d+:\s*', '', stripped, flags=re.IGNORECASE)
        stripped = re.sub(r'^Answer:\s*', '', stripped, flags=re.IGNORECASE | re.MULTILINE)
        stripped = re.sub(r'Question:\s*', '', stripped, flags=re.IGNORECASE)
        
        # Remove excessive whitespace
        stripped = re.sub(r'\n{3,}', '\n\n', stripped)
        stripped = stripped.strip()
        
        # Deduplicate sentences while preserving format
        deduped = deduplicate_sentences_preserving_format(stripped)
        return deduped

    return str(raw_response)


@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form.get('message', '').strip()
    use_uploaded = request.form.get('use_uploaded', 'false').lower() == 'true'
    
    if not user_input:
        return jsonify({
            'response': 'Please provide a question.',
            'source': 'uncertain',
            'dataset_source': None,
            'similarity_score': 0.0
        })
    
    response, source, metadata = get_chatbot_response(user_input, use_uploaded_files=use_uploaded)
    formatted = format_chatbot_response(response, user_input)
    
    return jsonify({
        'response': formatted,
        'source': source,
        'dataset_source': metadata.get('dataset_source'),
        'similarity_score': metadata.get('similarity_score', 0.0),
        'is_grounded': metadata.get('is_grounded', False)
    })


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file uploads for indexing."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check file type
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_FILE_TYPES:
        return jsonify({
            'error': f'Unsupported file type. Supported types: {", ".join(SUPPORTED_FILE_TYPES)}'
        }), 400
    
    try:
        # Save file
        filename = secure_filename(file.filename)
        file_path = Path(app.config['UPLOAD_FOLDER']) / filename
        file.save(str(file_path))
        
        # Process and index
        success = add_uploaded_file(file_path, filename)
        
        if success:
            return jsonify({
                'message': f'File "{filename}" uploaded and indexed successfully.',
                'filename': filename
            })
        else:
            return jsonify({'error': 'Failed to process file'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/status')
def status():
    """Check chatbot readiness status."""
    try:
        from .chatbot import is_chatbot_ready
    except ImportError:
        from chatbot import is_chatbot_ready
    
    status_info = is_chatbot_ready()
    return jsonify({
        'status': 'ready' if status_info['ready'] else 'not_ready',
        'ready': status_info['ready'],
        'has_dataset': status_info['has_dataset'],
        'has_llm': status_info['has_llm'],
        'has_uploaded_files': status_info['has_uploaded_files'],
        'dataset_size': status_info['dataset_size'],
        'mode': status_info['model_name']
    })


@app.route('/debug')
def debug():
    """Debug endpoint to check chatbot state."""
    try:
        from .chatbot import is_chatbot_ready
        from .rag_pipeline import _main_dataset, _main_vectorizer, _main_chunks, _uploaded_chunks
    except ImportError:
        from chatbot import is_chatbot_ready
        from rag_pipeline import _main_dataset, _main_vectorizer, _main_chunks, _uploaded_chunks
    
    status_info = is_chatbot_ready()
    
    debug_info = {
        'status': status_info,
        'main_dataset_empty': _main_dataset is None or _main_dataset.empty if _main_dataset is not None else True,
        'main_vectorizer_none': _main_vectorizer is None,
        'main_chunks_count': len(_main_chunks) if _main_chunks else 0,
        'uploaded_chunks_count': len(_uploaded_chunks) if _uploaded_chunks else 0,
    }
    
    return jsonify(debug_info)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
