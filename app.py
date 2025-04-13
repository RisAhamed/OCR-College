import os
import time
import requests
import io
import json
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from PIL import Image
import threading
import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'ocr_secret_key')

# Configure folders and settings
UPLOAD_FOLDER = 'uploads'
DOWNLOAD_FOLDER = 'downloads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILES = 10
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create necessary directories
for folder in [UPLOAD_FOLDER, DOWNLOAD_FOLDER, RESULTS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Global variables
model = None
tokenizer = None
pipe = None
model_loading = False
model_loaded = False

# Use Qwen 2.5 72B model
current_model_name = "Qwen/Qwen2.5-72B"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    """
    Load the Qwen 2.5 72B model using both pipeline and direct loading
    """
    global model, tokenizer, pipe, model_loading, model_loaded
    
    if model_loading:
        return "Model is already being loaded"
    if model_loaded:
        return "Model is already loaded"
    
    model_loading = True
    try:
        print(f"Loading Qwen model {current_model_name}...")
        
        # Load tokenizer and model directly
        tokenizer = AutoTokenizer.from_pretrained(
            current_model_name,
            trust_remote_code=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            current_model_name,
            trust_remote_code=True,
            device_map="auto"
        )
        
        # Create pipeline for easier text generation
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto"
        )
        
        if torch.cuda.is_available():
            print("Using GPU for inference")
        else:
            print("Using CPU for inference")
        
        model_loaded = True
        print("Qwen model loaded successfully")
        return "Model loaded successfully"
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return f"Error loading model: {e}"
        
    finally:
        model_loading = False

def ensure_model_loaded():
    """
    Ensure the model is loaded before processing
    """
    global model_loaded
    if not model_loaded:
        load_model()
    return model_loaded

def process_image_local(image_path):
    """
    Process an image using Qwen model to extract text with high accuracy
    """
    global model, tokenizer, pipe, model_loaded
    
    if not ensure_model_loaded():
        return {"error": "Could not load the model. Please try again."}
    
    try:
        # Open and process the image
        image = Image.open(image_path).convert("RGB")
        
        # Create a specific prompt for accurate account document text extraction
        prompt = """Please analyze this account document image and extract all text exactly as it appears.
        Requirements:
        1. Extract all numbers, text, and special characters precisely as shown
        2. Maintain the exact column structure of the document
        3. Separate columns with commas (CSV format)
        4. Preserve all decimal points and numerical formatting
        5. Keep headers and labels exactly as they appear
        6. Extract data row by row, maintaining the table structure
        7. Do not interpret or modify any values
        8. Include all dates in their original format
        
        Format the output as a CSV table where:
        - Each row of data is on a new line
        - Columns are separated by commas
        - Preserve empty cells with commas
        
        Extract the text now:"""
        
        # Process with Qwen model using pipeline
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image}
                ]
            }
        ]
        
        # Generate response using pipeline
        response = pipe(messages, max_new_tokens=1024, do_sample=False)
        extracted_text = response[0]['generated_text']
        
        # Process the text to ensure proper CSV format
        processed_text = process_account_text(extracted_text)
        
        return {
            "raw_text": extracted_text,
            "generated_text": processed_text
        }
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return {"error": f"Error processing image: {str(e)}"}

def process_account_text(text):
    """
    Process extracted text to ensure proper CSV formatting
    """
    lines = []
    current_table = []
    
    # Split into lines and clean up
    raw_lines = text.strip().split('\n')
    
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
            
        # Remove any CSV formatting instructions that might have been generated
        if line.lower().startswith(('csv', 'table', 'format', 'column')):
            continue
            
        # Check if line contains actual data
        if any(char.isalnum() for char in line):
            # If line already contains commas, assume it's properly formatted
            if ',' in line:
                current_table.append(line)
            else:
                # Split by multiple spaces and join with commas
                parts = [part.strip() for part in line.split('  ') if part.strip()]
                current_table.append(','.join(parts))
    
    return '\n'.join(current_table)

@app.route('/upload', methods=['POST'])
def upload_file():
    if not ensure_model_loaded():
        flash('Could not load the model. Please try again.')
        return redirect(url_for('index'))
    
    if 'files[]' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    
    files = request.files.getlist('files[]')
    if not files or files[0].filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    
    if len(files) > MAX_FILES:
        flash(f'Maximum {MAX_FILES} files allowed. Only the first {MAX_FILES} will be processed.')
        files = files[:MAX_FILES]
    
    results = []
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            try:
                result = process_image_local(file_path)
                if 'error' in result:
                    flash(f"Error processing {filename}: {result['error']}")
                else:
                    results.append({
                        'filename': filename,
                        'raw_text': result['raw_text'],
                        'generated_text': result['generated_text']
                    })
            except Exception as e:
                flash(f'Error processing {filename}: {str(e)}')
            finally:
                # Clean up uploaded file
                if os.path.exists(file_path):
                    os.remove(file_path)
    
    if results:
        # Save results to CSV
        csv_filename = f'account_data_{timestamp}.csv'
        csv_path = os.path.join(RESULTS_FOLDER, csv_filename)
        
        # Prepare data for CSV
        csv_data = []
        for result in results:
            for line in result['generated_text'].split('\n'):
                if line.strip():
                    # First column is always the source filename
                    row_data = {'Source_File': result['filename']}
                    
                    # Split the line by comma and preserve all parts
                    parts = line.split(',')
                    for i, part in enumerate(parts):
                        row_data[f'Column_{i+1}'] = part.strip()
                    
                    csv_data.append(row_data)
        
        # Save to CSV
        if csv_data:
            df = pd.DataFrame(csv_data)
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            flash(f'Results saved to {csv_filename}')
            
            return render_template('index.html',
                                model_loaded=model_loaded,
                                current_model=current_model_name,
                                ocr_results=results,
                                csv_download_path=csv_filename)
    
    flash('No valid results were generated')
    return redirect(url_for('index'))

@app.route('/')
def index():
    # Load model when the app starts
    if not model_loaded and not model_loading:
        threading.Thread(target=load_model).start()
    return render_template('index.html',
                         model_loaded=model_loaded,
                         current_model=current_model_name)

@app.route('/results/<path:filename>')
def download_file(filename):
    """Download processed results"""
    return send_from_directory(RESULTS_FOLDER, filename, as_attachment=True)

if __name__ == '__main__':
    # Load model when the app starts
    threading.Thread(target=load_model).start()
    app.run(debug=True, host='0.0.0.0')
