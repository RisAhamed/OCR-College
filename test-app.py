import os
import time
import json
import threading
import subprocess

import torch
import pandas as pd
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from transformers import AutoProcessor, AutoModelForImageTextToText

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = "ocr_secret_key"

# Configure upload folder and model cache directory
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
MODEL_DIR = 'models'  # This is used as the cache directory

# Create necessary directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Global variables for model and processor
model = None
processor = None
model_loading = False
model_loaded = False

# Set the Hugging Face repository identifier for the model
current_model_name = "Qwen/Qwen2.5-VL-32B-Instruct-AWQ"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def load_model():
#     """
#     Loads the model directly from Hugging Face using the repository identifier.
#     The model is downloaded (if not already cached) into the MODEL_DIR folder.
#     """
#     global model, processor, model_loading, model_loaded, current_model_name

#     if model_loading:
#         return "Model is already being loaded"
#     if model_loaded:
#         return "Model is already loaded"
    
#     model_loading = True
#     try:
#         print(f"Loading model '{current_model_name}' from Hugging Face (cache directory: {MODEL_DIR})")
        
#         # Upgrade autoawq if needed (required for quantized models)
#         try:
#             subprocess.check_call(["pip", "install", "autoawq>=0.1.8", "--upgrade"])
#             print("autoawq upgraded successfully.")
#         except Exception as e:
#             print(f"Warning: Could not upgrade autoawq: {e}")
        
#         # Load processor and model directly from Hugging Face
#         processor = AutoProcessor.from_pretrained(
#             current_model_name,
#             trust_remote_code=True,
#             cache_dir=MODEL_DIR
#         )
#         model = AutoModelForImageTextToText.from_pretrained(
#             current_model_name,
#             trust_remote_code=True,
#             device_map="auto",
#             cache_dir=MODEL_DIR
#         )
        
#         model_loaded = True
#         print("Model loaded successfully from Hugging Face.")
#         return "Model loaded successfully"
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         model_loading = False
#         return f"Error loading model: {e}"
#     finally:
#         model_loading = False
from transformers import AutoProcessor, AutoModelForImageTextToText

def load_model():
    """
    Load the Qwen model directly from Hugging Face using its repository ID.
    The model will be downloaded to the MODEL_DIR cache (if not already cached)
    and quantization settings are disabled by setting quantization_config to None.
    """
    global model, processor, model_loading, model_loaded, current_model_name

    if model_loading:
        return "Model is already being loaded"
    if model_loaded:
        return "Model is already loaded"
    
    model_loading = True
    try:
        print(f"Loading model '{current_model_name}' from Hugging Face (cache directory: {MODEL_DIR})")
        
        # (Remove autoawq upgrade subprocess to avoid dependency conflicts)
        # Load processor and model directly from Hugging Face, with caching in MODEL_DIR.
        processor = AutoProcessor.from_pretrained(
            current_model_name,
            trust_remote_code=True,
            cache_dir=MODEL_DIR
        )
        model = AutoModelForImageTextToText.from_pretrained(
            current_model_name,
            trust_remote_code=True,
            device_map="auto",
            cache_dir=MODEL_DIR,
            quantization_config=None  # Disable quantization skipping
        )
        
        model_loaded = True
        print("Model loaded successfully from Hugging Face.")
        return "Model loaded successfully"
    except Exception as e:
        print(f"Error loading model: {e}")
        model_loading = False
        return f"Error loading model: {e}"
    finally:
        model_loading = False

def process_image_local(image_path):
    """
    Processes an image by sending it along with a prompt to the loaded model.
    The model is expected to return text that is then formatted into CSV (tabular) form.
    """
    global model, processor, model_loaded
    
    if not model_loaded:
        return {"error": "Model not loaded. Please wait for the model to load."}
    
    try:
        # Open the image and convert it to RGB
        image = Image.open(image_path).convert("RGB")
        
        # Define the prompt for extracting text exactly as seen
        prompt = (
            "Please extract all text from this image exactly as it appears, preserving all formatting, "
            "numbers, and special characters. Format the output as a CSV table with clear column separations using commas. "
            "Do not add any interpretations, just extract the raw text as is."
        )
        
        # Prepare inputs (the model accepts both image and text)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
        
        # Generate output using the model
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512)
        
        # Decode the output text
        extracted_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        print(f"Extracted raw text: {extracted_text}")
        
        # Optionally, format the extracted text into CSV-like table format
        tabular_text = format_text_to_table(extracted_text)
        
        return {"raw_text": extracted_text, "generated_text": tabular_text}
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return {"error": f"Error processing image: {str(e)}"}

def format_text_to_table(text):
    """
    Formats the extracted text into a CSV-like structure.
    Attempts to convert tabs or multiple spaces into comma-separated columns.
    """
    try:
        lines = text.strip().split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # If the line contains commas, assume it's already CSV formatted
            if ',' in line:
                formatted_lines.append(line)
            # Convert tabs to commas
            elif '\t' in line:
                formatted_lines.append(','.join(part.strip() for part in line.split('\t')))
            # Convert multiple spaces to commas
            elif '  ' in line:
                import re
                parts = re.split(r'\s{2,}', line)
                parts = [part.strip() for part in parts if part.strip()]
                formatted_lines.append(','.join(parts))
            else:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    except Exception as e:
        print(f"Error formatting text: {e}")
        return text

@app.route('/upload', methods=['POST'])
def upload_file():
    global model_loaded
    
    if not model_loaded:
        load_model()
        if not model_loaded:
            flash('Model not loaded yet. Please wait for the model to load or reload the page.')
            return redirect(url_for('index'))
    
    if 'files[]' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    
    files = request.files.getlist('files[]')
    if not files or files[0].filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    if len(files) > 10:
        flash('Maximum 10 files allowed. Only the first 10 will be processed.')
        files = files[:10]
    
    results = []
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
                        'raw_text': result.get("raw_text", ""),
                        'generated_text': result.get("generated_text", "")
                    })
            except Exception as e:
                flash(f'Error processing {filename}: {str(e)}')
    
    ocr_results = []
    for item in results:
        ocr_results.append({
            'Source File': item['filename'],
            'Raw Text': item.get('raw_text', ''),
            'Tabular Text': item.get('generated_text', '')
        })
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    csv_path = f'results/ocr_results_{len(results)}_{timestamp}.csv'
    
    if ocr_results:
        rows = []
        for result in results:
            tabular_text = result.get('generated_text', '')
            if tabular_text:
                for line in tabular_text.split('\n'):
                    if line.strip():
                        row = {'Source File': result['filename']}
                        if ',' in line:
                            parts = line.split(',')
                            for i, part in enumerate(parts):
                                row[f'Column {i+1}'] = part.strip()
                        else:
                            row['Content'] = line.strip()
                        rows.append(row)
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            flash(f'Results saved to {csv_path}')
        else:
            flash('No valid data extracted from images')
    
    return render_template('index.html', 
                           model_loaded=model_loaded, 
                           current_model=current_model_name,
                           ocr_results=ocr_results,
                           csv_download_path=csv_path)

def list_available_models():
    """Since we're only using one model, return that."""
    return [current_model_name]

@app.route('/')
def index():
    global model_loaded, model_loading
    if not model_loaded and not model_loading:
        threading.Thread(target=load_model).start()
    return render_template('index.html', 
                           model_loaded=model_loaded, 
                           current_model=current_model_name)

@app.route('/load_model', methods=['POST'])
def load_model_route():
    result = load_model()
    return {"status": result}

@app.route('/change_model', methods=['POST'])
def change_model():
    global current_model_name, model_loaded
    data = request.get_json()
    if not data or 'model_name' not in data:
        return {"status": "error", "message": "No model name provided"}
    new_model = data['model_name']
    if not check_model_downloaded(new_model):
        return {"status": "error", "message": f"Model {new_model} not found locally"}
    current_model_name = new_model
    model_loaded = False
    threading.Thread(target=load_model).start()
    return {"status": "success", "message": f"Changing to model: {new_model}"}

@app.route('/results/<path:filename>')
def download_file(filename):
    return send_from_directory('results', filename, as_attachment=True)

if __name__ == '__main__':
    threading.Thread(target=load_model).start()
    app.run(debug=True)
