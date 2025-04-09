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
from transformers import AutoProcessor, VisionEncoderDecoderModel

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = "ocr_secret_key"

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
MODEL_DIR = 'models'

# Create necessary directories if not exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Global variables for model and processor
model = None
processor = None
model_loading = False
model_loaded = False
# Set the specific model path
MODEL_PATH = "C:\\Users\\riswa\\Desktop\\AI\\OCR-College\\models\\models--Qwen--Qwen2.5-VL-32B-Instruct-AWQ\\snapshots\\66c370b74a18e7b1e871c97918f032ed3578dfef"
current_model_name = "Qwen/Qwen2.5-VL-32B-Instruct-AWQ"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Update the default model to one that's known to work well with OCR
# current_model_name = "microsoft/trocr-base-handwritten"  # Changed from Qwen model

def check_model_downloaded(model_name):
    """
    Check if the model has been downloaded locally
    """
    model_dir = os.path.join(os.getcwd(), MODEL_DIR)
    
    # More robust check for model existence
    # First, check the standard Hugging Face cache directory structure
    hf_cache_pattern = os.path.join(model_dir, "models--" + model_name.replace('/', '--'))
    if os.path.exists(hf_cache_pattern):
        # Check for model files inside this directory
        for root, dirs, files in os.walk(hf_cache_pattern):
            if 'config.json' in files:
                return True
    
    # Also check other possible paths
    possible_paths = [
        os.path.join(model_dir, model_name.replace('/', '--')),
        os.path.join(model_dir, model_name.split('/')[-1]),
        os.path.join(model_dir, model_name),
        os.path.join(model_dir, "models--" + model_name.replace('/', '--'))
    ]
    
    # Also search subdirectories inside the model folder
    for root, dirs, files in os.walk(model_dir):
        if model_name.replace('/', '--') in root or model_name.split('/')[-1] in root:
            return True
    
    # Check possible paths
    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            # Check if this directory contains model files
            if any(os.path.exists(os.path.join(path, f)) for f in ['config.json', 'pytorch_model.bin']):
                return True
    
    return False

# Global flag to track if any model has been loaded at least once
model_ever_loaded = False

# Update the model loading function to better handle different model types
def load_model():
    """
    Load the Qwen model for image processing and text extraction.
    Uses the specific model path directly.
    """
    global model, processor, model_loading, model_loaded, model_ever_loaded
    
    if model_loading:
        return "Model is already being loaded"
    
    if model_loaded:
        return "Model is already loaded"
    
    model_loading = True
    
    try:
        print(f"Loading Qwen model from: {MODEL_PATH}")
        
        # First, try to install or upgrade autoawq
        try:
            import subprocess
            print("Checking and upgrading autoawq package...")
            subprocess.check_call(["pip", "install", "autoawq>=0.1.8", "--upgrade"])
            print("autoawq package upgraded successfully")
        except Exception as e:
            print(f"Warning: Could not upgrade autoawq package: {e}")
        
        # Load the model directly from the specific path
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # Use specific classes for Qwen VL model
            processor = AutoTokenizer.from_pretrained(
                MODEL_PATH,
                trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH, 
                trust_remote_code=True,
                device_map="auto",
                # Skip quantization config if there are issues
                ignore_mismatched_sizes=True
            )
            
        except Exception as e1:
            print(f"Failed with primary loading method: {e1}")
            try:
                # Try with a different approach - load without quantization
                from transformers import AutoTokenizer, AutoModelForCausalLM
                
                processor = AutoTokenizer.from_pretrained(
                    MODEL_PATH,
                    trust_remote_code=True
                )
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_PATH, 
                    trust_remote_code=True,
                    device_map="auto",
                    quantization_config=None  # Explicitly disable quantization
                )
            except Exception as e2:
                print(f"Failed with secondary loading method: {e2}")
                # Try with a simpler approach
                try:
                    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
                    
                    # Load config first
                    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
                    # Disable quantization in config
                    if hasattr(config, 'quantization_config'):
                        config.quantization_config = None
                    
                    processor = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
                    model = AutoModelForCausalLM.from_pretrained(
                        MODEL_PATH,
                        config=config,
                        trust_remote_code=True,
                        device_map="auto"
                    )
                except Exception as e3:
                    print(f"Failed with tertiary loading method: {e3}")
                    raise Exception(f"All model loading attempts failed. Please install autoawq>=0.1.8 manually.")
        
        model_loaded = True
        model_ever_loaded = True
        print("Qwen model loaded successfully")
        return "Qwen model loaded successfully"
    except Exception as e:
        print(f"Error loading Qwen model: {e}")
        model_loading = False
        return f"Error loading Qwen model: {e}"
    finally:
        model_loading = False

def process_image_local(image_path):
    """
    Process the image using the Qwen model to extract text.
    """
    global model, processor, model_loaded
    
    if not model_loaded:
        return {"error": "Qwen model not loaded. Please wait for the model to load."}
    
    try:
        # Open and process the image
        image = Image.open(image_path).convert("RGB")
        
        # Create a specific prompt for the Qwen VL model to extract text
        prompt = """Please extract all text from this image exactly as it appears, preserving all formatting, numbers, and special characters.
        Format the output as a table with clear column separations using commas.
        Do not add any interpretations, just extract the raw text as is."""
        
        # Check if we're using pipeline or direct model
        if isinstance(model, type(lambda: None).__class__) or str(type(model)).find('pipeline') != -1:
            # Pipeline approach
            try:
                # Format messages for pipeline
                messages = [
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "image": image}
                    ]}
                ]
                
                # Process with pipeline
                result = model(messages)
                extracted_text = result[0]["generated_text"]
            except Exception as e:
                # Try alternative pipeline format
                result = model(image, prompt)
                extracted_text = result
        else:
            # Direct model approach
            import torch
            
            # Process the image with Qwen VL model
            inputs = processor.from_list_format([
                {"image": image},
                {"text": prompt}
            ], return_tensors="pt").to(model.device)
            
            # Generate text with the model
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=512,
                    do_sample=False
                )
            
            # Decode the generated text
            extracted_text = processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        raw_text = extracted_text.strip()
        print(f"Extracted raw text: {raw_text}")  # Debug print
        
        # Convert the raw text to tabular format
        tabular_text = format_text_to_table(raw_text)
        
        return {"raw_text": raw_text, "generated_text": tabular_text}
    except Exception as e:
        print(f"Error in processing image with Qwen: {str(e)}")
        return {"error": f"Error processing image: {str(e)}"}

def format_text_to_table(text):
    """
    Format the extracted text into a tabular structure.
    This function preserves the exact text but ensures it's in CSV format.
    """
    try:
        # Split the text into lines
        lines = text.strip().split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            
            # If the line already has commas, keep it as is
            if ',' in line:
                formatted_lines.append(line)
            # If it has tabs, convert to CSV
            elif '\t' in line:
                formatted_lines.append(','.join(part.strip() for part in line.split('\t')))
            # If it has multiple spaces that might indicate columns
            elif '  ' in line:
                # Split by multiple spaces (2 or more)
                import re
                parts = re.split(r'\s{2,}', line)
                parts = [part.strip() for part in parts if part.strip()]
                formatted_lines.append(','.join(parts))
            else:
                # Single column - keep as is
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    except Exception as e:
        print(f"Error formatting text: {e}")
        return text  # Return original text if formatting fails

# Update the upload_file function to save CSV properly
@app.route('/upload', methods=['POST'])
def upload_file():
    global model_loaded
    
    if not model_loaded:
        # Try to load the model if it's not loaded yet
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
    
    # Prepare data for display and CSV file
    ocr_results = []
    for item in results:
        ocr_results.append({
            'Source File': item['filename'],
            'Raw Text': item.get('raw_text', ''),
            'Tabular Text': item.get('generated_text', '')
        })
    
    # Save results as CSV with timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    csv_path = f'results/ocr_results_{len(results)}_{timestamp}.csv'
    
    if ocr_results:
        # Create a DataFrame with the tabular text
        rows = []
        for result in results:
            tabular_text = result.get('generated_text', '')
            if tabular_text:
                # Split the tabular text into lines
                for line in tabular_text.split('\n'):
                    if line.strip():
                        # Create a row with the source file and the line content
                        row = {'Source File': result['filename']}
                        
                        # If the line is CSV-formatted, parse it
                        if ',' in line:
                            parts = line.split(',')
                            for i, part in enumerate(parts):
                                row[f'Column {i+1}'] = part.strip()
                        else:
                            row['Content'] = line.strip()
                        
                        rows.append(row)
        
        if rows:
            df = pd.DataFrame(rows)
            # Save with encoding that supports special characters
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            flash(f'Results saved to {csv_path}')
        else:
            flash('No valid data extracted from images')
    
    available_models = list_available_models()
    # Render the index template with the OCR results for display
    return render_template('index.html', 
                          model_loaded=model_loaded, 
                          current_model=current_model_name,
                          available_models=available_models,
                          ocr_results=ocr_results,
                          csv_download_path=csv_path)

# Simplify the list_available_models function since we're only using one model
def list_available_models():
    """
    Since we're only using the specific Qwen model, just return that model
    """
    return [current_model_name]

@app.route('/')
def index():
    global model_loaded, model_loading, model_ever_loaded
    
    # Always try to load the model if it's not loaded
    if not model_loaded and not model_loading:
        threading.Thread(target=load_model).start()
    
    return render_template('index.html', 
                          model_loaded=model_loaded, 
                          current_model=current_model_name)

# Keep only the load_model route for manual loading if needed
@app.route('/load_model', methods=['POST'])
def load_model_route():
    result = load_model()
    return {"status": result}

# Remove the change_model route and get_models route since we're only using one model
# @app.route('/change_model', methods=['POST'])
# def change_model():
#     ...

# @app.route('/models', methods=['GET'])
# def get_models():
#     ...

@app.route('/change_model', methods=['POST'])
def change_model():
    """
    API endpoint to change the current model
    """
    global current_model_name, model_loaded
    
    data = request.get_json()
    if not data or 'model_name' not in data:
        return {"status": "error", "message": "No model name provided"}
    
    new_model = data['model_name']
    
    # Check if model exists
    if not check_model_downloaded(new_model):
        return {"status": "error", "message": f"Model {new_model} not found locally"}
    
    # Reset model state
    current_model_name = new_model
    model_loaded = False
    
    # Start loading the new model
    threading.Thread(target=load_model).start()
    
    return {"status": "success", "message": f"Changing to model: {new_model}"}

@app.route('/results/<path:filename>')
def download_file(filename):
    """
    Route to download result files
    """
    return send_from_directory('results', filename, as_attachment=True)

if __name__ == '__main__':
    # Load the model when the app starts
    threading.Thread(target=load_model).start()
    app.run(debug=True)
