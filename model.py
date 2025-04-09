import os
import argparse
from transformers import AutoProcessor, VisionEncoderDecoderModel

def download_and_save_model(model_name, output_dir="models"):
    """
    Download and save a Hugging Face model locally
    
    Args:
        model_name (str): The name of the model on Hugging Face
        output_dir (str): Directory to save the model
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"Downloading model: {model_name}")
    
    # Create the output directory if it doesn't exist
    Current_dir = os.getcwd()
    if not os.path.exists(Current_dir + "/" + output_dir):
        os.makedirs(Current_dir + "/" + output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Download and save the processor
        print("Downloading processor...")
        processor = AutoProcessor.from_pretrained(model_name, cache_dir=output_dir)
        
        # Download and save the model
        print("Downloading model...")
        # Changed from AutoModelForCausalLM to VisionEncoderDecoderModel
        model = VisionEncoderDecoderModel.from_pretrained(model_name, cache_dir=output_dir)
        
        print(f"Model '{model_name}' downloaded and saved successfully to '{output_dir}'")
        return True
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False

def list_available_models():
    """
    List some recommended OCR models for handwritten text
    """
    models = [
        "microsoft/trocr-base-handwritten",
        "microsoft/trocr-small-handwritten",
        "microsoft/trocr-large-handwritten",
        "facebook/nougat-base",
        "google/pix2struct-docvqa-base"
    ]
    
    print("Recommended OCR models for handwritten text:")
    for model in models:
        print(f"- {model}")
    
    print("\nYou can also explore more models at: https://huggingface.co/models?pipeline_tag=image-to-text")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and save a Hugging Face model locally")
    parser.add_argument("--model", type=str, default="microsoft/trocr-base-handwritten",
                        help="Name of the model on Hugging Face")
    parser.add_argument("--output", type=str, default="models",
                        help="Directory to save the model")
    parser.add_argument("--list", action="store_true",
                        help="List recommended OCR models")
    
    args = parser.parse_args()
    
    if args.list:
        list_available_models()
    else:
        download_and_save_model(args.model, args.output)