import os
import subprocess
import sys

def setup_environment():
    print("Setting up virtual environment...")
    
    # Create virtual environment
    subprocess.run(["python", "-m", "venv", "venv"], check=True)
    
    # Determine the path to the activate script based on OS
    if os.name == 'nt':  # Windows
        activate_script = os.path.join("venv", "Scripts", "activate")
        activate_cmd = f"{activate_script} && "
    else:  # Unix/Linux/Mac
        activate_script = os.path.join("venv", "bin", "activate")
        activate_cmd = f"source {activate_script} && "
    
    # Install requirements
    print("Installing dependencies...")
    if os.name == 'nt':  # Windows
        subprocess.run(f"{activate_cmd} pip install -r requirements.txt", shell=True, check=True)
    else:
        subprocess.run(f"{activate_cmd} pip install -r requirements.txt", shell=True, check=True, executable='/bin/bash')
    
    print("Setup complete! You can now run the application.")
    print("\nTo activate the virtual environment:")
    if os.name == 'nt':  # Windows
        print(f"    {activate_script}")
    else:
        print(f"    source {activate_script}")
    
    print("\nTo run the application:")
    if os.name == 'nt':  # Windows
        print("    python app.py")
    else:
        print("    python app.py")

if __name__ == "__main__":
    setup_environment()