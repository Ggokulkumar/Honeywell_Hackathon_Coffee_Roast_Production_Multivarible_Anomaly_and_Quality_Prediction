"""
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
lightgbm==4.0.0
streamlit==1.25.0
plotly==5.15.0
seaborn==0.12.2
matplotlib==3.7.2
joblib==1.3.1
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    requirements = [
        "pandas==2.0.3",
        "numpy==1.24.3", 
        "scikit-learn==1.3.0",
        "lightgbm==4.0.0",
        "streamlit==1.25.0",
        "plotly==5.15.0",
        "seaborn==0.12.2",
        "matplotlib==3.7.2",
        "joblib==1.3.1"
    ]
    
    print("Installing required packages...")
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")

def create_project_structure():
    directories = [
        "data",
        "models", 
        "output",
        "logs"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def run_complete_setup():
    print("🚀 Setting up Roastmaster's AI Assistant...")
    print("=" * 50)
    
    install_requirements()
    print("\n" + "=" * 50)
    
    create_project_structure()
    print("\n" + "=" * 50)
    
    print("📊 Generating coffee roasting dataset...")
    try:
        exec(open('generate_dataset.py').read())
        print("✅ Dataset generated successfully!")
    except FileNotFoundError:
        print("⚠️  Please ensure generate_dataset.py is in the current directory")
    
    print("\n" + "=" * 50)
    
    print("🤖 Training machine learning models...")
    try:
        exec(open('train_model.py').read())
        print("✅ Models trained successfully!")
    except FileNotFoundError:
        print("⚠️  Please ensure train_model.py is in the current directory")
    
    print("\n" + "=" * 50)
    
    print("🎉 Setup completed successfully!")
    print("\nTo run the application:")
    print("streamlit run app.py")
    
    print("\nProject files:")
    print("- FNB_Coffee_Roast_Dataset.csv (Generated dataset)")
    print("- coffee_quality_model.pkl (Trained quality model)")
    print("- coffee_anomaly_model.pkl (Trained anomaly model)")
    print("- coffee_scaler.pkl (Feature scaler)")
    print("- coffee_preprocessors.pkl (Label encoders)")
    print("- model_evaluation_results.png (Model performance)")

if __name__ == "__main__":
    run_complete_setup()

import os
import subprocess
import sys

def quick_start():
    """Quick start script to run everything in sequence"""
    
    print("☕ Roastmaster's AI Assistant - Quick Start")
    print("=" * 50)
    
    steps = [
        ("📊 Generating Dataset", "python generate_dataset.py"),
        ("🤖 Training Models", "python train_model.py"), 
        ("🚀 Starting Dashboard", "streamlit run app.py")
    ]
    
    for step_name, command in steps:
        print(f"\n{step_name}...")
        print("-" * 30)
        
        if "streamlit" in command:
            print("🌐 Opening Streamlit dashboard in browser...")
            print("Use Ctrl+C to stop the application")
        
        try:
            if "streamlit" in command:
                subprocess.run(command.split())
            else:
                result = subprocess.run(command.split(), capture_output=True, text=True)
                if result.returncode == 0:
                    print("✅ Success!")
                else:
                    print(f"❌ Error: {result.stderr}")
                    break
        except FileNotFoundError:
            print(f"❌ Error: Could not find required files")
            print("Please ensure all Python files are in the current directory")
            break
        except KeyboardInterrupt:
            print("\n👋 Application stopped by user")
            break

if __name__ == "__main__":
    quick_start()

dockerfile_content = """
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python generate_dataset.py
RUN python train_model.py

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
"""

docker_compose_content = """
version: '3.8'

services:
  roastmaster:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./output:/app/output
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
"""

print("Setup files created!")
print("\nTo save these configurations:")
print("1. Save requirements as 'requirements.txt'")
print("2. Save setup script as 'setup.py'") 
print("3. Save quick start as 'run_project.py'")
print("4. Save Dockerfile content if using Docker")
print("5. Save docker-compose.yml content if using Docker Compose")