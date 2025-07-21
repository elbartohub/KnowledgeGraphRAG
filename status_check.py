#!/usr/bin/env python3
"""
Application Health Check and Status Script
"""

import os
import sys
import requests
import time
import subprocess
import signal
import psutil
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    try:
        import flask
        import google.generativeai as genai
        import chromadb
        import sentence_transformers
        import jieba
        import spacy
        import networkx
        import numpy
        import sklearn
        print("✅ All required Python packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Run: pip install -r requirements.txt")
        return False

def check_spacy_model():
    """Check if the Chinese spaCy model is installed"""
    print("🔍 Checking spaCy Chinese model...")
    
    try:
        import spacy
        nlp = spacy.load("zh_core_web_sm")
        print("✅ Chinese spaCy model (zh_core_web_sm) is available")
        return True
    except OSError:
        print("❌ Chinese spaCy model not found")
        print("Run: python -m spacy download zh_core_web_sm")
        return False

def check_environment():
    """Check environment variables"""
    print("🔍 Checking environment variables...")
    
    google_api_key = os.getenv('GOOGLE_API_KEY')
    if not google_api_key:
        print("❌ GOOGLE_API_KEY not set")
        print("Set your Google API key in .env file or environment")
        return False
    else:
        print("✅ GOOGLE_API_KEY is set")
    
    secret_key = os.getenv('SECRET_KEY')
    if not secret_key:
        print("⚠️  SECRET_KEY not set (will use default)")
    else:
        print("✅ SECRET_KEY is set")
    
    return True

def check_application_running():
    """Check if the Flask application is running"""
    print("🔍 Checking if application is running...")
    
    try:
        response = requests.get('http://localhost:5001', timeout=5)
        if response.status_code == 200:
            print("✅ Application is running on http://localhost:5001")
            return True
        else:
            print(f"⚠️  Application responded with status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException:
        print("❌ Application is not running or not accessible")
        return False

def find_flask_processes():
    """Find running Flask processes"""
    flask_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            if 'app.py' in cmdline or ('python' in proc.info['name'] and 'flask' in cmdline):
                flask_processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return flask_processes

def start_application():
    """Start the Flask application"""
    print("🚀 Starting the application...")
    
    # Change to the application directory
    app_dir = Path(__file__).parent
    os.chdir(app_dir)
    
    # Start the application
    try:
        # Start in background
        process = subprocess.Popen([
            sys.executable, 'app.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print(f"📱 Application started with PID: {process.pid}")
        print("⏳ Waiting for application to start...")
        
        # Wait for the application to start (up to 30 seconds)
        for i in range(30):
            time.sleep(1)
            if check_application_running():
                print("✅ Application started successfully!")
                return True
        
        print("❌ Application failed to start within 30 seconds")
        return False
        
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        return False

def stop_application():
    """Stop running Flask applications"""
    print("🛑 Stopping Flask applications...")
    
    processes = find_flask_processes()
    if not processes:
        print("ℹ️  No Flask processes found")
        return
    
    for proc in processes:
        try:
            print(f"Stopping process {proc.pid}")
            proc.terminate()
            proc.wait(timeout=5)
        except (psutil.NoSuchProcess, psutil.TimeoutExpired):
            try:
                proc.kill()
            except psutil.NoSuchProcess:
                pass
    
    print("✅ Flask applications stopped")

def show_status():
    """Show comprehensive application status"""
    print("=" * 60)
    print("🔍 Gemini RAG Application Status Check")
    print("=" * 60)
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Check spaCy model
    spacy_ok = check_spacy_model()
    
    # Check environment
    env_ok = check_environment()
    
    # Check if running
    running = check_application_running()
    
    # Show Flask processes
    processes = find_flask_processes()
    if processes:
        print(f"\n📊 Found {len(processes)} Flask process(es):")
        for proc in processes:
            print(f"  PID: {proc.pid}, Command: {' '.join(proc.cmdline())}")
    
    # Check directory structure
    print("\n📁 Directory structure:")
    important_dirs = ['chroma_db', 'keyword_elements', 'templates', 'uploads', 'models']
    for dir_name in important_dirs:
        if os.path.exists(dir_name):
            print(f"  ✅ {dir_name}/")
        else:
            print(f"  ❌ {dir_name}/ (missing)")
    
    # Check important files
    print("\n📄 Important files:")
    important_files = ['app.py', 'config.py', 'requirements.txt', '.env']
    for file_name in important_files:
        if os.path.exists(file_name):
            print(f"  ✅ {file_name}")
        else:
            print(f"  ❌ {file_name} (missing)")
    
    print("\n" + "=" * 60)
    
    # Overall status
    if deps_ok and spacy_ok and env_ok:
        if running:
            print("🎉 Application is ready and running!")
            print("🌐 Access at: http://localhost:5001")
        else:
            print("⚠️  Application is ready but not running")
            print("Run: python app.py or use --start option")
    else:
        print("❌ Application has configuration issues")
        print("Fix the issues above before running")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Knowledge Graph RAG Application Management')
    parser.add_argument('--start', action='store_true', help='Start the application')
    parser.add_argument('--stop', action='store_true', help='Stop the application')
    parser.add_argument('--restart', action='store_true', help='Restart the application')
    parser.add_argument('--status', action='store_true', help='Show status (default)')
    
    args = parser.parse_args()
    
    if args.stop or args.restart:
        stop_application()
    

    if args.start or args.restart:
        deps_ok = check_dependencies()
        spacy_ok = check_spacy_model()
        env_ok = check_environment()
        gitignore_ok = check_gitignore()
        if deps_ok and spacy_ok and env_ok and gitignore_ok:
            start_application()
        else:
            print("❌ Cannot start due to missing requirements or .gitignore exclusions")
            sys.exit(1)

    if args.status or not any([args.start, args.stop, args.restart]):
        show_status()
        print("\nChecking .gitignore exclusions...")
        check_gitignore()
def check_gitignore():
    """Check .gitignore for uploads_old/ and app.log exclusions"""
    print("🔍 Checking .gitignore exclusions...")
    from pathlib import Path
    gitignore_path = Path('.gitignore')
    if not gitignore_path.exists():
        print("❌ .gitignore not found!")
        return False
    content = gitignore_path.read_text()
    ok = True
    for item in ['uploads_old/', 'app.log']:
        if item not in content:
            print(f"❌ Missing exclusion: {item}")
            ok = False
    if ok:
        print("✅ .gitignore exclusions are correct.")
    return ok

if __name__ == "__main__":
    main()
