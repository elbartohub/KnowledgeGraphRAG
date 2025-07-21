#!/bin/bash

# Gemini RAG Startup Script

echo "🚀 Starting Gemini RAG Application"
echo "=================================="

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating from template..."
    cp .env.example .env
    echo "📝 Please edit .env file and add your GOOGLE_API_KEY"
    echo "🔑 Get your API key from: https://makersuite.google.com/app/apikey"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "📦 Installing requirements..."
pip install -r requirements.txt

# Start the application directly
echo "🚀 Starting the Gemini RAG application..."
python app.py
