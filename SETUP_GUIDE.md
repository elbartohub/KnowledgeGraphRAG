
# Quick Setup Guide

## 🎯 Your Knowledge Graph RAG system is almost ready!


### ✅ What's Done:
- ✅ Virtual environment created (`venv`)
- ✅ All dependencies installed
- ✅ Project structure ready
- ✅ `.env` file created


### 🔧 Next Steps:


#### 1. Get Your Google API Key
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated API key


#### 2. Update Your .env File
Edit the `.env` file and replace `your_google_api_key_here` with your actual API key:


```bash
# Edit .env file
nano .env
# or
code .env
# or
open .env
```


Change this line:
```
GOOGLE_API_KEY=your_google_api_key_here
```
To:
```
GOOGLE_API_KEY=your_actual_api_key_here
```


#### 3. Start the Application
```bash
# Activate virtual environment
source venv/bin/activate

# Start the server
python app.py
```


#### 4. Open Your Browser
Navigate to: `http://localhost:5000`

---


## 🚀 You're All Set!

Once you add your Google API key, you can:
- Upload documents (PDF, DOCX, TXT)
- Ask questions about your documents
- Get AI-powered answers with source citations
- Visualize the knowledge graph of entities and relationships
- Manage keywords (whitelist/blacklist) and reload them


## 🎉 Features You'll Have:
- **Modern Responsive Web Interface** with drag-and-drop uploads
- **AI-Powered Q&A** using Google Gemini
- **Smart Document Search** with vector embeddings
- **Source Attribution** showing which documents and chunks were used
- **Multi-format Support** for PDF, DOCX, TXT
- **Knowledge Graph Visualization** of entities and relationships
- **Keyword Management** for whitelist/blacklist and statistics
- **Consistent Section Layout** for all main features
- **Keyboard Shortcuts** for fast question input

Enjoy your new Knowledge Graph RAG system! 🚀
