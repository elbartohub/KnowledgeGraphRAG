# Gemini answer generation function
def generate_answer_with_gemini(query, relevant_chunks):
    """
    Generate an answer using Gemini API given a query and relevant document chunks.
    """
    try:
        # Prepare context for Gemini
        context = "\n\n".join([chunk['content'] for chunk in relevant_chunks])
        # Detect Traditional Chinese in query
        def contains_traditional_chinese(text):
            # Basic check for Traditional Chinese characters (subset of CJK)
            # This can be improved with a more robust check if needed
            trad_chars = set("的一是不了人我在有他這個們中來上大為和國地到以說時要就出會可也你對生能而子那得於著下自之年過發後作裡用道行所然家種事成方多經麼去法學如都同現當沒動面起看定天分還進好小部其些主樣理心她本前開但因只從想實日軍者意無力它與長把機十民第公此已工使情明性知全三又關點正業外將兩高間由問很最重並物手應戰向頭文體政美相見被利什二等產或新己制身果加西斯月話合回特代內信表化老給世位次度門任常先海通教兒原東聲提立及比員解水名真論處走義各入幾口認條平系氣題活爾更別打女變四神總何電數安少報才結反受目太量再感建務功持至市走價眾利".encode('utf-8').decode('utf-8'))
            return any(char in trad_chars for char in text)
        if contains_traditional_chinese(query):
            prompt = f"請用繁體中文回答所有問題，並根據提供的內容。\n\n內容：\n{context}\n\n問題：{query}\n\n答案："
        else:
            prompt = f"Answer the following question based on the provided context in traditional Chinese.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
        response = model.generate_content(prompt)
        # Defensive: handle response structure
        if hasattr(response, 'text') and response.text:
            return response.text.strip()
        elif hasattr(response, 'candidates') and response.candidates:
            # Fallback for candidate-based responses
            return response.candidates[0].text.strip()
        else:
            logger.warning("A.I. response did not contain text.")
            return "No answer generated."
    except Exception as e:
        logger.error(f"Error in generate_answer_with_gemini: {str(e)}")
        return f"Error generating answer: {str(e)}"
import os
import tempfile
from flask import Flask, request, render_template, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
import google.generativeai as genai
import chromadb
from sentence_transformers import SentenceTransformer
import PyPDF2
import docx
from dotenv import load_dotenv
import uuid
import logging
import config  # Import our configuration
import networkx as nx
import numpy as np
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import json
import jieba

# Fix tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')

# Configure upload folder
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = config.MAX_FILE_SIZE

# Ensure upload directory exists
os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GraphRAG imports (after logger is configured)
try:
    import spacy
    SPACY_AVAILABLE = True
    # Try to load the spacy model
    try:
        nlp = spacy.load(config.SPACY_MODEL)
        logger.info(f"Loaded spaCy model: {config.SPACY_MODEL}")
    except OSError:
        logger.warning(f"spaCy model '{config.SPACY_MODEL}' not found. Entity extraction will be limited.")
        # Fallback to basic entity extraction without spaCy
        SPACY_AVAILABLE = False
        nlp = None
except ImportError:
    logger.warning("spaCy not available. Entity extraction will be limited.")
    SPACY_AVAILABLE = False
    nlp = None

# Configure Gemini
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY not found in environment variables")
    raise ValueError("Please set GOOGLE_API_KEY in your .env file")

genai.configure(api_key=GOOGLE_API_KEY)

# Create models directory if it doesn't exist
os.makedirs(config.MODELS_PATH, exist_ok=True)
os.makedirs(config.CACHE_DIR, exist_ok=True)

def initialize_collection_with_embedding_check():
    """Initialize ChromaDB collection with embedding dimension check"""
    try:
        # Try to get existing collection
        collection = chroma_client.get_collection(name=config.COLLECTION_NAME)
        
        # Test if embedding dimensions match by trying a dummy embedding
        test_text = "test"
        test_embedding = embedding_model.encode([test_text])[0].tolist()
        embedding_dim = len(test_embedding)
        
        # Try to add a test embedding to check compatibility
        try:
            collection.add(
                embeddings=[test_embedding],
                documents=[test_text],
                ids=["dimension_test"]
            )
            # If successful, remove the test entry
            collection.delete(ids=["dimension_test"])
            logger.info(f"Using existing collection with embedding dimension: {embedding_dim}")
            return collection
            
        except Exception as dim_error:
            if "dimension" in str(dim_error).lower():
                logger.warning(f"Embedding dimension mismatch. Creating new collection...")
                # Delete the old collection
                chroma_client.delete_collection(name=config.COLLECTION_NAME)
                # Create new collection
                collection = chroma_client.create_collection(name=config.COLLECTION_NAME)
                logger.info(f"Created new collection with embedding dimension: {embedding_dim}")
                return collection
            else:
                raise dim_error
                
    except Exception as e:
        # Collection doesn't exist, create new one
        logger.info("Creating new collection...")
        collection = chroma_client.create_collection(name=config.COLLECTION_NAME)
        test_embedding = embedding_model.encode(["test"])[0].tolist()
        logger.info(f"Created collection with embedding dimension: {len(test_embedding)}")
        return collection

# Initialize components using config settings
embedding_model = SentenceTransformer(config.EMBEDDING_MODEL, cache_folder=config.CACHE_DIR)
chroma_client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
collection = initialize_collection_with_embedding_check()

def initialize_jieba_custom_dictionary():
    """Initialize jieba with custom dictionary from whitelist keywords"""
    if not config.USE_JIEBA_SEGMENTATION:
        return
    
    try:
        # Add whitelist keywords to jieba dictionary to ensure they are recognized as single units
        logger.info("Loading custom keywords into jieba dictionary...")
        
        # Load whitelist from config
        if hasattr(config, 'ENTITY_WHITELIST') and config.ENTITY_WHITELIST:
            keyword_count = 0
            for keyword in config.ENTITY_WHITELIST:
                if keyword and len(keyword.strip()) > 0:
                    # Add keyword to jieba dictionary
                    # Parameters: word, freq (frequency weight), tag (optional part-of-speech tag)
                    jieba.add_word(keyword.strip(), freq=1000)  # High frequency to ensure recognition
                    keyword_count += 1
            
            logger.info(f"Added {keyword_count} custom keywords to jieba dictionary")
        
        # Also load directly from whitelist files if they exist
        if hasattr(config, 'LOADED_WHITELIST') and config.LOADED_WHITELIST:
            file_keyword_count = 0
            for keyword in config.LOADED_WHITELIST:
                if keyword and len(keyword.strip()) > 0:
                    jieba.add_word(keyword.strip(), freq=1000)
                    file_keyword_count += 1
            
            logger.info(f"Added {file_keyword_count} file-based keywords to jieba dictionary")
        
        # Add some common technical terms that should be treated as single units
        technical_terms = [
            "DeepMind", "OpenAI", "ChatGPT", "GPT-4", "BERT", "Transformer",
            "人工智能", "機器學習", "深度學習", "神經網路", "自然語言處理", "計算機視覺",
            "多普勒超聲心動圖", "白開水"  # Add the specific terms from whitelist
        ]
        
        for term in technical_terms:
            jieba.add_word(term, freq=1000)
        
        logger.info(f"Jieba custom dictionary initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing jieba custom dictionary: {e}")

# Initialize jieba with custom dictionary
initialize_jieba_custom_dictionary()

# Allowed file extensions
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ====== 自動擴充白名單功能 ======
def auto_expand_whitelist(text):
    """自動擴充白名單：抽取高頻新詞並加入白名單與 custom 檔案"""
    try:
        import jieba
        words = list(jieba.cut(text))
        freq = Counter([w.strip() for w in words if len(w.strip()) > 1])
        top_keywords = [w for w, c in freq.most_common(20)]
        whitelist = set(config.ENTITY_WHITELIST)
        new_keywords = [w for w in top_keywords if w not in whitelist]
        if new_keywords:
            config.ENTITY_WHITELIST.extend(new_keywords)
            custom_path = 'keyword_elements/whitelist_custom.txt'
            with open(custom_path, 'a', encoding='utf-8') as f:
                for kw in new_keywords:
                    f.write(kw + '\n')
            logger.info(f"自動新增 {len(new_keywords)} 個新關鍵詞到白名單: {new_keywords}")
        else:
            logger.info("沒有新關鍵詞需要加入白名單")
    except Exception as e:
        logger.error(f"自動擴充白名單失敗: {e}")

def extract_text_from_file(file_path, filename):
    """Extract text from uploaded files"""
    try:
        if filename.lower().endswith('.pdf'):
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        
        elif filename.lower().endswith('.docx'):
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        
        elif filename.lower().endswith('.txt'):
            try:
                # 優先嘗試 utf-8
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read()
            except UnicodeDecodeError:
                try:
                    import chardet
                    with open(file_path, 'rb') as f:
                        raw = f.read()
                        result = chardet.detect(raw)
                        encoding = result['encoding'] if result['encoding'] else 'utf-8'
                        text = raw.decode(encoding, errors='ignore')
                        logger.info(f"Detected encoding for {filename}: {encoding}")
                        return text
                except Exception as ce:
                    logger.error(f"chardet failed to detect encoding for {filename}: {ce}")
                    return None
        
        else:
            return None
    except Exception as e:
        logger.error(f"Error extracting text from {filename}: {str(e)}")
        return None

def detect_language(text):
    """Detect if text is primarily Chinese or English with Traditional Chinese support"""
    # Count Chinese characters (CJK Unified Ideographs including Traditional Chinese)
    # Basic CJK: U+4E00-U+9FFF
    # Extended: U+3400-U+4DBF, U+20000-U+2A6DF, U+2A700-U+2B73F, U+2B740-U+2B81F, U+2B820-U+2CEAF
    chinese_chars = len(re.findall(r'[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]', text))
    total_chars = len(re.sub(r'\s', '', text))
    
    if total_chars == 0:
        return 'en'
    
    chinese_ratio = chinese_chars / total_chars
    return 'zh' if chinese_ratio > 0.3 else 'en'

def detect_chinese_variant(text):
    """Detect if Chinese text is primarily Simplified or Traditional"""
    if not config.ENABLE_TRADITIONAL_CHINESE:
        return 'simplified'
    
    # Common Traditional Chinese characters that differ from Simplified
    traditional_indicators = re.findall(r'[繁體漢語台灣香港澳門書學車門開關時間國際]', text)
    # Common Simplified Chinese characters
    simplified_indicators = re.findall(r'[简体汉语台湾香港澳门书学车门开关时间国际]', text)
    
    if len(traditional_indicators) > len(simplified_indicators):
        return 'traditional'
    else:
        return 'simplified'

def normalize_chinese_text(text, target_variant='simplified'):
    """Convert between Simplified and Traditional Chinese for better segmentation"""
    if not config.TRADITIONAL_CHINESE_CONVERSION:
        return text
    
    try:
        import opencc
        if target_variant == 'simplified':
            # Convert Traditional to Simplified
            converter = opencc.OpenCC('t2s')  # Traditional to Simplified
        else:
            # Convert Simplified to Traditional
            converter = opencc.OpenCC('s2t')  # Simplified to Traditional
        
        return converter.convert(text)
    except ImportError:
        logger.warning("OpenCC not available. Traditional Chinese conversion disabled.")
        return text
    except Exception as e:
        logger.warning(f"Chinese text conversion failed: {e}")
        return text

def get_sentence_boundaries(text, language='auto'):
    """Get sentence boundaries for proper text segmentation with enhanced Chinese support"""
    if language == 'auto':
        language = detect_language(text)
    
    if language == 'zh':
        # Chinese sentence delimiters - more comprehensive
        if config.CHINESE_PUNCTUATION:
            delims = '|'.join(re.escape(d) for d in config.CHINESE_SENTENCE_DELIMITERS)
            sentence_delims = f'[{delims}]+'
        else:
            sentence_delims = r'[。！？；\n]+'
    else:
        # English sentence delimiters
        sentence_delims = r'[.!?;\n]+'
    
    boundaries = []
    for match in re.finditer(sentence_delims, text):
        boundaries.append(match.end())
    
    return boundaries

def chunk_text_intelligent(text, chunk_size=None, overlap=None, language='auto'):
    """
    Intelligently split text into chunks with proper sentence/word boundaries
    Uses jieba for Chinese word segmentation and respects sentence boundaries
    """
    if chunk_size is None:
        chunk_size = config.CHUNK_SIZE
    if overlap is None:
        overlap = config.CHUNK_OVERLAP
    
    if language == 'auto':
        language = detect_language(text)
    
    # Get sentence boundaries
    sentence_boundaries = get_sentence_boundaries(text, language)
    
    # If no sentence boundaries found, use traditional chunking
    if not sentence_boundaries:
        return chunk_text_simple(text, chunk_size, overlap)
    
    chunks = []
    current_chunk = ""
    current_pos = 0
    
    # Add final boundary at end of text
    if sentence_boundaries[-1] < len(text):
        sentence_boundaries.append(len(text))
    
    for boundary in sentence_boundaries:
        sentence = text[current_pos:boundary].strip()
        
        # If adding this sentence would exceed chunk size
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            # Save current chunk
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            # Start new chunk with overlap from previous chunk
            if config.USE_JIEBA_SEGMENTATION and language == 'zh':
                # Enhance jieba for Traditional Chinese support
                text_for_segmentation = current_chunk
                
                # Convert Traditional Chinese to Simplified for better jieba segmentation
                if config.TRADITIONAL_CHINESE_CONVERSION:
                    chinese_variant = detect_chinese_variant(current_chunk)
                    if chinese_variant == 'traditional':
                        text_for_segmentation = normalize_chinese_text(current_chunk, 'simplified')
                
                # Use jieba for Chinese text overlap with configurable mode
                if config.JIEBA_MODE == 'search':
                    words = list(jieba.cut_for_search(text_for_segmentation))
                elif config.JIEBA_MODE == 'full':
                    words = list(jieba.cut(text_for_segmentation, cut_all=True, HMM=config.JIEBA_HMM))
                else:
                    words = list(jieba.cut(text_for_segmentation, HMM=config.JIEBA_HMM))
                
                overlap_words = words[-min(len(words)//4, overlap//10):] if words else []
                current_chunk = ''.join(overlap_words) + sentence
            else:
                # Use character-based overlap for English
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + sentence
        else:
            # Add sentence to current chunk
            current_chunk += sentence
        
        current_pos = boundary
    
    # Add final chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # If no chunks were created, fall back to simple chunking
    if not chunks:
        return chunk_text_simple(text, chunk_size, overlap)
    
    return [chunk for chunk in chunks if len(chunk.strip()) > 20]  # Filter very short chunks

def chunk_text_simple(text, chunk_size, overlap):
    """Simple character-based chunking (fallback method)"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
        if start >= len(text):
            break
    return chunks

def chunk_text(text, chunk_size=None, overlap=None):
    """
    Main text chunking function - chooses appropriate method based on configuration
    """
    if config.USE_JIEBA_SEGMENTATION and config.PRIMARY_LANGUAGE in ['zh', 'auto']:
        return chunk_text_intelligent(text, chunk_size, overlap, config.PRIMARY_LANGUAGE)
    else:
        return chunk_text_simple(text, chunk_size or config.CHUNK_SIZE, overlap or config.CHUNK_OVERLAP)

def add_document_to_vector_store(text, filename):
    """Add document chunks to ChromaDB"""
    try:
        chunks = chunk_text(text)
        doc_id = str(uuid.uuid4())
        
        for i, chunk in enumerate(chunks):
            if chunk.strip():  # Only add non-empty chunks
                embedding = embedding_model.encode([chunk])[0].tolist()
                chunk_id = f"{doc_id}_chunk_{i}"
                
                collection.add(
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{"filename": filename, "chunk_id": i, "doc_id": doc_id}],
                    ids=[chunk_id]
                )
        
        logger.info(f"Added {len(chunks)} chunks from {filename} to vector store")
        return True
    except Exception as e:
        logger.error(f"Error adding document to vector store: {str(e)}")
        return False

def retrieve_relevant_chunks(query, k=None):
    """Retrieve relevant document chunks for a query"""
    if k is None:
        k = config.RETRIEVAL_K
    try:
        query_embedding = embedding_model.encode([query])[0].tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        relevant_chunks = []
        if results['documents'] and results['documents'][0]:
            for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                # Skip entries where document or metadata is None
                if doc is None or metadata is None:
                    continue
                relevant_chunks.append({
                    'content': doc,
                    'filename': metadata.get('filename', 'Unknown'),
                    'chunk_id': metadata.get('chunk_id', 0)
                })
        logger.info(f"Query: '{query}' | Found {len(relevant_chunks)} relevant chunks.")
        for i, chunk in enumerate(relevant_chunks[:5]):
            logger.info(f"Chunk {i+1}: filename={chunk['filename']}, content_sample={chunk['content'][:100]}")
        return relevant_chunks
    except Exception as e:
        logger.error(f"Error retrieving relevant chunks: {str(e)}")
        return []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        
        # Save file to uploads folder
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Extract text
            text = extract_text_from_file(filepath, filename)
            logger.info(f"Extracted text from {filename}: {len(text) if text else 0} chars")
            if text:
                # Add to vector store
                result = add_document_to_vector_store(text, filename)
                if result:
                    flash(f'Successfully uploaded and processed {filename}')
                    logger.info(f"Successfully processed {filename}")
                else:
                    flash(f'Error processing {filename}')
                    logger.error(f"Error processing {filename}")
                # Log current document/chunk count after upload
                try:
                    results = collection.get()
                    logger.info(f"Current collection: {len(results['ids'])} chunks, {len(set([m.get('filename') for m in results['metadatas'] if m]))} documents")
                    for m in results['metadatas']:
                        logger.info(f"Chunk metadata: {m}")
                except Exception as e:
                    logger.error(f"Error logging collection state after upload: {str(e)}")
                # Remove file if needed
                if not config.KEEP_UPLOADED_FILES and os.path.exists(filepath):
                    os.unlink(filepath)
            else:
                flash(f'Could not extract text from {filename}')
                logger.error(f"Could not extract text from {filename}")
                if os.path.exists(filepath):
                    os.unlink(filepath)
        except Exception as e:
            flash(f'Error processing {filename}: {str(e)}')
            logger.error(f"Exception during upload of {filename}: {str(e)}")
            if os.path.exists(filepath):
                os.unlink(filepath)
                
    else:
        flash('Invalid file type. Please upload PDF, DOCX, or TXT files.')
    
    return redirect(url_for('index'))  # Already correct, no change needed

@app.route('/query', methods=['POST'])
def query_documents():
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        if not query:
            return jsonify({'error': 'Please provide a query'}), 400
        # Retrieve relevant chunks
        relevant_chunks = retrieve_relevant_chunks(query)
        logger.info(f"/query endpoint: {len(relevant_chunks)} chunks returned for query '{query}'")
        if not relevant_chunks:
            logger.warning(f"No relevant chunks found for query: '{query}'")
            return jsonify({
                'answer': 'I could not find any relevant information in the uploaded documents to answer your question.',
                'sources': []
            })
        # Generate answer with Gemini
        answer = generate_answer_with_gemini(query, relevant_chunks)
        # Prepare source information
        sources = [{'filename': chunk['filename'], 'content': chunk['content'][:200] + '...'
                   if len(chunk['content']) > 200 else chunk['content']}
                  for chunk in relevant_chunks]
        return jsonify({
            'answer': answer,
            'sources': sources
        })
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/clear', methods=['POST'])
def clear_documents():
    try:
        # Get all document IDs first
        results = collection.get()
        
        if results['ids']:
            # Delete all documents by their IDs
            collection.delete(ids=results['ids'])
            flash(f'Successfully cleared {len(results["ids"])} document chunks from the database')
        else:
            flash('No documents found to clear')
        
        return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"Error clearing documents: {str(e)}")
        flash(f'Error clearing documents: {str(e)}')
        return redirect(url_for('index'))

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'message': 'RAG application is running'})

@app.route('/documents')
def get_documents():
    """Get list of all documents in the collection"""
    try:
        # Get all documents from ChromaDB
        results = collection.get()
        
        # Process documents to group by filename
        documents = {}
        if results['metadatas']:
            for metadata in results['metadatas']:
                filename = metadata.get('filename', 'Unknown')
                doc_id = metadata.get('doc_id', 'unknown')
                
                if filename not in documents:
                    documents[filename] = {
                        'id': doc_id,
                        'filename': filename,
                        'chunks': 0,
                        'created_at': None
                    }
                documents[filename]['chunks'] += 1
        
        # Convert to list
        document_list = list(documents.values())
        
        return jsonify({
            'documents': document_list,
            'total': len(document_list)
        })
        
    except Exception as e:
        logger.error(f"Error getting documents: {str(e)}")
        return jsonify({'error': f'Error getting documents: {str(e)}'}), 500

@app.route('/documents/<doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    """Delete a specific document from the collection"""
    try:
        # Get all items with this doc_id
        results = collection.get(where={"doc_id": doc_id})
        
        if not results['ids']:
            return jsonify({'error': 'Document not found'}), 404
        
        # Delete all chunks for this document
        collection.delete(where={"doc_id": doc_id})
        
        logger.info(f"Deleted document with ID: {doc_id}")
        return jsonify({'message': 'Document deleted successfully'})
        
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        return jsonify({'error': f'Error deleting document: {str(e)}'}), 500

@app.route('/documents/<doc_id>/preview')
def preview_document(doc_id):
    """Get preview of document chunks"""
    try:
        # Get all chunks for this document
        results = collection.get(where={"doc_id": doc_id})
        
        if not results['documents']:
            return jsonify({'error': 'Document not found'}), 404
        
        # Get metadata for the first chunk to get filename
        filename = results['metadatas'][0].get('filename', 'Unknown') if results['metadatas'] else 'Unknown'
        
        return jsonify({
            'filename': filename,
            'doc_id': doc_id,
            'chunks': results['documents'],
            'total_chunks': len(results['documents'])
        })
        
    except Exception as e:
        logger.error(f"Error previewing document: {str(e)}")
        return jsonify({'error': f'Error previewing document: {str(e)}'}), 500

# ====== GraphRAG Functions ======

def is_valid_entity(entity_text, entity_type):
    """Check if an entity should be included based on filtering rules"""
    if not entity_text or len(entity_text.strip()) == 0:
        return False
    
    entity_text = entity_text.strip()
    entity_lower = entity_text.lower()
    
    # Check whitelist first - if it's whitelisted, always include
    if entity_text in config.ENTITY_WHITELIST or entity_lower in [w.lower() for w in config.ENTITY_WHITELIST]:
        return True
    
    # Check blacklist - if it's blacklisted, never include
    if entity_text in config.ENTITY_BLACKLIST or entity_lower in [w.lower() for w in config.ENTITY_BLACKLIST]:
        return False
    
    # Filter numbers-only entities
    if config.FILTER_NUMBERS_ONLY and entity_text.replace('.', '').replace(',', '').replace(' ', '').isdigit():
        return False
    
    # Filter very short entities (unless they're technical abbreviations)
    if config.FILTER_SHORT_ENTITIES and len(entity_text) < 3:
        # Allow common technical abbreviations
        technical_abbrevs = ['AI', 'ML', 'OS', 'DB', 'UI', 'UX', 'IT', 'IP', 'IO', 'CPU', 'GPU', 'RAM', 'SSD', 'HDD']
        if entity_text.upper() not in technical_abbrevs:
            return False
    
    # Filter pure cardinal numbers (like "1", "2", "100") unless they're part of a larger concept
    if entity_type in ['CARDINAL', 'ORDINAL'] and entity_text.replace(',', '').replace('.', '').isdigit():
        return False
    
    # Filter common stop words and generic terms
    if config.FILTER_COMMON_WORDS:
        common_words = {'system', 'method', 'approach', 'technique', 'solution', 'process', 'function', 'data', 'information', 'result', 'results', 'analysis', 'study', 'research', 'paper', 'article', 'work', 'problem', 'issue', 'case', 'example', 'model', 'framework', 'structure', 'design', 'development', 'implementation', 'application', 'use', 'usage', 'user', 'users', 'time', 'times', 'way', 'ways', 'part', 'parts', 'place', 'places', 'thing', 'things', 'people', 'person', 'group', 'groups', 'team', 'teams', 'company', 'companies', 'business', 'businesses', 'organization', 'organizations', 'service', 'services', 'product', 'products', 'tool', 'tools', 'technology', 'technologies', 'software', 'hardware', 'computer', 'computers', 'internet', 'web', 'website', 'websites', 'online', 'digital', 'electronic', 'automatic', 'manual', 'basic', 'advanced', 'simple', 'complex', 'new', 'old', 'good', 'bad', 'best', 'better', 'worse', 'worst', 'high', 'low', 'large', 'small', 'big', 'little', 'long', 'short', 'fast', 'slow', 'quick', 'easy', 'hard', 'difficult', 'important', 'significant', 'major', 'minor', 'main', 'primary', 'secondary', 'first', 'second', 'third', 'last', 'final', 'initial', 'original', 'current', 'recent', 'latest', 'modern', 'traditional', 'conventional', 'standard', 'normal', 'regular', 'usual', 'common', 'rare', 'unique', 'special', 'specific', 'general', 'overall', 'total', 'complete', 'full', 'partial', 'limited', 'unlimited', 'free', 'paid', 'open', 'closed', 'public', 'private', 'local', 'global', 'national', 'international', 'worldwide', 'universal'}
        if entity_lower in common_words:
            return False
    
    return True

def get_better_entity_label(entity_text, original_label):
    """Get a more appropriate label for whitelisted or special entities"""
    entity_lower = entity_text.lower()
    
    # Technical abbreviations and concepts
    tech_terms = {'ai', 'ml', 'dl', 'nlp', 'cv', 'gpu', 'cpu', 'api', 'sql', 'http', 'json', 'xml', 'html', 'css', 'js'}
    if entity_lower in tech_terms:
        return 'TECH'
    
    # Algorithms and methods
    algorithms = {'dfs', 'bfs', 'cnn', 'rnn', 'lstm', 'gan', 'svm', 'knn', 'pca'}
    if entity_lower in algorithms:
        return 'ALGORITHM'
    
    # Programming languages
    languages = {'python', 'java', 'c++', 'javascript', 'react', 'vue', 'angular', 'django', 'flask'}
    if entity_lower in languages:
        return 'PROGRAMMING_LANG'
    
    # Cloud and enterprise
    cloud_terms = {'aws', 'gcp', 'ibm', 'sap', 'crm', 'erp'}
    if entity_lower in cloud_terms:
        return 'TECH_SERVICE'
    
    # Keep original label if no better match
    return original_label

def extract_entities_basic(text):
    """Simplified entity extraction using specific patterns with deduplication, frequency control, and filtering"""
    entities = []
    seen_entities = set()  # To avoid duplicates
    entity_counts = {}  # Track frequency per entity
    
    # Specific, conservative patterns for different entity types
    patterns = {
        # Only capture well-known organizations
        'ORG': r'\b(?:Apple\s+Inc\.?|Microsoft\s+Corporation|Google\s+LLC|Amazon|Facebook|Meta|IBM|Oracle|Intel|AMD|NVIDIA|Tesla|SpaceX)\b',
        # Only capture well-known locations  
        'GPE': r'\b(?:China|USA|America|Japan|Korea|Taiwan|Hong Kong|Singapore|Beijing|Shanghai|Tokyo|Seoul|New York|London|Paris|California|Washington|Seattle|Boston|Silicon Valley)\b',
        # More specific technical terms (only well-established problems)
        'SPECIFIC_PROBLEM': r'\b(?:Kissing\s+number|Traveling\s+salesman|Knapsack|Vehicle\s+routing|Set\s+cover)\s+problem\b',
        # Only include truly classic algorithms, not proprietary names
        'CLASSIC_ALGORITHM': r'\b(?:PageRank|QuickSort|MergeSort|Dijkstra\'?s?\s*algorithm?)\b',
    }
    
    for entity_type, pattern in patterns.items():
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            # Clean up the match
            entity_text = match.group().strip()
            entity_key = entity_text.lower()  # Use lowercase for deduplication
            
            # Apply filtering rules
            if not is_valid_entity(entity_text, entity_type):
                continue
def add_document_to_vector_store(text, filename):
    """Add document chunks to ChromaDB"""
    try:
        chunks = chunk_text(text)
        doc_id = str(uuid.uuid4())
        logger.info(f"Preparing to add {len(chunks)} chunks for {filename} (doc_id={doc_id})")
        for i, chunk in enumerate(chunks):
            if chunk.strip():  # Only add non-empty chunks
                embedding = embedding_model.encode([chunk])[0].tolist()
                chunk_id = f"{doc_id}_chunk_{i}"
                metadata = {"filename": filename, "chunk_id": i, "doc_id": doc_id}
                collection.add(
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[metadata],
                    ids=[chunk_id]
                )
                logger.info(f"Appended chunk {chunk_id} for {filename}: {metadata}")
        logger.info(f"Added {len(chunks)} chunks from {filename} to vector store")
        return True
    except Exception as e:
        logger.error(f"Error adding document to vector store: {str(e)}")
        return False
        whitelist_keywords, _ = config.load_all_keywords()
        if whitelist_keywords:
            for keyword in whitelist_keywords:
                # Create case-insensitive pattern for exact word boundary matching
                keyword_pattern = r'\b' + re.escape(keyword) + r'\b'
                matches = re.finditer(keyword_pattern, text, re.IGNORECASE)
                
                for match in matches:
                    entity_text = match.group().strip()
                    entity_key = entity_text.lower()
                    
                    # Skip if already seen
                    if entity_key in seen_entities:
                        continue
                        
                    # Count occurrences 
                    if entity_key not in entity_counts:
                        entity_counts[entity_key] = 0
                    entity_counts[entity_key] += 1
                    
                    # Only add if not overused
                    if entity_counts[entity_key] <= 3:  # Allow more for whitelist terms
                        seen_entities.add(entity_key)
                        
                        # Determine entity type based on keyword characteristics
                        entity_label = 'WHITELISTED_TERM'
                        if any(org_term in keyword.lower() for org_term in ['inc', 'corporation', 'company', 'ltd', 'llc', 'technologies', 'tech']):
                            entity_label = 'ORG'
                        elif keyword in ['DeepMind', 'OpenAI', 'Google', 'Microsoft', 'Meta', 'Facebook', 'Tesla', 'NVIDIA', 'AMD', 'Intel', 'IBM', 'Amazon', 'Apple']:
                            entity_label = 'ORG'
                        elif keyword.isupper() and len(keyword) <= 10:  # Acronyms like AI, ML, GPU
                            entity_label = 'TECH_ACRONYM'
                        elif any(tech_term in keyword.lower() for tech_term in ['ai', 'ml', 'algorithm', 'neural', 'learning', 'intelligence']):
                            entity_label = 'TECH_TERM'
                        
                        entities.append({
                            'text': entity_text,
                            'label': entity_label,
                            'start': match.start(),
                            'end': match.end()
                        })
    except Exception as e:
        logger.warning(f"Error extracting whitelist entities: {e}")
    
    # Additional pattern for mathematical/scientific notation (more specific)
    math_patterns = {
        'THEOREM': r'\b(?:Fermat\'s|Euler\'s|Newton\'s|Pythagoras\'|Bayes\')\s+(?:theorem|law|principle)\b',
        'FAMOUS_CONSTANT': r'\b(?:Euler\'s\s+number|Golden\s+ratio|Pi|Planck\s+constant)\b',
    }
    
    for entity_type, pattern in math_patterns.items():
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            entity_text = match.group().strip()
            entity_key = entity_text.lower()
            
            # Apply filtering rules
            if not is_valid_entity(entity_text, entity_type):
                continue
            
            # Count and limit mathematical terms too
            if entity_key not in entity_counts:
                entity_counts[entity_key] = 0
            entity_counts[entity_key] += 1
            
            if entity_key not in seen_entities and entity_counts[entity_key] <= 2:
                seen_entities.add(entity_key)
                entities.append({
                    'text': entity_text,
                    'label': entity_type,
                    'start': match.start(),
                    'end': match.end()
                })
    
    return entities

def extract_entities_spacy(text):
    """Enhanced entity extraction using spaCy NLP model with custom patterns, frequency control, and filtering"""
    if not SPACY_AVAILABLE or nlp is None:
        return extract_entities_basic(text)
    
    try:
        doc = nlp(text)
        entities = []
        seen_entities = set()  # To avoid duplicates
        entity_counts = {}  # Track frequency per entity
        
        # Standard spaCy entities
        for ent in doc.ents:
            # Include more entity types from spaCy
            valid_types = ['PERSON', 'ORG', 'GPE', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE', 'PRODUCT', 'CARDINAL', 'ORDINAL']
            if ent.label_ in valid_types:
                entity_text = ent.text.strip()
                entity_key = entity_text.lower()
                
                # Apply filtering rules
                if not is_valid_entity(entity_text, ent.label_):
                    continue
                
                # Count occurrences
                if entity_key not in entity_counts:
                    entity_counts[entity_key] = 0
                entity_counts[entity_key] += 1
                
                # Only add if we haven't seen this entity and not overused
                if (entity_key not in seen_entities and 
                    entity_counts[entity_key] <= 3):  # Limit to 3 occurrences per document
                    
                    # Get better label for whitelisted items
                    better_label = get_better_entity_label(entity_text, ent.label_)
                    
                    seen_entities.add(entity_key)
                    entities.append({
                        'text': entity_text,
                        'label': better_label,
                        'start': ent.start_char,
                        'end': ent.end_char
                    })
        
        # Add custom concept extraction for technical terms (with its own deduplication)
        concept_entities = extract_technical_concepts(text)
        
        # Merge concept entities with spaCy entities, avoiding duplicates
        for concept_entity in concept_entities:
            concept_key = concept_entity['text'].lower()
            if concept_key not in seen_entities:
                seen_entities.add(concept_key)
                entities.append(concept_entity)
        
        return entities
    except Exception as e:
        logger.warning(f"Error in spaCy entity extraction: {e}")
        return extract_entities_basic(text)

def extract_technical_concepts(text):
    """Extract technical concepts and multi-word phrases with deduplication, context awareness, and filtering"""
    entities = []
    seen_entities = set()  # To avoid duplicates
    entity_counts = {}  # Track frequency per entity
    
    # Enhanced patterns for technical concepts with better context awareness
    technical_patterns = [
        # Specific patterns for well-known mathematical problems (more precise)
        (r'\b(?:Kissing\s+number|Traveling\s+salesman|Knapsack|Bin\s+packing|Graph\s+coloring|Satisfiability|Vehicle\s+routing|Set\s+cover|Vertex\s+cover)\s+problem\b', 'FAMOUS_PROBLEM'),
        # Mathematical objects with proper names
        (r'\b(?:Fibonacci|Euler|Gaussian|Poisson|Bernoulli|Pascal)\s+(?:number|sequence|distribution|triangle|constant)\b', 'MATH_OBJECT'),
        # Scientific methods and algorithms (more specific with context)
        (r'\b(?:genetic\s+algorithm|simulated\s+annealing|particle\s+swarm\s+optimization|ant\s+colony\s+optimization|gradient\s+descent|neural\s+network|random\s+forest|support\s+vector\s+machine)\b', 'ALGORITHM'),
        # Research areas and fields
        (r'\b(?:machine\s+learning|artificial\s+intelligence|computer\s+science|data\s+science|operations\s+research|combinatorial\s+optimization|deep\s+learning|reinforcement\s+learning)\b', 'FIELD'),
        # Mathematical theorems and principles (more specific)
        (r'\b(?:Fermat\'s|Euler\'s|Newton\'s|Gauss\'s|Bayes\')\s+(?:theorem|law|principle|conjecture|lemma)\b', 'THEOREM'),
        # Classic algorithm names (only well-established computer science algorithms)
        (r'\b(?:PageRank|QuickSort|MergeSort|Dijkstra\'?s?\s*(?:algorithm)?|Bellman-Ford|Floyd-Warshall)\s*(?:algorithm)?\b', 'CLASSIC_ALGORITHM'),
        # Programming and software concepts
        (r'\b(?:object-oriented\s+programming|functional\s+programming|design\s+pattern|software\s+architecture|data\s+structure)\b', 'PROGRAMMING_CONCEPT'),
    ]
    
    for pattern, entity_type in technical_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            entity_text = match.group().strip()
            entity_key = entity_text.lower()  # Use lowercase for deduplication
            
            # Apply filtering rules
            if not is_valid_entity(entity_text, entity_type):
                continue
            
            # Count occurrences
            if entity_key not in entity_counts:
                entity_counts[entity_key] = 0
            entity_counts[entity_key] += 1
            
            # Only add if we haven't seen this entity before, it's meaningful, and not overused
            if (entity_key not in seen_entities and 
                entity_counts[entity_key] <= 3):  # Limit to 3 occurrences per document
                
                seen_entities.add(entity_key)
                entities.append({
                    'text': entity_text,
                    'label': entity_type,
                    'start': match.start(),
                    'end': match.end()
                })
    
    return entities

def extract_chinese_entities(text):
    """Extract Chinese entities using jieba segmentation and Chinese-specific patterns with Traditional Chinese support"""
    if not config.USE_JIEBA_SEGMENTATION:
        return []
    
    entities = []
    seen_entities = set()
    entity_counts = {}
    
    # Handle Traditional Chinese for better segmentation
    text_for_segmentation = text
    original_text = text
    
    if config.TRADITIONAL_CHINESE_CONVERSION:
        chinese_variant = detect_chinese_variant(text)
        if chinese_variant == 'traditional':
            # Convert to Simplified for jieba segmentation, but keep original for entity text
            text_for_segmentation = normalize_chinese_text(text, 'simplified')
    
    # Use jieba to segment Chinese text with configurable mode
    if config.JIEBA_MODE == 'search':
        words = list(jieba.cut_for_search(text_for_segmentation))
    elif config.JIEBA_MODE == 'full':
        words = list(jieba.cut(text_for_segmentation, cut_all=True, HMM=config.JIEBA_HMM))
    else:
        words = list(jieba.cut(text_for_segmentation, HMM=config.JIEBA_HMM))
    
    # Enhanced Chinese technical terms patterns (supports both Simplified and Traditional)
    chinese_patterns = [
        # AI and ML terms (Traditional Chinese variants included)
        (r'人工智[能慧]|機器學習|深度學習|神經網[絡路]|卷積神經網[絡路]|循環神經網[絡路]|長短期記憶|生成對抗網[絡路]', 'TECH'),
        (r'自然[語语]言[處处]理|[計计]算機視覺|[語语]音[識识]別|圖像[識识]別|模式[識识]別|知[識识]圖譜', 'TECH'),
        # Computer science terms
        (r'算法|[數数]據[結结]構|操作系[統统]|[數数]據[庫库]|分布式系[統统]|雲[計计]算|大[數数]據|[區区][塊块][鏈链]', 'TECH'),
        (r'[軟软][體件]工程|系[統统][結架]構|[設设][計计]模式|面向[對对]象|函[數数]式[編程]程|微服[務务][結架]構', 'TECH'),
        # Development terms
        (r'前端[開开][發发]|後端[開开][發发]|全[棧栈][開开][發发]|移[動动][開开][發发]|Web[開开][發发]|API接口|[數数]據[庫库][優优]化', 'TECH'),
        # Security and networking
        (r'網[絡络]安全|信息安全|防火[牆墙]|加密算法|[數数][字位][簽签]名|網[絡络][協协][議议]|TCP/IP', 'TECH'),
        # Data science
        (r'[數数]據分析|[數数]據[挖挘]掘|商[業业]智能|[預预][測测]分析|[統统][計计][學学][習习]|回[歸归]分析|聚[類类]分析', 'TECH'),
        # Mathematical concepts
        (r'概率[論论]|[線线]性代[數数]|微[積积][分册]|離散[數数][學学]|圖[論论]|[數数][論论]|[組组]合[數数][學学]|最[優优]化理[論论]', 'MATH'),
        # Companies and organizations (Chinese) - including Traditional names
        (r'阿里巴巴|[騰腾][訊讯]|百度|[華华][為为]|小米|字[節节][跳跳][動动]|美[團团]|滴滴|京[東东]|網易', 'ORG'),
        (r'清华大学|北京大学|中科院|中国科学院|哈尔滨工业大学|上海交通大学', 'ORG'),
        # Locations (Chinese)
        (r'北京|上海|深圳|广州|杭州|南京|成都|西安|武汉|天津|重庆|苏州', 'GPE'),
        (r'中关村|浦东新区|南山区|高新区|经济技术开发区|软件园|科技园', 'GPE'),
        # Research areas
        (r'量子计算|生物信息学|计算生物学|医学影像|精准医疗|智能制造|工业互联网', 'FIELD'),
    ]
    
    # Apply Chinese patterns
    for pattern, entity_type in chinese_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            entity_text = match.group().strip()
            entity_key = entity_text.lower()
            
            # Basic validation for Chinese entities
            if len(entity_text) < 2 or len(entity_text) > 20:
                continue
                
            # Count occurrences
            if entity_key not in entity_counts:
                entity_counts[entity_key] = 0
            entity_counts[entity_key] += 1
            
            # Add if not seen and not overused
            if (entity_key not in seen_entities and 
                entity_counts[entity_key] <= 3):
                
                seen_entities.add(entity_key)
                entities.append({
                    'text': entity_text,
                    'label': entity_type,
                    'start': match.start(),
                    'end': match.end()
                })
    
    # Extract compound Chinese technical terms using jieba
    compound_terms = []
    for i in range(len(words) - 1):
        if len(words[i]) >= 2 and len(words[i+1]) >= 2:
            compound = words[i] + words[i+1]
            # Check if compound makes sense and is technical
            if (compound in config.ENTITY_WHITELIST or
                any(tech_term in compound for tech_term in ['算法', '系统', '网络', '数据', '智能', '计算', '分析', '技术', '工程', '开发'])):
                compound_terms.append({
                    'text': compound,
                    'label': 'COMPOUND_CONCEPT',
                    'start': text.find(compound),
                    'end': text.find(compound) + len(compound)
                })
    
    # Add compound terms if not duplicates
    for term in compound_terms:
        term_key = term['text'].lower()
        if (term_key not in seen_entities and len(term['text']) >= 4 and
            term['start'] != -1):  # Make sure the term was found in text
            seen_entities.add(term_key)
            entities.append(term)
    
    return entities

def extract_entities_enhanced(text):
    """Enhanced entity extraction that combines spaCy, basic patterns, and Chinese extraction"""
    logger.info(f"extract_entities_enhanced called. Text sample: '{text[:100]}'")
    all_entities = []
    seen_entities = set()
    # Get entities from spaCy (if available)
    if SPACY_AVAILABLE and nlp is not None:
        try:
            spacy_entities = extract_entities_spacy(text)
            if spacy_entities is None:
                logger.error("extract_entities_spacy returned None! Should return empty list.")
                spacy_entities = []
            for entity in spacy_entities:
                entity_key = entity['text'].lower()
                if entity_key not in seen_entities:
                    seen_entities.add(entity_key)
                    all_entities.append(entity)
        except Exception as e:
            logger.warning(f"Error in spaCy extraction: {e}")
    # Get entities from basic patterns
    basic_entities = extract_entities_basic(text)
    if basic_entities is None:
        logger.error("extract_entities_basic returned None! Should return empty list.")
        basic_entities = []
    for entity in basic_entities:
        entity_key = entity['text'].lower()
        if entity_key not in seen_entities:
            seen_entities.add(entity_key)
            all_entities.append(entity)
    # Get Chinese entities if enabled and text contains Chinese
    if config.USE_JIEBA_SEGMENTATION and detect_language(text) == 'zh':
        try:
            chinese_entities = extract_chinese_entities(text)
            if chinese_entities is None:
                logger.error("extract_chinese_entities returned None! Should return empty list.")
                chinese_entities = []
            for entity in chinese_entities:
                entity_key = entity['text'].lower()
                if entity_key not in seen_entities:
                    seen_entities.add(entity_key)
                    all_entities.append(entity)
        except Exception as e:
            logger.warning(f"Error in Chinese entity extraction: {e}")
    # Apply final filtering and whitelist/blacklist
    filtered_entities = []
    for entity in all_entities:
        if is_valid_entity(entity['text'], entity['label']):
            entity_text = entity['text'].strip()
            if entity_text in config.ENTITY_BLACKLIST:
                continue
            if entity_text in config.ENTITY_WHITELIST:
                entity['label'] = get_better_entity_label(entity_text, entity['label'])
            filtered_entities.append(entity)
    if filtered_entities is None:
        logger.error("extract_entities_enhanced filtered_entities is None! Should return empty list instead.")
        filtered_entities = []
    logger.info(f"extract_entities_enhanced returning {len(filtered_entities)} entities.")
    return filtered_entities[:config.MAX_ENTITIES_PER_DOCUMENT] if filtered_entities else []  # Limit total entities

def build_knowledge_graph():
    """Build knowledge graph from all documents in the collection"""
    try:
        # Get all documents from ChromaDB
        results = collection.get()
        
        # Defensive: handle NoneType for documents and metadatas
        if results is None or 'documents' not in results or results['documents'] is None:
            logger.error("ChromaDB returned None for documents. Returning empty graph.")
            return {'nodes': [], 'links': [], 'error': 'No documents found'}
        if 'metadatas' not in results or results['metadatas'] is None:
            logger.error("ChromaDB returned None for metadatas. Returning empty graph.")
            return {'nodes': [], 'links': [], 'error': 'No metadatas found'}
        if not results['documents']:
            logger.info("No documents present in ChromaDB. Returning empty graph.")
            return {'nodes': [], 'links': [], 'error': 'No documents present'}
        
        # Group chunks by document to avoid duplicates
        documents_by_id = {}
        documents_by_filename = {}
        
        logger.info("Grouping document chunks...")
        
        # Defensive: ensure both lists are iterable and of same length
        documents_list = results['documents'] if isinstance(results['documents'], list) else []
        metadatas_list = results['metadatas'] if isinstance(results['metadatas'], list) else []
        if len(documents_list) != len(metadatas_list):
            logger.error(f"Mismatch in documents ({len(documents_list)}) and metadatas ({len(metadatas_list)}) length. Returning empty graph.")
            return {'nodes': [], 'links': [], 'error': 'Mismatch in documents and metadatas length'}
        for i, (doc_text, metadata) in enumerate(zip(documents_list, metadatas_list)):
            filename = metadata.get('filename', f'Document_{i}')
            doc_id = metadata.get('doc_id', f'doc_{i}')
            
            # Group by doc_id first, then by filename as fallback
            if doc_id not in documents_by_id:
                documents_by_id[doc_id] = {
                    'filename': filename,
                    'chunks': [],
                    'combined_text': ''
                }
            
            documents_by_id[doc_id]['chunks'].append(doc_text)
            documents_by_id[doc_id]['combined_text'] += ' ' + doc_text
        
        # If we still have duplicates by filename, merge them
        for doc_id, doc_data in documents_by_id.items():
            filename = doc_data['filename']
            if filename in documents_by_filename:
                # Merge with existing document
                documents_by_filename[filename]['chunks'].extend(doc_data['chunks'])
                documents_by_filename[filename]['combined_text'] += ' ' + doc_data['combined_text']
                documents_by_filename[filename]['doc_ids'].append(doc_id)
            else:
                documents_by_filename[filename] = {
                    'filename': filename,
                    'chunks': doc_data['chunks'],
                    'combined_text': doc_data['combined_text'],
                    'doc_ids': [doc_id]
                }
        
        logger.info(f"Grouped into {len(documents_by_filename)} unique documents")
        
        # Extract entities from combined document texts
        document_entities = {}
        all_entities = Counter()
        
        logger.info("Extracting entities from documents...")
        
        for filename, doc_data in documents_by_filename.items():
            # Use the first doc_id as the primary identifier
            primary_doc_id = doc_data['doc_ids'][0]
            combined_text = doc_data['combined_text']
            
            # Extract entities from the combined text using enhanced extraction
            entities = extract_entities_enhanced(combined_text)
            if entities is None:
                logger.error(f"Entity extraction returned None for document '{filename}'. Skipping entity processing for this document.")
                entities = []
            logger.info(f"Entities type: {type(entities)}, value: {entities}")
            # Filter entities by frequency and type
            doc_entity_list = []
            for entity in entities:
                entity_key = f"{entity['text']}_{entity['label']}"
                all_entities[entity_key] += 1
                doc_entity_list.append(entity)
            # Limit entities per document
            doc_entity_list = doc_entity_list[:config.MAX_ENTITIES_PER_DOCUMENT]
            document_entities[primary_doc_id] = {
                'filename': filename,
                'entities': doc_entity_list,
                'text': combined_text,
                'all_doc_ids': doc_data['doc_ids']
            }
        
        logger.info(f"Extracted {len(all_entities)} unique entities")
        
        # Filter entities by minimum frequency and maximum frequency (to avoid spam)
        frequent_entities = {
            entity: count for entity, count in all_entities.items() 
            if config.MIN_ENTITY_FREQUENCY <= count <= config.MAX_ENTITY_FREQUENCY
        }
        
        logger.info(f"Filtered to {len(frequent_entities)} frequent entities (min: {config.MIN_ENTITY_FREQUENCY}, max: {config.MAX_ENTITY_FREQUENCY})")
        
        # Build graph
        G = nx.Graph()
        
        # Add document nodes (one per unique document)
        for doc_id, doc_data in document_entities.items():
            G.add_node(doc_id, 
                      type='document', 
                      name=doc_data['filename'], 
                      size=10,
                      color='#10b981')
        
        # Add entity nodes and connections
        entity_to_docs = defaultdict(set)
        
        for doc_id, doc_data in document_entities.items():
            for entity in doc_data['entities']:
                entity_key = f"{entity['text']}_{entity['label']}"
                
                if entity_key in frequent_entities:
                    entity_id = f"entity_{entity_key}"
                    
                    # Add entity node if not exists
                    if not G.has_node(entity_id):
                        G.add_node(entity_id,
                                  type='entity',
                                  name=entity['text'],
                                  label=entity['label'],
                                  size=5,
                                  color='#6366f1')
                    
                    # Connect entity to document
                    G.add_edge(doc_id, entity_id, weight=1, type='contains')
                    entity_to_docs[entity_id].add(doc_id)
        
        # Add entity-entity relationships based on co-occurrence
        entity_list = [n for n in G.nodes() if G.nodes[n]['type'] == 'entity']
        
        for i, entity1 in enumerate(entity_list):
            for entity2 in entity_list[i+1:]:
                # Calculate co-occurrence strength
                docs1 = entity_to_docs[entity1]
                docs2 = entity_to_docs[entity2]
                
                intersection = len(docs1.intersection(docs2))
                union = len(docs1.union(docs2))
                
                if union > 0:
                    strength = intersection / union
                    
                    if strength >= config.MIN_RELATIONSHIP_STRENGTH:
                        G.add_edge(entity1, entity2, 
                                  weight=strength, 
                                  type='co_occurs')
        
        # Limit graph size for performance
        if len(G.nodes()) > config.MAX_GRAPH_NODES:
            # Keep most connected nodes
            node_degrees = dict(G.degree())
            top_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:config.MAX_GRAPH_NODES]
            nodes_to_keep = set([node[0] for node in top_nodes])
            G = G.subgraph(nodes_to_keep).copy()
        
        if len(G.edges()) > config.MAX_GRAPH_EDGES:
            # Keep strongest edges
            edge_weights = [(u, v, d['weight']) for u, v, d in G.edges(data=True) if 'weight' in d]
            edge_weights.sort(key=lambda x: x[2], reverse=True)
            edges_to_keep = edge_weights[:config.MAX_GRAPH_EDGES]
            
            G_filtered = nx.Graph()
            G_filtered.add_nodes_from(G.nodes(data=True))
            for u, v, weight in edges_to_keep:
                G_filtered.add_edge(u, v, weight=weight)
            G = G_filtered
        
        # Convert to format expected by D3.js
        nodes = []
        for node_id, node_data in G.nodes(data=True):
            nodes.append({
                'id': node_id,
                'name': node_data['name'],
                'type': node_data['type'],
                'label': node_data.get('label', ''),
                'size': node_data.get('size', 5),
                'color': node_data.get('color', '#6366f1'),
                'connections': G.degree(node_id)
            })
        
        links = []
        for u, v, edge_data in G.edges(data=True):
            links.append({
                'source': u,
                'target': v,
                'weight': edge_data.get('weight', 1),
                'type': edge_data.get('type', 'related')
            })
        
        logger.info(f"Built knowledge graph with {len(nodes)} nodes and {len(links)} links")
        
        return {
            'nodes': nodes,
            'links': links,
            'stats': {
                'total_documents': len([n for n in nodes if n['type'] == 'document']),
                'total_entities': len([n for n in nodes if n['type'] == 'entity']),
                'total_relationships': len(links)
            }
        }
        
    except Exception as e:
        logger.error(f"Error building knowledge graph: {str(e)}")
        return {'nodes': [], 'links': [], 'error': str(e)}

@app.route('/graph-data')
def get_graph_data():
    """Endpoint to get knowledge graph data"""
    try:
        graph_data = build_knowledge_graph()
        return jsonify(graph_data)
    except Exception as e:
        logger.error(f"Error getting graph data: {str(e)}")
        return jsonify({'error': f'Error generating graph data: {str(e)}'}), 500

@app.route('/graph-data/<doc_id>')
def get_document_graph_data(doc_id):
    """Endpoint to get knowledge graph data for a specific document"""
    try:
        # Get all graph data first
        full_graph_data = build_knowledge_graph()
        
        if not full_graph_data['nodes']:
            return jsonify({'nodes': [], 'links': [], 'stats': {'total_documents': 0, 'total_entities': 0, 'total_relationships': 0}})
        
        # Find the specific document node
        selected_doc_node = None
        for node in full_graph_data['nodes']:
            if node['id'] == doc_id and node['type'] == 'document':
                selected_doc_node = node
                break
        
        if not selected_doc_node:
            return jsonify({'error': 'Document not found'}), 404
        
        # Find all entities connected to this document
        connected_entity_ids = set()
        for link in full_graph_data['links']:
            if link['source'] == doc_id:
                connected_entity_ids.add(link['target'])
            elif link['target'] == doc_id:
                connected_entity_ids.add(link['source'])
        
        # Create filtered nodes (document + connected entities)
        filtered_nodes = [selected_doc_node]
        for node in full_graph_data['nodes']:
            if node['type'] == 'entity' and node['id'] in connected_entity_ids:
                filtered_nodes.append(node)
        
        # Create filtered links (only links involving the selected document)
        filtered_links = []
        for link in full_graph_data['links']:
            if (link['source'] == doc_id or link['target'] == doc_id) or \
               (link['source'] in connected_entity_ids and link['target'] in connected_entity_ids):
                filtered_links.append(link)
        
        # Calculate stats
        stats = {
            'total_documents': 1,
            'total_entities': len([n for n in filtered_nodes if n['type'] == 'entity']),
            'total_relationships': len(filtered_links)
        }
        
        return jsonify({
            'nodes': filtered_nodes,
            'links': filtered_links,
            'stats': stats
        })
        
    except Exception as e:
        logger.error(f"Error getting document graph data: {str(e)}")
        return jsonify({'error': f'Error generating document graph data: {str(e)}'}), 500
# ====== End GraphRAG Functions ======

# ====== Configuration Endpoints ======

@app.route('/config/entity-extraction')
def get_entity_extraction_config():
    """Get current entity extraction configuration"""
    try:
        return jsonify({
            'whitelist': config.ENTITY_WHITELIST,
            'blacklist': config.ENTITY_BLACKLIST,
            'filter_numbers_only': config.FILTER_NUMBERS_ONLY,
            'filter_short_entities': config.FILTER_SHORT_ENTITIES,
            'filter_common_words': config.FILTER_COMMON_WORDS,
            'min_entity_frequency': config.MIN_ENTITY_FREQUENCY,
            'max_entity_frequency': config.MAX_ENTITY_FREQUENCY,
            'max_entities_per_document': config.MAX_ENTITIES_PER_DOCUMENT
        })
    except Exception as e:
        logger.error(f"Error getting entity extraction config: {str(e)}")
        return jsonify({'error': f'Error getting configuration: {str(e)}'}), 500

@app.route('/config/entity-extraction/whitelist', methods=['POST'])
def update_entity_whitelist():
    """Update entity extraction whitelist"""
    try:
        data = request.get_json()
        new_whitelist = data.get('whitelist', [])
        
        # Validate input
        if not isinstance(new_whitelist, list):
            return jsonify({'error': 'Whitelist must be a list'}), 400
        
        # Update the whitelist (in memory for this session)
        config.ENTITY_WHITELIST = new_whitelist
        
        return jsonify({
            'message': 'Whitelist updated successfully',
            'whitelist': config.ENTITY_WHITELIST
        })
        
    except Exception as e:
        logger.error(f"Error updating entity whitelist: {str(e)}")
        return jsonify({'error': f'Error updating whitelist: {str(e)}'}), 500

@app.route('/config/entity-extraction/blacklist', methods=['POST'])
def update_entity_blacklist():
    """Update entity extraction blacklist"""
    try:
        data = request.get_json()
        new_blacklist = data.get('blacklist', [])
        
        # Validate input
        if not isinstance(new_blacklist, list):
            return jsonify({'error': 'Blacklist must be a list'}), 400
        
        # Update the blacklist (in memory for this session)
        config.ENTITY_BLACKLIST = new_blacklist
        
        return jsonify({
            'message': 'Blacklist updated successfully',
            'blacklist': config.ENTITY_BLACKLIST
        })
        
    except Exception as e:
        logger.error(f"Error updating entity blacklist: {str(e)}")
        return jsonify({'error': f'Error updating blacklist: {str(e)}'}), 500

@app.route('/config/keywords/reload', methods=['POST'])
def reload_keywords():
    """Reload keywords from keyword_elements directory"""
    try:
        # Reload keywords from files
        new_whitelist, new_blacklist = config.load_all_keywords()
        
        # Update the config module's keyword lists
        config.LOADED_WHITELIST = new_whitelist
        config.LOADED_BLACKLIST = new_blacklist
        config.ENTITY_WHITELIST = config.ENTITY_WHITELIST_BASE + new_whitelist
        config.ENTITY_BLACKLIST = config.ENTITY_BLACKLIST_BASE + new_blacklist
        
        return jsonify({
            'success': True,
            'message': 'Keywords reloaded successfully',
            'whitelist_count': len(config.ENTITY_WHITELIST),
            'blacklist_count': len(config.ENTITY_BLACKLIST),
            'loaded_whitelist_count': len(new_whitelist),
            'loaded_blacklist_count': len(new_blacklist)
        })
    except Exception as e:
        logger.error(f"Error reloading keywords: {str(e)}")
        return jsonify({'error': f'Error reloading keywords: {str(e)}'}), 500

@app.route('/config/keywords/stats')
def get_keyword_stats():
    """Get statistics about current keywords"""
    try:
        return jsonify({
            'whitelist': {
                'total_count': len(config.ENTITY_WHITELIST),
                'base_count': len(config.ENTITY_WHITELIST_BASE),
                'loaded_count': len(config.LOADED_WHITELIST),
                'sample': config.ENTITY_WHITELIST[:10]  # First 10 as sample
            },
            'blacklist': {
                'total_count': len(config.ENTITY_BLACKLIST),
                'base_count': len(config.ENTITY_BLACKLIST_BASE),
                'loaded_count': len(config.LOADED_BLACKLIST),
                'sample': config.ENTITY_BLACKLIST[:10]  # First 10 as sample
            },
            'settings': {
                'auto_load_keywords': config.AUTO_LOAD_KEYWORDS,
                'keyword_elements_dir': config.KEYWORD_ELEMENTS_DIR,
                'keyword_file_encoding': config.KEYWORD_FILE_ENCODING
            }
        })
    except Exception as e:
        logger.error(f"Error getting keyword stats: {str(e)}")
        return jsonify({'error': f'Error getting keyword stats: {str(e)}'}), 500

@app.route('/config/keywords/files')
def get_keyword_files():
    """Get list of available keyword files and their contents"""
    try:
        import os
        import glob
        
        files_info = []
        
        if os.path.exists(config.KEYWORD_ELEMENTS_DIR):
            # Get all txt files
            all_files = glob.glob(os.path.join(config.KEYWORD_ELEMENTS_DIR, "*.txt"))
            
            for filepath in all_files:
                filename = os.path.basename(filepath)
                file_type = 'whitelist' if filename.startswith('whitelist_') else 'blacklist' if filename.startswith('blacklist_') else 'other'
                
                # Get file stats
                try:
                    keywords = config.load_keywords_from_file(filepath)
                    keyword_count = len(keywords)
                    file_size = os.path.getsize(filepath)
                    
                    files_info.append({
                        'filename': filename,
                        'filepath': filepath,
                        'type': file_type,
                        'keyword_count': keyword_count,
                        'file_size': file_size,
                        'sample_keywords': keywords[:5]  # First 5 keywords as sample
                    })
                except Exception as e:
                    files_info.append({
                        'filename': filename,
                        'filepath': filepath,
                        'type': file_type,
                        'error': str(e)
                    })
        
        return jsonify({
            'keyword_files': files_info,
            'total_files': len(files_info),
            'directory': config.KEYWORD_ELEMENTS_DIR
        })
    except Exception as e:
        logger.error(f"Error getting keyword files: {str(e)}")
        return jsonify({'error': f'Error getting keyword files: {str(e)}'}), 500

# ====== End Configuration Endpoints ======

if __name__ == '__main__':
    app.run(debug=config.DEBUG, host=config.HOST, port=config.PORT)
