"""
Configuration settings for the Gemini RAG application
"""

# Document processing settings
CHUNK_SIZE = 400  # Smaller chunks for Chinese text (characters are denser, need shorter chunks)
CHUNK_OVERLAP = 80  # Proportional overlap for Chinese (20% of chunk size)

# Retrieval settings
RETRIEVAL_K = 5  # Number of relevant chunks to retrieve for each query

# File upload settings
UPLOAD_FOLDER = "./uploads"  # Directory to store uploaded files
KEEP_UPLOADED_FILES = True  # Whether to keep uploaded files after processing
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB max file size
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

# Embedding model settings
# Options for different languages:
# - 'all-MiniLM-L6-v2': Good for English, limited Chinese support
# - 'paraphrase-multilingual-MiniLM-L12-v2': Better multilingual including Chinese
# - 'distiluse-base-multilingual-cased': Good multilingual model
# - 'sentence-transformers/LaBSE': Excellent for Chinese (109 languages)
# - 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2': Very good for Chinese

# Current active embedding model:
EMBEDDING_MODEL = 'sentence-transformers/LaBSE'  # Best for Chinese and multilingual

# Alternative options (uncomment to use):
# EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # Faster, English-focused
# EMBEDDING_MODEL = 'distiluse-base-multilingual-cased'  # Good multilingual balance
# EMBEDDING_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'  # Good multilingual performance

# Note: TAIDE models are generative models, not embedding models
# For Chinese RAG systems, LaBSE or multilingual models work best

# Gemini model settings
# Options for generation models:
# - 'gemini-1.5-flash': Fast Google model
# - 'gemini-pro': More capable Google model
# - 'taide': Could integrate TAIDE for Chinese generation (requires custom implementation)
GEMINI_MODEL = 'gemini-1.5-flash'  # Gemini model to use

# ChromaDB settings
CHROMA_DB_PATH = "./chroma_db"  # Path to store ChromaDB data
COLLECTION_NAME = "documents"  # Name of the ChromaDB collection

# Local model storage settings
MODELS_PATH = "./models"  # Local directory to store embedding models
CACHE_DIR = "./models/cache"  # Cache directory for model downloads

# Flask settings
DEBUG = True  # Set to False in production
HOST = '0.0.0.0'
PORT = 5001

# Logging settings
LOG_LEVEL = 'INFO'

# Language settings
PRIMARY_LANGUAGE = 'zh'  # Primary language: 'zh' for Chinese, 'en' for English, 'auto' for auto-detection
USE_JIEBA_SEGMENTATION = True  # Use jieba for Chinese word segmentation (install: pip install jieba)

# Chinese text processing settings (supports both Simplified and Traditional Chinese)
CHINESE_PUNCTUATION = True  # Handle Chinese punctuation marks for better text chunking
CHINESE_SENTENCE_DELIMITERS = ['。', '！', '？', '；', '：', '…', '——']
CHINESE_COMMA_DELIMITERS = ['，', '、', '；']
JIEBA_MODE = 'default'  # Options: 'default', 'search', 'full' - jieba cutting mode
JIEBA_HMM = True  # Use Hidden Markov Model for unknown words in jieba

# Traditional Chinese support settings
ENABLE_TRADITIONAL_CHINESE = True  # Enable Traditional Chinese text processing
TRADITIONAL_CHINESE_CONVERSION = True  # Enable Simplified/Traditional conversion for better segmentation
ENABLE_ENTITY_EXTRACTION = True  # Enable/disable entity extraction
SPACY_MODEL = 'zh_core_web_sm'  # spaCy model for Chinese text - install with: python -m spacy download zh_core_web_sm
# Alternative models: 'en_core_web_sm' for English, 'xx_ent_wiki_sm' for multilingual

# Entity extraction settings
MIN_ENTITY_FREQUENCY = 1  # Reduced to capture more concepts (was 2)
MAX_ENTITY_FREQUENCY = 10  # Maximum frequency to prevent spam entities
MAX_ENTITIES_PER_DOCUMENT = 100  # Increased to capture more entities (was 50)
ENTITY_TYPES = ['PERSON', 'ORG', 'GPE', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE', 'PRODUCT', 'CARDINAL', 'ORDINAL', 'CONCEPT', 'TECHNICAL_TERM', 'COMPOUND_CONCEPT', 'MATH_OBJECT', 'METHOD', 'FAMOUS_PROBLEM', 'FIELD', 'MATH_TERM', 'FORMULA']  # Enhanced entity types

# Entity filtering settings
FILTER_NUMBERS_ONLY = True  # Filter out entities that are numbers only (e.g., "123", "2024")
FILTER_SHORT_ENTITIES = True  # Filter out very short entities (less than 3 characters)
FILTER_COMMON_WORDS = True  # Filter out common stop words and generic terms

# Keyword management settings
KEYWORD_ELEMENTS_DIR = "./keyword_elements"  # Directory containing keyword files
AUTO_LOAD_KEYWORDS = True  # Automatically load keywords from text files
KEYWORD_FILE_ENCODING = 'utf-8'  # Encoding for keyword files

def load_keywords_from_file(filepath):
    """Load keywords from a text file, ignoring comments and empty lines"""
    import os
    keywords = []
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding=KEYWORD_FILE_ENCODING) as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if line and not line.startswith('#'):
                        keywords.append(line)
        except Exception as e:
            print(f"Warning: Could not load keywords from {filepath}: {e}")
    return keywords

def load_all_keywords():
    """Load all keywords from the keyword_elements directory"""
    import os
    import glob
    
    whitelist = []
    blacklist = []
    
    if not AUTO_LOAD_KEYWORDS:
        return whitelist, blacklist
    
    if not os.path.exists(KEYWORD_ELEMENTS_DIR):
        print(f"Warning: Keyword directory {KEYWORD_ELEMENTS_DIR} not found")
        return whitelist, blacklist
    
    # Load whitelist files
    whitelist_files = glob.glob(os.path.join(KEYWORD_ELEMENTS_DIR, "whitelist_*.txt"))
    for filepath in whitelist_files:
        keywords = load_keywords_from_file(filepath)
        whitelist.extend(keywords)
        print(f"Loaded {len(keywords)} whitelist keywords from {os.path.basename(filepath)}")
    
    # Load blacklist files
    blacklist_files = glob.glob(os.path.join(KEYWORD_ELEMENTS_DIR, "blacklist_*.txt"))
    for filepath in blacklist_files:
        keywords = load_keywords_from_file(filepath)
        blacklist.extend(keywords)
        print(f"Loaded {len(keywords)} blacklist keywords from {os.path.basename(filepath)}")
    
    # Remove duplicates while preserving order
    whitelist = list(dict.fromkeys(whitelist))
    blacklist = list(dict.fromkeys(blacklist))
    
    print(f"Total loaded: {len(whitelist)} whitelist, {len(blacklist)} blacklist keywords")
    return whitelist, blacklist

# Load keywords from files
if AUTO_LOAD_KEYWORDS:
    try:
        LOADED_WHITELIST, LOADED_BLACKLIST = load_all_keywords()
    except Exception as e:
        print(f"Warning: Could not load keywords from files: {e}")
        LOADED_WHITELIST, LOADED_BLACKLIST = [], []
else:
    LOADED_WHITELIST, LOADED_BLACKLIST = [], []

# Entity whitelist - entities that should always be extracted regardless of other filters
# This combines hardcoded keywords with those loaded from files
ENTITY_WHITELIST_BASE = [
    # Mathematical and scientific concepts (English) - core essentials
    'AI', 'ML', 'GPU', 'CPU', 'API', 'SQL', 'HTTP', 'JSON', 'XML', 'HTML', 'CSS', 'JS',
    # Famous algorithms and methods (English) - core essentials
    'DFS', 'BFS', 'CNN', 'RNN', 'LSTM', 'GAN', 'SVM', 'KNN', 'PCA',
    # Programming languages and frameworks (English) - core essentials
    'Python', 'Java', 'C++', 'JavaScript', 'React', 'Vue', 'Angular', 'Django', 'Flask',
    # Cloud and tech companies (short forms) - core essentials
    'AWS', 'GCP', 'IBM', 'SAP', 'CRM', 'ERP',
    # Mathematical constants and objects (English) - core essentials
    'Pi', 'E', 'φ', 'Phi', 'NP', 'P',
    # Chinese technical terms (core essentials)
    '人工智能', '机器学习', '深度学习', '神经网络', '算法', '数据库', '云计算', '大数据',
    '区块链', '物联网', '虚拟现实', '增强现实', '自然语言处理', '计算机视觉', '数据分析',
    '软件工程', '系统架构', '前端开发', '后端开发', '移动开发', '网络安全', '数据挖掘',
]

# Combine base whitelist with loaded keywords
ENTITY_WHITELIST = ENTITY_WHITELIST_BASE + LOADED_WHITELIST

# Entity blacklist - entities that should never be extracted
# This combines hardcoded keywords with those loaded from files
ENTITY_BLACKLIST_BASE = [
    # Common generic words that might be picked up (English) - core essentials
    'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
    # Generic tech terms that are too broad (English) - core essentials
    'system', 'method', 'approach', 'technique', 'solution', 'process', 'function',
    # Common numbers and dates that add no value - core essentials
    '2023', '2024', '2025', '100', '1000', 'first', 'second', 'third',
    # Common Chinese generic words and characters - core essentials
    '的', '是', '在', '有', '个', '这', '那', '了', '和', '对', '也', '就', '可以', '能够', '可能',
    '一个', '一些', '很多', '非常', '比较', '比如', '例如', '通过', '由于', '因为', '所以', '但是',
    '然后', '现在', '今天', '明天', '昨天', '时候', '时间', '地方', '问题', '情况', '方面', '方式',
    # Generic Chinese tech terms that are too broad - core essentials
    '系统', '方法', '技术', '方案', '过程', '功能', '效果', '结果', '原因', '目标', '任务', '工作',
    '研究', '开发', '设计', '实现', '应用', '使用', '操作', '管理', '控制', '处理', '分析', '计算',
]

# Combine base blacklist with loaded keywords
ENTITY_BLACKLIST = ENTITY_BLACKLIST_BASE + LOADED_BLACKLIST

# Graph generation settings
MIN_RELATIONSHIP_STRENGTH = 0.1  # Minimum co-occurrence strength for relationships
MAX_GRAPH_NODES = 200  # Maximum nodes in the graph for performance
MAX_GRAPH_EDGES = 300  # Maximum edges in the graph for performance

# Graph visualization settings
GRAPH_LAYOUT_ITERATIONS = 100  # Number of iterations for graph layout
