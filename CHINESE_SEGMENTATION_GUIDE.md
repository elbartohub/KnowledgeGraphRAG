# Chinese Text Segmentation and Keyword Management Guide

## Overview

This enhanced Gemini RAG system now includes advanced Chinese text segmentation and flexible keyword management capabilities. You can customize entity extraction by managing keywords through text files and a web interface.

## Key Improvements

### 1. Chinese Text Segmentation

**Enhanced Features:**
- **Jieba Integration**: Uses jieba for proper Chinese word segmentation
- **Language Detection**: Automatically detects Chinese vs English text
- **Smart Chunking**: Respects sentence boundaries and word boundaries
- **Configurable Modes**: Multiple jieba segmentation modes (default, search, full)
- **Chinese Punctuation**: Proper handling of Chinese punctuation marks

**Configuration Settings:**
```python
# In config.py
PRIMARY_LANGUAGE = 'zh'  # or 'en' or 'auto'
USE_JIEBA_SEGMENTATION = True
JIEBA_MODE = 'default'  # 'default', 'search', 'full'
JIEBA_HMM = True
CHINESE_PUNCTUATION = True
CHINESE_SENTENCE_DELIMITERS = ['。', '！', '？', '；', '：', '…', '——']
```

### 2. Flexible Keyword Management

**File-Based System:**
- Keywords stored in `keyword_elements/` directory
- Separate files for different categories
- Support for comments and easy editing
- Auto-reload capability

**File Structure:**
```
keyword_elements/
├── whitelist_english_tech.txt    # English technical terms
├── whitelist_chinese_tech.txt    # Chinese technical terms  
├── whitelist_custom.txt          # Your custom keywords
├── blacklist_english.txt         # English words to exclude
├── blacklist_chinese.txt         # Chinese words to exclude
└── README.md                     # Documentation
```

## Using the Keyword Management System

### 1. Managing Keywords via Files

**Adding Keywords:**
1. Open the appropriate file in `keyword_elements/`
2. Add one keyword per line
3. Use `#` for comments
4. Save the file
5. Reload keywords via web interface or restart application

**Example file content:**
```
# Technical terms
人工智能
机器学习
深度学习
# Companies
谷歌
百度
腾讯
```

### 2. Web Interface Controls

**Available Actions:**
- **Reload Keywords**: Refresh keywords from files without restarting
- **View Statistics**: See current keyword counts and samples
- **Test Extraction**: Test entity extraction on sample text

**Accessing the Interface:**
1. Start the application: `python app.py`
2. Open http://localhost:5001
3. Scroll down to "Keyword Management" section

### 3. API Endpoints

**Reload Keywords:**
```bash
curl -X POST http://localhost:5001/config/keywords/reload
```

**Get Statistics:**
```bash
curl http://localhost:5001/config/keywords/stats
```

**Test Extraction:**
```bash
curl -X POST http://localhost:5001/config/keywords/test \
  -H "Content-Type: application/json" \
  -d '{"text": "这是一个测试文本，包含人工智能和机器学习技术。"}'
```

## Configuration Guide

### 1. Basic Chinese Support Setup

**Install Dependencies:**
```bash
pip install jieba
python -m spacy download zh_core_web_sm
```

**Update config.py:**
```python
PRIMARY_LANGUAGE = 'zh'
USE_JIEBA_SEGMENTATION = True
SPACY_MODEL = 'zh_core_web_sm'
CHUNK_SIZE = 400  # Smaller for Chinese
CHUNK_OVERLAP = 80
```

### 2. Advanced Configuration

**Jieba Modes:**
- `'default'`: Standard segmentation (recommended)
- `'search'`: More fine-grained for search applications
- `'full'`: All possible word combinations

**Chunk Size Guidelines:**
- **Chinese**: 300-500 characters (denser information)
- **English**: 800-1200 characters
- **Mixed**: 400-600 characters

**Entity Extraction Settings:**
```python
MIN_ENTITY_FREQUENCY = 1
MAX_ENTITY_FREQUENCY = 10
MAX_ENTITIES_PER_DOCUMENT = 100
FILTER_SHORT_ENTITIES = True
FILTER_NUMBERS_ONLY = True
```

## Best Practices

### 1. Keyword Management

**Whitelist Strategy:**
- Add specific technical terms, company names, product names
- Include both English and Chinese equivalents
- Focus on domain-specific vocabulary
- Regular review and updates

**Blacklist Strategy:**
- Common words that add no value (的, 是, the, and)
- Generic technical terms (系统, method, approach)
- Numbers and dates that aren't meaningful
- Overly broad categories

### 2. Text Processing

**For Chinese Documents:**
- Use shorter chunk sizes (300-500 characters)
- Enable jieba segmentation
- Set primary language to 'zh' or 'auto'
- Include Chinese technical terms in whitelist

**For Mixed Language Documents:**
- Use 'auto' language detection
- Include keywords in both languages
- Test extraction on sample content
- Adjust chunk sizes based on content density

### 3. Performance Optimization

**Large Document Collections:**
- Limit max entities per document
- Use appropriate chunk sizes
- Regular keyword cleanup
- Monitor extraction performance

**Graph Performance:**
- Limit graph nodes (MAX_GRAPH_NODES = 200)
- Limit graph edges (MAX_GRAPH_EDGES = 300)
- Filter low-strength relationships

## Troubleshooting

### Common Issues

**Issue**: Chinese text not segmenting properly
**Solution**: 
1. Verify jieba is installed: `pip install jieba`
2. Check `USE_JIEBA_SEGMENTATION = True`
3. Ensure `PRIMARY_LANGUAGE = 'zh'` or `'auto'`

**Issue**: Too many irrelevant entities
**Solution**:
1. Add generic terms to blacklist
2. Increase `MIN_ENTITY_FREQUENCY`
3. Decrease `MAX_ENTITIES_PER_DOCUMENT`
4. Review and clean whitelist

**Issue**: Missing important entities
**Solution**:
1. Add specific terms to whitelist
2. Check if terms are in blacklist
3. Decrease `MIN_ENTITY_FREQUENCY`
4. Test extraction on sample text

**Issue**: spaCy model not found
**Solution**:
```bash
python -m spacy download zh_core_web_sm
```

### Testing and Validation

**Web Testing:**
1. Use the "Test Entity Extraction" section in the web interface
2. Try both Chinese and English text samples
3. Verify entity types and labels

## Example Usage

### 1. Adding Domain-Specific Keywords

**For Finance Domain:**
```bash
# Edit keyword_elements/whitelist_custom.txt
echo "区块链" >> keyword_elements/whitelist_custom.txt
echo "数字货币" >> keyword_elements/whitelist_custom.txt
echo "金融科技" >> keyword_elements/whitelist_custom.txt
```

**For Medical Domain:**
```bash
# Create new domain file
cat > keyword_elements/whitelist_medical.txt << EOF
# Medical terms
医疗诊断
治疗方案
临床试验
药物研发
医学影像
精准医疗
EOF
```

### 2. Testing and Validation

**Python Testing:**
```python
import config
from app import extract_entities_enhanced, detect_language

# Test Chinese text
chinese_text = "腾讯和阿里巴巴在人工智能领域的竞争日趋激烈。"
entities = extract_entities_enhanced(chinese_text)
print(f"Detected language: {detect_language(chinese_text)}")
print(f"Entities: {entities}")
```

**Web Interface Testing:**
1. Navigate to keyword management section
2. Enter test text: "Google DeepMind开发了AlphaGo人工智能系统"
3. Click "Test" to see extraction results
4. Verify both English and Chinese entities are extracted

This enhanced system provides powerful, flexible keyword management while maintaining excellent performance for both Chinese and English content processing.
