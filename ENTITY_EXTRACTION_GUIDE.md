# Entity Extraction Configuration

This document explains how to customize entity extraction in the GraphRAG system.

## Features

### 1. Automatic Filtering
- **Numbers Only**: Pure numbers (like "100", "2024") are automatically filtered out
- **Short Entities**: Very short entities (< 3 characters) are filtered unless whitelisted
- **Common Words**: Generic stop words and common terms are filtered out
- **Frequency Control**: Prevents any entity from appearing too many times

### 2. Whitelist System
Entities in the whitelist are always extracted, regardless of other filtering rules.

**Default Whitelist includes:**
- Technical abbreviations: AI, ML, GPU, CPU, API, SQL, HTTP, JSON, XML, HTML, CSS, JS
- Algorithms: DFS, BFS, CNN, RNN, LSTM, GAN, SVM, KNN, PCA  
- Programming languages: Python, Java, C++, JavaScript, React, Vue, Angular, Django, Flask
- Cloud services: AWS, GCP, IBM, SAP, CRM, ERP
- Math constants: Pi, E, φ, Phi, NP, P

### 3. Blacklist System
Entities in the blacklist are never extracted.

**Default Blacklist includes:**
- Common stop words: the, and, or, but, in, on, at, to, for, of, with, by
- Generic terms: system, method, approach, technique, solution, process, function
- Common numbers: 2023, 2024, 2025, 100, 1000, first, second, third

## API Endpoints

### Get Current Configuration
```bash
GET /config/entity-extraction
```

### Update Whitelist
```bash
POST /config/entity-extraction/whitelist
Content-Type: application/json

{
  "whitelist": ["AI", "ML", "Your-Custom-Term", "Another-Term"]
}
```

### Update Blacklist
```bash
POST /config/entity-extraction/blacklist
Content-Type: application/json

{
  "blacklist": ["spam-term", "noise-word", "unwanted-entity"]
}
```

## Entity Types

The system assigns better labels to recognized entities:

- **TECH**: AI, ML, GPU, CPU, API, etc.
- **ALGORITHM**: CNN, RNN, DFS, BFS, etc.
- **PROGRAMMING_LANG**: Python, JavaScript, Java, etc.
- **TECH_SERVICE**: AWS, GCP, IBM, SAP, etc.
- **FAMOUS_PROBLEM**: Kissing number problem, Traveling salesman problem, etc.
- **FIELD**: Machine learning, artificial intelligence, etc.
- **ORG**: Companies and organizations
- **PERSON**: People's names
- **GPE**: Geographic locations

## Configuration Settings

In `config.py`, you can adjust:

```python
# Entity filtering settings
FILTER_NUMBERS_ONLY = True          # Filter pure numbers
FILTER_SHORT_ENTITIES = True        # Filter short entities  
FILTER_COMMON_WORDS = True          # Filter generic terms
MIN_ENTITY_FREQUENCY = 1            # Minimum frequency to include
MAX_ENTITY_FREQUENCY = 10           # Maximum frequency to prevent spam
MAX_ENTITIES_PER_DOCUMENT = 100     # Max entities per document
```

## Customization Tips

1. **For Technical Documents**: Add domain-specific terms to the whitelist
2. **For Academic Papers**: Include author names, university names, specific algorithms
3. **For Business Documents**: Include company names, product names, industry terms
4. **To Reduce Noise**: Add common but meaningless terms to the blacklist

## Example Usage

```python
# Add custom terms for a machine learning document
custom_whitelist = [
    "Transformer", "BERT", "GPT", "ResNet", "VGG", 
    "PyTorch", "TensorFlow", "Scikit-learn",
    "CUDA", "cuDNN", "OpenCV"
]

# Update via API
import requests
requests.post('http://localhost:5001/config/entity-extraction/whitelist', 
              json={'whitelist': custom_whitelist})
```

This ensures that important technical terms are always captured in your knowledge graph, while keeping noise to a minimum.
