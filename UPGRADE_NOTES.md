
# Upgrade and Maintenance Notes

## Project: Knowledge Graph RAG


## Current Status
The Chinese text segmentation, keyword management, knowledge graph visualization, and Q&A system are fully functional and have been successfully tested. All major components are working as expected.


## Known Issues and Recommended Improvements


### 1. NumPy 2.0 Compatibility Warnings
**Issue**: The system shows warnings about `np.NaN` being removed in NumPy 2.0, originating from ChromaDB.
```
ERROR: `np.NaN` was removed in the NumPy 2.0 release. Use `np.nan` instead.
```


**Status**: This is a known upstream issue with ChromaDB version 0.4.15 and NumPy 2.x compatibility. It does not affect core functionality.


**Recommendations**:
- Monitor ChromaDB updates for NumPy 2.0 compatibility
- Consider pinning NumPy to version <2.0 if warnings become problematic
- The warnings do not affect functionality
- Check .gitignore for exclusion of logs and uploads_old/


### 2. Dependency Version Updates
**Current versions**:
- ChromaDB: 0.4.15 (update recommended)
- Flask: 2.3.3 (update recommended)
- Google Generative AI: 0.3.2 (update recommended)


**Recommended updates**:
```bash
pip install --upgrade chromadb flask google-generativeai
```


### 3. Performance Optimizations


#### ChromaDB Embedding Queue Errors
**Issue**: "pop from empty list" errors in ChromaDB embedding queue.
**Impact**: Minimal - internal ChromaDB queue management issues.
**Solution**: Will be resolved with ChromaDB updates.


#### Jieba Import Warning
**Issue**: pkg_resources deprecation warning from jieba.
**Impact**: Minimal - cosmetic warning only.
**Solution**: Monitor jieba updates or suppress specific warnings.


### 4. Potential Enhancements


#### UI/UX Improvements
- Add progress indicators for document processing
- Implement real-time keyword file editing in the web interface
- Add export functionality for extracted entities and graphs
- Implement batch document processing
- Improve responsive design and section layout


#### Advanced Features
- Add support for more document formats (markdown, rtf, etc.)
- Implement document similarity analysis
- Add entity relationship strength scoring
- Implement advanced graph filtering and clustering
- Add multilingual support for more languages


#### Performance Improvements
- Add caching for frequently accessed documents
- Implement background processing for large documents
- Add database connection pooling
- Optimize graph generation algorithms


## Maintenance Tasks


### Regular Updates
1. **Monthly**: Check for dependency updates
2. **Quarterly**: Review and update keyword lists in `keyword_elements/`
3. **As needed**: Monitor ChromaDB and NumPy compatibility
4. **Check .gitignore for proper exclusions**


### Backup Recommendations
- Backup `chroma_db/` directory regularly
- Version control `keyword_elements/` files
- Keep backups of uploaded documents
- Exclude `uploads_old/` and logs from version control


### Monitoring
- Monitor disk usage in `chroma_db/` and `uploads/` directories
- Watch for memory usage during large document processing
- Monitor API usage and rate limits for Google Gemini


## Testing Checklist


When making updates, test:
- [ ] Document upload and processing
- [ ] Chinese text segmentation with jieba
- [ ] Entity extraction with spaCy
- [ ] Keyword management (reload, stats, files)
- [ ] Graph generation and visualization
- [ ] Query and response generation
- [ ] API endpoints functionality
- [ ] Responsive UI and section layout


## Environment Setup Notes


### Required Environment Variables
```bash
GOOGLE_API_KEY=your_gemini_api_key_here
FLASK_SECRET_KEY=your_flask_secret_key_here
```


### Required System Dependencies
```bash
# For Chinese language support
python -m spacy download zh_core_web_sm
```


### Docker Considerations
If containerizing, ensure:
- Volume mounts for `chroma_db/` and `uploads/`
- Proper handling of Chinese fonts in browser
- Environment variable injection
- Model download during container build
- Exclude logs and uploads_old/ from image


## Future Roadmap


### Short-term (1-3 months)
- Update dependencies to resolve NumPy warnings
- Implement real-time keyword editing
- Add document similarity features
- Improve responsive UI and section layout


### Medium-term (3-6 months)
- Add multi-language support beyond English/Chinese
- Implement advanced analytics dashboard
- Add user authentication and document management


### Long-term (6+ months)
- Implement distributed processing for large datasets
- Add machine learning model fine-tuning capabilities
- Develop plugin architecture for custom processing
