# README for Keyword Elements Management

This folder contains text files for managing entity extraction keywords in the GraphRAG system.

## File Structure

### Whitelist Files (Keywords to ALWAYS extract)
- `whitelist_english_tech.txt` - English technical terms, programming languages, frameworks, etc.
- `whitelist_chinese_tech.txt` - Chinese technical terms, companies, universities, locations
- `whitelist_custom.txt` - Your custom domain-specific keywords

### Blacklist Files (Keywords to NEVER extract)
- `blacklist_english.txt` - Common English words to exclude (articles, prepositions, etc.)
- `blacklist_chinese.txt` - Common Chinese words to exclude (虚词, 连接词, etc.)

## How to Use

1. **Adding Keywords**: Simply add one keyword per line in the appropriate file
2. **Comments**: Lines starting with `#` are treated as comments and ignored
3. **Auto-reload**: The system automatically loads these files when the application starts
4. **Runtime Updates**: You can update keywords while the application is running

## File Format

```
# This is a comment
keyword1
keyword2
multi word keyword
中文关键词
```

## Tips

- **Be Specific**: Add specific technical terms rather than generic words
- **Domain Focus**: Create separate custom files for different domains if needed
- **Regular Updates**: Review and update keywords based on your document content
- **Test First**: Use the test scripts to verify keyword extraction before deployment

## Custom Files

You can create additional keyword files following this naming convention:
- `whitelist_[domain].txt` - for domain-specific whitelists
- `blacklist_[domain].txt` - for domain-specific blacklists

The system will automatically detect and load any `.txt` files in this directory.
