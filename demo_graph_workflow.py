#!/usr/bin/env python3
"""
Demonstration script showing how extracted keywords become graph nodes
and how jieba handles query segmentation in the DeepMind RAG system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import networkx as nx
import jieba
import json
from collections import defaultdict

def create_sample_knowledge_graph(keywords):
    """Create a sample knowledge graph from extracted keywords"""
    G = nx.Graph()
    
    # Add nodes (keywords become graph nodes)
    for keyword in keywords:
        G.add_node(keyword, type='concept')
    
    # Create sample relationships based on co-occurrence or semantic similarity
    # In a real implementation, this would be based on document analysis
    sample_relationships = [
        ('高壓氧療法', 'HBO', 'synonym'),
        ('糖尿病', '下肢潰瘍', 'causes'),
        ('心肌舒張功能', '心肌', 'part_of'),
        ('脈衝波多普勒超聲心動圖', '心血管功能', 'measures'),
        ('運動康復', '運動員', 'benefits'),
        ('DeepMind', 'AI', 'develops'),
        ('人工智慧', '醫學', 'applied_to')
    ]
    
    # Add edges with relationship types
    for source, target, rel_type in sample_relationships:
        if source in keywords and target in keywords:
            G.add_edge(source, target, relationship=rel_type)
    
    return G

def demonstrate_query_processing():
    """Demonstrate how queries are processed with jieba segmentation"""
    
    print("🔍 QUERY PROCESSING DEMONSTRATION")
    print("=" * 60)
    
    # Load custom keywords into jieba
    custom_keywords = [
        "HBO治療", "HBO 治療", "高壓氧治療", "多普勒超聲心動圖", 
        "DeepMind", "白開水", "人工智慧", "機器學習"
    ]
    
    for keyword in custom_keywords:
        jieba.add_word(keyword, freq=1000)
    
    # Test queries
    test_queries = [
        "什麼是HBO治療的原理？",
        "DeepMind開發了哪些AI技術？",
        "多普勒超聲心動圖如何檢測心臟功能？",
        "高壓氧治療對糖尿病患者有效嗎？",
        "白開水對身體有什麼好處？"
    ]
    
    print("📝 Query Segmentation Results:")
    for i, query in enumerate(test_queries, 1):
        segments = list(jieba.cut(query))
        print(f"  {i}. Query: {query}")
        print(f"     Segments: {' | '.join(segments)}")
        
        # Extract key terms for graph search
        key_terms = [seg for seg in segments if len(seg.strip()) > 1 and seg not in ['？', '的', '是', '了', '有', '在', '對', '如何', '什麼', '哪些']]
        print(f"     Key terms for graph search: {key_terms}")
        print()

def main():
    """Main demonstration function"""
    
    print("🎯 DEEPMIND RAG SYSTEM - KEYWORD EXTRACTION & GRAPH DEMO")
    print("=" * 70)
    
    # Sample extracted keywords (from Gemini API processing)
    sample_keywords = [
        '高壓氧療法', '糖尿病', '心肌舒張功能', '心肌', 'HBO',
        '脈衝波多普勒超聲心動圖', '組織多普勒超聲心動圖', '左心室', '二尖瓣', '右心室',
        '運動康復', '運動員', '康復醫學', '物理治療', '組織修復',
        'DeepMind', 'AlphaGo', 'AlphaFold', 'AI', '人工智慧',
        '機器學習', '深度學習', '神經網絡', '強化學習', '蛋白質折疊'
    ]
    
    print("📊 STEP 1: DOCUMENT PROCESSING")
    print("-" * 40)
    print("✅ Documents uploaded to /uploads folder")
    print("✅ Gemini API extracts top 50 keywords per document")
    print(f"✅ Example extracted keywords: {len(sample_keywords)} terms")
    print("   Sample: " + ", ".join(sample_keywords[:10]) + ", ...")
    
    # Create knowledge graph
    print(f"\n🕸️ STEP 2: KNOWLEDGE GRAPH CREATION")
    print("-" * 40)
    G = create_sample_knowledge_graph(sample_keywords)
    print(f"✅ Created knowledge graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    print(f"✅ Nodes represent extracted keywords")
    print(f"✅ Edges represent relationships between concepts")
    
    # Show graph structure
    print(f"\n📈 Graph Structure Sample:")
    for i, (node1, node2, data) in enumerate(list(G.edges(data=True))[:5]):
        rel_type = data.get('relationship', 'related')
        print(f"   {node1} --[{rel_type}]--> {node2}")
    
    print(f"\n🔍 STEP 3: QUERY PROCESSING")
    print("-" * 40)
    demonstrate_query_processing()
    
    print("💡 SYSTEM WORKFLOW SUMMARY:")
    print("=" * 70)
    print("1. 📄 Document Upload → Gemini extracts keywords → Graph nodes")
    print("2. 🔍 User Query → Jieba segments → Search graph → Retrieve docs")
    print("3. 🤖 Found docs → Gemini generates answer | No docs → Fallback message")
    print()
    print("🎯 KEY BENEFITS:")
    print("   ✅ Separates document analysis (Gemini) from query processing (jieba)")
    print("   ✅ Graph nodes focus on document concepts")
    print("   ✅ Jieba preserves important query terms")
    print("   ✅ Handles both Chinese and English terminology")
    
if __name__ == "__main__":
    main()
