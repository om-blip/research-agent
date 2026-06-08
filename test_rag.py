from rag.chunker import chunk_text

chunks = chunk_text('This is a test sentence about AI research. ' * 50, 'https://test.com')
print(f'Chunker works: {len(chunks)} chunks created')
print(f'First chunk preview: {chunks[0].page_content[:80]}')
print(f'Metadata: {chunks[0].metadata}')