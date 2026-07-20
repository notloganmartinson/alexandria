import pytest
from alexandria.extraction import RoutingChunker

@pytest.fixture
def chunker():
    # Use smaller chunk sizes for easier testing of the fallback logic
    return RoutingChunker(fallback_chunk_size=50, fallback_overlap=10)

def test_routing_logic(chunker, mocker):
    """Ensure files are routed to the correct engine based on extension."""
    # Mock the internal methods to just return their names for verification
    mocker.patch.object(chunker, '_chunk_code_ast', return_value="ast_route")
    mocker.patch.object(chunker, '_chunk_documentation', return_value="doc_route")
    mocker.patch.object(chunker, '_chunk_fallback', return_value="fallback_route")

    assert chunker.process_file("script.py", "") == "ast_route"
    assert chunker.process_file("readme.md", "") == "doc_route"
    assert chunker.process_file("data.csv", "") == "fallback_route"
    assert chunker.process_file("unknown_file", "") == "fallback_route"

def test_markdown_chunking(chunker):
    """Test that Markdown is split by headers correctly."""
    markdown_content = """# Header 1
This is the first section.

## Header 2
This is the second section."""

    chunks = chunker._chunk_documentation("test.md", markdown_content)
    
    assert len(chunks) == 2
    assert chunks[0]["metadata"]["header"] == "# Header 1"
    assert "first section" in chunks[0]["content"]
    
    assert chunks[1]["metadata"]["header"] == "## Header 2"
    assert "second section" in chunks[1]["content"]

def test_fallback_chunking(chunker):
    """Test standard character splitting with overlaps."""
    # Create a string of 100 characters (2x our mock chunk size)
    content = "A" * 100 
    
    chunks = chunker._chunk_fallback("test.txt", content)
    
    # 100 chars / 50 chunk size with 10 overlap means it should take 3 chunks
    assert len(chunks) > 1
    assert chunks[0]["metadata"]["chunk_type"] == "fallback_character"
    assert len(chunks[0]["content"]) <= 50

def test_ast_python_chunking(chunker):
    """Test that valid Python code extracts functional blocks."""
    python_code = """
def my_test_function():
    print("Hello Alexandria")
    return True

class MyTestClass:
    def method(self):
        pass
"""
    # tree-sitter-language-pack uses 'python' as the identifier
    chunks = chunker._chunk_code_ast("test.py", python_code, "python")
    
    # We expect one function definition and one class definition
    assert len(chunks) == 2
    assert chunks[0]["metadata"]["node_type"] == "function_definition"
    assert "my_test_function" in chunks[0]["content"]
    assert chunks[1]["metadata"]["node_type"] == "class_definition"
    assert "MyTestClass" in chunks[1]["content"]
