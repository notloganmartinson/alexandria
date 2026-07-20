"""
Alexandria Extraction & Chunking Engine

This module serves as the primary dispatcher for processing incoming documents.
It inspects file extensions and routes them to the optimal semantic chunker:
1. AST-aware chunking for code (preserves functions/classes).
2. Hierarchical chunking for documentation (Markdown).
3. Recursive character fallback for unknown types.
"""

import pathlib
import re
import logging
from typing import List, Dict, Any, Optional

# Attempt to load the pre-compiled tree-sitter wheels
try:
    from tree_sitter_language_pack import get_parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logging.warning("tree_sitter_languages not found. Code chunking will degrade to fallback.")

logger = logging.getLogger(__name__)

class RoutingChunker:
    def __init__(self, fallback_chunk_size: int = 1000, fallback_overlap: int = 200):
        self.fallback_chunk_size = fallback_chunk_size
        self.fallback_overlap = fallback_overlap

        # 1. Map file extensions to Tree-sitter language identifiers
        self.code_extensions = {
            '.py': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'tsx',
            '.rs': 'rust',
            '.go': 'go',
            '.cpp': 'cpp',
            '.c': 'c',
            '.java': 'java',
            '.cs': 'c_sharp',
            '.rb': 'ruby'
        }
        
        # 2. Define supported documentation extensions
        self.doc_extensions = {'.md', '.mdx', '.txt', '.rst'}

    def process_file(self, file_path: str, content: str) -> List[Dict[str, Any]]:
        """
        Main dispatcher: routes the content based on file extension.
        """
        path = pathlib.Path(file_path)
        ext = path.suffix.lower()

        logger.info(f"Routing document: {file_path} (Extension: {ext})")

        if ext in self.code_extensions and TREE_SITTER_AVAILABLE:
            language_id = self.code_extensions[ext]
            return self._chunk_code_ast(file_path, content, language_id)
        
        elif ext in self.doc_extensions:
            return self._chunk_documentation(file_path, content)
        
        else:
            return self._chunk_fallback(file_path, content)

    def _chunk_code_ast(self, file_path: str, content: str, language_id: str) -> List[Dict[str, Any]]:
        """
        The Code Route: Parses the code into an AST and extracts top-level and 
        class-level constructs (functions, classes, methods).
        """
        try:
            parser = get_parser(language_id)
            tree = parser.parse(content.encode('utf-8'))
        except Exception as e:
            logger.error(f"Failed to parse {file_path} with tree-sitter: {e}. Using fallback.")
            return self._chunk_fallback(file_path, content)

        chunks = []
        lines = content.split('\n')
        
        # We want to extract functions, classes, and methods. 
        # Node types vary slightly by language, but these are standard general identifiers.
        target_node_types = {
            'function_definition', 'class_definition', 'method_definition', 
            'function_declaration', 'class_declaration', 'method_declaration'
        }

        def traverse(node):
            if node.type in target_node_types:
                start_line = node.start_point[0]
                end_line = node.end_point[0]
                
                # Reconstruct the exact string for this node
                node_content = '\n'.join(lines[start_line:end_line + 1])
                
                chunks.append({
                    "content": node_content,
                    "metadata": {
                        "file_path": file_path,
                        "chunk_type": "code_ast",
                        "node_type": node.type,
                        "start_line": start_line + 1,
                        "end_line": end_line + 1,
                        "language": language_id
                    }
                })
            else:
                for child in node.children:
                    traverse(child)

        traverse(tree.root_node)

        # If a file had no extractable functions/classes (e.g., a pure script or config),
        # gracefully degrade to fallback so we don't lose the data.
        if not chunks:
            return self._chunk_fallback(file_path, content)

        return chunks

    def _chunk_documentation(self, file_path: str, content: str) -> List[Dict[str, Any]]:
        """
        The Documentation Route: Uses recursive Markdown splitting.
        Looks for Header -> Double Newline -> Single Newline.
        """
        chunks = []
        
        # Split primarily by Markdown headers
        sections = re.split(r'(^#+\s+.*)', content, flags=re.MULTILINE)
        
        current_section = sections[0].strip()
        
        for i in range(1, len(sections), 2):
            header = sections[i].strip()
            body = sections[i+1].strip() if i+1 < len(sections) else ""
            
            combined_content = f"{header}\n\n{body}".strip()
            
            if combined_content:
                chunks.append({
                    "content": combined_content,
                    "metadata": {
                        "file_path": file_path,
                        "chunk_type": "markdown_section",
                        "header": header
                    }
                })

        # Add initial preamble if it exists before any headers
        if current_section:
            chunks.insert(0, {
                "content": current_section,
                "metadata": {
                    "file_path": file_path,
                    "chunk_type": "markdown_preamble",
                    "header": "Preamble"
                }
            })

        return chunks

    def _chunk_fallback(self, file_path: str, content: str) -> List[Dict[str, Any]]:
        """
        The Fallback Route: Standard recursive character splitting.
        Guarantees that large unsupported files won't overflow the LLM context.
        """
        chunks = []
        start_idx = 0
        text_length = len(content)

        while start_idx < text_length:
            end_idx = min(start_idx + self.fallback_chunk_size, text_length)
            
            # Try to snap to the nearest double newline to avoid cutting mid-sentence
            if end_idx < text_length:
                nearest_newline = content.rfind("\n\n", start_idx, end_idx)
                if nearest_newline != -1 and nearest_newline > start_idx + (self.fallback_chunk_size // 2):
                    end_idx = nearest_newline

            chunk_text = content[start_idx:end_idx].strip()
            if chunk_text:
                chunks.append({
                    "content": chunk_text,
                    "metadata": {
                        "file_path": file_path,
                        "chunk_type": "fallback_character",
                        "start_char": start_idx,
                        "end_char": end_idx
                    }
                })
            
            # FIX: If we've reached the end of the file, break the loop to prevent infinite cycling
            if end_idx >= text_length:
                break
            
            start_idx = end_idx - self.fallback_overlap

        return chunks
# Example usage for local testing
if __name__ == "__main__":
    chunker = RoutingChunker()
    # Replace with an actual read of a file
    # sample_chunks = chunker.process_file("example.py", "def my_func():\n    pass\n")
    # print(sample_chunks)
