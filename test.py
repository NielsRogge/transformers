import re
import os
from pathlib import Path
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.utils import direct_transformers_import

def get_docstring_from_class(class_name):
    """Get docstring from a Transformers class by name."""
    try:
        # Try to get the class from transformers
        if class_name.startswith("transformers."):
            class_name = class_name.split(".")[-1]
            
        if class_name.startswith("Auto"):
            cls = getattr(AutoModel, class_name, None) or \
                  getattr(AutoConfig, class_name, None) or \
                  getattr(AutoTokenizer, class_name, None)
        else:
            transformers = direct_transformers_import("src/transformers")
            cls = getattr(transformers, class_name, None)
            
        if cls is None:
            return f"Could not find docstring for {class_name}"
            
        return cls.__doc__ or f"No docstring available for {class_name}"
    except Exception as e:
        return f"Error getting docstring for {class_name}: {str(e)}"

def process_markdown_file(file_path):
    """Process a markdown file and replace [[autodoc]] directives with actual docstrings."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Regular expression to match [[autodoc]] directives with optional method specifications
    autodoc_pattern = r'(?:## [^\n]+\n)?\[\[autodoc\]\]\s+([^\n]+?)(?:\s+-\s+([^\n]+))?$'
    
    def replace_autodoc(match):
        class_name = match.group(1).strip()
        methods = match.group(2).split('\n') if match.group(2) else None
        
        docstring = get_docstring_from_class(class_name)
        
        if methods:
            return f"{docstring}\n\nMethods: {', '.join(methods)}"
        return docstring
    
    processed_content = re.sub(autodoc_pattern, replace_autodoc, content, flags=re.MULTILINE)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(processed_content)

def process_docs_directory(docs_dir="docs/source/en"):
    """Process all markdown files in the English docs directory."""
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise ValueError(f"Documentation directory {docs_dir} does not exist")
        
    for md_file in docs_path.rglob("*.md"):
        if "model_doc" in str(md_file):  # Focus on model documentation
            print(f"Processing {md_file}")
            process_markdown_file(md_file)
            print(f"Completed processing {md_file}")

if __name__ == "__main__":
    process_docs_directory()