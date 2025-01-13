import os
import re
from transformers.utils.doc import add_start_docstrings, add_end_docstrings, add_code_sample_docstrings
import importlib

def get_docstring_from_class(class_name):
    try:
        # Split the class name into module path and class name
        module_parts = class_name.split('.')
        class_name_only = module_parts[-1]
        module_path = '.'.join(module_parts[:-1])
        
        # Import the module
        module = importlib.import_module(module_path)
        
        # Get the class
        cls = getattr(module, class_name_only)
        
        # Get the docstring
        docstring = cls.__doc__
        if docstring is None:
            return f"Could not find docstring for {class_name}"
        
        return docstring
    except (ImportError, AttributeError) as e:
        return f"Could not find docstring for {class_name}: {str(e)}"

def process_markdown_file(file_path):
    try:
        print(f"Processing {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find all [[autodoc]] directives
        pattern = r'\[\[autodoc\]\]\s+([^\s]+)'
        matches = re.finditer(pattern, content)
        
        # Replace each [[autodoc]] directive with the actual docstring
        for match in matches:
            class_name = match.group(1)
            docstring = get_docstring_from_class(class_name)
            content = content.replace(match.group(0), docstring)
        
        # Write the modified content back to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Completed processing {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

def process_directory(directory_path):
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                process_markdown_file(file_path)

if __name__ == "__main__":
    # Add the transformers source directory to the Python path
    import sys
    sys.path.append(os.path.abspath("src"))
    
    # Process all markdown files in docs/source/en
    process_directory("docs/source/en")