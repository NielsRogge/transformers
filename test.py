import os
import re
import importlib


def get_docstring_from_class(class_name):
    try:
        if '.' in class_name:
            # For any class in transformers (e.g. models.ernie.modeling_ernie.ErnieForPreTrainingOutput or pipelines.ArgumentHandler)
            first_part = class_name.split('.')[0]
            module_path = f'transformers.{first_part}'
            remaining_path = '.'.join(class_name.split('.')[1:])
            
            module = importlib.import_module(module_path)
            
            # Navigate through the remaining path
            obj = module
            for part in remaining_path.split('.'):
                obj = getattr(obj, part)
            
            docstring = obj.__doc__
            if docstring:
                return f"{class_name}\n{docstring}"
        else:
            # Try importing directly from transformers as fallback
            module = importlib.import_module('transformers')
            cls = getattr(module, class_name)
            docstring = cls.__doc__
            if docstring:
                return f"{class_name}\n{docstring}"

        return f"{class_name}"
    except Exception as e:
        return f"[[autodoc]] {class_name}: {str(e)}"


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