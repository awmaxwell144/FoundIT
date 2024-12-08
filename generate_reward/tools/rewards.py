
def format_reward(unformatted):

    # Flags to track whether we're inside the Python or comment block
    inside_python_block = False
    inside_docstring = False
    extracted_code = []

    lines = unformatted.splitlines()
    # Iterate through each line and check for the boundaries
    for line in lines:
        # Detect start and end of triple double quotes (docstring)
        if '"""' in line:
            inside_docstring = not inside_docstring
            continue  # Skip the line containing """

        # Skip lines inside the docstring block
        if inside_docstring:
            continue

        # Start of the Python block
        if "```python" in line:
            inside_python_block = True
            continue  # Skip the line with ```python

        # End of the Python block
        if inside_python_block and "```" in line:
            inside_python_block = False
            continue  # Skip the line with ```

        # Add lines inside the Python block to the extracted code
        if inside_python_block:
            extracted_code.append(line)
            
    # ensure common import statements are included
    extracted_code = ensure_imports(extracted_code)
    # Join the lines with newline characters to form a single string
    return "\n".join(extracted_code)


def ensure_imports(lines):
    imports_to_check = [
        "import jax.numpy as jnp",
        "import numpy as np"
    ]
    
    # Check for each required import and add if missing
    for imp in imports_to_check:
        if imp not in lines:
            lines.insert(0, imp)
    
    return lines

def write_to_py(output_file, code):
    with open(output_file, 'w') as py_file:
            py_file.writelines(code)