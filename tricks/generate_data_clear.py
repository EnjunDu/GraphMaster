# ./tricks/generate_data_clear.py
import json
import re

def fix_messy_json(content):
    """
    Fix the messed up JSON text and return a list of multiple dictionaries.
    Preprocessing steps:
        1. Remove the outermost brackets, if any;
        2. If the entire content is enclosed in quotes (i.e. a string), remove the outer quotes;
        3. Convert escaped newlines to actual newlines;
        4. Split each object using a specific delimiter, making sure each object starts with { and ends with } before parsing.
    """
    content = content.strip()
    # If the content starts with [ and ends with ], remove the outermost brackets
    if content.startswith('[') and content.endswith(']'):
        content = content[1:-1].strip()

    # If the content is enclosed in quotation marks, remove the outermost quotation marks.
    if content.startswith('"') and content.endswith('"'):
        content = content[1:-1].strip()

    # Convert escaped newline characters (\n) to actual newline characters
    content = content.replace(r'\n', '\n')

    # Sometimes the delimiter between objects is "},\n {", we use a special delimiter to separate each object
    delimiter = "|DELIM|"
    content = content.replace('},\n  {', '}' + delimiter + '{')

    # Split the string into individual objects
    parts = content.split(delimiter)

    objs = []
    for i, part in enumerate(parts):
        part = part.strip()
        # If the part does not start with {, add
        if not part.startswith('{'):
            part = '{' + part
        # If the section does not end with }, add
        if not part.endswith('}'):
            part = part + '}'
        # Remove any extra commas that may be leading or trailing
        part = part.strip(', \n')
        try:
            obj = json.loads(part)
            objs.append(obj)
        except Exception as e:
            print(f"Error parsing object {i}: {e}")
            print("content:", part)
    return objs

def reformat_json_file(input_file, output_file):
    """
    Reads a scrambled JSON file, converts it to standard format, and writes it to an output file.
    Standard format example:
    [
        {
            "node_id": 0,
            "label": "2",
            "text": "Title: ... Abstract: ...",
            "neighbors": [8, 14, 258, 435, 544]
        },
        {
            "node_id": 1,
            "label": "5",
            "text": "Title: ... Abstract: ...",
            "neighbors": [344]
        },
        ...
    ]
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # First try to parse the entire file content directly
    try:
        data = json.loads(content)
    except Exception as e:
        print("There is an error when parsing JSON directly. Try to correct it manually. Error message:", e)
        data = None

    objs = []
    if data is None:
        # If direct parsing fails, call the manual correction function
        objs = fix_messy_json(content)
    else:
        # If the parsing is successful, determine the data structure
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                # It is already a standard list dictionary format, so you can use it directly
                objs = data
            elif len(data) > 0 and isinstance(data[0], str):
                # If the list contains strings, assume it is a garbled text and try to fix it.
                messy = data[0]
                objs = fix_messy_json(messy)
            else:
                print("Unsupported JSON data structure!")
                return
        else:
            print("Unsupported JSON data structure!")
            return



    # Write the results to the output file (standard format, 4 spaces indented, Chinese characters retained)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(objs, f, indent=4, ensure_ascii=False)
    print(f"Conversion completed, standard JSON format has been written {output_file}")

if __name__ == '__main__':
    # Please specify the input and output file paths according to the actual situation.
    input_file = 'cora_augmented_output (3).json'
    output_file = 'output.json'
    reformat_json_file(input_file, output_file)
