#!/bin/bash
# dumpApp.sh - Dump all relevant source code files from the repository into projectFiles.txt
#
# This script recursively searches the current directory for files with relevant extensions
# (e.g., js, py, html, css) and excludes directories like venv, .git, and node_modules.
#
# For each file found, it writes to projectFiles.txt the following format:
#
# content of [<file path>]:
# <file content>
#
# (followed by an empty line)

# Set the output file
OUTPUT_FILE="projectFiles.txt"

# Clear the output file if it exists
> "$OUTPUT_FILE"

# Use find to locate files with the desired extensions in the current directory,
# excluding common directories that are not part of the source code.
find . -type f \( -name "*.js" -o -name "*.py" -o -name "*.html" -o -name "*.css" -o -name "*.cfg" -o -name "*.yml" \) \
    -not -path "./venv/*" \
    -not -path "./.git/*" \
    -not -path "./node_modules/*" | while IFS= read -r file; do
    echo "content of [$file]:" >> "$OUTPUT_FILE"
    cat "$file" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"  # Append an empty line between files
done

echo "Source code dump complete. Output written to $OUTPUT_FILE"
