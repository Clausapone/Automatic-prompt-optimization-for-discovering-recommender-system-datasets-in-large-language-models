import csv
import io
import sys

# Set ranges
ranges = [
    (218365, 219395),
    (192380, 200405),
    (209167, 219173),
]

def process_line(line):
    line_clean = line.rstrip('\n').rstrip('\r')
    if not line_clean:
         return line
         
    reader = csv.reader([line_clean])
    try:
        row = next(reader)
    except StopIteration:
        return line
        
    if len(row) < 4:
        # If row is malformed, return as is
        return line
        
    new_row = row[:]
    # Modify
    # row[1] is generated interaction
    # row[2] is real interaction
    # row[3] is error flag
    new_row[1] = row[2] # Set Gen to Real
    new_row[-1] = "0"   # Set Flag to 0
    
    line_buffer = io.StringIO()
    # Use simple lineterminator to avoid double spacing in output
    line_writer = csv.writer(line_buffer, quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
    line_writer.writerow(new_row)
    new_line = line_buffer.getvalue()
    return new_line

file_path = '../gepa/meta-llama_Llama-3.2-3B-books-bookcrossing_interaction_results.csv'


print(f"Reading {file_path}...")
try:
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        all_lines = f.readlines()
except Exception as e:
    print(f"Error reading file: {e}")
    sys.exit(1)

total_modified = 0
total_lines = len(all_lines)
print(f"Total lines in file: {total_lines}")

for start, end in ranges:
    # Adjust for 0-indexing
    # range is inclusive in user request (1-based), so in python list (0-based):
    # start line -> index start-1
    # end line -> index end-1
    # slice [start-1 : end] covers indices start-1 up to end-1 (inclusive)
    
    start_idx = start - 1
    end_idx = end 
    
    if start_idx < 0: start_idx = 0
    if end_idx > total_lines: end_idx = total_lines
    
    print(f"Processing range {start}-{end} (Indices {start_idx}-{end_idx})...")
    
    for i in range(start_idx, end_idx):
        if i < total_lines:
            original = all_lines[i]
            modified = process_line(original)
            all_lines[i] = modified
            total_modified += 1

print(f"Modified {total_modified} lines.")
print("Overwriting file...")

try:
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(all_lines)
    print("Successfully overwrote the file.")
except Exception as e:
    print(f"Error writing file: {e}")
    sys.exit(1)
