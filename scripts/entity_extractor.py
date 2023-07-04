import csv

def extract_entities(text_csv_file, entity_csv_file, output_csv_file):
    # Read the text CSV file
    text_data = {}
    with open(text_csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            text_id, text = row
            text_data[text_id] = text
    
    # Read the entity CSV file and extract the entities
    extracted_data = []
    with open(entity_csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            text_id, entity = row
            text = text_data.get(text_id)
            if text:
                begin = 0
                while begin != -1:
                    begin = text.find(entity, begin)
                    if begin != -1:
                        end = begin + len(entity)
                        chunk = text[begin:end]
                        extracted_data.append([text_id, 'DISEASE', begin, end, chunk])
                        begin = end
    
    # Write the extracted data to a new CSV file
    with open(output_csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['text_id', 'entity', 'begin', 'end', 'chunk'])
        writer.writerows(extracted_data)

import sys

# Check if the correct number of command-line arguments is provided
if len(sys.argv) != 4:
    print("Usage: python entity_extractor.py text_csv_file entity_csv_file output_file")
    sys.exit(1)

# Extract the filenames from the command-line arguments
text_csv_file = sys.argv[1]
entity_csv_file = sys.argv[2]
output_file = sys.argv[3]

extract_entities(text_csv_file, entity_csv_file, output_csv_file)

print("Successful!")
