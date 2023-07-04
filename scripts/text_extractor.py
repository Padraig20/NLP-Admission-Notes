import csv
import xml.etree.ElementTree as ET
import sys
import os

def extract_xml_to_csv(xml_file, csv_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['text_id', 'text'])

        for topic in root.findall('topic'):
            text_id = topic.get('number')
            text = topic.text.strip().replace('\n', '')

            writer.writerow([text_id, text])

    print(f"Data extracted from '{xml_file}' and saved to '{csv_file}' successfully.")


if len(sys.argv) != 3:
    print("Usage: python text_extractor.py input_file.xml output_file.csv")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

if not input_file.endswith(".xml"):
    print("Input file must have the extension '.xml'")
    sys.exit(1)

if not output_file.endswith(".csv"):
    print("Output file must have the extension '.csv'")
    sys.exit(1)

extract_xml_to_csv(input_file, output_file)

