import os
import pdfplumber
import pandas as pd

def merge_csv_files(input_folder, output_filename):
    all_data = pd.DataFrame()
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            csv_path = os.path.join(input_folder, filename)
            df = pd.read_csv(csv_path)
            all_data = all_data.append(df, ignore_index=True)

    all_data.to_csv(output_filename, index=False)

output_folder = f'together/'
output_filename = f'merged.csv'
merge_csv_files(output_folder, output_filename)
