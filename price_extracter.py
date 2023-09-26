import os
import pdfplumber
import pandas as pd

def extract_and_save_pdf_page(pdf_path, output_folder, start_page, end_page):
    with pdfplumber.open(pdf_path) as pdf:
        for i in range(start_page - 1, end_page):
            page = pdf.pages[i]
    #         # Assuming there's only one table per page
            table = page.extract_tables()[0]

    #         # Convert extracted table to a DataFrame
            df = pd.DataFrame(table[1:], columns=table[0])
    #         # Save DataFrame as CSV
            csv_filename = f"{output_folder}/page_{i + 1}.csv"
            df.to_csv(csv_filename, index=False)


def merge_csv_files(input_folder, output_filename):
    all_data = pd.DataFrame()
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            csv_path = os.path.join(input_folder, filename)
            df = pd.read_csv(csv_path)
            all_data = all_data.append(df, ignore_index=True)

    all_data.to_csv(output_filename, index=False)


# Specify the input PDF file and output folder
year = 2020
part = 'prices'
input_pdf_path = f'pdf_datasets/{year}.pdf'
output_folder = f'output/{year}/{part}'
start_page = 621
end_page = start_page + 7

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Split the PDF into separate pages and save tables as CSV
extract_and_save_pdf_page(input_pdf_path, output_folder, start_page, end_page)

# Merge CSV files
output_filename = f'output/{year}/{part}.csv'
merge_csv_files(output_folder, output_filename)
