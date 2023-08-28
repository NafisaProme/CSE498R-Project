import os
import pdfplumber
import pandas as pd


def extract_and_save_pdf_page(pdf_path, output_folder, start_page, end_page):
    with pdfplumber.open(pdf_path) as pdf:
        for i in range(start_page - 1, end_page):
            page = pdf.pages[i]
            # Assuming there's only one table per page
            table = page.extract_tables()[0]

            # Convert extracted table to a DataFrame
            df = pd.DataFrame(table[1:], columns=table[0])

            # Save DataFrame as CSV
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

    # Transpose the DataFrame
    transposed_data = all_data.transpose()

    # Save transposed DataFrame as CSV
    transposed_output_filename = 'transposed_output.csv'
    transposed_data.to_csv(transposed_output_filename, index=False)


# Specify the input PDF file and output folder
input_pdf_path = 'pdf_datasets\\2022\\Yearbook 2022 (PDF).pdf'
output_folder = 'output/2023'
start_page = 640
end_page = 645

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Split the PDF into separate pages and save tables as CSV
extract_and_save_pdf_page(input_pdf_path, output_folder, start_page, end_page)

# Merge CSV files
output_filename = 'merged_output.csv'
merge_csv_files(output_folder, output_filename)
