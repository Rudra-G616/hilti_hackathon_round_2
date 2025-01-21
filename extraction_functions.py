import os
import shutil
from typing import List, Dict
from PyPDF2 import PdfReader

def move_pdfs(source_folder, destination_folder):
    
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Traverse through the source folder and its subfolders
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith(".pdf"):  # Check if the file is a PDF
                # Full path of the PDF file
                file_path = os.path.join(root, file)
            
                # Move the file to the destination folder
            shutil.copy(file_path, os.path.join(destination_folder, file))

    print("All PDFs have been moved to the new folder.")


# def extract_text_from_pdfs(folder_path, output_file):
#     """
#     Extract text from PDFs while attempting to ignore tables and images.
    
#     Args:
#         folder_path (str): Path to the folder containing PDF files
#         output_file (str): Path where the output text file will be saved
#     """
#     all_text = []
    
#     try:
#         for filename in os.listdir(folder_path):
#             if filename.lower().endswith('.pdf'):
#                 pdf_path = os.path.join(folder_path, filename)
                
#                 print(f"Processing: {filename}")
                
#                 try:
#                     with pdfplumber.open(pdf_path) as pdf:
#                         # Add filename as header in output
#                         all_text.append(f"\n\n{'='*50}\n{filename}\n{'='*50}\n")
                        
#                         for page_num, page in enumerate(pdf.pages, 1):
#                             # Extract tables to identify their bounding boxes
#                             tables = page.find_tables()
#                             table_bboxes = [table.bbox for table in tables]
                            
#                             # Extract text while excluding table areas
#                             words = page.extract_words()
#                             filtered_text = []
                            
#                             for word in words:
#                                 # Check if word is within any table boundary
#                                 word_midpoint = ((word['x0'] + word['x1'])/2, 
#                                                (word['y0'] + word['y1'])/2)
                                
#                                 in_table = any(
#                                     bbox[0] <= word_midpoint[0] <= bbox[2] and
#                                     bbox[1] <= word_midpoint[1] <= bbox[3]
#                                     for bbox in table_bboxes
#                                 )
                                
#                                 if not in_table:
#                                     filtered_text.append(word['text'])
                            
#                             if filtered_text:
#                                 all_text.append(f"\n----- Page {page_num} -----\n")
                                
#                                 # Reconstruct text with proper spacing
#                                 text = ' '.join(filtered_text)
                                
#                                 # Clean up common PDF extraction artifacts
#                                 text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
#                                 text = re.sub(r'([a-z])-\s([a-z])', r'\1\2', text)  # Fix hyphenation
                                
#                                 all_text.append(text)
                            
#                 except Exception as e:
#                     print(f"Error processing {filename}: {str(e)}")
#                     continue
        
#         # Write all extracted text to output file
#         with open(output_file, 'w', encoding='utf-8') as f:
#             f.write('\n'.join(all_text))
            
#         print(f"\nText extraction completed. Output saved to: {output_file}")
        
#     except Exception as e:
#         print(f"An error occurred: {str(e)}")

######################################## Improved extraction_function ########################################

def path_finder(file_name, folder_path):
    """
    Find the full path of a file in a directory tree.
    
    Args:
        file_name (str): Name of the file to search for
        folder_path (str): Path to the root folder to start the search
    
    Returns:
        str: Full path of the file if found, else 'File not found'
    """
    for root, dirs, files in os.walk(folder_path):
        if file_name in files:
            full_path = os.path.join(root, file_name)
            return os.path.relpath(full_path, folder_path)
    
    return "File not found"



def extract_text_from_pdfs(folder_path, output_file, src_folder):
    """
    Extract text from all PDFs in the specified folder and save to a single text file.
    
    Args:
        folder_path (str): Path to the folder containing PDF files
        output_file (str): Path where the output text file will be saved
    """
    # Create a list to store extracted text
    all_text = []
    
    try:
        # Iterate through all files in the folder
        for filename in os.listdir(folder_path):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(folder_path, filename)
                
                print(f"Processing: {filename}")
                
                try:
                    # Create PDF reader object
                    reader = PdfReader(pdf_path)
                    
                    # Add filename as header in output
                    # all_text.append(f"\n\n{'>'*10}\n{filename}\n{'='*50}\n")
                    
                    # Extract text from each page
                    for page_num, page in enumerate(reader.pages, 1):
                        text = page.extract_text()
                        if text:
                            all_text.append(f"<file_path : {path_finder(filename, src_folder)}>\n<filename : {filename}>\n<page : {page_num}>\n")
                            all_text.append(text)
                            
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    continue
        
        # Write all extracted text to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(all_text))
            
        print(f"\nText extraction completed. Output saved to: {output_file}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")