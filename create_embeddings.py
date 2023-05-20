"""
Description: Embedding generation and file processing script. Add files to data/raw to generate embeddings. Supported filetypes: .docx, .pdf

"""

import os
import pandas as pd
from src.file_manager import CustomPdfReader, CustomWordReader
from src.ai import EmbeddingAI
from src.embedding_manager import EmbeddingManager

EMBEDDING_MODEL = "text-embedding-ada-002"
N_TOKENS = 675

# Set up directory paths
data_raw = "\\data\\raw"
embedded = "\\data\\embedded"
processed = "\\data\\processed"
cwd = os.getcwd()
directory_in = cwd + data_raw
directory_out = cwd + embedded
directory_processed = cwd + processed

# Initialize EmbeddingAI instance
embedding_ai = EmbeddingAI(n_tokens=N_TOKENS, model_embedding=EMBEDDING_MODEL)

# Get a list of files in the input directory
files = []
for filename in os.listdir(directory_in):
    # Construct the absolute file path
    filepath = os.path.join(directory_in, filename)
    files.append(filepath)

# Process each file
for f in files:
    text = ""
    # Get file name and extension
    directory_path, file_name_with_extension = os.path.split(f)
    file_name, file_extension = os.path.splitext(file_name_with_extension)

    # Extract text from supported file types
    if file_extension == ".docx":
        text = CustomWordReader.extract_text_from_docx(f)
    elif file_extension == ".pdf":
        text = CustomPdfReader.extract_text_from_pdf(f)
    else:
        print(f"Ignoring file {f} - Unsupported file extension")
        continue

    # Split text into chunks
    text_chunks = embedding_ai.get_text_chunks(text)

    embeddings = []
    # Generate embeddings for each text chunk
    for text in text_chunks:
        embeddings.append(embedding_ai.get_embedding(text))

    # Create DataFrame from embeddings and export to file
    df = EmbeddingManager.get_embedding_df(embeddings)
    EmbeddingManager.invoke_embedding_export(df, directory_out, file_name, N_TOKENS)

    # Move processed file to the designated directory
    EmbeddingManager.invoke_move_raw_to_processed_file(directory_in, directory_processed, file_name_with_extension)
