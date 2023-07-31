"""
Description: Embedding generation and file processing script. Add files to data/raw to generate embeddings. Supported filetypes: .docx, .pdf

"""
import os
import pandas as pd
from src.file_manager import CustomPdfReader, CustomWordReader, CSVHandler
from src.ai import EmbeddingAI, EmbeddingSearchAI
from src.embedding_manager import EmbeddingManager

EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL ="gpt-3.5-turbo"
N_TOKENS = 500

# Set up directory paths
data_crawled = "\\data\\crawled"
embedded = "\\data\\embedded"
processed = "\\data\\processed"
cwd = os.getcwd()
directory_in = cwd + data_crawled
directory_out = cwd + embedded
directory_processed = cwd + processed

# Initialize EmbeddingAI instance
embedding_ai = EmbeddingAI(n_tokens=N_TOKENS, model_embedding=EMBEDDING_MODEL)


#######################################
#           Embedding Creation
#######################################

# Get a list of files in the input directory
files = []
for filename in os.listdir(directory_in):
    # Construct the absolute file path
    filepath = os.path.join(directory_in, filename)
    files.append(filepath)

# Process each file
for f in files:
    df_list = []
    # Get file name and extension
    directory_path, file_name_with_extension = os.path.split(f)
    file_name, file_extension = os.path.splitext(file_name_with_extension)

    # Extract text from supported file types
    if file_extension == ".csv":
        print("working with csv")
        df = CSVHandler.load_csv_to_df(f)
        df_list.append(df)
    else:
        print(f"Ignoring file {f} - Unsupported file extension")
        continue
    
    for df in df_list:
        df["text"] = df.apply(lambda row: embedding_ai.get_text_chunks(row["article_text"]), axis = 1)
        df = df.explode("text", ignore_index=True)
        df["embedding"] = df.apply(lambda row : embedding_ai.get_embedding(row["text"])[1], axis = 1)
        EmbeddingManager.invoke_embedding_export(df, directory_out, file_name)

    # Move processed file to the designated directory
    EmbeddingManager.invoke_move_raw_to_processed_file(directory_in, directory_processed, file_name_with_extension)

embedding_manager = EmbeddingManager()
df = embedding_manager.get_embedding_all(directory_out)

system_message = "Du bist ein hilfreicher Assistent, der mit Fragen zu Schweizer gesetzen hilft"
query_introduction = "Dir steht eine Datenbank zur Verfügung und du erhältst die relevantesten Daten zur Frage. Wenn die Daten aus der Datenbank nicht zufriedenstellend waren, versuche eine globalere antwort zu geben"
embedding_search_ai = EmbeddingSearchAI(n_tokens=N_TOKENS, model_embedding=EMBEDDING_MODEL, model_chat=GPT_MODEL, query_system_message=system_message)

running = True
while running:
    user_input = "placeholder"
    try:
        user_input = input(r"Enter your question (or 'e' to exit): ")
    except EOFError as e:
        print(e)


    

    if user_input.lower() in ["e", "exit"]:
        running = False
        print("Exiting the program...")
    else:
        # Process the user input
        print(f"\nQuestion: \n{user_input}")
        answer = embedding_search_ai.ask(query=user_input, query_introduction=query_introduction, df=df, print_message=True)
        print(f"\nAnswer: \n{answer}")