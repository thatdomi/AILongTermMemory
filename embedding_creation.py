import os
import pandas as pd
from src.file_manager import CustomPdfReader, CustomWordReader
from src.ai import EmbeddingAI
from src.embedding_manager import EmbeddingManager

data_raw = "\\data\\raw"
embedded = "\\data\\embedded"
processed = "\\data\\processed"
cwd = os.getcwd()
directory_in = cwd+data_raw
directory_out = cwd+embedded
directory_processed = cwd+processed


model = "text-embedding-ada-002"
tokens = 675

embedding_ai = EmbeddingAI(n_tokens=tokens, model_embedding=model)

files = []
# Iterate over all files in the directory
for filename in os.listdir(directory_in):
    # Construct the absolute file path
    filepath = os.path.join(directory_in, filename)
    files.append(filepath)

# f = files[0]
#text = CustomPdfReader.extract_text_from_pdf(f)
#text_chunks = embedding_ai.get_text_chunks(text)

for f in files:
    text = CustomWordReader.extract_text_from_docx(f)
    text_chunks = embedding_ai.get_text_chunks(text)

    embeddings = []
    for text in text_chunks:
        embeddings.append(embedding_ai.get_embedding(text))

    directory_path, file_name_with_extension = os.path.split(f)
    file_name, file_extension = os.path.splitext(file_name_with_extension)

    df = EmbeddingManager.get_embedding_df(embeddings)
    EmbeddingManager.invoke_embedding_export(df, directory_out, file_name, tokens)

    EmbeddingManager.invoke_move_raw_to_processed_file(directory_in, directory_processed, file_name_with_extension)

#df['ada_embedding'] = df.combined.apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))
#df.to_csv('output/embedded_1k_reviews.csv', index=False)





#df = pd.read_csv('output/embedded_1k_reviews.csv')
#df['ada_embedding'] = df.ada_embedding.apply(eval).apply(np.array)
