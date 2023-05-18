import pandas as pd
import numpy as np
import os
import openai
import tiktoken
from scipy import spatial
from openai.embeddings_utils import get_embedding, cosine_similarity
from src.ai import EmbeddingSearchAI
from src.embedding_manager import EmbeddingManager

data_raw = "\\data\\raw"
embedded = "\\data\\embedded"
cwd = os.getcwd()
directory_in = cwd+data_raw
directory_out = cwd+embedded
embedding_file = "20230518-bund_embeddings-675.csv"
embedding_file_fullpath = directory_out + "\\" + embedding_file

EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL ="gpt-3.5-turbo"
tokens = 675

embedding_manager = EmbeddingManager()
df = embedding_manager.get_embedding_all(directory_out)

#strings, relatednesses = strings_ranked_by_relatedness(query, df, top_n=5)
#for string, relatedness in zip(strings, relatednesses):
#    print(f"{relatedness=:.3f}")
#    display(string)

system_message = "You answer questions about Laws of the Swiss Confederation"
query_introduction = "You get the most relevant results from a database below to answer the question. If the answer cannot be found in the database, write 'I was not able to find an answer.' and try to give a more general answer about the topic"
embedding_search_ai = EmbeddingSearchAI(n_tokens=tokens, model_embedding=EMBEDDING_MODEL, model_chat=GPT_MODEL, query_system_message=system_message)

query = "What are the most important laws in switzerland?"
embedding_search_ai.ask(query=query, query_introduction=query_introduction, df=df, print_message=True)

#query = "Welche Bevölkungsgruppe wurde in den letzten Jahren Religiöser?"
#ask(query=query, print_message=True)