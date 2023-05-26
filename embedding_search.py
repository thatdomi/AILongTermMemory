import pandas as pd
import numpy as np
import os
import openai
from scipy import spatial
from src.ai import EmbeddingSearchAI
from src.embedding_manager import EmbeddingManager

EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL ="gpt-3.5-turbo"
N_TOKENS = 675

data_raw = "\\data\\raw"
embedded = "\\data\\embedded"
processed = "\\data\\processed"
cwd = os.getcwd()
directory_in = cwd+data_raw
directory_out = cwd+embedded
directory_processed = cwd+processed

openai.organization = os.environ.get("OPENAI_ORG", "")
openai.api_key = os.environ.get("OPENAI_API_KEY", "")

embedding_manager = EmbeddingManager()
df = embedding_manager.get_embedding_all(directory_out)


system_message = "You answer questions about technical documentations"
query_introduction = "You get the most relevant results from a database below to answer the question. If the answer cannot be found in the database, write 'I was not able to find an answer.' and try to give a more general answer about the topic. Answer the question in the language in which it was asked."
embedding_search_ai = EmbeddingSearchAI(n_tokens=N_TOKENS, model_embedding=EMBEDDING_MODEL, model_chat=GPT_MODEL, query_system_message=system_message)

running = True
while running:
    user_input = input("Enter your question (or 'e' to exit): ")

    if user_input.lower() in ["e", "exit"]:
        running = False
        print("Exiting the program...")
    else:
        # Process the user input
        print(f"\nQuestion: \n{user_input}")
        answer = embedding_search_ai.ask(query=user_input, query_introduction=query_introduction, df=df, print_message=False)
        print(f"\nAnswer: \n{answer}")