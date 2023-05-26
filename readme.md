# OpenAI Embeddings Demo
This project shows the use of OpenAI embeddings to extend the capabilities of ChatGPT to answer questions about specific technical documentations.

## Requirements
- OpenAI API Key
- Setup a python venv and install the requirements in requirements.txt

## Create Embeddings
Put Documents (.pdf or .docx) in the data/raw folder and run create_embeddings.py to create embeddings

## Ask Questions
Run embedding_search.py to ask questions about the documents you created embeddings for. You may have to change the system string in embedding_search.py to expand answers to non technical documentations.

## Code
This code is an adaption from the examples provided by OpenAI here: https://github.com/openai/openai-cookbook/tree/main/examples
