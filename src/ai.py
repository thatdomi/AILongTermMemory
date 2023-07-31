import pandas as pd
import openai
import tiktoken
import os
from scipy import spatial

class EmbeddingAI:
    def __init__(self, n_tokens: int, model_embedding: str):
        # Initialize any required variables or resources
        self.n_tokens = n_tokens
        self.model_embedding = model_embedding
        self.encoding = tiktoken.encoding_for_model(self.model_embedding)
        self.tokenizer = tiktoken.get_encoding(self.encoding.name)
        
    ### tokenization
    def __create_chunks(self, text):
        tokens = self.tokenizer.encode(text)
        n = self.n_tokens
        """Yield successive n-sized chunks from text."""
        i = 0
        while i < len(tokens):
            # Find the nearest end of sentence within a range of 0.7 * n and 1.3 * n tokens
            j = min(i + int(1.3 * n), len(tokens))
            while j > i + int(0.7 * n):
                # Decode the tokens and check for full stop or newline
                chunk = self.tokenizer.decode(tokens[i:j])
                if chunk.endswith(".") or chunk.endswith("\n"):
                    break
                j -= 1
            # If no end of sentence found, use n tokens as the chunk size
            if j == i + int(0.5 * n):
                j = min(i + n, len(tokens))
            yield tokens[i:j]
            i = j
    
    def get_text_chunks(self, text) -> list:
        chunks = self.__create_chunks(text)
        return [self.tokenizer.decode(chunk) for chunk in chunks]
    
    def get_tokencount_in_text(self, text) -> int:
        tokens = self.tokenizer.encode(text)
        return len(tokens)

    ### Embeddings
    def get_embedding(self, text) -> tuple:
        #text = text.replace("\n", " ")
        embedding = openai.Embedding.create(input = [text], model=self.model_embedding)['data'][0]['embedding']
        print(embedding)
        return (text, embedding)
    
    #def num_tokens_from_string(string: str, encoding_name: str) -> int:
    #    """Returns the number of tokens in a text string."""
    #    tokenizer = tiktoken.get_encoding(encoding_name)
    #    num_tokens = len(tokenizer.encode(string))
    #    return num_tokens

class AdminCHEmbeddingAI(EmbeddingAI):
    def __init__(self, n_tokens: int, model_embedding: str):
        super().__init__(n_tokens, model_embedding)

class EmbeddingSearchAI(EmbeddingAI):
    def __init__(self, n_tokens: int, model_embedding: str, model_chat: str, query_system_message: str):
        super().__init__(n_tokens, model_embedding)
        self.model_chat = model_chat
        self.query_system_message = query_system_message

    def get_num_tokens(self, text: str) -> int:
        """Return the number of tokens in a string."""
        encoding = tiktoken.encoding_for_model(self.model_chat)
        return len(encoding.encode(text))

    def get_strings_ranked_by_relatedness(
        self,
        query: str,
        df: pd.DataFrame,
        relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
        top_n: int = 100
    ) -> tuple[list[str], list[float]]:
        """Returns a list of strings and relatednesses, sorted from most related to least."""
        query_embedding_response = openai.Embedding.create(
            model=self.model_embedding,
            input=query,
        )
        query_embedding = query_embedding_response["data"][0]["embedding"]
        strings_and_relatednesses = [
            (row["text"], relatedness_fn(query_embedding, row["embedding"]))
            for i, row in df.iterrows()
        ]
        strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
        strings, relatednesses = zip(*strings_and_relatednesses)
        return strings[:top_n], relatednesses[:top_n]

    def query_message(
        self,
        query: str,
        query_introduction: str,
        df: pd.DataFrame,
        token_budget: int
    ) -> str:
        """Return a message for GPT, with relevant source texts pulled from a dataframe."""
        strings, relatednesses = self.get_strings_ranked_by_relatedness(query, df)
        introduction = query_introduction 
        question = f"\n\nQuestion: {query}"
        message = introduction + question
        counter = 0
        for string in strings:
            counter = counter+1
            next_article = f'\n\nDatabase Result {counter}:\n"""\n{string}\n"""'
            if (
                self.get_num_tokens(message + next_article)
                > token_budget
            ):
                break
            else:
                message += next_article
                
        return message

    def ask(
        self,
        query: str,
        query_introduction: str,
        df: pd.DataFrame,
        token_budget: int = 4096 - 500,
        print_message: bool = False,
    ) -> str:
        """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
        message = self.query_message(query, query_introduction, df, token_budget=token_budget)
        if print_message:
            print(message)
        messages = [
            {"role": "system", "content": self.query_system_message},
            {"role": "user", "content": message},
        ]
        response = openai.ChatCompletion.create(
            model=self.model_chat,
            messages=messages,
            temperature=0
        )
        response_message = response["choices"][0]["message"]["content"]
        return response_message

