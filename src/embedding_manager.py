import pandas as pd
import ast
import os
from datetime import datetime
import shutil

class EmbeddingManager:
    @staticmethod
    def get_embedding_df(embeddings):
        columns = ["text", "embedding"]
        df = pd.DataFrame(embeddings, columns=columns)
        return df

    @staticmethod
    def invoke_move_raw_to_processed_file(directory_in:str, directory_processed:str, filename_with_extension:str):
        source_path= f"{directory_in}\\{filename_with_extension}"
        destination_path = f"{directory_processed}\\{filename_with_extension}"
        try:
            shutil.move(source_path, destination_path)
            print(f"File moved from {source_path} to {destination_path}")
        except FileNotFoundError:
            print("Source file not found. Please check the file path.")
        except PermissionError:
            print("Permission denied. Unable to copy the file.")
        except Exception as e:
            print("An error occurred:", str(e))

    @staticmethod
    def invoke_embedding_export(df: pd.DataFrame, directory_out: str, filename: str, tokens: int) -> str:
        current_datetime = datetime.now().strftime("%Y%m%d")
        full_filepath = directory_out+"\\"+current_datetime+"-"+filename+"-"+str(tokens)+".csv"
        
        base_path, ext = os.path.splitext(full_filepath)
        counter = 1
        new_filename = full_filepath
        while os.path.exists(new_filename):
            new_filename = f"{base_path}_{counter}{ext}"
            counter += 1
        
        df.to_csv(new_filename, sep=",", encoding="utf-8", index=False, )
        print(f"DataFrame exported to {new_filename}")
        
        return new_filename
            

    @staticmethod
    def get_embedding_single(path: str) -> pd.DataFrame:
        """Reads embedding file and converts embeddings from string to list"""
        try:
            with open(path, "r", errors = 'backslashreplace') as file:
                print(f"reading file :{path}")
                df = pd.read_csv(file, sep=",", encoding="utf-8")
        except FileNotFoundError:
            print("File not found. Please check the file path.")
        except pd.errors.ParserError:
            print("Error parsing the CSV file. Please ensure it is in the correct format.")
        except Exception as e:
            print("An error occurred while reading file :", str(e))

        try:
            df['embedding'] = df['embedding'].apply(ast.literal_eval)
        except SyntaxError:
            print("Error converting embeddings. Invalid format in 'embedding' column.")
            return pd.DataFrame()
        except Exception as e:
            print("An error occurred while converting embeddings:", str(e))
            return pd.DataFrame()
        return df

    @staticmethod
    def get_embedding_all(dir_path: str) -> pd.DataFrame:
        df_all = None
        for filename in os.listdir(dir_path):
            filepath = os.path.join(dir_path, filename)  # Construct full file path
            df = EmbeddingManager.get_embedding_single(filepath)
            if not df.empty:
                if df_all is None:
                    df_all = df.copy()
                else:
                    df_all = pd.concat([df_all,df], ignore_index=True)
        return df_all