from PyPDF2 import PdfReader
import docx

class CustomPdfReader:
    @staticmethod
    def extract_text_from_page(page) -> str:
        try:
            text = page.extract_text()
        except Exception as e:
            print(f"Error extracting text from page: {e}")
            return ""
        return text
    
    @staticmethod
    def cleanup_text(text) -> str:
        cleaned_text = text.replace("\n\n", "\n")
        cleaned_text = ' '.join(cleaned_text.split())
        return cleaned_text

    @staticmethod
    def extract_text_from_pdf(path) -> str:
        try:
            reader = PdfReader(path)
        except FileNotFoundError:
            print(f"File not found: {path}")
            return ""
        except Exception as e:
            print(f"Error reading PDF file: {e}")
            return ""
        
        extracted_text = ""
        for page in reader.pages:
            page_text = CustomPdfReader.extract_text_from_page(page)
            if page_text:
                cleaned_text = CustomPdfReader.cleanup_text(page_text)
                extracted_text += cleaned_text + " "
        
        return extracted_text.strip()

class CustomWordReader:
    @staticmethod
    def extract_text_from_docx(path) -> str:
        try:
            doc = docx.Document(path)
        except FileNotFoundError:
            print(f"File not found: {path}")
            return ""
        except Exception as e:
            print(f"Error reading DOCX file: {e}")
            return ""
        
        extracted_text = '\n'.join(para.text for para in doc.paragraphs)
        return extracted_text
