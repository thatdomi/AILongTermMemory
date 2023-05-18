from PyPDF2 import PdfReader
import docx

class CustomPdfReader:
    @staticmethod
    def extract_text_from_page(page) -> str:
        text = page.extract_text()
        return text
    
    @staticmethod
    def cleanup_text(text) -> str:
        clean_text = text.replace("  ", " ").replace("\n", "; ").replace(';', ' ')
        return clean_text

    @staticmethod
    def extract_text_from_pdf(path) -> str:
        full_text = ""
        reader = PdfReader(path)
        for i in range(0, len(reader.pages)):
            text = CustomPdfReader.extract_text_from_page(reader.pages[i])
            clean_text = CustomPdfReader.cleanup_text(text)
            full_text += clean_text
    
        return full_text 

class CustomWordReader:
    @staticmethod
    def extract_text_from_docx(path) -> str:
        doc = docx.Document(path)
        fullText = []
        for para in doc.paragraphs:
            fullText.append(para.text)
        return '\n'.join(fullText)
        
