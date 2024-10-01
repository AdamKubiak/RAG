import tiktoken
import fitz
from io import BytesIO
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

class PdfPreprocessor():
    
    def __init__(self, pdfFileFromRequest, pdfFileName, chunk_size) -> None:
        if not pdfFileFromRequest:
            raise Exception("No file uploaded")
        
        self.pdf_File = pdfFileFromRequest
        self.chunk_size = chunk_size
        self.pdf_Filename = pdfFileName
        
        self.document_list = []
        
    @staticmethod
    def num_tokens_from_string(string: str, encoding_name: str) -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def _open_and_read_pdf(self) -> list[dict]:
        pdf_stream = BytesIO(self.pdf_File.read())  # Read the file into a BytesIO stream
        doc = fitz.open(stream=pdf_stream, filetype="pdf")
        
        for page_number, page in enumerate(doc):
            text = page.get_text()
            self.document_list.append({"page_number": page_number,
                                    "page_char_count": len(text),
                                    "page_word_count": len(text.split()),
                                    "page_sentance_count": len(text.split(". ")),
                                    "filename": self.pdf_Filename,
                                    'page_content': text})

    def _from_dict_to_document(self):
        result = []
        
        for _, doc in enumerate(self.document_list):
            object_doc = Document(page_content=doc['page_content'],
                                metadata={"page_number": doc['page_number'],
                                    "page_char_count": doc['page_char_count'],
                                    "page_word_count": doc["page_word_count"],
                                    "page_sentance_count": doc["page_sentance_count"],
                                    "filename": self.pdf_Filename,})
            result.append(object_doc)
        self.document_list = result

    def _create_chunks(self):
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=self.chunk_size, 
        chunk_overlap=0)
        
        self.document_list = text_splitter.split_documents(self.document_list)

    def _update_token_count(self):
        dict_splits = []
        for doc in self.document_list:
            dict_split = doc.dict()
            dict_split['metadata']['split_token_count'] = self.num_tokens_from_string(dict_split['page_content'], "cl100k_base")
            dict_splits.append(Document(page_content=dict_split['page_content'], metadata=dict_split['metadata']))
        self.document_list = dict_splits

    def _filter_small_token_count(self):
        self.document_list = [x for x in self.document_list if x.dict()['metadata']['split_token_count'] > 30]

    def preprocess_pdf(self):
        self._open_and_read_pdf()
        self._from_dict_to_document()
        self._create_chunks()
        self._update_token_count()
        self._filter_small_token_count()
        
        return self.document_list
    
    
    
    