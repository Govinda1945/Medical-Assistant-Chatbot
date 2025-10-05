from langchain.document_loaders import PyPDFLoader, DirectoryLoader  #for loading pdf files and files are in directories
from langchain.text_splitter import RecursiveCharacterTextSplitter   #for chunking operation
from langchain.embeddings import HuggingFaceEmbeddings


#extract the data from pdf files

def load_pdf_file(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

#split the data into chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


# Download the Embedding model from huggingface
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  #embedding model with 384 dimensonal dense (needed to create pinecode vector database)
    return embeddings