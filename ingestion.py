import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

if __name__ == "__main__":
    loader = PDFPlumberLoader("/Users/gauravkandel/Desktop/LangChain/mlops.pdf")
    document = loader.load()

    print("Splitting....")

    text_splitter = CharacterTextSplitter(chunk_size = 1000,chunk_overlap=0)
    texts = text_splitter.split_documents(document)

    embedding = OpenAIEmbeddings(model = "text-embedding-3-large")
    PineconeVectorStore.from_documents(texts,embedding,index_name = os.environ['INDEX_NAME'])
    print("Done")
    


