from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import Docx2txtLoader

def load_documents(path):

    loader = DirectoryLoader(
        path,
        glob="**/*",
        loader_cls=Docx2txtLoader
    )

    documents = loader.load()

    return documents