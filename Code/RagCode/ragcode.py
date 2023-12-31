# -*- coding: utf-8 -*-
import os
import uuid

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.storage import InMemoryStore
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferMemory


os.environ["OPENAI_API_KEY"] = "valid openai api key"

from langchain.document_loaders import DirectoryLoader



loader = DirectoryLoader('regulation_data', glob="**/[!.]*",show_progress=True)
    # Set glob="**/[!.]*" to load all files except hidden ones
    # Set show_progress=True to display loading progress
docs = loader.load()
if len(docs):
    print("Files loaded successfully")


text_splitter = RecursiveCharacterTextSplitter(
    separators = ["\r\n", "\n\n", "\n\n\u3000", "\n\u3000\u3000"], # customized separators
    chunk_size = 400,
    chunk_overlap = 200
)

docs_splits = text_splitter.split_documents(docs)
print("len(docs_splits): ",len(docs_splits)) 



vectorstore = Chroma(
    collection_name="regulations",
    embedding_function=OpenAIEmbeddings() #ada by default
)

#### Document Database Settings

store = InMemoryStore() # Store documents in memory
id_key = "doc_id" # Document id key


#### Set Retriever, Specify Document Database and Vector Database
# The retriever (empty to start)
retriever = MultiVectorRetriever(
    vectorstore=vectorstore, 
    docstore=store, 
    id_key=id_key,
)

#### Document Data Processing

doc_ids = [str(uuid.uuid4()) for _ in docs_splits]

child_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 150, 
    chunk_overlap = 50, 
    )

sub_docs_splits = []

for i, doc in enumerate(docs_splits):
    _id = doc_ids[i]
    _sub_docs = child_text_splitter.split_documents([doc]) 
    for _doc in _sub_docs:
        _doc.metadata[id_key] = _id 
    sub_docs_splits.extend(_sub_docs)



retriever.vectorstore.add_documents(sub_docs_splits)
retriever.docstore.mset(list(zip(doc_ids, docs_splits)))

#### similarity search
similar_docs = retriever.vectorstore.similarity_search("your questions")
relevant_docs = retriever.get_relevant_documents("your questions")



memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(OpenAI(model_name='gpt-3.5-turbo', temperature=0), retriever, memory=memory)

result = qa({"question": "your questions"})
print("Q:","your questions")
print(result)

