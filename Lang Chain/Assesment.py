import os
from langchain import VectorDBQA,OpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
import chromadb
import tiktoken
import nltk

os.environ["OPENAI_API_KEY"] = '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'

file = TextLoader("Cristiano_Ronaldo.txt")
documents = file.load()
print (f"Having {len(documents)} document")
print (f"Length of the characters in that Document {len(documents[0].page_content)} ")

text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
texts = text_splitter.split_documents(documents)

num_total_characters = sum([len(x.page_content) for x in texts])
print (f"Now it have {len(texts)} documents that have an average of {num_total_characters / len(texts):,.0f} characters (smaller pieces)")

embeddings = OpenAIEmbeddings()

docsearch = Chroma.from_documents(texts,embeddings)


qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever(), return_source_documents=True)
query = "details of ronadlo's family?"
result = qa({"query": query})
print(f'ASNWER : {result["result"]}')
print(f'REVELANT DOCUMENT :{result["source_documents"]}')
