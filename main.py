import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
load_dotenv()

model = ChatGroq(model="qwen/qwen3.6-27b",temperature=0.5)

loader = WebBaseLoader(
    web_paths= ("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs= dict(
        parse_only = bs4.SoupStrainer(
            class_ = ["post-content","post-title","post-header"]
        )
    )
)
parser = StrOutputParser()
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)
retriever = vectorstore.as_retriever()

prompt = ChatPromptTemplate.from_template("""
You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {context}

Answer:
""")
#ÖNEMLİ FONKSİYON, İŞE YARAYABİLİR
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
   | prompt
   | model
   | parser
)

if __name__ == '__main__':
    for chunk in chain.stream("what is maximum inner product search?"):
        print(chunk,end=" ",flush = True)