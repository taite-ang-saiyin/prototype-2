import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
import pytesseract
from pdf2image import convert_from_path
from langchain.schema import Document
from pinecone import Pinecone
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Pinecone as pine
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import ChatPromptTemplate,PromptTemplate
from langchain import hub
from langchain.schema.runnable import RunnableLambda,RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import openai
import logging
from langchain.schema import SystemMessage,HumanMessage
from openai import OpenAI
from fastapi import FastAPI
import uvicorn

app=FastAPI()

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    

def Extract_text_from_digital_pdf(pdf_path):
  loader=PyMuPDFLoader(pdf_path)
  docs=loader.load()
  return docs

def Extract_text_from_scanned_pad(pdf_path):
  images = convert_from_path(pdf_path)
  docs_ocr = []
  for page_num, img in enumerate(images, start=1):
      text = pytesseract.image_to_string(img)
      doc = Document(
          page_content=text,
          metadata={"page": page_num}
      )
      docs_ocr.append(doc)
  return docs_ocr

def check_for_digital_pdf(pdf_path):
  docs=Extract_text_from_digital_pdf(pdf_path)
  for doc in docs:
    print(doc.page_content)
    if(doc.page_content):
      return True
    else:
      return False

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def Split_chunks(docs):
  text_splitter=RecursiveCharacterTextSplitter(chunk_size=3000,chunk_overlap=500)
  splits=text_splitter.split_documents(docs)
  return splits

def document_processing(pdfs_path):
  pdfs_path=pdfs_path
  all_docs=[]
  for file in os.listdir(pdfs_path):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(pdfs_path, file)
            print(f"Processing: {pdf_path}")
            if(check_for_digital_pdf(pdf_path)):
              docs=Extract_text_from_digital_pdf(pdf_path)
              all_docs=all_docs+docs
              print("length of the list:",len(all_docs))
              print("processing for digital_pdf")
            else:
              docs=Extract_text_from_scanned_pad(pdf_path)
              all_docs=all_docs+docs
              print("processing for scanned pdf")
  return all_docs

def embedding_output(splits):
  model = SentenceTransformer("all-MiniLM-L6-v2")
  embeddings = model.encode([d.page_content for d in splits], show_progress_bar=True)
  return embeddings

def VectorStoring(splits,embeddings):
  records = []
  for d, e in zip(splits, embeddings):
    records.append({
        "id": str(d.metadata['page']),
        "values": e,
        "metadata": {'text': d.page_content}
    })
  index=pc.Index('new-hybrid-index')
  index.upsert(
      vectors=records,
      namespace="p3-p6-namespace"
  )


def gpt_function(prompt_text:str)->str:
  completion = client.chat.completions.create(
    model="gpt-4o",
    store=True,
    messages=[
      {"role": "user", "content": str(prompt_text)}
    ]
  )
  return completion.choices[0].message.content

embedding_model=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  
index=pc.Index('new-hybrid-index')
vectorstore = PineconeVectorStore(index_name='new-hybrid-index', embedding=embedding_model, namespace="p3-p6-namespace")

retriever1 = RunnableLambda(lambda query: vectorstore.similarity_search(str(query)))

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)
llm=RunnableLambda(gpt_function)
prompt = ChatPromptTemplate.from_template(
    "You are a teacher assistant in charge of making notes that would be used to teach students of Primary level in Singapore Science. Ensure your notes follow a good format that would introduce the topic from foundation up. You are to strictly use the context provided to you and answer the questions and not miss out any details. Think and generate like a teacher conducting a class for Primary school kids.\n\ncontext:{context}\n\nquestoin:{question}"
)

rag_chain = (
    {"context": retriever1 | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
@app.get("/output/{input_msg}")
def get_result(input_msg:str):
    input_sen=input(input_msg)
    response = rag_chain.invoke(input_sen)
    return {"message":response}
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 4000))  # ✅ Render expects port 10000
    uvicorn.run(app, host="0.0.0.0", port=port)     
