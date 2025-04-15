import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
import pytesseract
from pydantic import BaseModel
from pdf2image import convert_from_path
from langchain.schema import Document
from pinecone import Pinecone
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Pinecone as pine
from langchain-huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import ChatPromptTemplate,PromptTemplate
from langchain import hub
from langchain.schema.runnable import RunnableLambda,RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import openai
import logging
from langchain.schema import SystemMessage,HumanMessage
import uvicorn
from openai import OpenAI
from fastapi import FastAPI,HTTPException
import weaviate
from weaviate.classes.init import Auth
from langchain_openai import ChatOpenAI
from typing import List, Dict
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
  client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
  )
  completion = client.chat.completions.create(
    model="gpt-4o",
    store=True,
    messages=[
      {"role": "user", "content": str(prompt_text)}
    ]
  )
  return completion.choices[0].message.content

def getnoteresponse(user_input):
    embedding_model=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  
    index=pc.Index('new-hybrid-index')
    vectorstore = PineconeVectorStore(index_name='new-hybrid-index', embedding=embedding_model, namespace="final-note-namespace")
    
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
    response = rag_chain.invoke(user_input)
    return response
    
@app.get("/output/{input_msg}")
def get_result(input_msg:str):
    response = getnoteresponse(input_msg)
    return {"message":response}

class QuestionRequest(BaseModel):
    level: str
    difficulty: str
    subject: str

def retrieve_notes_from_weaviate(query: str) -> List[Dict]:
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url="ggw6eupshybj5ol0gqa8w.c0.asia-southeast1.gcp.weaviate.cloud",
        auth_credentials=Auth.api_key(api_key="AzOaydnWlANzR8mAioN8HfGHRgreaLHRtEkb"),
        headers={"X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"]}
    )

    science_questions = client.collections.get("ScienceQuestion")

    response = science_questions.query.hybrid(
        query=query,
        limit=30,
        return_properties=["question", "subject", "type", "level", "topic", "difficulty", "year", "school"]
    )

    retrieved_data = [obj.properties for obj in response.objects]
    client.close()

    return retrieved_data

def debug_context(x):
    print("🔎 Retrieved Context:\n")
    print(x)
    return x
    
def format_docs_1(docs):
    return "\n\n".join(doc['question'] for doc in docs)
    
def getQuizResponse(level,difficulty,subject):
    retriever1 = RunnableLambda(
        lambda query: retrieve_notes_from_weaviate(query)
    )
    
    debug_node = RunnableLambda(debug_context)
    client = OpenAI(
      api_key=os.environ["OPENAI_API_KEY"]
    )
    input_sen=f"give me 30 {difficulty} mcqs on {subject} for {level} level"
    llm=RunnableLambda(gpt_function)
    prompt = ChatPromptTemplate.from_template(
        "You are a teacher assistant in charge of giving questions that would be used to teach students of Primary level in Singapore Science. Ensure your questions follow a good format that would introduce the topic from foundation up. You are to strictly use the context provided to you and answer the questions and not miss out any details. Think and generate like a teacher conducting a class for Primary school kids.Don't include introduction or conclusion and just give only the extracted questions from the context.\n\ncontext:{context}\n\nquestoin:{question}"
    )
    rag_chain = (
        {"context": retriever1 | format_docs_1, "question": RunnablePassthrough()}
        | debug_node
        | prompt
        | llm
        | StrOutputParser()
    )
    result=rag_chain.invoke(input_sen)
    return result
@app.post("/generate")
async def generate_mcqs(request: QuestionRequest):
    try:
        result=getQuizResponse(request.level,request.difficulty,request.subject)
        return {"result":result}
    except Exception as e:
        raise HTTPException (status_code=500,detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 4000))  # ✅ Render expects port 10000
    uvicorn.run(app, host="0.0.0.0", port=port)     
