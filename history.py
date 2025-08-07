from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.document_loaders import NotionDBLoader
from typing import List
from langchain.schema import Document
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
import faiss
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
load_dotenv()

embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
db = FAISS.load_local("faiss_index", embeddings=embedding_model,allow_dangerous_deserialization=True)


model = ChatOpenAI(model='gpt-4', temperature=0.3, max_completion_tokens=500)

chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful assistant for Recogni Company."),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessage(content="""Use the context below to answer the question.
If the context is not helpful, answer using your own knowledge.

Context:
{context}

Question:
{question}""")
])

chat_history = []

while True:
    user_input = input('You: ')
    if user_input.strip().lower() == "exit":
        break

    results = db.similarity_search(user_input,k=5)
    context = "\n\n".join([doc.page_content for doc in results])

    prompt =  chat_prompt.invoke({'chat_history':chat_history,'context':context,'question':user_input})

    result = model.invoke(prompt)
    print("AI: ",result.content) 

    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=result.content))

print(chat_history)