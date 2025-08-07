from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder 
from openai import OpenAI
import time 
from sentence_transformers import CrossEncoder
from langchain.callbacks.base import BaseCallbackHandler
reranker = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2-v2') 
#cross-encoder/ms-marco-MiniLM-L-6-v2

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.tokens = []

    def on_llm_new_token(self, token: str, **kwargs):
        self.tokens.append(token)
        self.container.markdown("".join(self.tokens))

load_dotenv()

@st.cache_resource
def load_vector_store():
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
    db = FAISS.load_local("faiss_index", embeddings=embedding_model, allow_dangerous_deserialization=True)
    return db, embedding_model

retriever_template = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    ("human", 
     "Based ONLY on the conversation above, generate a concise and effective search query "
     "that can be used to retrieve the most relevant documents from a knowledge base. "
     "If the user's question is self-contained and doesn't require additional context, return it as-is.")
])

prompt_template = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a helpful and knowledgeable assistant at Recogni. "
     "Use the provided context below to answer the user’s question. "
     "If the context is insufficient or not relevant, fall back to your own general knowledge "
     "but make it clear when doing so.\n\n"
     "Context:\n{context}"),
    
    MessagesPlaceholder(variable_name="chat_history"),
    
    ("human", "{input}")
])

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2') 

def rerank_documents(query, documents):
    pairs = [[query, doc.page_content] for doc in documents]
    scores = reranker.predict(pairs)

    doc_score_pairs = list(zip(documents, scores))
    reranked = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
    
    return [doc for doc, score in reranked]

def query_rag(query: str, db: FAISS, chat_history, container, model_name="gpt-4"):
    start_time = time.time()
    results = db.as_retriever(search_type="similarity", search_kwargs={"k": 20})  

    handler = StreamHandler(container)
    retriever_model = ChatOpenAI(model=model_name, temperature=0.2) 
    qa_model = ChatOpenAI(model=model_name, temperature=0.2, streaming=True, callbacks=[handler])
    parser = StrOutputParser()

    lc_chat_history = [
        HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
        for m in chat_history
    ]

    standalone_query = (retriever_template | retriever_model | parser).invoke({
        'chat_history': lc_chat_history,
        'input': query
    })

    retrieved_docs = results.invoke(standalone_query)
    reranked_docs = rerank_documents(standalone_query, retrieved_docs)
    top_docs = reranked_docs[:5] 
    context = "\n\n".join([doc.page_content for doc in top_docs])

    final_prompt = prompt_template.invoke({
        'context': context,
        'input': query,
        'chat_history': lc_chat_history
    })

    qa_model.invoke(final_prompt) 
    streamed_text = "".join(handler.tokens)
    chat_history.append({"role": "assistant", "content": streamed_text})

    end_time = time.time()
    response_time = end_time - start_time

    return streamed_text, top_docs, response_time

st.set_page_config(page_title="RAG Chat", layout="centered")
st.title("🧠 RecoMind")

query = st.chat_input("What's up mate?")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4"

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query:
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        response_box = st.empty()
        db, _ = load_vector_store()
        answer, sources, response_time = query_rag(query, db, st.session_state.chat_history, response_box)

