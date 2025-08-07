import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

@st.cache_resource
def load_vector_store():
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
    db = FAISS.load_local("faiss_index", embeddings=embedding_model, allow_dangerous_deserialization=True)
    return db, embedding_model

def query_rag(query: str, db: FAISS, model_name="gpt-4"):
    results = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    retrieved_docs = results.invoke(query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt_template = PromptTemplate(
        template="""
        You are a helpful assistant of Recogni Company. You need to answer the question using the context
        If you do not have proper or good amount of context then just then respond as a general assistant using your own knowledge.

        Context:
        {context}

        Question:
        {query}
        """,
        input_variables=['query', 'context']
    ) 

    prompt = prompt_template.format(query=query, context=context)

    model = ChatOpenAI(model=model_name, temperature=0.3)
    response = model.invoke(prompt)

    return response.content, retrieved_docs

st.set_page_config(page_title="RAG Chat", layout="centered")
st.title("🧠 RecoMind")

query = st.text_input("Enter your query:", placeholder="e.g. What is pyxis?")

if query:
    with st.spinner("Querying vector DB and generating answer..."):
        db, _ = load_vector_store()
        answer, sources = query_rag(query, db)

    st.markdown("## ✅ Answer")
    st.write(answer)

    st.markdown("### 📚 Retrieved Contexts")
    for i, doc in enumerate(sources, 1):
        with st.expander(f"Context {i}"):
            st.markdown(f"```\n{doc.page_content}\n```")

