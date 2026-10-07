from dotenv import load_dotenv
import sys

# Prevent infinite freeze if accidentally launched with 'streamlit run main.py'
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx():
        import streamlit as st
        st.error("⚠️ **Incorrect command**: You ran `streamlit run main.py`.")
        st.info("The Duova AI web application is in **`duova_app.py`**.\n\nPlease stop this process in your terminal (`Ctrl + C`) and run:\n```bash\nstreamlit run duova_app.py\n```")
        st.stop()
except Exception:
    pass

load_dotenv()

from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from create_DB import index_pdf

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


vectorstore = Chroma(persist_directory="ChromaDB", embedding_function=embedding_model)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 10}
)


llm = ChatGroq(
    model="openai/gpt-oss-120b"
)

# prompt template

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant. Answer the question accurately and clearly based on the provided context.
If the question is a greeting or general inquiry, respond politely and explain what the document covers.
Only if the context contains zero relevant information to answer or address the question, explain that the document does not contain that specific detail.""",
        ),
        (
            "human",
            """Context:
{context}

Question: {question}

Answer:""",
        ),
    ]
)

print("Rag system created")

print("press 0 to exit")

print("type 'upload' to add a PDF from your laptop")

while True:
 
    query = input("You: ")
 
    if query == "0":
        print("Thank you for using!")
        break
 
    if query.lower() == "upload":
        pdf_path = input("Enter path to your PDF file: ").strip('"')
        index_pdf(pdf_path)
        continue
 
    docs = retriever.invoke(query)
 
    context= "\n\n".join(
        [doc.page_content for doc in docs]
    )
 
    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })
 
    response = llm.invoke(final_prompt)
 
    print(f"\n AI: {response.content}")
 