"""
app.py

WHAT THIS FILE DOES:
1. Loads the database that ingest.py already built
2. Shows a simple web page with a text box
3. When you type a question:
     - finds the most relevant chunks from your papers
     - sends them + your question to a free LLM (Groq/Llama 3)
     - shows you the answer AND which paper/page it came from

Run it like this in your terminal (after running ingest.py once):
    streamlit run app.py

You need a free Groq API key first: https://console.groq.com
"""

import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

DB_FOLDER = "chroma_db"

# ----- PAGE SETUP -----
st.set_page_config(page_title="AI Research Assistant", page_icon="📄")
st.title("📄 AI Research Assistant")
st.write("Ask a question about the research papers you've loaded. Answers are grounded in the actual paper text.")

# ----- GET API KEY FROM USER (kept in memory only, never saved to a file) -----
groq_api_key = st.sidebar.text_input("Enter your Groq API key", type="password")

if not groq_api_key:
    st.info("Enter your free Groq API key in the sidebar to get started. Get one at console.groq.com")
    st.stop()

os.environ["GROQ_API_KEY"] = groq_api_key


@st.cache_resource
def load_database():
    """
    Loads the already-built Chroma database from disk.
    @st.cache_resource means this only runs ONCE, not every time you ask a question —
    otherwise it would be slow, reloading the whole database each time.
    """
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=DB_FOLDER, embedding_function=embedding_model)
    return vector_db


# A custom prompt template — this is the instruction we give the LLM
# so it answers ONLY from the retrieved chunks, not from its own memory.
PROMPT_TEMPLATE = """
Use the following excerpts from research papers to answer the question.
The excerpts may describe the answer using different wording than the question — read carefully and connect related terms before concluding the answer isn't present.
Only say "I couldn't find this in the loaded papers" if none of the excerpts are relevant at all. If the excerpts partially address the question, answer with what they say and note what's missing.

Excerpts:
{context}

Question: {question}

Answer:
"""

prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])

# Load the database (cached, so this is fast after the first run)
vector_db = load_database()

# Set up the LLM — using Groq's free Llama 3 model
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

# RetrievalQA chain: this is LangChain's built-in way of wiring together
# "search the database" + "send results to the LLM" in one step.
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_db.as_retriever(search_kwargs={"k": 6}),  # retrieve top 6 chunks
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True,  # this is what lets us show citations
)

# ----- THE ACTUAL QUESTION BOX -----
question = st.text_input("Your question:")

if question:
    with st.spinner("Searching papers and generating answer..."):
        result = qa_chain.invoke({"query": question})

    st.subheader("Answer")
    st.write(result["result"])

    st.subheader("Sources")

    # Track which (file, page) combos we've already shown, to avoid duplicates
    seen_sources = set()

    for doc in result["source_documents"]:
        raw_source = doc.metadata.get("source", "unknown file")
        # Normalize path separators (Windows uses \, Linux uses /) before extracting filename
        clean_filename = os.path.basename(raw_source.replace("\\", "/"))

        # Page numbers from PyPDFLoader are 0-indexed, so add 1 for the number a human expects
        raw_page = doc.metadata.get("page", None)
        page_number = raw_page + 1 if isinstance(raw_page, int) else "unknown"

        source_key = (clean_filename, page_number)
        if source_key in seen_sources:
            continue  # skip duplicate chunk from the same file/page
        seen_sources.add(source_key)

        with st.expander(f"{clean_filename} — page {page_number}"):
            st.write(doc.page_content)
