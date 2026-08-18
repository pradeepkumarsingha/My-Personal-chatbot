import os
import pickle
import streamlit as st
from dotenv import load_dotenv

# 1. Page Configuration (Must be the very first Streamlit command)
st.set_page_config(page_title="Pradeep's Personal AI Bot", page_icon="🤖", layout="centered")

# Load Environment Variables
load_dotenv()

# --- RECOVERY IMPORT PATTERN ---
try:
    from langchain.retrievers import EnsembleRetriever
except ModuleNotFoundError:
    # Fallback to direct utility file structure if parent LangChain package lacks root mapping
    from langchain_classic.retrievers import EnsembleRetriever 

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

CHROMA_DIR = "chroma_db"
BM25_PKL_PATH = os.path.join(CHROMA_DIR, "bm25_corpus.pkl")

# 2. Cache resources so they only load ONCE when the app starts up
@st.cache_resource
def initialize_hybrid_retriever():
    if not os.path.exists(CHROMA_DIR) or not os.path.exists(BM25_PKL_PATH):
        st.error("⚠️ Database elements missing! Please run 'python createDB.py' in your terminal first.")
        st.stop()
        
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Dense Stream
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Sparse Stream
    with open(BM25_PKL_PATH, "rb") as f:
        bm25_corpus_data = pickle.load(f)
        
    from langchain_core.documents import Document
    bm25_docs = [
        Document(page_content=item["page_content"], metadata=item["metadata"]) 
        for item in bm25_corpus_data
    ]
    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    bm25_retriever.k = 3
    
    # Blended Stream
    return EnsembleRetriever(retrievers=[chroma_retriever, bm25_retriever], weights=[0.6, 0.4])

# Initialize backend instances
hybrid_retriever = initialize_hybrid_retriever()

# 3. Setup LLM (Check for free Groq key)
if not os.getenv("GROQ_API_KEY"):
    st.error("🔑 GROQ_API_KEY missing in your .env file! Please add it to talk to the bot.")
    st.stop()

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.3, max_tokens=500)

# 4. Prompt Engineering Setup
system_prompt = (
    "You are an elite, professional, and friendly AI assistant representing Pradeep Kumar Singh.\n"
    "Your job is to answer portfolio visitors' questions accurately using only the provided context below.\n"
    "Keep responses concise, clear, and highly professional.\n"
    "If someone asks for contact details, projects, or strengths, use the context directly.\n"
    "If you do not know the answer based on the context, politely say you don't have that information.\n\n"
    "Context:\n{context}"
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# 5. Persistent Session Chat History Memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- STREAMLIT UI DESIGN ---
st.title("🤖 Chat with My Portfolio AI")
st.caption("Ask questions about my projects, technical stack, skills, or professional experience!")
st.divider()

# Display ongoing chat history logs on screen
for msg in st.session_state.chat_history.messages:
    role = "user" if msg.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Handle live User Inputs
if user_input := st.chat_input("Ask me something (e.g., 'What are your core projects?')"):
    
    # 1. Show user message
    with st.chat_message("user"):
        st.markdown(user_input)
        
    # 2. Fetch context via Hybrid Ensemble Strategy
    retrieved_docs = hybrid_retriever.invoke(user_input)
    context_str = format_docs(retrieved_docs)
    
    # 3. Formulate Chain
    brain_chain = prompt_template | llm
    conversational_chain = RunnableWithMessageHistory(
        brain_chain,
        lambda session_id: st.session_state.chat_history,
        input_messages_key="input",
        history_messages_key="history"
    )
    
    # 4. Generate & Stream/Render Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = conversational_chain.invoke(
                {"input": user_input, "context": context_str},
                config={"configurable": {"session_id": "portfolio_session"}}
            )
            st.markdown(response.content)