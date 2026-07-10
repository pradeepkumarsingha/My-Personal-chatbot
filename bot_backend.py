import os
import pickle
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever 
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# Load API Keys
load_dotenv()

CHROMA_DIR = "chroma_db"
BM25_PKL_PATH = os.path.join(CHROMA_DIR, "bm25_corpus.pkl")

# 1. Initialize the Free Embedding Model (Same as createDB.py)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Setup Dense Retriever (ChromaDB)
vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 3. Setup Sparse Retriever (BM25)
if not os.path.exists(BM25_PKL_PATH):
    raise FileNotFoundError("⚠️ Could not find bm25_corpus.pkl. Please run createDB.py first!")

with open(BM25_PKL_PATH, "rb") as f:
    bm25_corpus_data = pickle.load(f)

# Reconstruct LangChain Document objects from serialized data
from langchain_core.documents import Document
bm25_docs = [
    Document(page_content=item["page_content"], metadata=item["metadata"]) 
    for item in bm25_corpus_data
]
bm25_retriever = BM25Retriever.from_documents(bm25_docs)
bm25_retriever.k = 3

# 4. Construct Hybrid Ensemble Retriever
# Assigning weights: 60% semantic relevance, 40% exact keyword matching
hybrid_retriever = EnsembleRetriever(
    retrievers=[chroma_retriever, bm25_retriever],
    weights=[0.6, 0.4]
)

# 5. Initialize the LLM via Free Groq API
# Using the blazing fast llama-3.1-8b-instant model (14,400 requests/day free tier)
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.4,
    max_tokens=500
)

# 6. Define System Prompt Template
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

# 7. Create Context-Aware Document Formatting Chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 8. Manage Memory (Chat History)
message_history_store = {}

def get_session_history(session_id: str):
    if session_id not in message_history_store:
        message_history_store[session_id] = ChatMessageHistory()
    return message_history_store[session_id]

# 9. Execution Function
# 9. Asynchronous Streaming Execution Function
# 9. Asynchronous Streaming Execution Function
async def ask_personal_bot_stream(user_query: str, session_id: str = "portfolio_user"):
    # 1. Retrieve documents using Hybrid Search (Keep synchronous if database lacks async native bindings)
    retrieved_documents = hybrid_retriever.invoke(user_query)
    context_str = format_docs(retrieved_documents)
    
    # 2. Re-formulate the core runnable pipeline
    brain_chain = prompt_template | llm
    
    conversational_chain = RunnableWithMessageHistory(
        brain_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )
    
    # 3. Stream chunks using LangChain's built-in async stream engine
    async for chunk in conversational_chain.astream(
        {"input": user_query, "context": context_str},
        config={"configurable": {"session_id": session_id}}
    ):
        # ChatGroq returns AIMessageChunk objects; extract the raw text content
        if hasattr(chunk, "content"):
            yield chunk.content
        elif isinstance(chunk, str):
            yield chunk


if __name__ == "__main__":
    print("🤖 Bot Ready for Testing!")
    # Test keyword extraction (e.g. checking specific projects inside projects.json)
    q1 = "Tell me about Krushi Sathi project and his skills."
    print(f"\nUser: {q1}\nBot: {ask_personal_bot(q1)}")
    
    # Test follow-up context memory
    q2 = "What are his contact details?"
    print(f"\nUser: {q2}\nBot: {ask_personal_bot(q2)}")