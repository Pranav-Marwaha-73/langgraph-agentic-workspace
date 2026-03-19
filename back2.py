import os
import tempfile
from typing import TypedDict, Annotated, Optional, Dict, Any
import contextvars
import streamlit as st
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import StateGraph, START, END

from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool, BaseTool
from langchain_core.runnables import RunnableConfig

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mcp_adapters.client import MultiServerMCPClient

from dotenv import load_dotenv

import requests
import asyncio
import threading
# 🛡️ SECURE RAM VARIABLES (Invisible to Supabase)
openai_key_var = contextvars.ContextVar('openai_key', default="")
stock_key_var = contextvars.ContextVar('stock_key', default="")
load_dotenv()

# ==========================================
# 1. Async Loop Setup & Core LLM/Embeddings
# ==========================================
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()

def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)

def run_async(coro):
    return _submit_async(coro).result()

def submit_async_task(coro):
    return _submit_async(coro)


# ==========================================
# 2. RAG Memory Store (Per Thread)
# ==========================================
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}

def _get_retriever(thread_id: Optional[str]):
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None

def ingest_pdf(file_bytes: bytes, thread_id: str, openai_key: str, filename: Optional[str] = None) -> dict:
    """Build a FAISS retriever for the uploaded PDF and store it for the thread."""
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")
    if not openai_key:
        raise ValueError("OpenAI API key is missing!")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        # Initialize user-specific embeddings!
        user_embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_key)
        
        vector_store = FAISS.from_documents(chunks, user_embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return _THREAD_METADATA[str(thread_id)]
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS

def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})

def remove_document(thread_id: str):
    """Clear the indexed document for a specific thread."""
    thread_key = str(thread_id)
    if thread_key in _THREAD_RETRIEVERS:
        del _THREAD_RETRIEVERS[thread_key]
    if thread_key in _THREAD_METADATA:
        del _THREAD_METADATA[thread_key]

# ==========================================
# 3. Tools Definition
# ==========================================
search_tool = DuckDuckGoSearchRun(
    name="DuckDuckGoSearchRun",
    description="Search the web for current information"
)

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Perform a basic arithmetic operation on two numbers. Supported operations: add, sub, mul, div"""
    try:
        if operation == "add": result = first_num + second_num
        elif operation == "sub": result = first_num - second_num
        elif operation == "mul": result = first_num * second_num
        elif operation == "div":
            if second_num == 0: return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_stock_price(symbol: str) -> dict: # <-- Removed config from arguments!
    """Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')."""
    api_key = stock_key_var.get() # <-- Read securely from RAM
    
    if not api_key:
        return {"error": "No Alpha Vantage API key provided in the sidebar."}
        
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    r = requests.get(url)
    return r.json()

@tool
async def rag_tool(query: str, thread_id: str) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    # Use ainvoke for true async retrieval
    result = await retriever.ainvoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }

# MCP Setup
async def _fetch_mcp_tools_async():
    # We build the client INSIDE the async function so it attaches to the correct thread
    client = MultiServerMCPClient({
        "ExpenseTracker": {
            "transport": "streamable_http",
            "url": "https://redundant-fuchsia-roundworm.fastmcp.app/mcp"
        }
    })
    return await client.get_tools()

def load_mcp_tools():
    try:
        # Run the entire process safely in our background loop
        tools = run_async(_fetch_mcp_tools_async())
        print(f"\n✅ SUCCESS: Loaded {len(tools)} tools from remote MCP server!\n")
        return tools
    except Exception as e:
        print(f"\n❌ MCP CONNECTION ERROR: {e}\n")
        return []

mcp_tools = load_mcp_tools()

# Combine all tools
tools = [search_tool, get_stock_price, calculator, rag_tool, *mcp_tools]

# ==========================================
# 4. State & Nodes
# ==========================================
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    title: Optional[str]

# Pass config into the node to extract the thread_id
async def chat_node(state: ChatState, config: RunnableConfig):
    """LLM node that may answer or request a tool call."""
    
    # Extract thread_id from the runtime config
    thread_id = config.get("configurable", {}).get("thread_id", "unknown")
    user_email = config.get("configurable", {}).get("user_id", "unknown_email") # <--- Get the email
    # Grab the key securely from RAM, not the config!
    openai_key = openai_key_var.get()

    if not openai_key:
        from langchain_core.messages import AIMessage
        return {"messages": [AIMessage(content="⚠️ Please enter your OpenAI API Key in the sidebar to chat.")]}
    
    # Initialize the LLM dynamically and bind the tools
    user_llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_key, model_kwargs={"stream_options": {"include_usage": True}})
    user_llm_with_tools = user_llm.bind_tools(tools) if tools else user_llm
    
    system_message = SystemMessage(
        content=(
            "You are a highly capable AI assistant. "
            f"The current thread_id is `{thread_id}` and the current user is `{user_email}`. "
            "If the user asks questions about an uploaded PDF or document, you MUST use the `rag_tool` "
            "and pass this exact thread_id to it." 
            "CRITICAL PRIVACY RULE: If you use any Expense Tracking tools, you MUST always pass the user's email "
            f"(`{user_email}`) to the tool so they only see their own private expenses."
            "You also have access to web search, stock prices, "
            "a calculator, and expense tracking tools."
        )
    )
    
    # Prepend the system prompt to the user's messages
    messages = [system_message] + state["messages"]
    
    # We pass config down so callbacks and usage tracking continue to work
    try:
        response = await user_llm_with_tools.ainvoke(messages, config=config)
        return {"messages": [response]}
    except Exception as e:
        error_str = str(e)
        from langchain_core.messages import AIMessage
        
        if "401" in error_str or "Incorrect API key" in error_str:
            err_msg = "⚠️ **Authentication Error:** The OpenAI API key provided is incorrect. Please update it in the sidebar."
        elif "insufficient_quota" in error_str or "429" in error_str:
            err_msg = "⚠️ **Billing Error:** Your OpenAI API key has run out of credits or hit a rate limit."
        else:
            err_msg = f"⚠️ **System Error:** {error_str}"
            
        # Return the error as an official AI Message so Supabase saves it!
        return {"messages": [AIMessage(content=err_msg)]}


tool_node = ToolNode(tools) if tools else None

async def generate_title_node(state: ChatState):
    openai_key = openai_key_var.get()
    if not openai_key:
        return {"title": "New Chat"}
        
    title_llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_key)
    
    messages = state['messages']
    first_message = messages[0].content
    prompt = f"""Generate a very short concise title (upto 3 words) Summarizing the Title of this message.
                 Your work is Just to generate the title, not to answer the question.
                 Message: {first_message}"""
    
# MAGIC HAPPENS HERE: Catch the error so it doesn't break the background process
    try:
        response = await title_llm.ainvoke(prompt)
        return {"title": response.content.strip()}
    except Exception as e:
        print(f"Title generation failed (likely bad API key): {e}")
        return {"title": "New Chat"} # Graceful fallback

async def route_start(state: ChatState):
    if len(state['messages']) == 1:
        return ["chat_node", "generate_title_node"]
    return ["chat_node"]



# -------------------
# 5. Checkpointer (Supabase / Postgres)
# -------------------

# -------------------
# 5. Checkpointer (Supabase / Postgres)
# -------------------
@st.cache_resource
def get_checkpointer():
    """
    Streamlit safely caches the result of this function.
    We push the actual async initialization into our background thread 
    so LangGraph can find a running event loop!
    """
    print("🗄️ Initializing Global Database Pool...")
    db_uri = os.environ.get("DATABASE_URL")
    if not db_uri:
        raise ValueError("DATABASE_URL environment variable is missing!")

    # 2. Wrap the strict async requirements in a coroutine
    async def _setup_async_components():
        # 🛡️ THE FIX: Instantiate the pool INSIDE the async thread
        # This guarantees all internal Locks are bound to the correct background event loop!
        pool = AsyncConnectionPool(
            conninfo=db_uri,
            max_size=20, 
            kwargs={"autocommit": True},
            open=False
        )
        
        await pool.open(wait=True)
        # Because this is inside an 'async def', get_running_loop() will succeed!
        cp = AsyncPostgresSaver(pool) 
        await cp.setup()
        return cp

    # 3. Fire it into our background thread and wait for the initialized object
    return run_async(_setup_async_components())

# Streamlit fetches the safely cached singleton
checkpointer = get_checkpointer()


graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("generate_title_node", generate_title_node)
graph.add_node("tools", tool_node)

graph.add_conditional_edges(START, route_start)
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge('tools', 'chat_node')
graph.add_edge("chat_node", END)
graph.add_edge("generate_title_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

# ==========================================
# 6. Helpers
# ==========================================
# ==========================================
async def _alist_user_threads(user_email: str):
    all_threads = set()
    
    # MAGIC HAPPENS HERE: We tell Supabase to only return checkpoints 
    # where the user_id in the configurable dict matches the logged-in email!
    async for checkpoint in checkpointer.alist(None, filter={"user_id": user_email}):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
        
    return list(all_threads)

def retrieve_user_threads(user_email: str):
    """Call this from Streamlit to populate the sidebar for the specific user."""
    return run_async(_alist_user_threads(user_email))