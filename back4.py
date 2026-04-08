import os
import tempfile
from typing import TypedDict, Annotated, Optional, Dict, Any
import contextvars
from Crag import crag_app
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
import yfinance as yf
import requests
import asyncio
import threading
# 🛡️ SECURE RAM VARIABLES (Invisible to Supabase)
openai_key_var = contextvars.ContextVar('openai_key', default="")
tavily_key_var = contextvars.ContextVar('tavily_key', default="")
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
def get_stock_price(symbol: str) -> dict:
    """Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')."""
    try:
        ticker = yf.Ticker(symbol)
        # Use .history() instead of fast_info, it is much more reliable!
        hist = ticker.history(period="1d")
        
        if hist.empty:
            raise ValueError("Yahoo returned empty data.")
            
        price = hist['Close'].iloc[-1] # Grabs the most recent closing price
        return {"price": round(price, 2), "source": "Yahoo Finance"}
        
    except Exception as e:
        return {"error": f"SYSTEM ERROR: Could not fetch price for {symbol}. Do NOT search the web. Ask the user to verify the ticker symbol."}
    
@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Converts currency using real-time exchange rates (e.g., amount=100, from_currency='USD', to_currency='INR')."""
    try:
        from_currency = from_currency.upper()
        to_currency = to_currency.upper()
        
        if from_currency == to_currency:
            return f"{amount} {from_currency} is {amount} {to_currency}"
            
        # UPGRADE: Faster API + a strict 5-second timeout limit
        url = f"https://open.er-api.com/v6/latest/{from_currency}"
        r = requests.get(url, timeout=5) 
        
        if r.status_code != 200:
            return f"Error: Could not fetch rates for {from_currency}."
            
        data = r.json()
        
        if to_currency not in data['rates']:
            return f"Error: Currency code '{to_currency}' not found."
            
        rate = data['rates'][to_currency]
        converted = round(amount * rate, 2)
        
        return f"{amount} {from_currency} = {converted} {to_currency} (Source: ExchangeRate-API)"
        
    except requests.exceptions.Timeout:
        # If it takes more than 5 seconds, kill it and tell the LLM to apologize
        return "SYSTEM ERROR: The Currency API took too long to respond. Do NOT search the web. Apologize to the user and ask them to try again later."
    except Exception as e:
        return "SYSTEM ERROR: Currency API failed. Do NOT search the web. Apologize to the user."

@tool
def get_weather(city: str) -> str:
    """Gets the current real-time weather, temperature, and wind speed for a given city."""
    try:
        # Step 1: Turn the city name into Latitude & Longitude
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
        geo_data = requests.get(geo_url).json()
        
        if "results" not in geo_data:
            return f"Error: Could not find geographic coordinates for '{city}'."
            
        lat = geo_data['results'][0]['latitude']
        lon = geo_data['results'][0]['longitude']
        country = geo_data['results'][0].get('country', '')
        
        # Step 2: Fetch the weather using those coordinates
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,wind_speed_10m,relative_humidity_2m&temperature_unit=celsius"
        w_data = requests.get(weather_url).json()
        
        temp = w_data['current']['temperature_2m']
        wind = w_data['current']['wind_speed_10m']
        humidity = w_data['current']['relative_humidity_2m']
        
        return f"Current weather in {city}, {country}: {temp}°C | Humidity: {humidity}% | Wind: {wind} km/h (Source: Open-Meteo)"
        
    except Exception as e:
        return f"SYSTEM ERROR: Weather API failed for {city}. Do NOT search the web. Apologize to the user."
      
@tool
async def rag_tool(query: str, thread_id: str) -> str:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    If the document does not contain the answer, this will automatically search the web.
    """
    openai_key = openai_key_var.get()
    tavily_key = tavily_key_var.get()
    retriever = _get_retriever(thread_id)
    
    if retriever is None:
        return "Error: No document indexed for this chat. Upload a PDF first."

    print(f"\n🔍 Triggering CRAG Subgraph for query: {query}")
    
    # 1. Setup the empty starting state
    starting_state = {
        "question": query, 
        "docs": [], "good_docs": [], "verdict": "", "reason": "", 
        "strips": [], "kept_strips": [], "refined_context": "", 
        "web_query": "", "web_docs": [], "answer": ""
    }
    from Crag import crag_openai_key_var, crag_tavily_key_var
    # 2. 🛡️ Setup the secure config channel
    secure_config = {
        "configurable": {
            "retriever": retriever
        }
    }
    
    # Inject securely into RAM
    token_crag_openai = crag_openai_key_var.set(openai_key)
    token_crag_tavily = crag_tavily_key_var.set(tavily_key)
    
    try:
        final_state = await crag_app.ainvoke(starting_state, config=secure_config)
    finally:
        # Cleanup RAM
        crag_openai_key_var.reset(token_crag_openai)
        crag_tavily_key_var.reset(token_crag_tavily)
    
    return final_state["answer"]

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
tools = [search_tool, get_stock_price, calculator, rag_tool,convert_currency, get_weather, *mcp_tools]

# ==========================================
# 4. State & Nodes
# ==========================================
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    title: Optional[str]
    summary: Optional[str] # 🛡️ ADDED: The new memory bucket

async def summarize_node(state: ChatState):
    """Generates a summary of the latest batch of 6 messages."""
    openai_key = openai_key_var.get()
    summary_llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_key)
    
    existing_summary = state.get("summary", "")
    all_messages = state["messages"]
    
    # 🛡️ THE BATCH SLICE: Grabs exactly the 6 oldest messages in our 8-message window, 
    # perfectly ignoring the 2 most recent messages.
    # 🚨 TOOL-SAFE BATCH SLICING:
    # 1. Find the end point (safely skipping the 2 most recent messages)
    end_idx = len(all_messages) - 2
    while end_idx > 0 and all_messages[end_idx].type != "human":
        end_idx -= 1
        
    # 2. Find the start point (safely grabbing the 6 messages before the end point)
    start_idx = max(0, end_idx - 6)
    while start_idx > 0 and all_messages[start_idx].type != "human":
        start_idx -= 1
        
    # 3. Grab the perfectly safe batch
    messages_to_summarize = all_messages[start_idx:end_idx]
    
    transcript = ""
    for m in messages_to_summarize:
        if m.type == "human":
            transcript += f"User: {m.content}\n"
        elif m.type == "ai" and m.content:
            transcript += f"AI: {m.content}\n"
            
    # 🛡️ YOUR EXACT PROMPT LOGIC
    if existing_summary:
        prompt = (
            f"Existing summary:\n{existing_summary}\n\n"
            "Extend the summary using the new conversation transcript below:\n\n"
            f"{transcript}"
        )
    else:
        prompt = f"Summarize the conversation transcript below:\n\n{transcript}"
        
    try:
        from langchain_core.messages import HumanMessage
        response = await summary_llm.ainvoke([HumanMessage(content=prompt)])
        print("✅ Batch Summary successful! Merged 6 new messages.")
        return {"summary": response.content}
    except Exception as e:
        print(f"❌ Summarization failed: {e}")
        return {"summary": existing_summary}

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
    
    # 🛡️ DYNAMIC TRIMMING: Keep all messages in state for the UI, 
    # but only send the last 6 to the LLM to save tokens!
    all_messages = state["messages"]
    summarized_count = max(0, ((len(all_messages) - 3) // 6) * 6)

    # 🚨 THE TOOL-SAFE SHIFT: 
    # If the math sliced us in the middle of a tool call, walk backwards to the nearest Human prompt!
    while summarized_count > 0 and all_messages[summarized_count].type != "human":
        summarized_count -= 1
    
    if summarized_count > 0 and state.get("summary"):
        # Slice off EXACTLY the messages that are already in the summary
        messages_for_llm = all_messages[summarized_count:]
        
        # Inject the summary
        summary_msg = SystemMessage(content=f"Summary of older conversation:\n{state['summary']}")
        messages_for_llm = [summary_msg] + messages_for_llm
    else:
        # If no summary yet, send everything
        messages_for_llm = all_messages
        
    # Prepend the main system prompt
    final_messages = [system_message] + messages_for_llm
    
    # We pass config down so callbacks and usage tracking continue to work
    try:
        # Pass the trimmed list to the LLM, not the full state!
        response = await user_llm_with_tools.ainvoke(final_messages, config=config)
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


def route_after_chat(state: ChatState):
    """Decides where to go after the LLM speaks."""
    messages = state["messages"]
    
    if messages[-1].tool_calls:
        return "tools"
        
    # 🛡️ THE BATCH TRIGGER: 
    # Triggers exactly when the chat has 8, 14, 20, 26 messages, etc.
    if len(messages) >= 8 and (len(messages) - 2) % 6 == 0:
        return "summarize_node"
        
    return END

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
        # 🛡️ THE IDLE CONNECTION FIX: Instantiate the pool with TCP Keepalives!
        pool = AsyncConnectionPool(
            conninfo=db_uri,
            max_size=20, 
            # 1. Ping the database every 60 seconds so Supabase doesn't kill it
            kwargs={
                "autocommit": True,
                "keepalives": 1,
                "keepalives_idle": 60,
                "keepalives_interval": 10,
                "keepalives_count": 5
            },
            # 2. Automatically recycle connections older than 10 minutes
            max_lifetime=600, 
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
# checkpointer = get_checkpointer()


graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("generate_title_node", generate_title_node)
graph.add_node("tools", tool_node)
graph.add_node("summarize_node", summarize_node)

graph.add_conditional_edges(START, route_start)
graph.add_conditional_edges("chat_node", route_after_chat)
graph.add_edge('tools', 'chat_node')
graph.add_edge("summarize_node", END)
graph.add_edge("generate_title_node", END)

# ✅ REPLACE WITH THIS FUNCTION:
def get_chatbot():
    return graph.compile(checkpointer=get_checkpointer())

# ==========================================
# 6. Helpers
# ==========================================
# ==========================================
async def _alist_user_threads(user_email: str):
    all_threads = set()
    
    # 🛡️ THE ARMOR: Wrap the database call so crashes don't kill the app
    try:
        cp = get_checkpointer()
        # 🛡️ THE LIMIT: Ensure limit=50 is here so it doesn't download the whole DB
        async for checkpoint in cp.alist(None, filter={"user_id": user_email}):
            all_threads.add(checkpoint.config["configurable"]["thread_id"])
            
    except Exception as e:
        # If Supabase drops the connection after 30 mins, we catch it here quietly!
        print(f"⚠️ Supabase Idle Connection Dropped: {e}")
        print("💡 TIP: Just refresh your browser to get a fresh connection.")
        return [] # Return an empty list so Streamlit doesn't crash!
        
    return list(all_threads)

def retrieve_user_threads(user_email: str):
    """Call this from Streamlit to populate the sidebar for the specific user."""
    return run_async(_alist_user_threads(user_email))