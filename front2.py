from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
import os
from streamlit_oauth import OAuth2Component
import streamlit as st
import uuid
import queue
from back2 import (
    chatbot, retrieve_user_threads, submit_async_task,
    ingest_pdf, thread_document_metadata, remove_document,
    openai_key_var, stock_key_var
)

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
# Add remove_document to your import list!

# 1. Configure the browser tab (MUST be the first Streamlit command!)
st.set_page_config(page_title="LangGraph Agentic AI RAG Based System", page_icon="🤖")

# 2. Helper function to load the external CSS file
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 3. Inject the CSS into the app
load_css("style.css")

# ==========================================
# 🔐 GOOGLE SSO LOGIN SYSTEM
# ==========================================
CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")
REDIRECT_URI = os.environ.get("REDIRECT_URI", "http://localhost:8501")

# Initialize the Google OAuth Component
oauth2 = OAuth2Component(
    CLIENT_ID, 
    CLIENT_SECRET, 
    "https://accounts.google.com/o/oauth2/v2/auth", 
    "https://oauth2.googleapis.com/token", 
    "https://oauth2.googleapis.com/token", 
    "https://oauth2.googleapis.com/revoke"
)

# If the user is NOT logged in, show the login screen and STOP the app
if "user_email" not in st.session_state:
    st.title("🛡️ Welcome to Agentic AI")
    st.write("Please sign in with your Google account to access your secure workspace.")
    
    # Create the stylish Google Login button
    result = oauth2.authorize_button(
        name="Sign in with Google",
        icon="https://upload.wikimedia.org/wikipedia/commons/5/53/Google_%22G%22_Logo.svg",
        redirect_uri=REDIRECT_URI,
        scope="openid email profile",
        key="google_login",
        use_container_width=True
    )
    
    if result:
        # Grab the raw token Google handed the browser
        raw_token = result["token"]["id_token"]
        
        try:
            # 🛡️ THE FIX: Cryptographically verify the token!
            # This checks the signature, the expiration, and the CLIENT_ID audience.
            payload = id_token.verify_oauth2_token(
                raw_token, 
                google_requests.Request(), 
                CLIENT_ID,
                clock_skew_in_seconds=10 # <--- ADD THIS LINE!
            )
            
            # If the code reaches here, the token is mathematically proven to be real.
            st.session_state["user_email"] = payload["email"]
            st.session_state["user_name"] = payload.get("name", "User")
            st.rerun() # Refresh the page to unlock the app!
            
        except ValueError as e:
            # 🛡️ DEBUG MODE: Print the exact reason Google rejected the token
            st.error(f"🚨 Token Rejected by Google. Reason: {str(e)}")
            st.stop()
            
    st.stop()

# ==========================================
# ✅ MAIN APP UNLOCKED
# ==========================================
# If the code reaches here, the user is successfully logged in!
# 🛡️ THE FIX: Sleek, dark frosted glass profile card
st.sidebar.markdown(f"""
<div style="background-color: rgba(0, 0, 0, 0.4); padding: 15px; border-radius: 10px; margin-bottom: 20px; border: 1px solid rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px);">
    <div style="color: white; font-weight: bold; font-size: 1.05rem; margin-bottom: 5px;">
        👤 {st.session_state['user_name']}
    </div>
    <div style="color: #cbd5e1; font-size: 0.85rem; word-break: break-all;">
        {st.session_state['user_email']}
    </div>
</div>
""", unsafe_allow_html=True)

if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.rerun()

# ... (The rest of your chat UI, file uploader, and sidebar logic stays down here!) ...
# =========================== Utilities ===========================

def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(thread_id)
    st.session_state['message_history'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)
    if thread_id not in st.session_state['thread_titles']:
        st.session_state['thread_titles'][thread_id] = "New Chat"

def load_conversation(thread_id):
    # Inject user_email here so LangGraph can find the secure checkpoint
    config = {'configurable': {'thread_id': thread_id, 'user_id': st.session_state["user_email"]}}
    state = chatbot.get_state(config=config)
    return state.values.get('messages', [])

def get_title_from_state(thread_id):
    # Inject user_email here as well
    config = {'configurable': {'thread_id': thread_id, 'user_id': st.session_state["user_email"]}}
    state = chatbot.get_state(config=config)
    return state.values.get('title', "New Chat")
# ======================= Session Initialization ===================

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_titles' not in st.session_state:
    st.session_state['thread_titles'] = {}

if 'chat_threads' not in st.session_state:
    # Pass the logged-in email to the backend!
    st.session_state['chat_threads'] = retrieve_user_threads(st.session_state["user_email"])

if 'ingested_docs' not in st.session_state:
    st.session_state['ingested_docs'] = {}

if 'thread_id' not in st.session_state:
    reset_chat()

add_thread(st.session_state['thread_id'])

# Grab the active thread key and document tracking dict
thread_key = str(st.session_state['thread_id'])
thread_docs = st.session_state['ingested_docs'].setdefault(thread_key, {})

# ============================ Sidebar ============================

st.sidebar.title('LangGraph Chatbot')

st.sidebar.header("🔑 API Keys")
user_openai_key = st.sidebar.text_input("OpenAI API Key", type="password", placeholder="sk-...")
user_stock_key = st.sidebar.text_input("Alpha Vantage Key", type="password", placeholder="Optional...")

# Stop the app from running the chat if they haven't provided an OpenAI key
if not user_openai_key:
    st.sidebar.warning("Please enter your OpenAI API Key to start chatting.")
    st.title("Welcome to your Autonomous Workspace 🧠")
    st.info("Please enter your OpenAI API key in the sidebar to activate your Agent.")
    st.stop()

st.sidebar.caption(f"**Thread ID:** `{thread_key[:12]}...`")

if st.sidebar.button('New Chat', use_container_width=True):
    reset_chat()
    st.rerun()

st.sidebar.divider()

# --- RAG UI ---
doc_meta = thread_document_metadata(thread_key)

if doc_meta:
    # If the backend already has it, show success
    # 🛡️ THE FIX: Premium "Active File" indicator
    st.sidebar.markdown(f"""
    <div style="background-color: rgba(0, 0, 0, 0.4); padding: 15px; border-radius: 10px; border: 1px solid rgba(167, 243, 208, 0.3); backdrop-filter: blur(10px); margin-bottom: 15px;">
        <div style="color: #a7f3d0; font-weight: bold; font-size: 0.9rem; margin-bottom: 5px; word-break: break-word;">
            📄 {doc_meta.get('filename')}
        </div>
        <div style="color: #cbd5e1; font-size: 0.85rem; font-style: italic;">
            ✅ {doc_meta.get('chunks')} chunks indexed
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # THE NEW FIX: Add a button to clear the database for this thread
    if st.sidebar.button("❌ Remove PDF", use_container_width=True):
        remove_document(thread_key)
        st.rerun() # Refresh the UI to show the uploader again!
        
else:
    st.sidebar.markdown("""
        <div style="background-color: rgba(0, 0, 0, 0.2); padding: 15px; border-radius: 10px; border: 1px dashed rgba(255, 255, 255, 0.3); backdrop-filter: blur(10px); text-align: center; margin-bottom: 15px;">
            <div style="color: white; font-size: 0.95rem; font-weight: 600;">
                📂 No PDF indexed for this chat yet.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Render the uploader
    uploaded_pdf = st.sidebar.file_uploader(
        "Upload a PDF for this chat", 
        type=["pdf"], 
        key=f"uploader_{thread_key}"
    )
    
    if uploaded_pdf:
    # 5MB limit = 5 * 1024 * 1024 bytes
        if uploaded_pdf.size > 5242880:
            st.sidebar.error("⚠️ File is too large! Please upload a PDF smaller than 5MB to prevent server overload.")
        else:
                # Require a button click before indexing!
            if st.sidebar.button("Process Document", use_container_width=True):
                with st.sidebar.status("Indexing PDF...", expanded=True) as status_box:
                    ingest_pdf(
                        file_bytes=uploaded_pdf.getvalue(),
                        thread_id=thread_key,
                        openai_key=user_openai_key,
                        filename=uploaded_pdf.name,
                    )
                    status_box.update(label="✅ PDF indexed", state="complete", expanded=False)
                
                st.rerun() 
# --------------
# -------------------------------------------------

st.sidebar.divider()

# History Tracker
st.sidebar.header('My Conversations')

for thread_id in reversed(st.session_state['chat_threads']):
    title = st.session_state['thread_titles'].get(thread_id, "New Chat")
    if title == "New Chat":
        title = get_title_from_state(thread_id)
        st.session_state['thread_titles'][thread_id] = title

    if st.sidebar.button(title, key=f"btn_{thread_id}"):
        st.session_state['thread_id'] = thread_id
        st.session_state['ingested_docs'].setdefault(str(thread_id), {}) # Ensure doc dict exists
        
        messages = load_conversation(thread_id)
        temp_messages = []
        tools_in_turn = [] 

        for msg in messages:
            if isinstance(msg, HumanMessage):
                temp_messages.append({'role': 'user', 'content': msg.content})
            
            elif isinstance(msg, AIMessage):
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tools_in_turn.append(tc.get('name', 'tool'))
                
                if msg.content and msg.content.strip():
                    temp_messages.append({
                        'role': 'assistant', 
                        'content': msg.content,
                        'tools': list(tools_in_turn) 
                    })
                    tools_in_turn = [] 

        st.session_state['message_history'] = temp_messages
        st.rerun()

# ============================ Main UI ============================
# 🛡️ THE FIX: Wrap the intro text in our custom frosted glass HTML card
# ============================ Main UI ============================

# 🛡️ THE FIX: The "Heavy Glass" Premium Container
st.markdown("""
<div style="background-color: rgba(255, 255, 255, 0.15); backdrop-filter: blur(12px); padding: 2.5rem; border-radius: 15px; border: 1px solid rgba(255, 255, 255, 0.3); box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1); margin-bottom: 2rem;">
    <h1 style="color: #0f172a; margin-top: 0;">🤖 LangGraph Multi-Agent RAG System</h1>
    <h3 style="color: #1e293b;">What can your Agentic AI do?</h3>
    <ol style="color: #1e293b; line-height: 1.8; font-size: 1.1rem; font-weight: 500;">
        <li><b style="color: #0f172a;">Web Search:</b> I can search the web for current information on various topics.</li>
        <li><b style="color: #0f172a;">Stock Prices:</b> I can fetch the latest stock price for any given company (e.g., AAPL for Apple, NVDA for Nvidia).</li>
        <li><b style="color: #0f172a;">Calculator:</b> I can perform basic arithmetic operations such as addition, subtraction, multiplication, and division.</li>
        <li><b style="color: #0f172a;">Expense Tracking:</b> I can help you track expenses by adding new entries, listing expenses within a date range, and summarizing expenses by category.</li>
        <li><b style="color: #0f172a;">RAG:</b> If you upload a PDF or document, I can extract relevant information from it.</li>
    </ol>
</div>
""", unsafe_allow_html=True)

st.divider()

st.subheader("💬 Chat with your assistant")
# 1. Loading the conversation history 
for message in st.session_state['message_history']:

    if message["role"] == "assistant" and not message["content"].strip():
        continue

    with st.chat_message(message["role"]):
        if message.get("tools"):
            with st.status("✅ Tool finished", state="complete", expanded=False):
                for t in message["tools"]:
                    st.write(f"Used `{t}`")
        
        safe_content = message["content"].replace("$", "＄")
        st.markdown(safe_content)

        # 🛡️ THE NEW FIX: Redraw the caption permanently from history
        if message.get("tools") and "rag_tool" in message["tools"]:
            doc_meta = thread_document_metadata(thread_key)
            if doc_meta:
                st.caption(f"📄 **Source Document:** {doc_meta.get('filename')} ({doc_meta.get('chunks')} chunks)")
# User Input
user_input = st.chat_input('Type here')

if user_input:
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)
# Send Message to Backend
    CONFIG = {
        'configurable': {
            'thread_id': thread_key,
            'user_id': st.session_state["user_email"] # <--- Tags the DB row with their email
        },
        "metadata" : {"thread_id": thread_key},
        "run_name":"chat_turn"
    }

    assistant_container = st.chat_message("assistant")
    with assistant_container:
        status_holder = {"box": None}
        used_tools = [] 
        
        def ai_only_stream():
            event_queue = queue.Queue()

            async def run_stream():
                # 1. 🛡️ Set the variables and CAPTURE the tokens!
                token_openai = openai_key_var.set(user_openai_key)
                token_stock = stock_key_var.set(user_stock_key)
                try:
                    # 🛡️ INJECT KEYS INTO SECURE SERVER RAM
                    async for message_chunk, metadata in chatbot.astream(
                        {"messages": [HumanMessage(content=user_input)]},
                        config=CONFIG,
                        stream_mode="messages"
                    ):
                        event_queue.put((message_chunk, metadata))
                except Exception as exc:
                    event_queue.put(("error", exc))
                finally:
                    # 2. 🧹 GUARANTEED CLEANUP: Reset RAM using the tokens
                    # This completely eliminates cross-user context bleeding.
                    openai_key_var.reset(token_openai)
                    stock_key_var.reset(token_stock)
                    event_queue.put(None)

            submit_async_task(run_stream())

            while True:
                item = event_queue.get()
                if item is None:
                    break
                
                if isinstance(item, tuple) and item[0] == "error":
                    error_obj = item[1]
                    error_str = str(error_obj)
                    
                    # Check if it's an API Key error
                    if "401" in error_str or "Incorrect API key" in error_str:
                        yield "⚠️ **Authentication Error:** The OpenAI API key provided in the sidebar is incorrect or invalid. Please double-check it and try again."
                    elif "insufficient_quota" in error_str or "429" in error_str:
                        yield "⚠️ **Billing Error:** Your OpenAI API key has run out of credits or hit its rate limit."
                    else:
                        yield f"⚠️ **System Error:** {error_str}"
                        
                    break # Stop the stream safely without crashing the app!

                message_chunk, metadata = item

                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if tool_name not in used_tools:
                        used_tools.append(tool_name) 

                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` ...", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` ...", state="running", expanded=True
                        )
                
                if (
                    isinstance(message_chunk, AIMessage)
                    and metadata.get("langgraph_node") == "chat_node"
                ):
                    yield message_chunk.content.replace("$", "＄")

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )
            for t in used_tools:
                status_holder["box"].write(f"Used `{t}`")
                
        # 🛡️ THE NEW FIX: Draw the caption live inside the chat bubble
        if "rag_tool" in used_tools:
            doc_meta = thread_document_metadata(thread_key)
            if doc_meta:
                st.caption(f"📄 **Source Document:** {doc_meta.get('filename')} ({doc_meta.get('chunks')} chunks)")

        # Normal success! Save the AI message to history.
    st.session_state['message_history'].append({
        'role': 'assistant', 
        'content': ai_message,
        'tools': used_tools 
    })

    # 4. Handle Title Generation
    if st.session_state['thread_titles'].get(thread_key) == "New Chat":
        new_title = get_title_from_state(thread_key)
        if new_title and new_title != "New Chat":
            st.session_state['thread_titles'][thread_key] = new_title
            st.rerun()