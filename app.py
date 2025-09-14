# streamlit run app.py
"""
Nepal Government Assistant - Memory-Enhanced Version (FIXED)
- All bugs fixed and improvements added
- Better error handling and input validation
- Optimized memory management
"""

import os
import re
import mimetypes
import pickle
import numpy as np
import streamlit as st
import faiss
import time
import hashlib
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from google.api_core.exceptions import ResourceExhausted
import base64

from gemini_client import model
from faiss_check import embed_query, retrieve_similar_chunk

# ----------------------------
# Configuration
# ----------------------------
FAISS_PATH = "gov_index.faiss"
CHUNKS_PATH = "chunks.pkl"
MEMORY_PATH = "memory.json"
CACHE_SIZE = 100
MAX_HISTORY_CONTEXT = 10
MAX_MEMORY_SIZE = 1000  # Maximum messages to store in memory

st.set_page_config(
    page_title="Nepal Gov Assistant", 
    page_icon="🇳🇵", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------------------
# Memory Management Functions (FIXED)
# ----------------------------
def load_memory() -> Dict[str, Any]:
    """Load conversation history from memory.json with error handling"""
    if os.path.exists(MEMORY_PATH):
        try:
            with open(MEMORY_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Ensure structure is correct
                if not isinstance(data, dict):
                    return {"conversations": []}
                if "conversations" not in data:
                    data["conversations"] = []
                return data
        except (json.JSONDecodeError, FileNotFoundError, Exception) as e:
            st.warning(f"Memory file corrupted or unreadable. Starting fresh. Error: {str(e)}")
            return {"conversations": []}
    return {"conversations": []}

def save_memory(memory_data: Dict[str, Any]) -> bool:
    """Save conversation history to memory.json with size management"""
    try:
        # Limit memory size to prevent file from growing too large
        conversations = memory_data.get("conversations", [])
        if len(conversations) > MAX_MEMORY_SIZE:
            # Keep only the most recent messages
            memory_data["conversations"] = conversations[-MAX_MEMORY_SIZE:]
        
        # Create backup before saving
        if os.path.exists(MEMORY_PATH):
            backup_path = f"{MEMORY_PATH}.backup"
            with open(MEMORY_PATH, 'r', encoding='utf-8') as f:
                backup_data = f.read()
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(backup_data)
        
        # Save new data
        with open(MEMORY_PATH, 'w', encoding='utf-8') as f:
            json.dump(memory_data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        st.error(f"Failed to save memory: {str(e)}")
        # Try to restore from backup
        backup_path = f"{MEMORY_PATH}.backup"
        if os.path.exists(backup_path):
            try:
                with open(backup_path, 'r', encoding='utf-8') as f:
                    backup_data = json.load(f)
                with open(MEMORY_PATH, 'w', encoding='utf-8') as f:
                    json.dump(backup_data, f, ensure_ascii=False, indent=2)
                st.info("Restored from backup")
            except:
                pass
        return False

def add_to_memory(role: str, content: str, image_info: Optional[Dict] = None) -> None:
    """Add a new message to memory with validation"""
    if not content or not role:
        return
    
    memory = load_memory()
    
    message = {
        "timestamp": datetime.now().isoformat(),
        "role": role,
        "content": content[:5000]  # Limit content size
    }
    
    if image_info and isinstance(image_info, dict):
        # Store only metadata, not actual image data
        message["image"] = {
            "name": image_info.get("name", "unknown")[:100],
            "type": image_info.get("type", "unknown")[:50]
        }
    
    memory["conversations"].append(message)
    save_memory(memory)

def get_conversation_context(current_query: str, max_messages: int = MAX_HISTORY_CONTEXT) -> str:
    """Get relevant conversation history for context with improved formatting"""
    memory = load_memory()
    conversations = memory.get("conversations", [])
    
    if not conversations:
        return ""
    
    # Filter out empty or invalid messages
    valid_conversations = [
        msg for msg in conversations 
        if msg.get("content") and msg.get("role") and len(msg.get("content", "")) > 0
    ]
    
    if not valid_conversations:
        return ""
    
    # Get recent messages (limit to max_messages)
    recent_messages = valid_conversations[-max_messages:] if len(valid_conversations) > max_messages else valid_conversations
    
    # Format conversation history
    context_parts = []
    for msg in recent_messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        
        # Clean and truncate content
        content = re.sub(r'\s+', ' ', content).strip()
        if len(content) > 200:
            content = content[:197] + "..."
        
        if msg.get("image"):
            image_name = msg['image'].get('name', 'uploaded')
            content = f"[Image: {image_name}] {content}"
        
        context_parts.append(f"{role.capitalize()}: {content}")
    
    return "\n".join(context_parts)

def analyze_conversation_patterns() -> Dict[str, Any]:
    """Analyze conversation patterns with error handling"""
    try:
        memory = load_memory()
        conversations = memory.get("conversations", [])
        
        if not conversations:
            return {"total_messages": 0, "top_topics": [], "has_images": False}
        
        # Analyze common topics
        topics = {}
        for msg in conversations:
            if msg.get("role") == "user" and msg.get("content"):
                content_lower = msg.get("content", "").lower()
                # Common government service keywords
                keywords = ["citizenship", "passport", "license", "certificate", "tax", 
                           "registration", "application", "document", "fee", "office",
                           "nagarikta", "pramanpatra", "darta", "kar", "karyalaya"]
                for keyword in keywords:
                    if keyword in content_lower:
                        topics[keyword] = topics.get(keyword, 0) + 1
        
        # Get most discussed topics
        top_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            "total_messages": len(conversations),
            "top_topics": top_topics,
            "has_images": any(msg.get("image") for msg in conversations)
        }
    except Exception as e:
        st.warning(f"Could not analyze patterns: {str(e)}")
        return {"total_messages": 0, "top_topics": [], "has_images": False}

# ----------------------------
# Dark Theme CSS (Fixed Unicode)
# ----------------------------
def inject_dark_theme():
    """Inject beautiful dark theme with fixed encoding."""
    return """
    <style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global dark theme */
    .stApp {
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container styling */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 95%;
    }
    
    /* Chat container with glassmorphism */
    .chat-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Title styling */
    .main-title {
        text-align: center;
        color: #ffffff;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 20px rgba(255, 255, 255, 0.3);
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .subtitle {
        text-align: center;
        color: #b0b0b0;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Input container styling */
    .input-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Memory stats card */
    .memory-stats {
        background: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .memory-stats h4 {
        color: #667eea;
        margin-bottom: 0.5rem;
    }
    
    .memory-stats p {
        color: #b0b0b0;
        font-size: 0.9rem;
        margin: 0.25rem 0;
    }
    
    /* Streamlit input styling */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 2px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 15px !important;
        color: #ffffff !important;
        font-size: 1.1rem !important;
        padding: 12px 20px !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.3) !important;
    }
    
    /* Action buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 12px 25px !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        font-family: 'Inter', sans-serif !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5) !important;
    }
    
    /* Chat messages styling */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 15px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        margin: 1rem 0 !important;
        backdrop-filter: blur(10px) !important;
    }
    
    /* Metrics styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 1rem 0;
    }
    
    /* File uploader styling */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 15px !important;
        border: 2px dashed rgba(255, 255, 255, 0.3) !important;
        padding: 1rem !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    </style>
    """

# ----------------------------
# Load Resources (FIXED)
# ----------------------------
@st.cache_data(max_entries=CACHE_SIZE, ttl=3600)
def cached_embed_query(query_hash: str, query_text: str):
    """Cached query embedding with error handling"""
    try:
        return embed_query(query_text)
    except Exception as e:
        st.error(f"Embedding error: {str(e)}")
        return None

@st.cache_data(max_entries=CACHE_SIZE, ttl=7200)
def cached_get_contexts(query_hash: str, query_text: str, index_size: int) -> List[str]:
    """Get cached contexts with error handling"""
    try:
        query_embedding = cached_embed_query(query_hash, query_text)
        if query_embedding is None:
            return []
        contexts = retrieve_similar_chunk(query_embedding, ALL_CHUNKS, index, top_k=3)
        return contexts if contexts else []
    except Exception as e:
        st.error(f"Context retrieval error: {str(e)}")
        return []

@st.cache_resource(show_spinner=True)
def load_resources() -> Tuple[Any, List]:
    """Load FAISS index and chunks with validation"""
    if not os.path.exists(FAISS_PATH):
        raise RuntimeError("❌ gov_index.faiss not found. Please generate it first.")
    
    if not os.path.exists(CHUNKS_PATH):
        raise RuntimeError("❌ chunks.pkl not found. Please generate chunks first.")
    
    try:
        index = faiss.read_index(FAISS_PATH)
        
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
        
        # Validate loaded resources
        if index is None or not chunks:
            raise RuntimeError("Invalid index or chunks data")
        
        return index, chunks
    except Exception as e:
        raise RuntimeError(f"Failed to load resources: {str(e)}")

# Initialize resources with error handling
try:
    index, ALL_CHUNKS = load_resources()
except RuntimeError as e:
    st.error(str(e))
    st.info("Please ensure all required files are in place and run the setup script first.")
    st.stop()

# ----------------------------
# Core Functions (FIXED)
# ----------------------------
def detect_language(query: str) -> str:
    """Detect language with improved accuracy"""
    if not query:
        return "english"
    
    # Check for Devanagari script
    if re.search(r'[\u0900-\u097F]', query):
        return "nepali"
    
    nepali_words = [
        "malai", "mero", "timro", "hamro", "tapai", "ma", "timi", "uni",
        "garnu", "garne", "gareko", "garera", "garchhu", "garcha",
        "parne", "parcha", "pareko", "parera", "paryo",
        "huncha", "hune", "bhayo", "thiyo", "cha", "chha", "chaina", 
        "chaincha", "chahincha", "chahiyeko", "chahiye",
        "kasari", "kaha", "kahile", "kati", "kun", "ke", "ko", "ka",
        "nagarikta", "nagrita", "pramanpatra", "janma", "passport",
        "sahayog", "karyalaya", "sarkar", "shulka", "dastavej",
        "bataideu", "bhannu", "dekhau", "sikau", "help"
    ]
    
    query_lower = query.lower()
    words = query_lower.split()
    total_words = len(words)
    
    if total_words == 0:
        return "english"
    
    nepali_count = sum(1 for word in nepali_words if word in query_lower)
    
    # Improved detection logic
    nepali_ratio = nepali_count / max(total_words, 1)
    
    if nepali_ratio >= 0.25 or nepali_count >= 3:
        return "semi"
    elif nepali_count >= 1 and total_words <= 5:
        return "semi"
    
    return "english"

def safe_api_call(func, *args, **kwargs) -> Optional[Any]:
    """Safe API call with exponential backoff"""
    max_retries = 3
    base_delay = 1
    
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except ResourceExhausted:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + (0.1 * time.time() % 1)  # Add jitter
                st.warning(f"⏱️ Rate limit hit. Waiting {delay:.1f}s before retry...")
                time.sleep(delay)
            else:
                st.error("API rate limit exceeded. Please try again later.")
                return None
        except Exception as e:
            st.error(f"API Error: {str(e)}")
            return None
    
    return None

def build_memory_aware_prompt(user_query: str, contexts: List[str], language: str, conversation_history: str) -> str:
    """Build prompt with conversation memory context"""
    # Safely handle contexts
    context_text = ""
    if contexts:
        valid_contexts = [str(c)[:500] for c in contexts[:2] if c]
        context_text = "\n".join(valid_contexts)
    
    # Analyze conversation patterns
    patterns = analyze_conversation_patterns()
    
    # Build conversation context string
    conv_context = ""
    if conversation_history:
        conv_context = f"\n\nPrevious conversation context:\n{conversation_history}"
    
    # Add pattern insights if available
    pattern_context = ""
    if patterns.get("top_topics"):
        topics = ", ".join([f"{topic[0]} ({topic[1]} times)" for topic in patterns["top_topics"][:3]])
        pattern_context = f"\nUser has previously asked about: {topics}"
    
    prompts = {
        "english": f"""You are a helpful Nepal Government Assistant. Consider the conversation history to provide contextual and consistent responses.

Relevant documents: {context_text}{conv_context}{pattern_context}

Current question: {user_query}

Provide a helpful, contextual response considering what was discussed before:""",
        
        "semi": f"""You are a helpful Nepal Government Assistant. Answer in mixed English-Nepali style. Consider previous conversation for context.

Relevant documents: {context_text}{conv_context}{pattern_context}

Current question: {user_query}

Provide a helpful response considering previous discussion:""",
        
        "nepali": f"""तपाईं नेपाल सरकार सहायक हुनुहुन्छ। पहिलेको कुराकानीलाई ध्यान दिंदै उत्तर दिनुहोस्।

सान्दर्भिक कागजातहरू: {context_text}{conv_context}{pattern_context}

हालको प्रश्न: {user_query}

पहिलेको छलफललाई ध्यान दिंदै उपयोगी उत्तर दिनुहोस्:"""
    }
    
    return prompts.get(language, prompts["english"])

def get_answer(user_query: str, image_data: Optional[Dict] = None) -> str:
    """Get answer with conversation memory context and error handling"""
    if not user_query and not image_data:
        return "Please ask a question or upload an image."
    
    try:
        # Get conversation history for context
        conversation_history = get_conversation_context(user_query or "")
        
        if image_data:
            # Validate image data
            if not isinstance(image_data, dict) or "data" not in image_data or "mime_type" not in image_data:
                return "Invalid image data. Please try uploading again."
            
            # Include conversation context in image analysis
            context_prompt = ""
            if conversation_history:
                context_prompt = f"\n\nConsidering our previous conversation:\n{conversation_history}\n\n"
            
            # Include user text with image analysis
            if user_query and user_query.strip():
                content = [
                    f"Government service question about this image.{context_prompt}User asks: {user_query}",
                    image_data
                ]
            else:
                content = [
                    f"What government service is shown in this image? Provide helpful information.{context_prompt}",
                    image_data
                ]
        else:
            language = detect_language(user_query)
            query_hash = hashlib.md5(user_query.encode()).hexdigest()
            contexts = cached_get_contexts(query_hash, user_query, index.ntotal)
            content = build_memory_aware_prompt(user_query, contexts, language, conversation_history)
        
        response = safe_api_call(model.generate_content, content)
        
        if response and hasattr(response, 'text') and response.text:
            return response.text.strip()
        else:
            return "Sorry, I couldn't process your request right now. Please try again in a moment."
            
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        st.error(error_msg)
        return "An error occurred. Please try again or rephrase your question."

# ----------------------------
# Session Management (FIXED)
# ----------------------------
def clear_chat():
    """Clear current session but preserve memory.json"""
    st.session_state.messages = []
    st.session_state.api_calls = 0
    # Clear input states properly
    for key in ['input_text', 'file_uploader', 'user_input_key']:
        if key in st.session_state:
            del st.session_state[key]

def clear_all_memory():
    """Clear both session and persistent memory"""
    clear_chat()
    # Clear memory.json directly without using save_memory function
    # Reset structure
    empty_data = {"conversations": []}
    
    # Overwrite file
    with open(MEMORY_PATH, "w") as f:
        json.dump(empty_data, f, indent=2)
    
    st.cache_data.clear()



# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_calls" not in st.session_state:
    st.session_state.api_calls = 0

if "user_input_key" not in st.session_state:
    st.session_state.user_input_key = 0

# ----------------------------
# Modern Dark UI Layout
# ----------------------------

# Inject dark theme
st.markdown(inject_dark_theme(), unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-title">🇳🇵 Nepal Government Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered guide with conversation memory for contextual assistance</p>', unsafe_allow_html=True)

# Create layout columns
col1, col2 = st.columns([3, 1])

with col2:
    # Memory statistics
    st.markdown('<div class="memory-stats">', unsafe_allow_html=True)
    st.markdown("### 🧠 Memory Status")
    
    patterns = analyze_conversation_patterns()
    total_msgs = patterns.get("total_messages", 0)
    
    st.metric("Total Messages", total_msgs)
    
    if patterns.get("top_topics"):
        st.markdown("**Top Topics:**")
        for topic, count in patterns["top_topics"]:
            st.caption(f"• {topic.capitalize()}: {count}x")
    
    if patterns.get("has_images"):
        st.caption("📷 Contains image interactions")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Performance metrics
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("### ⚡ Performance")
    st.metric("API Calls", st.session_state.api_calls, delta=None)
    progress_val = min(st.session_state.api_calls / 50, 1.0)
    st.progress(progress_val)
    st.markdown(f"**Usage:** {int(progress_val * 100)}% of free tier")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Controls
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("### 🛠️ Controls")
    
    if st.button("🗑️ Clear Session", use_container_width=True, help="Clear current chat but keep memory"):
        clear_chat()
        st.rerun()
    
    if st.button("🧹 Clear All Memory", use_container_width=True):
        clear_all_memory()
        st.success("Memory cleared!")
        time.sleep(1)
        st.rerun()

    
    st.markdown('</div>', unsafe_allow_html=True)

with col1:
    # Chat interface
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.markdown("### 💬 Conversation")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Combined Input Area
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    st.markdown("### 📝 Ask Your Question")
    
    # Create form for better input handling
    with st.form(key="input_form", clear_on_submit=True):
        # Text input
        user_input = st.text_area(
            "Your question:", 
            placeholder="Type your question about Nepal government services... (You can also attach an image below for context)",
            height=100,
            label_visibility="collapsed",
            key=f"text_input_{st.session_state.user_input_key}"
        )
        
        # File upload in the same container
        col_upload, col_status = st.columns([2, 1])
        
        with col_upload:
            uploaded_file = st.file_uploader(
                "📎 Attach Image (Optional)",
                type=['png', 'jpg', 'jpeg', 'gif', 'webp'],
                help="Upload an image to analyze along with your question",
                key=f"file_input_{st.session_state.user_input_key}"
            )
        
        with col_status:
            if uploaded_file:
                file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # MB
                if file_size > 5:  # Limit file size to 5MB
                    st.error("File too large (>5MB)")
                else:
                    st.markdown(f'<div class="file-status">📷 {uploaded_file.name[:20]}... ({file_size:.1f}MB)</div>', 
                              unsafe_allow_html=True)
        
        # Submit button
        submit_button = st.form_submit_button(
            "🚀 Ask Question", 
            type="primary",
            use_container_width=True
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

# Process combined input
if submit_button:
    # Validate input
    has_text = user_input and user_input.strip()
    has_image = uploaded_file is not None
    
    if not has_text and not has_image:
        st.warning("Please enter a question or upload an image.")
    else:
        # Initialize variables
        image_data = None
        image_info = None
        user_message = ""
        
        # Process image if uploaded
        if uploaded_file:
            try:
                # Check file size
                file_bytes = uploaded_file.read()
                file_size = len(file_bytes) / (1024 * 1024)
                
                if file_size > 5:
                    st.error("File size exceeds 5MB limit. Please upload a smaller image.")
                    st.stop()
                
                mime_type, _ = mimetypes.guess_type(uploaded_file.name)
                
                # Validate mime type
                valid_types = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/webp']
                if mime_type and mime_type in valid_types:
                        # Convert to base64 for Gemini API
                    image_data = {
                        "mime_type": mime_type,
                        "data": file_bytes
                    }
                    image_info = {"name": uploaded_file.name, "type": mime_type}
                    
                    # Create user message with both text and image info
                    if has_text:
                        user_message = f"📷 [Image: {uploaded_file.name}]\n{user_input.strip()}"
                    else:
                        user_message = f"📷 [Image uploaded: {uploaded_file.name}] Please analyze this image for government services."
                else:
                    st.error(f"Invalid image type. Please upload: {', '.join([t.split('/')[1].upper() for t in valid_types])}")
                    st.stop()
            except Exception as e:
                st.error(f"File processing error: {str(e)}")
                st.stop()
        else:
            # Text-only message
            user_message = user_input.strip()
        
        # Validate message length
        if len(user_message) > 10000:
            st.error("Message too long. Please keep it under 10,000 characters.")
            st.stop()
        
        # Add user message to session
        st.session_state.messages.append({"role": "user", "content": user_message})
        
        # Add to persistent memory
        add_to_memory("user", user_message, image_info)
        
        # Show thinking indicator
        with st.spinner("🤔 Processing with context awareness..."):
            # Get answer with both text and image
            if image_data:
                answer = get_answer(user_input.strip() if has_text else "", image_data)
            else:
                answer = get_answer(user_message)
            
            st.session_state.api_calls += 1
        
        # Add bot response to session
        st.session_state.messages.append({"role": "assistant", "content": answer})
        
        # Add to persistent memory
        add_to_memory("assistant", answer)
        
        # Increment input key to clear form
        st.session_state.user_input_key += 1
        
        # Rerun to show new messages
        st.rerun()

# ----------------------------
# Error Recovery Section
# ----------------------------
with st.sidebar:
    st.markdown("### 🔧 Troubleshooting")
    
    if st.button("🔄 Reset Application"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    if st.button("📥 Export Memory"):
        try:
            memory = load_memory()
            json_str = json.dumps(memory, indent=2, ensure_ascii=False)
            st.download_button(
                label="Download memory.json",
                data=json_str,
                file_name=f"memory_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        except Exception as e:
            st.error(f"Export failed: {str(e)}")
    
    st.markdown("---")
    st.caption("Version: 2.0 (Fixed)")
    st.caption(f"Memory file: {MEMORY_PATH}")
    st.caption(f"Index: {FAISS_PATH}")

# ----------------------------
# Beautiful Footer (Fixed)
# ----------------------------
st.markdown("""
<div style="
    margin-top: 3rem;
    padding: 2rem;
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(20px);
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    text-align: center;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
">
    <h3 style="color: #667eea; margin-bottom: 1rem;">🏛️ Nepal Government Assistant</h3>
    <p style="color: #b0b0b0; font-size: 1.1rem; margin-bottom: 1rem;">
        Memory-Enhanced Contextual AI Assistant
    </p>
    <div style="display: flex; justify-content: center; gap: 2rem; margin-bottom: 1rem;">
        <span style="color: #4caf50;">✅ Conversation Memory</span>
        <span style="color: #2196f3;">🧠 Context Awareness</span>
        <span style="color: #ff9800;">🖼️ Combined Input</span>
        <span style="color: #9c27b0;">🌙 Auto-Clear</span>
    </div>
    <p style="color: #888; font-size: 0.9rem;">
        Built with ❤️ for Nepal citizens | Powered by Gemini AI
    </p>
</div>
""", unsafe_allow_html=True)