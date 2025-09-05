import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
import os
import time
from typing import Optional, List, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="MedChat - Medical Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'api_token' not in st.session_state:
    st.session_state.api_token = ""
if 'model_initialized' not in st.session_state:
    st.session_state.model_initialized = False

# Configuration - Using environment variables with fallbacks
DB_FAISS_PATH = os.getenv("DB_FAISS_PATH", "vectorstores/db_faiss")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "mistralai/Mistral-7B-v0.1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: bold;
    }
    .chat-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        max-height: 500px;
        overflow-y: auto;
        border: 1px solid #dee2e6;
    }
    .user-message {
        background-color: #2E86AB;
        color: white;
        padding: 12px;
        border-radius: 15px;
        margin: 8px 0;
        text-align: right;
        max-width: 80%;
        margin-left: auto;
    }
    .bot-message {
        background-color: #e9ecef;
        color: #333;
        padding: 12px;
        border-radius: 15px;
        margin: 8px 0;
        text-align: left;
        max-width: 80%;
        margin-right: auto;
    }
    .source-info {
        font-size: 0.8rem;
        color: #6c757d;
        margin-top: 5px;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

custom_prompt_template = """Use the following medical information to answer the user's question accurately. 
If you don't know the answer based on the context, please say that you don't know. Don't make up information.

Context: {context}
Question: {question}

Provide a clear, concise, and medically accurate answer. If the question is not medical-related, politely redirect to medical topics.
Helpful answer:
"""

def set_custom_prompt() -> PromptTemplate:
    """Create and return a custom prompt template."""
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=['context', 'question']
    )
    return prompt

def validate_huggingface_token(token: str) -> bool:
    """Validate Hugging Face token format."""
    if not token or len(token) < 10:
        return False
    return True

def get_api_token() -> str:
    """Get API token from multiple sources with priority."""
    # 1. Check environment variable (for Render deployment)
    token = os.getenv('HUGGINGFACEHUB_API_TOKEN')
    if token and validate_huggingface_token(token):
        return token
    
    # 2. Check Streamlit secrets (alternative for other deployments)
    try:
        if hasattr(st.secrets, 'HUGGINGFACEHUB_API_TOKEN'):
            token = st.secrets.HUGGINGFACEHUB_API_TOKEN
            if validate_huggingface_token(token):
                return token
    except:
        pass
    
    # 3. Check session state (user input)
    if (hasattr(st.session_state, 'api_token') and 
        st.session_state.api_token and 
        validate_huggingface_token(st.session_state.api_token)):
        return st.session_state.api_token
    
    return ""

def load_llm(model_id: Optional[str] = None, temperature: Optional[float] = None) -> Optional[HuggingFaceHub]:
    """Load the Hugging Face LLM with configurable parameters."""
    model_id = model_id or DEFAULT_MODEL
    temperature = temperature or TEMPERATURE
    
    # Get API token
    api_token = get_api_token()
    if not api_token:
        st.error("❌ No valid Hugging Face API token found. Please provide one in the sidebar.")
        return None
    
    try:
        llm = HuggingFaceHub(
            repo_id=model_id,
            huggingfacehub_api_token=api_token,
            model_kwargs={
                "temperature": temperature,
                "max_length": 1024,
                "max_new_tokens": MAX_TOKENS,
                "top_p": 0.9,
                "repetition_penalty": 1.1
            }
        )
        return llm
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        logger.error(f"Error loading model: {str(e)}")
        return None

def initialize_embeddings() -> Optional[HuggingFaceEmbeddings]:
    """Initialize and return Hugging Face embeddings."""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}
        )
        return embeddings
    except Exception as e:
        st.error(f"❌ Error initializing embeddings: {str(e)}")
        logger.error(f"Error initializing embeddings: {str(e)}")
        return None

def load_vector_store(embeddings: HuggingFaceEmbeddings) -> Optional[FAISS]:
    """Load the FAISS vector store."""
    try:
        if not os.path.exists(DB_FAISS_PATH):
            st.error(f"❌ FAISS vector store not found at {DB_FAISS_PATH}. Please ensure it's uploaded to Render.")
            logger.error(f"FAISS vector store not found at {DB_FAISS_PATH}")
            return None
        
        db = FAISS.load_local(
            DB_FAISS_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        logger.info("Successfully loaded FAISS vector store")
        return db
    except Exception as e:
        st.error(f"❌ Error loading vector store: {str(e)}")
        logger.error(f"Error loading vector store: {str(e)}")
        return None

def initialize_qa_chain() -> Optional[RetrievalQA]:
    """Initialize the QA chain with all components."""
    try:
        # Initialize embeddings
        embeddings = initialize_embeddings()
        if embeddings is None:
            return None
        
        # Load vector store
        db = load_vector_store(embeddings)
        if db is None:
            return None
        
        # Load LLM
        llm = load_llm()
        if llm is None:
            return None
        
        # Set up prompt and QA chain
        qa_prompt = set_custom_prompt()
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': qa_prompt}
        )
        
        logger.info("Successfully initialized QA chain")
        return qa_chain
    except Exception as e:
        st.error(f"❌ Error initializing QA chain: {str(e)}")
        logger.error(f"Error initializing QA chain: {str(e)}")
        return None

def display_chat_messages():
    """Display all chat messages in the chat container."""
    chat_container = st.container()
    
    with chat_container:
        st.markdown("### 💬 Chat")
        
        if not st.session_state.messages:
            st.info("👋 Welcome to MedChat! Please initialize the bot from the sidebar to start chatting.")
        
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message">👤 {message["content"]}</div>', unsafe_allow_html=True)
            else:
                # Check if the message contains source information
                content = message["content"]
                if "Sources:" in content:
                    # Split the content and sources
                    parts = content.split("Sources:")
                    main_content = parts[0].strip()
                    sources = "Sources:" + parts[1]
                    st.markdown(f'<div class="bot-message">🤖 {main_content}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="source-info">{sources}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="bot-message">🤖 {content}</div>', unsafe_allow_html=True)

def process_user_query(user_input: str):
    """Process the user's query and generate a response."""
    if not user_input.strip():
        return
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message immediately
    with st.chat_message("user"):
        st.write(user_input)
    
    # Generate bot response
    with st.chat_message("assistant"):
        with st.spinner("🤖 Thinking..."):
            try:
                response = st.session_state.qa_chain({'query': user_input})
                answer = response["result"]
                
                # Add sources if available
                sources = response.get("source_documents", [])
                if sources:
                    source_info = "\n\n**Sources:**\n"
                    for i, source in enumerate(sources, 1):
                        source_name = source.metadata.get('source', 'Unknown document')
                        source_info += f"{i}. {source_name}\n"
                    answer += source_info
                
                # Add bot response to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
                # Display bot response
                st.write(answer)
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.error(error_msg)
                logger.error(f"Error processing query: {str(e)}")

def main():
    """Main application function."""
    # Sidebar for configuration
    with st.sidebar:
        st.title("⚙️ MedChat Settings")
        st.markdown("---")
        
        # Token management section
        st.subheader("🔑 API Token Management")
        
        # Check if token is already set in environment variables
        env_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')
        if env_token:
            st.info("✅ API token found in environment variables")
            st.session_state.api_token = env_token
        else:
            api_token = st.text_input(
                "Hugging Face API Token",
                type="password",
                help="Get your token from https://huggingface.co/settings/tokens",
                value=st.session_state.get('api_token', '')
            )
            
            if api_token:
                st.session_state.api_token = api_token
                if validate_huggingface_token(api_token):
                    st.success("✅ Token format looks good!")
                else:
                    st.error("❌ Invalid token format")
        
        st.markdown("---")
        
        # Model selection
        st.subheader("🤖 Model Configuration")
        
        model_option = st.selectbox(
            "Choose Model",
            ["mistralai/Mistral-7B-v0.1", "google/flan-t5-xl", "microsoft/DialoGPT-large"],
            index=0,
            help="Select which Hugging Face model to use"
        )
        
        # Temperature slider
        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=1.0,
            value=TEMPERATURE,
            help="Lower values for more factual responses, higher for more creative ones"
        )
        
        # Initialize button
        if st.button("🔄 Initialize Chat Bot", use_container_width=True):
            with st.spinner("Initializing MedChat..."):
                st.session_state.qa_chain = initialize_qa_chain()
                if st.session_state.qa_chain:
                    st.session_state.initialized = True
                    st.session_state.model_initialized = True
                    st.success("✅ MedChat initialized successfully!")
                else:
                    st.error("❌ Failed to initialize MedChat")
                    st.session_state.initialized = False
        
        st.markdown("---")
        
        # Status information
        st.subheader("📊 System Status")
        if st.session_state.initialized:
            st.success("✅ Bot initialized and ready")
        else:
            st.warning("⚠️ Bot not initialized")
        
        # Token status
        token_status = "✅ Available" if get_api_token() else "❌ Missing"
        st.write(f"API Token: {token_status}")
        
        # Vector store status
        vector_store_status = "✅ Found" if os.path.exists(DB_FAISS_PATH) else "❌ Missing"
        st.write(f"Vector Store: {vector_store_status}")
        
        st.markdown("---")
        st.info("""
        **Note:** 
        - Enter your Hugging Face API token
        - Ensure vector store is created
        - Click 'Initialize Chat Bot' to start
        - Medical questions only for accurate responses
        """)
    
    # Main chat interface
    st.markdown('<h1 class="main-header">🏥 MedChat - Medical Assistant</h1>', unsafe_allow_html=True)
    
    # Display chat messages
    display_chat_messages()
    
    # Chat input
    if st.session_state.initialized and st.session_state.qa_chain:
        user_input = st.chat_input("Ask your medical question here...")
        if user_input:
            process_user_query(user_input)
    else:
        st.warning("⚠️ Please initialize the chat bot from the sidebar to start chatting.")
    
    # Additional features
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("ℹ️ About MedChat", use_container_width=True):
            st.info("""
            **MedChat** - A medical assistant chatbot powered by:
            - Hugging Face models
            - LangChain for retrieval-augmented generation
            - FAISS vector database
            - Streamlit for web interface
            
            This tool is for informational purposes only and not a substitute for professional medical advice.
            """)
    
    with col3:
        if st.button("📊 Advanced Settings", use_container_width=True):
            st.info("""
            **Advanced Options:**
            - Change model parameters
            - Adjust temperature
            - Modify search parameters
            - Customize prompts
            """)

if __name__ == "__main__":
    main()
