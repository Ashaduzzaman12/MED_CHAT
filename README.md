# MedChat - Medical Assistant (Render Deployment)

A Streamlit-based medical chatbot using Hugging Face models and LangChain for retrieval-augmented generation, deployed on Render.

## Features

- 🤖 Medical question answering using Hugging Face models
- 🔍 FAISS vector store for efficient similarity search
- 💬 Interactive chat interface with Streamlit
- 🔐 Secure API token management
- 📚 Source attribution for answers
- 🚀 Deployed on Render for reliable hosting

## Deployment on Render

### Prerequisites

1. A Render account (https://render.com/)
2. A Hugging Face account with API token
3. Your FAISS vector store files

### Deployment Steps

1. **Fork or upload this repository** to your GitHub account

2. **Create a new Web Service on Render**:
   - Connect your GitHub repository
   - Select "Web Service" as the service type
   - Use the following settings:
     - **Name**: `medchat-app` (or your preferred name)
     - **Environment**: `Python 3`
     - **Region**: Choose the closest to your users
     - **Branch**: `main` (or your preferred branch)
     - **Root Directory**: Leave empty (unless your files are in a subdirectory)
     - **Build Command**: `chmod +x build.sh && ./build.sh`
     - **Start Command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`

3. **Configure Environment Variables**:
   - `HUGGINGFACEHUB_API_TOKEN`: Your Hugging Face API token
   - `DB_FAISS_PATH`: `vectorstores/db_faiss` (default)
   - `DEFAULT_MODEL`: `mistralai/Mistral-7B-v0.1` (default)
   - `EMBEDDING_MODEL`: `sentence-transformers/all-MiniLM-L6-v2` (default)
   - `MAX_TOKENS`: `512` (default)
   - `TEMPERATURE`: `0.3` (default)

4. **Deploy your application**:
   - Render will automatically build and deploy your application
   - The first build may take several minutes as it installs dependencies

5. **Upload your FAISS vector store**:
   - You'll need to include your FAISS vector store files in your repository
   - Create a `vectorstores/db_faiss` directory with all required files
   - Alternatively, use a cloud storage solution and modify the code to load from URL

## File Structure
