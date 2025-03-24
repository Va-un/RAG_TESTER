import streamlit as st
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import pandas as pd
import tempfile
import os
import uuid
from utils import Generate_Answer

# Set up the page configuration
st.set_page_config(page_title="Business Chatbot", layout="wide")
st.title("Business Chatbot")

# Initialize session state for conversations
if "conversations" not in st.session_state:
    st.session_state.conversations = {}
if "current_conversation_id" not in st.session_state:
    # Create a new conversation at the start
    conv_id = str(uuid.uuid4())
    st.session_state.current_conversation_id = conv_id
    st.session_state.conversations[conv_id] = []
if "vector_data" not in st.session_state:
    st.session_state.vector_data = []

# Helper function to save the current conversation
def save_current_conversation():
    curr_id = st.session_state.current_conversation_id
    st.session_state.conversations[curr_id] = st.session_state.messages

# Load current conversation messages
if "messages" not in st.session_state:
    st.session_state.messages = st.session_state.conversations[st.session_state.current_conversation_id]

# Sidebar: file upload, conversation management
with st.sidebar:
    st.header("Document Upload")
    
    # Streamlit file uploader in sidebar
    uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True, type=['csv', 'xlsx', 'txt', 'pdf'])
    
    # Process uploaded files
    if uploaded_files:
        # Initialize containers for all documents and content
        all_documents = []
        
        # Process each file
        for i, uploaded_file in enumerate(uploaded_files):
            st.subheader(f"File: {uploaded_file.name}")
            
            # Handle PDF files
            if uploaded_file.name.endswith('.pdf'):
                # Create a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                # Create document object with metadata
                content = ""
                try:
                    pdf_reader = PyPDF2.PdfReader(tmp_path)
                    for page_num in range(len(pdf_reader.pages)):
                        page_text = pdf_reader.pages[page_num].extract_text()
                        content += page_text
                        
                        # Add each page as a document with metadata
                        all_documents.append({
                            "content": page_text,
                            "metadata": {
                                "source": uploaded_file.name,
                                "page": page_num + 1
                            }
                        })
                    
                except Exception as e:
                    st.error(f"Error processing PDF: {e}")
                
                # Clean up temp file
                os.unlink(tmp_path)
            
            # Handle other file types (txt, csv, xlsx) as needed
            # Add similar processing for other file types here
        
        # Generate embeddings if we have documents
        if all_documents and st.button("Generate Embeddings"):
            with st.spinner("Generating embeddings..."):
                # Text splitting
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=2000)
                chunks = []
                
                for doc in all_documents:
                    doc_chunks = text_splitter.create_documents(
                        texts=[doc["content"]], 
                        metadatas=[doc["metadata"]]
                    )
                    chunks.extend(doc_chunks)
                
                # Initialize embedding model
                embedding_model_name = "all-mpnet-base-v2"
                embedding_model = SentenceTransformer(embedding_model_name)
                
                # Generate embeddings
                vector_data = []
                for idx, chunk in enumerate(chunks, 1):
                    text_content = chunk.page_content
                    embeddings = embedding_model.encode(text_content)
                    
                    metadata = chunk.metadata.copy()
                    metadata["text"] = text_content
                    
                    vector_data.append({
                        "id": idx,
                        "embedding": embeddings.tolist(),  # Convert to list for JSON serialization
                        "metadata": metadata
                    })
                
                # Save vector data to session state
                st.session_state.vector_data = vector_data
                
                # Create DataFrame and save
                df = pd.DataFrame({
                    "id": [item["id"] for item in vector_data],
                    "embedding": [item["embedding"] for item in vector_data],
                    "source": [item["metadata"]["source"] for item in vector_data],
                    "page": [item["metadata"]["page"] for item in vector_data],
                    "text": [item["metadata"]["text"] for item in vector_data]
                })
                
                # Save to CSV (optional - embeddings in CSV can be very large)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Vector CSV",
                    csv,
                    "vector_data.csv",
                    "text/csv",
                    key='download-csv'
                )
                
                st.success(f"Generated {len(vector_data)} vectors from your documents!")
    
    st.header("Conversations")
    
    # Build list of conversation options as (id, title)
    conv_options = []
    for conv_id, messages in st.session_state.conversations.items():
        # Use the first user message as a title, if available
        title = "New Conversation"
        for msg in messages:
            if msg["role"] == "user":
                title = msg["content"][:30] + "..." if len(msg["content"]) > 30 else msg["content"]
                break
        conv_options.append((conv_id, title))
    
    # Create a radio selection for existing conversations
    conversation_ids = [item[0] for item in conv_options]
    selected_conv = st.radio(
        "Select Conversation",
        options=conversation_ids,
        format_func=lambda conv_id: next((title for cid, title in conv_options if cid == conv_id), conv_id)
    )
    
    # Allow creation of a new conversation
    if st.button("New Conversation"):
        save_current_conversation()
        new_conv_id = str(uuid.uuid4())
        st.session_state.conversations[new_conv_id] = []
        st.session_state.current_conversation_id = new_conv_id
        st.session_state.messages = st.session_state.conversations[new_conv_id]
    
    # If the selected conversation differs from the current one, load it
    if selected_conv != st.session_state.current_conversation_id:
        save_current_conversation()
        st.session_state.current_conversation_id = selected_conv
        st.session_state.messages = st.session_state.conversations[selected_conv]

# Main chat area
# Display conversation messages
for msg in st.session_state.messages:
    if msg["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("user"):
            st.markdown(msg["content"])

# Capture user input via chat widget
user_input = st.chat_input("Type your message here...")

if user_input:
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)
        
    # Append the user's message to the conversation history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Generate response using vector data from session state
    bot_output = Generate_Answer(user_input, vector_data=st.session_state.vector_data)
    
    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(bot_output)
    
    # Append assistant response to conversation history
    st.session_state.messages.append({"role": "assistant", "content": bot_output})
    
    # Save the updated conversation
    save_current_conversation()