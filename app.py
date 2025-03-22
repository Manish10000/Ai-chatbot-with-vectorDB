import streamlit as st
import ollama
from sentence_transformers import SentenceTransformer
from faiss import IndexFlatIP
import numpy as np
from duckduckgo_search import DDGS

# Initialize models and vector database
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    faiss_index = IndexFlatIP(384)  # Match embedding dimension
    return embed_model, faiss_index

embedder, vector_db = load_models()

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Internet search function
def web_search(query):
    with DDGS() as ddgs:
        return [result for result in ddgs.text(query, max_results=3)]

# Response generation with fallback
def generate_response(user_input, context=""):
    prompt = f"Context: {context}\n\nQuestion: {user_input}\nAnswer:"
    
    try:
        response = ollama.chat(model='gemma:3b', messages=[
            {'role': 'user', 'content': prompt}
        ])
        answer = response['message']['content']
        
        if "I don't know" in answer or "not sure" in answer:
            raise ValueError("Need web search")
            
        return answer, True
    
    except Exception as e:
        results = web_search(user_input)
        context = "\n".join([r['body'] for r in results[:2]])
        response = ollama.chat(model='gemma3:latest', messages=[
            {'role': 'user', 'content': f"Using this: {context}\n\nAnswer: {user_input}"}
        ])
        return response['message']['content'], False

# Streamlit UI
st.title("Local AI Assistant")
pages = ["Chat", "Vector DB Viewer"]
page = st.sidebar.selectbox("Navigation", pages)

if page == "Chat":
    user_input = st.chat_input("Ask me anything...")
    
    if user_input:
        # Embed and search vector DB
        query_embed = embedder.encode(user_input)
        query_embed = query_embed.reshape(1, -1).astype('float32')
        
        if vector_db.ntotal > 0:
            scores, indices = vector_db.search(query_embed, 3)
            valid_indices = [i for i in indices[0] if i < len(st.session_state.chat_history)]
            context = "\n".join([st.session_state.chat_history[i] for i in valid_indices])
        else:
            context = ""
        
        # Generate response
        response, local_knowledge = generate_response(user_input, context)
        
        # Store in history and vector DB
        st.session_state.chat_history.extend([user_input, response])
        embeddings = embedder.encode([user_input, response])
        vector_db.add(embeddings.astype('float32'))
        
        # Display messages
        for msg in [user_input, response]:
            role = "user" if msg == user_input else "assistant"
            with st.chat_message(role):
                st.write(msg)
                
        if not local_knowledge:
            st.info("Web search was used for this response")

elif page == "Vector DB Viewer":
    st.header("Vector Database Contents")
    if len(st.session_state.chat_history) > 0:
        st.write("Stored Conversations:")
        for i, text in enumerate(st.session_state.chat_history):
            st.write(f"{i+1}. {text}")
    else:
        st.warning("Vector database is empty")

# Save chat history to disk periodically
if len(st.session_state.chat_history) % 5 == 0 and len(st.session_state.chat_history) > 0:
    np.save("chat_history.npy", np.array(st.session_state.chat_history))
    vector_db.write_index("faiss_index.bin")