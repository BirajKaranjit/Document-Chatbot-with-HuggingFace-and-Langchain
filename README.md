# Document-Chatbot-with-HuggingFace-and-Langchain
**Document Chatbot** is a web application designed to help users interact with documents and get answers to their queries. 
    It supports various file types like PDF, DOCX, TXT, and images. You can also request a call for further assistance as per your interest.
    The app uses **Retrieval-Augmented Generation (RAG)** to answer user queries. 
    It retrieves relevant information from uploaded documents using FAISS and Sentence-Transformers embeddings, then generates answers with the Mistral-7B-Instruct model from Hugging Face. 
    This approach enhances accuracy by combining document retrieval with generative response generation.
    vs
   **Technologies Used**
    - Streamlit: For building the web application
    - Hugging Face: For NLP models and embeddings
    -Langchain Community: For document loading, text splitting and vectorization
    - FAISS: For vector storage and retrieval
    - Google Cloud: For email services
    - Pytesseract: For OCR in images
    - PDFPlumber: For extracting text from PDFs
    - Python: For backend processing
    -Streamlit Cloud: For deployment
             
    **Features:**
    - Upload and process any specified documents in drop-down menu.
    - Ask questions and get contextual AI-generated answers.
    - Request a call for personalized assistance.
             
