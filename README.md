**📄 Document Chatbot with Hugging Face and Langchain**
Document Chatbot is a web application designed to help users interact with documents and get AI-powered answers to their queries.
It supports multiple file formats, including PDF, DOCX, TXT, and images, and allows users to request calls for further assistance.

This chatbot leverages Retrieval-Augmented Generation (RAG) to enhance response accuracy. It retrieves relevant information from uploaded documents using FAISS and Sentence-Transformers embeddings before generating answers with the Mistral-7B-Instruct model from Hugging Face.

🚀 Technologies Used
Streamlit – For building the interactive web application
Hugging Face – For NLP models and embeddings
Langchain Community – For document loading, text splitting, and vectorization
FAISS – For vector storage and retrieval
Google Cloud – For email services
PyTesseract – For OCR-based text extraction from images
PDFPlumber – For extracting text from PDFs
Python – Backend processing and logic
Streamlit Cloud – For deployment

✨ Features
✅ Upload and process documents via an interactive dropdown menu
✅ Ask AI-generated contextual questions and get relevant answers
✅ Support for multiple document types (PDF, DOCX, TXT, Images)
✅ Email notification system for call requests
✅ Simple UI built using Streamlit
