import streamlit as st
st.set_page_config(page_title="Document Chatbot", layout="wide")
import faiss
import os
import asyncio
import numpy as np
import pdfplumber
from docx import Document
from PIL import Image
from PyPDF2 import PdfReader
import pytesseract
from datetime import datetime, timedelta
import re
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint

# Load environment variables
load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")
GMAIL_USER = os.getenv("GMAIL_USER")
GMAIL_PASS = os.getenv("GMAIL_PASS")

# Apply full-page background image
background_image_path = './images/document_bg_wallpaper.jpg'
# background_image_path = 'https://c4.wallpaperflare.com/wallpaper/490/444/59/the-inscription-books-candle-compass-wallpaper-preview.jpg'
page_bg_img = f'''
<style>
[data-testid="stAppViewContainer"] > .main {{
    background: url("{background_image_path}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)


# load the image for the sidebar
sidebar_image_url1 = './images/document_bg_wallpaper.jpg'
st.sidebar.image(sidebar_image_url1)

# sidebar for navigation
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Chat", "History", "Request a Call", "About"])

sidebar_image_url2 ='./images/personal_doc.jpg'
st.sidebar.image(sidebar_image_url2)

# ensure async event loop is running
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Function to process uploaded files
def process_input(files, input_type):
    documents = ""
    if input_type == "Hyper-Link":
        loader = WebBaseLoader(files)
        web_documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = [doc.page_content for doc in text_splitter.split_documents(web_documents)]
    elif input_type == "PDF":
        for file in files:
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    documents += page.extract_text() or ""
    elif input_type == "DOCX":
        for file in files:
            doc = Document(file)
            documents += "\n".join([para.text for para in doc.paragraphs])
    elif input_type == "TXT":
        for file in files:
            documents += file.read().decode('utf-8')
    elif input_type == "Image":
        for file in files:
            image = Image.open(file)
            text = pytesseract.image_to_string(image)
            documents += text
    else:
        raise ValueError("Unsupported input type")
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(documents)
    hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = FAISS.from_texts(texts, hf_embeddings)
    st.session_state.vectorstore = vector_store

# Function to extract date from input
def extract_date_from_input(user_input):
    today = datetime.today()
    weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    user_input = user_input.lower().strip()
    
    if "today" in user_input:
        return today.strftime("%Y-%m-%d")
    if "tomorrow" in user_input:
        return (today + timedelta(days=1)).strftime("%Y-%m-%d")
    
    for i, day in enumerate(weekdays):
        if f"next {day}" in user_input:
            days_ahead = (i - today.weekday() + 7) % 7 or 7
            return (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
            
    # Check for specific weekdays without "next" i.e only "Monday", "Tuesday", etc.
    for i, day in enumerate(weekdays):
        if day in user_input:
            if i > today.weekday():
                # If the day is later in the week
                return (today + timedelta(days=(i - today.weekday()))).strftime("%Y-%m-%d")
            else:
                # If the day is earlier in the week (return next week's day)
                return (today + timedelta(days=(i - today.weekday() + 7) % 7)).strftime("%Y-%m-%d")
     
    # Check for "in X days"
    match = re.search(r"in (\d+) days", user_input)
    if match:
        days_ahead = int(match.group(1))
        return (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        
    # Check for date in YYYY-MM-DD format
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", user_input)
    if match:
        return match.group(0)
    return None

# Function to answer questions
def answer_question(query):
    if st.session_state.vectorstore is None:
        return "No document uploaded yet."
    llm = HuggingFaceEndpoint(repo_id='mistralai/Mistral-7B-Instruct-v0.2', token=HUGGINGFACE_API_KEY)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=st.session_state.vectorstore.as_retriever())
    result = qa.invoke({"query": query})
    return result.get("result", "No answer found.")

# Function to send confirmation email
def send_confirmation_email(name, email, date):
    msg = MIMEMultipart()
    msg['From'] = GMAIL_USER
    msg['To'] = email
    msg['Subject'] = "Call Request Confirmation"
    
    body = f"""
    Hi {name},
    
    Your call request has been submitted successfully✅. We will contact you on {date} for further processes.
    Till then if you like to ask any queries or re-configure the call request date, feel free to ask by replying to this email, or you can contact on the below listed phone number also.
    
    Best regards,
    Biraj Kumar Karanjit
    9866659797
    """
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(GMAIL_USER, GMAIL_PASS)
        server.sendmail(GMAIL_USER, email, msg.as_string())
        server.quit()
        st.success("Confirmation email sent successfully!")
    except Exception as e:
        st.error(f"Failed to send email: {e}")

# Function to handle call requests
def show_call_request_form():
    st.subheader("📞 Request a Call")
    with st.form("call_request_form"):
        name = st.text_input("Your Name:")
        phone = st.text_input("Phone Number:")
        email = st.text_input("Email Address:")
        preferred_date = st.text_input("Preferred Date (e.g., Tomorrow, Next Monday, YYYY-MM-DD):")
        submitted = st.form_submit_button("Submit")
        
        if submitted:
            extracted_date = extract_date_from_input(preferred_date)
            if extracted_date:
                send_confirmation_email(name, email, extracted_date)
                st.success(f"✅ Your request has been submitted successfully. You will be contacted on {extracted_date} for further processes.")
            else:
                st.error("Invalid date format. Please try again.")

# Function to display About section
def show_about_section():
    st.subheader("👨‍💻 About App")
    st.write("""
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
    - Upload and process any specified documents.
    - Ask questions and get AI-generated answers.
    - Request a call for personalized assistance.
             
    **Developed by:**
    - Name: Biraj Kumar Karanjit
    - Email: karanjitbiraj123@gmail.com
    - GitHub: https://github.com/BirajKaranjit
    - LinkedIn: https://www.linkedin.com/in/biraj-kumar-karanjit/
 
    """)

if menu == "Chat":
    st.subheader("💬 Chat with AI")
    file_type = st.selectbox("Select File Type", ["PDF", "DOCX", "TXT", "Image"])
    
    if file_type == "Hyper-Link":
        # Input box for the link
        link = st.text_input("Insert the link:")
        
        if link:
            # Process the link
            process_input(link, "Hyper-Link")
            user_query = st.text_input("Ask something about the link:")
            
            if user_query:
                response = answer_question(user_query)
                st.session_state.chat_history.append({"query": user_query, "response": response})
                st.write("🤖 AI:", response)
    else:
        # Handle file uploads for other types (PDF, DOCX, TXT, Image)
        uploaded_files = st.file_uploader("Upload Files", accept_multiple_files=True, type=["pdf", "docx", "txt", "png", "jpg", "jpeg"])
        
        if uploaded_files:
            process_input(uploaded_files, file_type)
            user_query = st.text_input("Ask something about the document:")
            
            if user_query:
                response = answer_question(user_query)
                st.session_state.chat_history.append({"query": user_query, "response": response})
                st.write("🤖 AI:", response)

elif menu == "History":
    st.subheader("📝 Chat History")
    for entry in st.session_state.chat_history:
        st.write(f"**User:** {entry['query']}")
        st.write(f"**AI:** {entry['response']}")

elif menu == "Request a Call":
    show_call_request_form()

elif menu == "About":
    show_about_section()

