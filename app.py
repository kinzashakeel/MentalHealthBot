import streamlit as st
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the PDF file and extract the text
def load_pdf(pdf_path):
    pdf_Text = ""
    with open(pdf_path, "rb") as file:
        reader = PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text = page.extract_text()
            pdf_Text += text + "\n"
    return pdf_Text

# Load the pre-trained model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("himmeow/vi-gemma-2b-RAG")
    model = AutoModelForCausalLM.from_pretrained(
        "himmeow/vi-gemma-2b-RAG",
        device_map="auto",
        torch_dtype=torch.fp16
    )
    if torch.cuda.is_available():
        model.to("cuda")
    return tokenizer, model

# Generate a response based on the user query
def generate_response(model, tokenizer, pdf_text, query):
    prompt = f"""
### Instruction and Input:
Based on the following context/document:
{pdf_text}
Please answer the question as a certified Psychologist or Therapist and resolve mental health issues of the user: {query}
### Response:
"""
    input_ids = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")
    
    outputs = model.generate(
        **input_ids,
        max_new_tokens=500, 
        no_repeat_ngram_size=5,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Streamlit interface
st.title("Mental Health Chatbot")
st.write("This chatbot can help address mental health issues by analyzing content from a document and answering your questions.")

# Load PDF and model
pdf_path = "smaller_dataset.pdf"  # Replace with your PDF file path
pdf_text = load_pdf(pdf_path)
tokenizer, model = load_model()

# User input
query = st.text_input("Enter your question here:")

if st.button("Generate Response"):
    if query:
        response = generate_response(model, tokenizer, pdf_text, query)
        st.subheader("Chatbot's Response:")
        st.write(response)
    else:
        st.write("Please enter a question to get a response.")
