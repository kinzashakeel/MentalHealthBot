from openai import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import pandas as pd
import PyPDF2
import os
from sentence_transformers import SentenceTransformer

load_dotenv() 

OPENAI_API_KEY= os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)

def csv_to_pdf(csv_file_path, pdf_file_path="data.pdf"):
    # Load CSV data
    df = pd.read_csv(csv_file_path)

    # Create a PDF
    c = canvas.Canvas(pdf_file_path, pagesize=letter)
    width, height = letter

    # Set up the PDF
    margin = 0.75 * inch
    table_width = width - 2 * margin
    table_height = height - 2 * margin
    row_height = 12
    col_width = table_width / len(df.columns)

    # Draw the table headers
    c.setFont("Helvetica-Bold", 12)
    y = height - margin - row_height
    for col in df.columns:
        c.drawString(margin, y, col)
        margin += col_width
    y -= row_height

    # Draw the table rows
    c.setFont("Helvetica", 10)
    for _, row in df.iterrows():
        margin = 0.75 * inch
        for col in df.columns:
            c.drawString(margin, y, str(row[col]))
            margin += col_width
        y -= row_height

    # Save the PDF
    c.save()


def extract_text_from_pdf(file_path):
    
    from langchain.document_loaders import PyPDFLoader
    #file_path=csv_to_pdf(file_path)
    # Load PDF
    loaders = [
       
        PyPDFLoader(file_path)
    ]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    
        # Split
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1500,
        chunk_overlap = 150
    )
    splits = text_splitter.split_documents(docs)
    
    print(splits)
    return splits

# Function to read data from a CSV file and create a vector store
def create_vector_store(file_path):
    
    splits=extract_text_from_pdf(file_path)
    
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    
    persist_directory = 'chroma.sqlite3'
    
    embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)
    
    
    vector_store = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory=persist_directory
)
    
    return vector_store


def generate_response(prompt, vector_store):
    try:
        # Search for relevant documents using the vector store
        docs = vector_store.similarity_search(query=prompt, k=5)

        # Combine the retrieved documents with the prompt
        context = "\n".join([doc.page_content for doc in docs])

        combined_prompt = [
            {
                "role": "system",
                "content": """
                Chat and respond in an informal style like humans chat with each other. Use emojis and a friendly tone. 
                Make it a conversation bot. Your responses should be formatted like a message in a messaging app. No special characters.
                Don't add Don't include anything like "Answer:" in start of response message. Just chat humanly. Greet only once at the start of the conversation; 
                don't say hi/hello/welcome/greetings again and again. Only Talk regarding mental health strictly, no other topic discussion.

                Act like a psychologist/therapist whose sole purpose is to provide support and guidance to individuals struggling with their mental health. 
                Your primary goal is to assess their mental health condition and offer effective countermeasures to improve their well-being.

                Engage in empathetic and non-judgmental conversations, actively listening to patients' concerns, understanding their emotions, 
                and providing insightful feedback. You should be equipped with the ability to identify different mental health conditions 
                such as anxiety, depression, or stress, and recommend appropriate coping strategies, self-care techniques, or professional help if necessary.

                Ask the patient's country first. Then talk with the patient according to the cultural issues of that region, city, or country.
                Only give short but meaningful responses. Only give one message at a time and wait for the patient's response; 
                make it like a conversation. Don't ask too many questions; it may irritate the patient. Only ask basic questions to know 
                the patient's mental health condition and then suggest remedies and measures for betterment of that specific mental health issue.
                
                Also, suggest any online psychologist service according to the region, city, or country of the patient. 
                Make messages short; don't give long responses. Reply messages in Urdu if the user's message is in Urdu; 
                otherwise, reply messages in English. 
                {context}
                Question: {prompt}
                """
            }
        ]

        # Insert the context and user prompt into the message content
        combined_prompt[0]["content"] = combined_prompt[0]["content"].format(context=context, prompt=prompt)

        # Generate a response using GPT-4
        response = client.chat.completions.create(
            model="gpt-4",
            messages=combined_prompt,
            max_tokens=1024,
            temperature=0.7
        )

        # Extract the content from the response
        return response.choices[0].message.content

    except Exception as e:
        print("Error:", e)
        return "I couldn't process your request. Please try again later."


import streamlit as st
from streamlit_chat import message
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import av
import numpy as np

from io import BytesIO
import requests
from gtts import gTTS
import time
import base64
import pyttsx3
import speech_recognition as sr
from audio_recorder_streamlit import audio_recorder
from dotenv import load_dotenv

def getResponse(user_input,vector_store):
    response=""
    if user_input.lower() == "quit":
        response= "Takecare Good Bye!"
    else:
         response = generate_response(user_input, vector_store)
    return response

import tempfile


def recognize_speech(temp_urdu_recording_path):

                    recognizer = sr.Recognizer()
                    with sr.AudioFile(temp_urdu_recording_path) as source:
                        urdu_recoded_voice = recognizer.record(source)
                        try:
                            text = recognizer.recognize_google(urdu_recoded_voice, language="en")
                            return text
                        except sr.UnknownValueError:
                            return "آپ کی آواز واضح نہیں ہے"
                        except sr.RequestError:
                            return "Sorry, my speech service is down"
                        
def speak_text(text):
    """Function to convert text to speech and play it."""
    tts = gTTS(text=text, lang='en')
    
    # Use a context manager to handle the temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
        tts.save(temp_file.name)
        temp_file_path = temp_file.name  # Store the file path for later use
    
    # Play the audio file and then clean up
    st.audio(temp_file_path, format='audio/mp3')
    
    # Ensure the file is properly closed before deleting
    os.remove(temp_file_path)




def handle_text_input(user_input):

    st.session_state.text_input = ""


    # Generate chatbot response
    file_path = "mh.pdf"  
    vector_store = create_vector_store(file_path)
    response = getResponse(user_input,vector_store)
    
    print(response)
            # Append user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": response})

    st.text_input = ""


def handle_voice_input(speech_text):
    
        # Generate chatbot response
        file_path = "mh.pdf"  
        vector_store = create_vector_store(file_path)
        response = getResponse(speech_text,vector_store)
        
        #if st.button("Speak"):
        #   speech_text = recognize_speech()
        print(speech_text)
        
        st.session_state.messages.append({"role": "user", "content": speech_text})
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        # Speak the response
        speak_text(response)
        

def main():
    """Main function to run the Streamlit app."""
    st.title("Text and Voice Chatbot using Gemini-1.5-flash & Streamlit")

    # Initialize session state for chat history and text input
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    

    recorder = audio_recorder(text='بولیۓ', icon_size="2x", icon_name="microphone-lines", key="recorder")
    # Handle text and voice input
    user_input = st.chat_input("Type your message:", key="text_input_field")
    if user_input:
        handle_text_input(user_input)
    elif recorder is not None:
            
            with st.container():
                col1, col2 = st.columns(2)

                with col2:
                    # Display the audio file
                    st.header('🧑')                                                                                                                                                                                                                                                                                                                                                                                                                                                          
                    st.audio(recorder)

                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_urdu_recording:
                        temp_urdu_recording.write(recorder)
                        temp_urdu_recording_path = temp_urdu_recording.name
                    
                    # Convert audio file to text
                    
                    #text = Urdu_audio_to_text(temp_urdu_recording_path)
                    #st.success( text)
                    recognizer = sr.Recognizer()
                    with sr.AudioFile(temp_urdu_recording_path) as source:
                        urdu_recoded_voice = recognizer.record(source)
                        try:
                            speech_text = recognizer.recognize_google(urdu_recoded_voice, language="en")
                        except sr.UnknownValueError:
                            return "آپ کی آواز واضح نہیں ہے"
                        except sr.RequestError:
                            return "Sorry, my speech service is down"
                    
                    # Remove the temporary file
                    os.remove(temp_urdu_recording_path)
                    #speech_text= recognize_speech(temp_urdu_recording_path)
                    handle_voice_input(speech_text)
   
    # Display previous chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.write(msg['content'])
            st.text_input = ""

if __name__ == "__main__":
    main()
    



