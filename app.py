import google.generativeai as genai
import streamlit as st
from streamlit_chat import message

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
import os
import openai

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]  

openai.api_key = OPENAI_API_KEY
#Model Initiation

#model= genai.GenerativeModel("gemini-1.5-flash")


def getResponse(user_input):

    test_messages = []

    system_message ="You are MindEase, a supportive virtual psychologist.Your role: listen empathetically, validate feelings, and offer safe, evidence-based coping strategies for anxiety, depression, stress, OCD, phobias, and related concerns. Guidelines:Be warm, calm, and non-judgmental.Always start with empathy, then suggest 1–2 practical techniques (e.g., breathing, journaling, CBT-style reframing).Encourage professional help when needed. Never prescribe or adjust medications.If user mentions self-harm or suicide: respond with empathy, urge them to seek urgent help, and share crisis resources (e.g., “Call 988 in the U.S. or your local emergency number”). Match the user’s language (reply in Hindi if they write in Hindi). Keep responses clear, short, and encouraging."
    test_messages.append({"role": "system", "content": system_message})
       

    test_messages.append({"role": "system", "content": user_input})
        #OpenAI Chat Completions
    response = openai.ChatCompletion.create(
                #model='ft:gpt-4o-mini-2024-07-18:sukkur-iba:mentalhealth:AGTajjiH', #can test it against gpt-3.5-turbo to see difference
                model= "gpt-5-mini",
                messages=test_messages,
                temperature=1
        )
    return response["choices"][0]["message"]["content"]

import tempfile
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
    response = getResponse(user_input)
            
    print(response)
            # Append user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": response})
            
    st.text_input = ""
    
        
            

def handle_voice_input(speech_text):

    #if st.button("Speak"):
    #   speech_text = recognize_speech()
        print(speech_text)
        
        st.session_state.messages.append({"role": "user", "content": speech_text})
        response = getResponse(speech_text)
        st.session_state.messages.append({"role": "assistant", "content": response})
        # Speak the response
        speak_text(response)
        

def main():
    """Main function to run the Streamlit app."""
    st.title("Mental Health Chatbot")

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
    

