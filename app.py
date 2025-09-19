import google.generativeai as genai
import streamlit as st
from streamlit_chat import message

import av
import numpy as np
import tempfile
import requests
from gtts import gTTS
import time
import base64
import pyttsx3
import speech_recognition as sr
from audio_recorder_streamlit import audio_recorder
import os
import openai
import datetime

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]  
openai.api_key = OPENAI_API_KEY

# -------------------- N8N Webhook Config --------------------
import requests

def send_to_n8n(user_message, bot_response):
    url = "https://kinzash.app.n8n.cloud/webhook/ai-chat"
    payload = {
        "user_message": user_message,
        "bot_response": bot_response
    }
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        print("Failed to send data to n8n:", e)

# -----------------------------------------------------------

#Model Initiation
#model= genai.GenerativeModel("gemini-1.5-flash")

def getResponse(user_input):
    test_messages = []
    system_message = "You are MindEase, a warm and supportive AI companion..."
    test_messages.append({"role": "system", "content": system_message})
    test_messages.append({"role": "user", "content": user_input})

    response = openai.ChatCompletion.create(
        model="gpt-5-mini",
        messages=test_messages,
        temperature=1
    )
     
    
    
    return response["choices"][0]["message"]["content"]

def speak_text(text):
    """Function to convert text to speech and play it."""
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
        tts.save(temp_file.name)
        temp_file_path = temp_file.name  
    st.audio(temp_file_path, format='audio/mp3')
    os.remove(temp_file_path)

def handle_text_input(user_input):
    st.session_state.text_input = ""
    response = getResponse(user_input)



    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": response})
    send_to_n8n(user_input, response)

def handle_voice_input(speech_text):
    st.session_state.messages.append({"role": "user", "content": speech_text})
    response = getResponse(speech_text)


    st.session_state.messages.append({"role": "assistant", "content": response})
    speak_text(response)
    send_to_n8n(speech_text, response)

def main():
    st.title("Mental Health Chatbot")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    recorder = audio_recorder(text='بولیۓ', icon_size="2x", icon_name="microphone-lines", key="recorder")
    user_input = st.chat_input("Type your message:", key="text_input_field")

    if user_input:
        handle_text_input(user_input)
    elif recorder is not None:
        with st.container():
            col1, col2 = st.columns(2)
            with col2:
                st.header('🧑')                                                                                                                                                                                                                                                                                                                                                                                                                                                          
                st.audio(recorder)

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_urdu_recording:
                    temp_urdu_recording.write(recorder)
                    temp_urdu_recording_path = temp_urdu_recording.name

                recognizer = sr.Recognizer()
                with sr.AudioFile(temp_urdu_recording_path) as source:
                    urdu_recoded_voice = recognizer.record(source)
                    try:
                        speech_text = recognizer.recognize_google(urdu_recoded_voice, language="en")
                    except sr.UnknownValueError:
                        return "آپ کی آواز واضح نہیں ہے"
                    except sr.RequestError:
                        return "Sorry, my speech service is down"

                os.remove(temp_urdu_recording_path)
                handle_voice_input(speech_text)

    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.write(msg['content'])
            st.text_input = ""

if __name__ == "__main__":
    main()
