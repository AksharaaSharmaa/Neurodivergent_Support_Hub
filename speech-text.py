import streamlit as st
import queue
import sounddevice as sd
import numpy as np
import os
from google.cloud import speech

# Set Google Cloud Credentials (Make sure your JSON key file path is correct)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "neurodivai-d33144d53259.json"

# Google Speech-to-Text Client
client = speech.SpeechClient()

# Queue to store audio chunks
audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    """Callback function to capture audio from the microphone."""
    if status:
        print(status)
    audio_queue.put(indata.copy())

def stream_speech():
    """Stream audio from the microphone and transcribe in real-time."""
    st.write("Listening... Speak now!")
    
    # Create a placeholder for the transcript
    transcript_placeholder = st.empty()
    current_transcript = ""

    # Stream audio input
    with sd.InputStream(samplerate=16000, blocksize=8000, dtype="int16",
                        channels=1, callback=audio_callback):
        
        # Configurations for Google Speech API
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
            enable_automatic_punctuation=True
        )

        streaming_config = speech.StreamingRecognitionConfig(
            config=config, interim_results=False  # Set to False to get only final results
        )

        # Generator to yield audio chunks
        def generate_audio():
            while True:
                data = audio_queue.get()
                yield speech.StreamingRecognizeRequest(audio_content=data.tobytes())

        # Send audio to Google Speech API
        responses = client.streaming_recognize(streaming_config, generate_audio())

        try:
            for response in responses:
                if response.results:
                    for result in response.results:
                        if result.is_final:
                            sentence = result.alternatives[0].transcript
                            current_transcript += sentence + " "
                            # Update the display with the complete transcript so far
                            transcript_placeholder.markdown(f"**Transcript:** {current_transcript}")

            return current_transcript

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            return current_transcript

# Streamlit UI
st.title("🎤 Real-Time Speech-to-Text using Google API")

if st.button("Start Listening 🎙️"):
    transcript = stream_speech()
    st.success("Done listening!")
    st.write("**Final Transcript:**", transcript)