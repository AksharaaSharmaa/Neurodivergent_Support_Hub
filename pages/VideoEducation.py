import streamlit as st
import sounddevice as sd
import numpy as np
import google.cloud.speech as speech
import google.generativeai as genai
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
import os
from datetime import datetime
from scipy.io.wavfile import write
from google.auth.credentials import Credentials
import io
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import pickle

# Set page config for dark mode
st.set_page_config(page_title="Neurodivergent YouTube Assistant", page_icon="🎥", layout="wide")

# Custom CSS for dark mode
st.markdown("""
<style>
    /* Dark mode enhancements */
    .stApp {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
    }
    
    .content-container {
        background-color: #2D2D2D;
        padding: 2rem;
        border-radius: 10px;
        border: 1px solid #3D3D3D;
        margin: 1rem 0;
    }
    
    .stExpander {
        background-color: #2D2D2D !important;
        border: 1px solid #3D3D3D !important;
    }
    
    /* Custom styling for text inputs */
    .stTextInput>div>div>input {
        background-color: #2D2D2D;
        color: white;
        border: 1px solid #3D3D3D;
    }
    
    /* Custom styling for select boxes */
    .stSelectbox>div>div>select {
        background-color: #2D2D2D;
        color: white;
        border: 1px solid #3D3D3D;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #252525;
    }
    
    /* Success message styling */
    .stSuccess {
        background-color: #1E4620;
        color: white;
    }
    
    /* Error message styling */
    .stError {
        background-color: #462020;
        color: white;
    }

    /* Custom styling for multiselect */
    .stMultiSelect>div>div>div {
        background-color: #2D2D2D;
        color: white;
        border: 1px solid #3D3D3D;
    }
</style>
""", unsafe_allow_html=True)

# Set up Google Cloud authentication using service account credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service-account.json"

# Configure Gemini API
genai.configure(api_key="AIzaSyCW6X3nK9yF4Q-XNN-2nl3j3wYfoCv32zc")

# Google Cloud Speech Client
client = speech.SpeechClient()

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
DURATION = 20

# Find Virtual Audio Cable device
DEVICE_ID = None
devices = sd.query_devices()
for i, dev in enumerate(devices):
    if "cable" in dev["name"].lower():
        DEVICE_ID = i
        st.sidebar.success(f"Using Virtual Cable: {dev['name']} (ID: {i})")
        break

if DEVICE_ID is None:
    st.error("Virtual Audio Cable not found! Make sure it's installed and set as output.")
    st.sidebar.markdown("""**Installation Instructions:** 1. Install Virtual Audio Cable: [Link to installation guide] 2. Set the Virtual Audio Cable as your audio output device. 3. Restart the app to enable recording.""")
    DEVICE_ID = sd.default.device

def get_video_id(url):
    """Extract video ID from YouTube URL."""
    parsed_url = urlparse(url)
    if parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
        query = parse_qs(parsed_url.query)
        return query.get('v', [None])[0]
    elif parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]
    return None

def get_video_transcript(video_id):
    """Get video transcript using YouTube Transcript API."""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        return ' '.join([item['text'] for item in transcript_list])
    except:
        return None

def authenticate():
    """Authenticate using OAuth 2.0 for Google Docs and Drive APIs."""
    creds = None
    SCOPES = ['https://www.googleapis.com/auth/documents', 'https://www.googleapis.com/auth/drive.file']

    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    return creds

def record_audio(duration=DURATION, sample_rate=SAMPLE_RATE):
    """Captures system audio from Virtual Audio Cable or fallback device."""
    with st.spinner(f"Listening to system audio for {duration} seconds..."):
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, 
                         channels=CHANNELS, dtype=np.int16, device=DEVICE_ID)
        sd.wait()
        
        wav_io = io.BytesIO()
        write(wav_io, sample_rate, recording)
        return wav_io.getvalue()

def transcribe_audio(audio_data):
    """Sends recorded audio to Google Speech-to-Text API."""
    audio = speech.RecognitionAudio(content=audio_data)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code="en-US",
        enable_automatic_punctuation=True
    )

    response = client.recognize(config=config, audio=audio)
    transcript = " ".join([result.alternatives[0].transcript for result in response.results])
    return transcript if transcript else "No speech detected."

def process_with_gemini(text, condition, content_formats):
    """
    Processes transcribed text with Gemini for condition-specific adaptation and visual content description.
    
    Args:
        text (str): The content to be processed
        condition (str): Accessibility requirements or specific conditions
        content_formats (list): List of desired output formats
    
    Returns:
        str: Processed content with visual descriptions and alternative formats
    """
    format_prompts = {
        "Bullet-point summary": "Create a concise bullet-point summary of the main points",
        "Full-length transcript": "Provide a detailed, organized transcript with headers and sections",
        "Visual breakdown": """
Create a detailed visual description of the content that includes:
- For any diagrams or visual elements:
  * Describe the layout and structure
  * List all elements with their properties
  * Explain step-by-step how the visual changes
  * Include ASCII art representations where helpful
- For spatial arrangements:
  * Describe the positioning of elements
  * Break down movements or changes
  * Highlight important visual relationships
- For technical concepts:
  * Create text-based visualizations (tables, trees, flowcharts)
  * Map out relationships between elements
  * Show progression and transformations
""",
        "Simplified language": "Rewrite the content using simple, clear language and short sentences"
    }
    
    selected_formats = [format_prompts[fmt] for fmt in content_formats]
    format_instructions = "\n".join(f"{i+1}. {fmt}" for i, fmt in enumerate(selected_formats))
    
    prompt = f"""
    As an assistant for someone with {condition}, analyze this content:
    {text}
    
    Please provide the content in the following formats:
    {format_instructions}
    
    For any visual elements, create detailed descriptions that:
    1. Break down the components step by step
    2. Create ASCII art diagrams where helpful, such as:
       - Structural layouts
       - Flow diagrams
       - Hierarchical relationships
       - Process steps
       - Data visualizations
    3. Describe spatial relationships and layouts clearly
    4. Explain any changes or transformations
    5. Use tables and structured formats for data
    6. Map out relationships between elements
    
    Example ASCII representation:
    ```
    Element A
    ├──[Step 1]──╮
    Element B    Element C
    │            ╰──[Step 2]──Element D
    └──[Step 3]──Element E
    
    Legend:
    ├── Primary connection
    ╰── Secondary connection
    [ ] Process/step
    ```
    
    Additionally, include:
    - Key concepts and definitions
    - Important sections or segments
    - Potential accessibility considerations
    - Visual descriptions of all elements
    - Recommended resources or alternatives
    
    Format each section with clear headings and maintain consistent organization throughout.
    """
    
    try:
        response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
        return response.text if response else "Unable to process the content."
    except Exception as e:
        return f"Error processing content: {str(e)}"

def save_to_google_docs(content, video_url, condition, content_formats):
    """Saves the processed content to Google Docs and Drive."""
    try:
        creds = authenticate()
        docs_service = build('docs', 'v1', credentials=creds)
        drive_service = build('drive', 'v3', credentials=creds)
        
        title = f"YouTube Summary - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        doc_body = {'title': title}
        doc = docs_service.documents().create(body=doc_body).execute()
        doc_id = doc.get('documentId')
        
        header = f"""
Video URL: {video_url}
Condition: {condition}
Content Formats: {', '.join(content_formats)}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

"""
        
        requests = [
            {
                'insertText': {
                    'location': {'index': 1},
                    'text': header + content
                }
            }
        ]
        
        docs_service.documents().batchUpdate(documentId=doc_id, body={'requests': requests}).execute()
        
        file_metadata = {'role': 'reader', 'type': 'anyone'}
        drive_service.permissions().create(fileId=doc_id, body=file_metadata).execute()
        
        return f"https://docs.google.com/document/d/{doc_id}/edit"
    
    except Exception as e:
        st.error(f"Error saving to Google Docs: {str(e)}")
        return None

# Streamlit UI
st.title("🎥 Neurodivergent YouTube Assistant")

# Create two columns for input fields
col1, col2 = st.columns(2)

with col1:
    video_url = st.text_input("Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")

with col2:
    condition = st.selectbox(
        "Select your neurodivergent condition:",
        ["ADHD", "Autism", "Dyslexia", "Anxiety", "Other"],
        help="This helps tailor the content to your needs"
    )

if condition == "Other":
    condition = st.text_input("Please specify your condition:")

# Content format selection
st.subheader("🎯 Content Format Preferences")
content_formats = st.multiselect(
    "Select your preferred content formats:",
    ["Bullet-point summary", "Full-length transcript", "Visual breakdown", "Simplified language"],
    default=["Bullet-point summary"],
    help="Choose how you'd like the content to be presented"
)

# Display video if URL is provided
if video_url:
    try:
        st.video(video_url)
    except Exception as e:
        st.error("Error loading video. Please check the URL.")

# Recording controls
col3, col4 = st.columns(2)
with col3:
    duration = st.slider("Recording Duration (seconds)", 10, 60, 20)
with col4:
    auto_save = st.checkbox("Auto-save to Google Docs", value=True)

# Main functionality
if st.button("Start Processing", key="start_button"):
    if not content_formats:
        st.warning("Please select at least one content format.")
    else:
        with st.spinner("Processing..."):
            video_id = get_video_id(video_url)
            if not video_id:
                st.error("Invalid YouTube URL")
            else:
                # Try to get subtitles first
                transcript = get_video_transcript(video_id)
                
                # If no subtitles available, fall back to audio recording
                if transcript is None:
                    st.info("No subtitles found. Recording audio instead...")
                    audio_data = record_audio(duration=duration)
                    transcript = transcribe_audio(audio_data)
                
                if transcript:
                    with st.expander("📝 Full Transcript", expanded=False):
                        st.write(transcript)

                    st.subheader("🎯 Adapted Content")
                    processed_content = process_with_gemini(transcript, condition, content_formats)
                    
                    # Display processed content in a custom container
                    st.markdown('<div class="content-container">', unsafe_allow_html=True)
                    st.markdown(processed_content)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Save to Google Docs if enabled
                    if auto_save:
                        with st.spinner("Saving to Google Docs..."):
                            doc_url = save_to_google_docs(processed_content, video_url, condition, content_formats)
                            if doc_url:
                                st.success(f"Content saved! [Open in Google Docs]({doc_url})")

# Enhanced sidebar with additional information
with st.sidebar:
    st.header("📋 Instructions")
    st.markdown(""" 1. Paste a YouTube video URL \n2. Select your condition \n3. Choose content formats \n4. Adjust recording duration \n5. Click 'Start Processing' \n6. View and save the adapted content """)
    
    st.header("🎯 Features")
    st.markdown(""" - Automatic subtitle extraction \n- Speech recognition fallback \n- Condition-specific content adaptation \n- Multiple content format options \n- Google Docs integration \n- Customizable recording duration \n- Sensory and trigger warnings \n- Resource recommendations """)
    
    st.header("⚙️ Settings")
    st.markdown("**Audio Device:**")
    st.info(f"Currently using: {sd.query_devices()[DEVICE_ID]['name']}")