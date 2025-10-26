import streamlit as st
import sounddevice as sd
import numpy as np
import google.generativeai as genai
import os
from datetime import datetime
from scipy.io.wavfile import write
import io
from rag_functions import *
try:
    from youtube_transcript_api import YouTubeTranscriptApi
except ImportError:
    st.error("Please install youtube-transcript-api: pip install youtube-transcript-api")
    YouTubeTranscriptApi = None
from urllib.parse import urlparse, parse_qs
from fpdf import FPDF
import tempfile
from gtts import gTTS
import base64

# Initialize session state variables
if 'processed_content' not in st.session_state:
    st.session_state.processed_content = None
if 'transcript' not in st.session_state:
    st.session_state.transcript = None
if 'video_url' not in st.session_state:
    st.session_state.video_url = None
if 'condition' not in st.session_state:
    st.session_state.condition = None
if 'content_formats' not in st.session_state:
    st.session_state.content_formats = None
if 'audio_file' not in st.session_state:
    st.session_state.audio_file = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'show_chat' not in st.session_state:
    st.session_state.show_chat = False
if 'use_dyslexic_font' not in st.session_state:
    st.session_state.use_dyslexic_font = False

# Set page config for dark mode
st.set_page_config(page_title="Neurodivergent YouTube Assistant", layout="wide")

# Add font toggle in sidebar
with st.sidebar:
    st.session_state.use_dyslexic_font = st.checkbox("Use OpenDyslexic Font", value=st.session_state.use_dyslexic_font)

# Load and embed OpenDyslexic font if enabled
if st.session_state.use_dyslexic_font:
    try:
        # Load the font file and encode it
        font_path = r"C:\Users\New User\OneDrive\Desktop\AccessiTubeAI\opendyslexic-0.91.12\opendyslexic-0.91.12\compiled\OpenDyslexic-Regular.otf"
        
        if os.path.exists(font_path):
            with open(font_path, "rb") as f:
                font_data = base64.b64encode(f.read()).decode()
            
            # Apply OpenDyslexic font to the entire app
            st.markdown(f"""
                <style>
                @font-face {{
                    font-family: 'OpenDyslexic';
                    src: url(data:font/otf;base64,{font_data}) format('opentype');
                    font-weight: normal;
                    font-style: normal;
                    font-display: swap;
                }}
                
                html, body, [class*="st-"], .stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6, 
                .stTextInput > div > div > input, .stSelectbox > div > div > select,
                .stMultiSelect > div > div > div, button, .stAlert, .stInfo, .stSuccess, .stError,
                .stWarning, .stExpander, .stTabs, .stTab, .stWidgetLabel, .stDownloadButton,
                .stFileUploader, .stProgress, .stSlider, .stRadio, .stCheckbox, .stNumberInput,
                .stDateInput, .stTimeInput, .stDataFrame, .stTable, .stImage, .stAudio, .stVideo,
                .content-container, .chat-container, .chat-message, .chat-label {{
                    font-family: 'OpenDyslexic', sans-serif !important;
                }}
                </style>
            """, unsafe_allow_html=True)
        else:
            st.sidebar.warning("⚠️ OpenDyslexic font file not found at specified path.")
    except Exception as e:
        st.sidebar.error(f"Error loading font: {str(e)}")

st.title("Neurodivergent YouTube Assistant")

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
    
    /* Video container styling */
    .video-container {
        background-color: #2D2D2D;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #3D3D3D;
        margin: 1rem 0;
    }
    
    /* Audio player styling */
    audio {
        width: 100%;
        margin: 1rem 0;
    }
    
    /* Chat container styling */
    .chat-container {
        background-color: #2D2D2D;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #3D3D3D;
        margin: 1rem 0;
        max-height: 500px;
        overflow-y: auto;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
    }
    
    .user-message {
        background-color: #3D5A80;
        margin-left: 20%;
    }
    
    .assistant-message {
        background-color: #2D3D4D;
        margin-right: 20%;
    }
    
    .chat-label {
        font-weight: bold;
        margin-bottom: 0.5rem;
        color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# Configure Gemini API
genai.configure(api_key="")

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
DURATION = 20

def text_to_speech_gtts(text, lang='en', slow=False):
    """
    Convert text to speech using gTTS and return audio file path.
    
    Args:
        text: Text to convert to speech
        lang: Language code (default 'en')
        slow: Whether to use slow speech (default False)
    
    Returns:
        Path to the generated audio file or None if failed
    """
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            audio_path = tmp_file.name
        
        # Generate speech
        tts = gTTS(text=text, lang=lang, slow=slow)
        tts.save(audio_path)
        
        return audio_path
    except Exception as e:
        st.error(f"Error generating speech: {str(e)}")
        return None

def get_audio_player_html(audio_path):
    """
    Generate HTML for audio player with autoplay option.
    
    Args:
        audio_path: Path to the audio file
    
    Returns:
        HTML string for audio player
    """
    try:
        with open(audio_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
        
        audio_base64 = base64.b64encode(audio_bytes).decode()
        
        audio_html = f"""
        <audio controls autoplay style="width: 100%; margin: 1rem 0;">
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            Your browser does not support the audio element.
        </audio>
        """
        return audio_html
    except Exception as e:
        st.error(f"Error creating audio player: {str(e)}")
        return None

# Find Virtual Audio Cable device
DEVICE_ID = None
devices = sd.query_devices()
for i, dev in enumerate(devices):
    if "cable" in dev["name"].lower():
        DEVICE_ID = i
        st.sidebar.success(f"Using Virtual Cable: {dev['name']} (ID: {i})")
        break

if DEVICE_ID is None:
    st.sidebar.warning("Virtual Audio Cable not found! Using default device.")
    st.sidebar.markdown("""**Installation Instructions:** 
1. Install Virtual Audio Cable
2. Set the Virtual Audio Cable as your audio output device
3. Restart the app to enable recording""")
    DEVICE_ID = sd.default.device[0] if isinstance(sd.default.device, tuple) else sd.default.device

def get_video_id(url):
    """Extract video ID from YouTube URL."""
    try:
        parsed_url = urlparse(url)
        if parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
            query = parse_qs(parsed_url.query)
            return query.get('v', [None])[0]
        elif parsed_url.hostname == 'youtu.be':
            return parsed_url.path[1:]
    except:
        return None
    return None

def get_video_transcript(video_id):
    """Get video transcript using YouTubeTranscriptApi.fetch() method."""
    if YouTubeTranscriptApi is None:
        st.error("YouTube Transcript API not available. Please install: pip install youtube-transcript-api")
        return None
    
    try:
        # Use the exact method you specified
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id)
        
        # Extract text from transcript snippets
        transcript_text = ' '.join([snippet.text for snippet in transcript])
        return transcript_text
    
    except Exception as e:
        st.warning(f"Could not fetch transcript: {str(e)}")
        st.info("Please try a video with English captions/subtitles enabled.")
        return None

def create_pdf(content, video_url, condition, content_formats):
    """Creates a PDF file with the processed content."""
    try:
        # Create PDF using FPDF
        pdf = FPDF()
        pdf.add_page()
        
        # Set font
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "YouTube Summary", ln=True)
        
        # Add header information
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"Video URL: {video_url}", ln=True)
        pdf.cell(0, 10, f"Condition: {condition}", ln=True)
        pdf.cell(0, 10, f"Content Formats: {', '.join(content_formats)}", ln=True)
        pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
        pdf.ln(10)
        
        # Add content
        pdf.set_font("Arial", "", 10)
        
        # Split content into lines to handle long text
        lines = content.split('\n')
        for line in lines:
            # Handle long lines by wrapping
            if len(line) > 0:
                try:
                    # Encode to latin-1, replacing characters that can't be encoded
                    line_encoded = line.encode('latin-1', 'replace').decode('latin-1')
                    pdf.multi_cell(0, 10, line_encoded)
                except:
                    # If encoding fails, skip the line
                    pdf.multi_cell(0, 10, "[Content could not be encoded]")
            else:
                pdf.ln(5)
        
        # Create a temporary file to save the PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            pdf_path = tmp_file.name
            
        # Save PDF to the temporary file
        pdf.output(pdf_path)
        
        return pdf_path
    
    except Exception as e:
        st.error(f"Error creating PDF: {str(e)}")
        return None

def record_audio(duration=DURATION, sample_rate=SAMPLE_RATE):
    """Captures system audio from Virtual Audio Cable or fallback device."""
    with st.spinner(f"Listening to system audio for {duration} seconds..."):
        try:
            recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, 
                             channels=CHANNELS, dtype=np.int16, device=DEVICE_ID)
            sd.wait()
            
            wav_io = io.BytesIO()
            write(wav_io, sample_rate, recording)
            return wav_io.getvalue()
        except Exception as e:
            st.error(f"Error recording audio: {str(e)}")
            return None

def transcribe_audio(audio_data):
    """Placeholder for audio transcription."""
    st.warning("Audio transcription requires additional services. Please use videos with subtitles.")
    return "Audio transcription is not available in this version. Please use videos with subtitles."

def process_with_gemini(text, condition, content_formats):
    """
    Processes transcribed text with Gemini for condition-specific adaptation and visual content description.
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
    2. Create ASCII art diagrams where helpful
    3. Describe spatial relationships and layouts clearly
    4. Explain any changes or transformations
    5. Use tables and structured formats for data
    6. Map out relationships between elements
    
    Additionally, include:
    - Key concepts and definitions
    - Important sections or segments
    - Potential accessibility considerations
    - Visual descriptions of all elements
    - Recommended resources or alternatives
    
    Format each section with clear headings and maintain consistent organization throughout.
    """
    
    try:
        response = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt)
        return response.text if response else "Unable to process the content."
    except Exception as e:
        return f"Error processing content: {str(e)}"

def chat_with_video(question, transcript, processed_content, condition):
    """
    Answers questions about the video using the transcript and processed content.
    
    Args:
        question: User's question
        transcript: Original video transcript
        processed_content: Processed and adapted content
        condition: User's neurodivergent condition
    
    Returns:
        Answer from Gemini
    """
    prompt = f"""
    You are an assistant helping someone with {condition} understand a YouTube video.
    
    Video Transcript:
    {transcript[:3000]}  # Limiting to avoid token limits
    
    Processed Content Summary:
    {processed_content[:2000]}
    
    User's Question: {question}
    
    Please answer the question based on the video content. Keep your answer:
    - Clear and concise
    - Tailored for someone with {condition}
    - Directly addressing their question
    - Include timestamps or specific references when possible
    
    If the question cannot be answered from the video content, politely let the user know.
    """
    
    try:
        response = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt)
        return response.text if response else "I'm unable to answer that question right now."
    except Exception as e:
        return f"Error generating answer: {str(e)}"

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
        video_id = get_video_id(video_url)
        if video_id:
            # Use iframe embed instead of st.video for better compatibility
            st.markdown(f"""
            <div class="video-container">
                <iframe width="100%" height="400" 
                src="https://www.youtube.com/embed/{video_id}" 
                frameborder="0" 
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                allowfullscreen>
                </iframe>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("Invalid YouTube URL. Please check the format.")
    except Exception as e:
        st.error(f"Error loading video: {str(e)}")

# Recording controls
col3, col4 = st.columns(2)
with col3:
    duration = st.slider("Recording Duration (seconds)", 10, 60, 20)
with col4:
    auto_save = st.checkbox("Generate PDF for download", value=True)

# TTS Settings
st.subheader("🔊 Text-to-Speech Settings")
tts_col1, tts_col2 = st.columns(2)
with tts_col1:
    tts_language = st.selectbox(
        "Language",
        options=[
            ("English", "en"),
            ("Spanish", "es"),
            ("French", "fr"),
            ("German", "de"),
            ("Italian", "it"),
            ("Portuguese", "pt"),
            ("Hindi", "hi")
        ],
        format_func=lambda x: x[0],
        help="Select the language for text-to-speech"
    )
with tts_col2:
    tts_slow = st.checkbox("Slow Speech", value=False, help="Enable slower speech rate")

# Main functionality
if st.button("🚀 Start Processing", key="start_button", use_container_width=True):
    if not content_formats:
        st.warning("Please select at least one content format.")
    elif not video_url:
        st.warning("Please enter a YouTube video URL.")
    else:
        with st.spinner("Processing..."):
            video_id = get_video_id(video_url)
            if not video_id:
                st.error("Invalid YouTube URL")
            else:
                # Try to get subtitles using the exact method specified
                transcript = get_video_transcript(video_id)
                
                # If no subtitles available, fall back to audio recording
                if transcript is None:
                    st.info("No subtitles found. Recording audio instead...")
                    audio_data = record_audio(duration=duration)
                    if audio_data:
                        transcript = transcribe_audio(audio_data)
                
                if transcript:
                    # Store in session state
                    st.session_state.transcript = transcript
                    st.session_state.video_url = video_url
                    st.session_state.condition = condition
                    st.session_state.content_formats = content_formats
                    
                    processed_content = process_with_gemini(transcript, condition, content_formats)
                    
                    # Store processed content in session state
                    st.session_state.processed_content = processed_content
                    
                    # Clear previous audio file and chat history
                    st.session_state.audio_file = None
                    st.session_state.chat_history = []
                    st.session_state.show_chat = False

# Display processed content if it exists in session state
if st.session_state.processed_content:
    # Show transcript in expander
    if st.session_state.transcript:
        with st.expander("📝 Full Transcript", expanded=False):
            st.write(st.session_state.transcript)

    st.subheader("🎯 Adapted Content")
    
    # Display processed content in a custom container
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    st.markdown(st.session_state.processed_content)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Text-to-Speech controls using gTTS
    st.subheader("🔊 Text-to-Speech")
    
    if st.button("▶️ Generate Audio", use_container_width=True):
        with st.spinner("Converting text to speech..."):
            # Generate audio file
            audio_path = text_to_speech_gtts(
                st.session_state.processed_content,
                lang=tts_language[1],
                slow=tts_slow
            )
            
            if audio_path:
                st.session_state.audio_file = audio_path
                st.success("Audio generated successfully!")
            else:
                st.error("Failed to generate audio")
    
    # Display audio player if audio file exists
    if st.session_state.audio_file and os.path.exists(st.session_state.audio_file):
        st.markdown("### Audio Player")
        
        # Read audio file
        with open(st.session_state.audio_file, "rb") as audio_file:
            audio_bytes = audio_file.read()
        
        # Use Streamlit's native audio player
        st.audio(audio_bytes, format='audio/mp3')
        
        # Download button for audio
        st.download_button(
            label="📥 Download Audio",
            data=audio_bytes,
            file_name=f"YouTube_Audio_{datetime.now().strftime('%Y%m%d_%H%M')}.mp3",
            mime="audio/mp3",
            use_container_width=True
        )
    
    # Create and offer PDF download if enabled
    if auto_save:
        with st.spinner("Creating PDF for download..."):
            pdf_path = create_pdf(
                st.session_state.processed_content, 
                st.session_state.video_url, 
                st.session_state.condition, 
                st.session_state.content_formats
            )
            if pdf_path:
                with open(pdf_path, "rb") as pdf_file:
                    pdf_bytes = pdf_file.read()
                
                # Create download button
                st.download_button(
                    label="📥 Download PDF Summary",
                    data=pdf_bytes,
                    file_name=f"YouTube_Summary_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                
                # Clean up the temporary file
                try:
                    os.unlink(pdf_path)
                except:
                    pass
    
    # Divider before chat section
    st.divider()
    
    # Chat with Video functionality
    st.subheader("💬 Chat with Video")
    
    # Toggle chat visibility
    if st.button("💬 Open Chat with Video" if not st.session_state.show_chat else "🔽 Close Chat", 
                 use_container_width=True, 
                 key="toggle_chat"):
        st.session_state.show_chat = not st.session_state.show_chat
    
    # Display chat interface if show_chat is True
    if st.session_state.show_chat:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Display chat history
        if st.session_state.chat_history:
            for i, chat in enumerate(st.session_state.chat_history):
                # User message
                st.markdown(f"""
                <div class="chat-message user-message">
                    <div class="chat-label">You:</div>
                    {chat['question']}
                </div>
                """, unsafe_allow_html=True)
                
                # Assistant message
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <div class="chat-label">Assistant:</div>
                    {chat['answer']}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("👋 Ask me anything about the video content!")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Chat input form
        with st.form(key="chat_form", clear_on_submit=True):
            user_question = st.text_input(
                "Ask a question about the video:",
                placeholder="E.g., What are the main points? Can you explain the concept of...?",
                key="chat_input"
            )
            
            col_submit, col_clear = st.columns([3, 1])
            with col_submit:
                submit_button = st.form_submit_button("Send 📤", use_container_width=True)
            with col_clear:
                clear_button = st.form_submit_button("Clear 🗑️", use_container_width=True)
        
        # Handle form submission
        if submit_button and user_question:
            with st.spinner("Thinking..."):
                answer = chat_with_video(
                    user_question,
                    st.session_state.transcript,
                    st.session_state.processed_content,
                    st.session_state.condition
                )
                
                # Add to chat history
                st.session_state.chat_history.append({
                    'question': user_question,
                    'answer': answer
                })
                
                # Rerun to display the new message
                st.rerun()
        
        # Handle clear button
        if clear_button:
            st.session_state.chat_history = []
            st.rerun()

# Enhanced sidebar with additional information
with st.sidebar:
    st.header("📋 Instructions")
    st.markdown("""
1. Paste a YouTube video URL
2. Select your condition
3. Choose content formats
4. Adjust recording duration
5. Configure TTS settings
6. Click 'Start Processing'
7. View, listen, and download
8. Chat with the video content
    """)
    
    st.header("🎯 Features")
    st.markdown("""
- ✅ Automatic subtitle extraction
- ✅ Condition-specific adaptation
- ✅ Multiple content formats
- ✅ Text-to-speech conversion (gTTS)
- ✅ Multi-language support
- ✅ Downloadable audio files
- ✅ PDF download functionality
- ✅ Visual content descriptions
- ✅ **Chat with video content**
- ✅ **OpenDyslexic font support**
    """)
    
    st.header("⚙️ Settings")
    st.markdown("**Audio Device:**")
    try:
        device_name = sd.query_devices()[DEVICE_ID]['name']
        st.info(f"Currently using: {device_name}")
    except:
        st.info("Using default audio device")
    
    st.header("ℹ️ About")
    st.markdown("""
This tool helps neurodivergent individuals better understand YouTube content through:
- Personalized content adaptation
- High-quality text-to-speech (gTTS)
- Multi-language audio support
- Visual descriptions
- Multiple format options
- Interactive Q&A chat
- Dyslexia-friendly font option

    """)
