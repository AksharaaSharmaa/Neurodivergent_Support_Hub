import streamlit as st
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from googleapiclient.discovery import build
from google.cloud import translate_v2 as translate
import json
import os
from google.cloud import secretmanager
import google.generativeai as genai
from googleapiclient.discovery import build

# Back button
if st.button("← Back to Home"):
    st.switch_page("HomePage.py")

# Configure page settings
st.title("ClearView Search")

# Updated CSS with better text visibility
st.markdown("""
<style>
    /* Base theme colors */
    :root {
        --text-color: #333333;
        --background-color: #ffffff;
        --link-color: #1a73e8;
        --border-color: #e9ecef;
    }

    /* Search result container */
    .search-result {
        background-color: var(--background-color);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border: 1px solid var(--border-color);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    /* Title styling */
    .result-title {
        color: var(--link-color);
        font-size: 1.2em;
        margin-bottom: 10px;
        font-weight: 600;
    }

    /* Link styling */
    .result-link {
        color: #34a853;
        font-size: 0.9em;
        margin-bottom: 15px;
    }

    /* Content styling */
    .result-content {
        line-height: 1.6;
        color: var(--text-color);
    }

    /* Language selector styling */
    .language-selector {
        padding: 10px;
        background-color: var(--background-color);
        border-radius: 5px;
        margin-bottom: 15px;
        border: 1px solid var(--border-color);
    }

    /* Button styling */
    .stButton>button {
        width: 100%;
        background-color: var(--link-color);
        color: white;
    }

    /* Ensure text visibility in dark mode */
    @media (prefers-color-scheme: dark) {
        :root {
            --text-color: #ffffff;
            --background-color: #1e1e1e;
            --link-color: #60a5fa;
            --border-color: #374151;
        }
    }

    /* Override Streamlit's default text colors */
    .stMarkdown, .stText {
        color: var(--text-color) !important;
    }
</style>
""", unsafe_allow_html=True)

# Function to access secrets from Google Secret Manager
def access_secret(secret_name):
    client = secretmanager.SecretManagerServiceClient()
    project_id = "neurodivai"
    secret_path = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
    response = client.access_secret_version(request={"name": secret_path})
    return response.payload.data.decode("UTF-8")

# Fetch secrets dynamically
SEARCH_ENGINE_ID = access_secret("SEARCH_ENGINE_ID")
GEMINI_API_KEY = access_secret("GEMINI_API_KEY")
GOOGLE_API_KEY = access_secret("GOOGLE_API_KEY")

# Initialize API clients
def init_google_apis():
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    search_service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
    
    try:
        credentials_info = json.loads(access_secret("SERVICE_ACCOUNT_JSON"))
        translate_client = translate.Client.from_service_account_info(credentials_info)
    except Exception as e:
        st.error(f"Error loading translation service: {str(e)}")
        translate_client = None

    return model, search_service, translate_client


# Supported languages
LANGUAGES = {
    'English': 'en',
    'Spanish': 'es',
    'French': 'fr',
    'German': 'de',
    'Japanese': 'ja'
}

# Fetch and parse webpage content
def fetch_webpage_content(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'iframe']):
            element.decompose()
            
        # Extract main content
        main_content = soup.find('main') or soup.find('article') or soup.find('body')
        
        if main_content:
            # Get all paragraphs and headings
            content_elements = main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            content = ' '.join([elem.get_text(strip=True) for elem in content_elements])
            return content[:15000]  # Limit content length for API
        return None
    except Exception as e:
        st.error(f"Error fetching webpage: {str(e)}")
        return None

# Search function
def perform_search(query, search_service):
    try:
        results = search_service.cse().list(
            q=query,
            cx=SEARCH_ENGINE_ID,
            num=5
        ).execute()
        return results.get('items', [])
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []

# Content processing with Gemini
def process_content(content, simplify_level, model):
    prompt = f"""
    Please analyze and restructure the following content to be more accessible for neurodivergent readers.
    Simplification Level: {simplify_level}
    
    Guidelines:
    1. {'Drastically simplify language' if simplify_level == 'High' else 'Moderately simplify language' if simplify_level == 'Medium' else 'Slightly simplify language'}
    2. Add clear structure with headers and sections
    3. Highlight key points in bold
    4. {'Remove all non-essential information' if simplify_level == 'High' else 'Keep important details' if simplify_level == 'Medium' else 'Maintain most content'}
    5. Use short paragraphs and clear transitions
    
    Content: {content}
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        return content

# Translate content
def translate_text(text, target_language, translate_client):
    if not translate_client or not text:
        return text
        
    try:
        result = translate_client.translate(
            text,
            target_language=target_language
        )
        return result['translatedText']
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return text

# Main app
def main():
    st.title("🔍 ClearView Search")
    st.markdown("### Making the web more accessible, one search at a time")
    
    # Initialize APIs
    model, search_service, translate_client = init_google_apis()
    
    if not search_service:
        st.error("Failed to initialize Google Custom Search API. Check your API key.")
        return
    
    # Layout configuration
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Search interface
        with st.form("search_form"):
            query = st.text_input(
                "Enter your search query",
                placeholder="What would you like to learn about?",
                help="Type your question or topic here"
            )
            
            cols = st.columns([1, 1])
            with cols[0]:
                simplify_level = st.select_slider(
                    "Simplification Level",
                    options=["Light", "Medium", "High"],
                    value="Medium"
                )
            with cols[1]:
                target_language = st.selectbox(
                    "Select Language",
                    options=list(LANGUAGES.keys()),
                    index=0
                )
            
            submitted = st.form_submit_button("Search 🔍")
    
    with col2:
        st.markdown("### Reading Preferences")
        font_size = st.slider("Font Size", min_value=12, max_value=24, value=16)
        line_spacing = st.slider("Line Spacing", min_value=1.0, max_value=2.0, value=1.6, step=0.1)
        
        # Apply reading preferences
        st.markdown(f"""
        <style>
            .result-content {{
                font-size: {font_size}px !important;
                line-height: {line_spacing} !important;
            }}
        </style>
        """, unsafe_allow_html=True)
    
    if submitted and query:
        with st.spinner("Searching and processing results..."):
            results = perform_search(query, search_service)
            
            if results:
                for result in results:
                    with st.container():
                        st.markdown(f"""
                        <div class="search-result">
                            <div class="result-title">{result['title']}</div>
                            <div class="result-link"><a href="{result['link']}" target="_blank">{result['link']}</a></div>
                        """, unsafe_allow_html=True)
                        
                        # Fetch and process full content
                        content = fetch_webpage_content(result['link']) or result['snippet']
                        processed_content = process_content(content, simplify_level, model)
                        
                        # Translate if needed
                        if target_language != 'English':
                            processed_content = translate_text(
                                processed_content,
                                LANGUAGES[target_language],
                                translate_client
                            )
                        
                        st.markdown(f'<div class="result-content">{processed_content}</div>', unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("No results found. Try modifying your search terms.")

    # Sidebar content
    with st.sidebar:
        st.markdown("### About ClearView Search 🎯")
        st.markdown("""
        ClearView Search is designed to make web content more accessible by:
        - Simplifying complex language
        - Adding clear structure
        - Highlighting key information
        - Offering multiple language support
        - Providing customizable reading experience
        """)
        
        st.markdown("### Search Tips 💡")
        st.markdown("""
        - Use specific keywords
        - Ask questions naturally
        - Try different phrasings
        - Start broad, then narrow down
        """)
        
        st.markdown("### Accessibility Features ⚡")
        st.markdown("""
        - Adjustable text size
        - Customizable line spacing
        - Multiple language support
        - Content simplification levels
        - Clean, distraction-free layout
        """)

if __name__ == "__main__":
    main()