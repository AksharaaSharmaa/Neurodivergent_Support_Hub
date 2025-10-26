# AccessiTube AI 🧠

AccessiTube AI is a web-based assistant built with Streamlit and Google's Gemini AI, designed to make YouTube video content more accessible for neurodivergent individuals. It transforms video transcripts into various accessible formats, provides text-to-speech output, and features a powerful "Chat with Video" function powered by a Retrieval-Augmented Generation (RAG) pipeline.

This tool is built to support users with conditions like **ADHD**, **Autism**, **Dyslexia**, and **Anxiety** by tailoring content to their specific needs.

-----

## 🎯 Key Features

  * **Adaptive Content Processing:** Takes a YouTube URL and generates summaries, breakdowns, and simplified text based on a selected neurodivergent condition.
  * **Multiple Content Formats:**
      * **Bullet-point summary:** For quick, digestible overviews.
      * **Full-length transcript:** For detailed reading.
      * **Visual breakdown:** Describes visual elements, diagrams, and spatial layouts in text (ASCII art, tables) for users who benefit from textual descriptions.
      * **Simplified language:** Rewrites complex topics into simple, clear sentences.
  * **Text-to-Speech (TTS):** Converts the processed content into audio using gTTS, with options for multiple languages and speech speed.
  * **Downloadable Formats:**
      * **Download PDF:** Save the adapted content as a PDF for offline viewing.
      * **Download Audio:** Save the generated TTS audio as an MP3 file.
  * **Accessibility First:**
      * **OpenDyslexic Font:** Includes a toggle to apply the OpenDyslexic font across the entire application for improved readability.
      * **Dark Mode UI:** A clean, high-contrast dark mode interface.
  * **Chat with Video (RAG):** An interactive chat interface that allows users to ask specific questions about the video. Answers are generated using a RAG pipeline to ensure they are accurate and grounded in the video's content.
  * **Audio Fallback:** If a video transcript is unavailable, the app can attempt to record system audio (via Virtual Audio Cable) and transcribe it.

-----

## ⚙️ How It Works: The RAG Pipeline

The "Chat with Video" feature is powered by a Retrieval-Augmented Generation (RAG) system. This ensures that the AI's answers are based *directly* on the video's content, not on its general knowledge, which prevents hallucinations and provides accurate, in-context answers.

The RAG pipeline (defined in `rag_functions.py`) works in two stages:

### 1\. Indexing Stage (When Content is Processed)

When you first process a video, the app prepares the transcript for a smart search:

1.  **Chunking:** The full transcript is broken down into small, overlapping text chunks (`chunk_text`). This retains semantic context.
2.  **Embedding:** Each chunk is converted into a numerical vector (an "embedding") using a `SentenceTransformer` model. These vectors represent the meaning of the text.
3.  **Indexing:** The vectors are loaded into a `FAISS` index. FAISS is a high-speed library for efficient similarity search, allowing the app to instantly find the most relevant text chunks for any given question.

### 2\. Retrieval & Generation Stage (When You Ask a Question)

When you type a question into the chat:

1.  **Retrieve:** Your question is also converted into an embedding. The app uses this query vector to search the `FAISS` index and retrieve the `top-k` (e.g., top 3) most relevant text chunks from the original transcript.
2.  **Augment:** The retrieved chunks (the "context") are stuffed into a prompt along with your question.
3.  **Generate:** This complete prompt is sent to the Gemini model. The model is instructed: "Answer this **[Question]** using *only* this **[Context]**."
4.  **Answer:** The model generates a response that is directly synthesized from the retrieved information, providing an accurate, timestamp-relevant answer.

This RAG approach allows the chat to answer highly specific questions like "What did the speaker say about [X concept] at [Y time]?" with precision.

-----

## 🔧 Installation & Setup

To run this project locally, follow these steps.

### 1\. Clone the Repository

```bash
git clone https://github.com/AksharaaSharma/Neurodivergent_Support_Hub.git
cd AccessiTube-AI
```

### 2\. Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3\. Install Requirements

The project uses several Python libraries. Create a `requirements.txt` file with the following content:

**requirements.txt**

```
streamlit
google-generativeai
youtube-transcript-api
sounddevice
scipy
numpy
fpdf
gtts
faiss-cpu
sentence-transformers
scikit-learn
```

Then, install them:

```bash
pip install -r requirements.txt
```

*(Note: Use `faiss-gpu` if you have a compatible NVIDIA GPU and CUDA installed for better performance.)*

### 4\. Configure API Key

The app requires a Google Gemini API key.

1.  Get your key from [Google AI Studio](https://aistudio.google.com/app/apikey).
2.  Create a file named `.env` in the root of your project directory.
3.  Add your API key to it:
    ```
    GOOGLE_API_KEY="AIzaSy...[YOUR_KEY]"
    ```
    The app is configured to load this key from `genai.configure(api_key=...)`. For better security, you should modify the code to load it from `os.environ`.

### 5\. Add OpenDyslexic Font

The app is hardcoded to find the OpenDyslexic font. For it to work, you must:

1.  [Download the OpenDyslexic font](https://opendyslexic.org/).
2.  Create a folder in your project directory (e.g., `fonts`).
3.  Place `OpenDyslexic-Regular.otf` inside that folder.
4.  Open `app.py` and **change the `font_path` variable** to point to your local file:
    ```python
    font_path = r"fonts/OpenDyslexic-Regular.otf" 
    ```

-----

## 🚀 How to Run

Once everything is installed and configured, run the Streamlit app from your terminal:

```bash
streamlit run app.py
```

Open your web browser and navigate to the local URL (usually `http://localhost:8501`).

-----

## Usage Guide

1.  **Paste URL:** Find a YouTube video and paste its URL into the input box.
2.  **Select Condition:** Choose the neurodivergent condition that best fits your needs.
3.  **Choose Formats:** Select one or more content formats (e.g., "Bullet-point summary," "Simplified language").
4.  **Process:** Click "🚀 Start Processing." The app will fetch the transcript and generate the adapted content.
5.  **Review:** Read the generated content in the main area.
6.  **Listen:** Click "▶️ Generate Audio" to create an audio version of the content.
7.  **Download:** Use the "📥 Download PDF" or "📥 Download Audio" buttons to save the content.
8.  **Chat:** Open the "💬 Chat with Video" section to ask specific questions about the video's content.

-----

## 🤝 Contributing

Contributions are welcome\! If you have ideas for new features, accessibility improvements, or bug fixes, please:

1.  Fork the repository.
2.  Create a new feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

-----

## 📝 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
