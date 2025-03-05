import streamlit as st
import streamlit.components.v1 as components
import importlib
import sys
from pathlib import Path

def set_custom_style():
    st.markdown("""
        <style>
        /* Modern Dark Theme */
        .stApp {
            background: linear-gradient(135deg, #13151a 0%, #1e2128 100%);
            color: #e0e0e0;
        }
        
        /* Hide Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Animated gradient header */
        .main-header {
            background: linear-gradient(-45deg, #6e45e2, #88d3ce, #d4267d, #6e45e2);
            background-size: 400% 400%;
            animation: gradient 15s ease infinite;
            padding: 3rem 2rem;
            border-radius: 20px;
            margin-bottom: 3rem;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
            border: 1px solid rgba(255, 255, 255, 0.18);
            backdrop-filter: blur(4px);
            text-align: center;
        }
        
        @keyframes gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        /* Glass morphism cards */
        .app-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 2rem;
            margin-bottom: 1.5rem;
            transition: all 0.3s ease;
        }
        
        .app-card:hover {
            transform: translateY(-10px);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        /* Neon text effects */
        .neon-text {
            color: #fff;
            text-shadow: 0 0 10px #6e45e2,
                         0 0 20px #6e45e2,
                         0 0 30px #6e45e2;
        }
        
        /* Enhanced button styling */
        .stButton > button {
            width: 100%;
            background: linear-gradient(45deg, #6e45e2, #88d3ce);
            color: white;
            border: none;
            padding: 1rem 2rem;
            border-radius: 10px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(110, 69, 226, 0.3);
        }
        
        .stButton > button:hover {
            transform: translateY(-3px) scale(1.02);
            box-shadow: 0 6px 20px rgba(110, 69, 226, 0.4);
        }
        
        /* Modern headings */
        h1 {
            font-size: 3.5rem;
            font-weight: 800;
            color: white !important;
            -webkit-text-fill-color: white !important;
            text-align: center;
            margin-bottom: 1rem;
        }
        
        h2 {
            color: white !important;
            -webkit-text-fill-color: white !important;
            font-weight: 700;
            margin-bottom: 1.5rem;
            text-align: center;
        }
        
        h3 {
            color: #6e45e2;
            font-weight: 600;
            margin-bottom: 1rem;
            text-align: center;
        }

        /* Custom footer */
        .footer {
            text-align: center;
            padding: 2rem;
            margin-top: 4rem;
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(16px);
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .footer p {
            text-align: center;
            margin: 0.5rem 0;
        }

        /* List styling */
        ul {
            list-style-type: none;
            padding: 0;
            text-align: center;
        }

        li {
            margin: 0.5rem 0;
            color: #e0e0e0;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="NeuroAI Hub",
        page_icon="🧠",
        layout="wide"
    )
    
    set_custom_style()
    show_home_page()

def show_home_page():
    # Animated Header
    st.markdown("""
        <div class="main-header">
            <h1>🧠 NeuroAI Hub</h1>
            <h2 class="neon-text">Next-Generation AI for Neurodivergent Minds</h2>
            <p style="font-size: 1.2rem; opacity: 0.9;">Empowering unique minds with cutting-edge artificial intelligence</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Main Features Section
    st.markdown("""
        <div class="app-card">
            <h2>🚀 Welcome to the Future of Cognitive Diversity</h2>
            <p style="font-size: 1.2rem; margin: 2rem 0; text-align: center;">
                Experience our revolutionary AI-powered platform designed to enhance, adapt, and transform 
                how neurodivergent individuals interact with technology. Our suite of tools is built with 
                deep learning algorithms that understand and adapt to your unique cognitive style.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Feature Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="app-card">
            <h3>📅 NeuroSchedule</h3>
            <p style="color: #88d3ce; margin-bottom: 1rem; text-align: center;">AI-Powered Time Management</p>
            <ul>
                <li>Smart energy-based scheduling</li>
                <li>Visual time tracking with AI insights</li>
                <li>Adaptive notification system</li>
                <li>Custom routine optimization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch NeuroSchedule ⚡"):
            st.switch_page("pages/Scheduler.py")
    
    with col2:
        st.markdown("""
        <div class="app-card">
            <h3>🏢 NeuroDivAI</h3>
            <p style="color: #88d3ce; margin-bottom: 1rem; text-align: center;">Workplace Enhancement Suite</p>
            <ul>
                <li>Real-time communication assistance</li>
                <li>Sensory environment optimizer</li>
                <li>Advanced task decomposition</li>
                <li>Smart accommodation planning</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch NeuroDivAI 🌟"):
            st.switch_page("pages/neurodivai.py")
    
    with col3:
        st.markdown("""
        <div class="app-card">
            <h3>🎥 NeuroTube AI</h3>
            <p style="color: #88d3ce; margin-bottom: 1rem; text-align: center;">Enhanced Learning Experience</p>
            <ul>
                <li>Dynamic content adaptation</li>
                <li>Multi-modal learning paths</li>
                <li>Focus-enhancing filters</li>
                <li>Personalized learning analytics</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch NeuroTube AI 🎯"):
            st.switch_page("pages/neurotube.py")
    
    # Search Section
    st.markdown("""
    <div class="app-card" style="margin-top: 3rem;">
        <h2>🔍 NeuroSearch</h2>
        <p style="text-align: center; font-size: 1.2rem; margin: 2rem 0;">
            Experience the world's first search engine built specifically for neurodivergent minds,
            powered by advanced AI algorithms that understand your unique way of processing information.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Experience NeuroSearch 🚀"):
        st.switch_page("pages/neurosearch.py")
    
    # Footer
    st.markdown("""
    <div class="footer">
        <h3 class="neon-text">Transforming Lives Through Technology</h3>
        <p style="font-size: 1.1rem;">© 2025 NeuroAI Hub | Pioneer in Neurodivergent AI Solutions</p>
        <p style="color: #88d3ce;">Made with 💜 by the community, for the community</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()