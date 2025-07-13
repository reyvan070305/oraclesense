import streamlit as st
import joblib
import re
import string
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import base64
import time

st.set_page_config(
    page_title="✨ Sentiment Oracle", 
    layout="wide", 
    page_icon="🔮",
    initial_sidebar_state="collapsed"
)

# Load model
model = joblib.load("naive_bayes_sentiment_model.pkl")

# Preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    return text

# Ultra Gacor CSS Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Poppins:wght@300;400;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(45deg, #0a0a0a, #1a1a2e, #16213e, #0f3460);
        background-size: 400% 400%;
        animation: gradientShift 8s ease infinite;
        color: white;
        font-family: 'Poppins', sans-serif;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 25px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        animation: shine 3s infinite;
    }
    
    @keyframes shine {
        0% { transform: translateX(-100%) translateY(-100%) rotate(30deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(30deg); }
    }
    
    .hero-title {
        font-family: 'Orbitron', monospace;
        font-size: 4rem;
        font-weight: 900;
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4, #ffeaa7);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: rainbow 3s ease-in-out infinite;
        text-shadow: 0 0 30px rgba(255,255,255,0.5);
        margin: 0;
        position: relative;
        z-index: 1;
    }
    
    @keyframes rainbow {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .hero-subtitle {
        font-size: 1.5rem;
        color: #ffffff;
        margin-top: 1rem;
        font-weight: 300;
        text-shadow: 0 0 20px rgba(255,255,255,0.3);
        position: relative;
        z-index: 1;
    }
    
    .glow-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.3);
        border: 1px solid rgba(255,255,255,0.1);
        position: relative;
        overflow: hidden;
    }
    
    .glow-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        animation: scan 2s infinite;
    }
    
    @keyframes scan {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    .input-section {
        background: rgba(0,0,0,0.3);
        border-radius: 15px;
        padding: 2rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
        margin: 2rem 0;
    }
    
    .section-title {
        font-family: 'Orbitron', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: #4ecdc4;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 0 0 20px rgba(78, 205, 196, 0.5);
    }
    
    .result-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        border-radius: 25px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 20px 40px rgba(255, 107, 107, 0.3);
        border: 1px solid rgba(255,255,255,0.2);
        transform: perspective(1000px) rotateX(5deg);
        transition: all 0.3s ease;
    }
    
    .result-card:hover {
        transform: perspective(1000px) rotateX(0deg) translateY(-10px);
        box-shadow: 0 30px 60px rgba(255, 107, 107, 0.4);
    }
    
    .result-card.positive {
        background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%);
        box-shadow: 0 20px 40px rgba(78, 205, 196, 0.3);
    }
    
    .result-card.positive:hover {
        box-shadow: 0 30px 60px rgba(78, 205, 196, 0.4);
    }
    
    .sentiment-emoji {
        font-size: 4rem;
        margin-bottom: 1rem;
        animation: bounce 2s infinite;
    }
    
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
        40% { transform: translateY(-10px); }
        60% { transform: translateY(-5px); }
    }
    
    .sentiment-label {
        font-size: 2.5rem;
        font-weight: 700;
        font-family: 'Orbitron', monospace;
        text-shadow: 0 0 20px rgba(255,255,255,0.5);
        margin-bottom: 1rem;
    }
    
    .confidence-score {
        font-size: 2rem;
        font-weight: 600;
        background: rgba(255,255,255,0.2);
        padding: 0.5rem 1rem;
        border-radius: 15px;
        display: inline-block;
        backdrop-filter: blur(5px);
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .stat-card {
        background: rgba(0,0,0,0.3);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(0,0,0,0.3);
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #4ecdc4;
        font-family: 'Orbitron', monospace;
    }
    
    .stat-label {
        font-size: 1rem;
        color: #ffffff;
        margin-top: 0.5rem;
    }
    
    .neon-button {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        border: none;
        border-radius: 25px;
        padding: 1rem 2rem;
        font-size: 1.2rem;
        font-weight: 700;
        color: white;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 0 20px rgba(255, 107, 107, 0.5);
        text-transform: uppercase;
        letter-spacing: 1px;
        font-family: 'Orbitron', monospace;
    }
    
    .neon-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(255, 107, 107, 0.6);
    }
    
    .upload-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        background: rgba(0,0,0,0.3);
        border-radius: 15px;
        margin-top: 3rem;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .particle {
        position: fixed;
        width: 4px;
        height: 4px;
        background: #4ecdc4;
        border-radius: 50%;
        pointer-events: none;
        animation: float 6s infinite ease-in-out;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); opacity: 1; }
        50% { transform: translateY(-20px) rotate(180deg); opacity: 0.5; }
    }
    
    /* Streamlit specific overrides */
    .stTextArea textarea {
        background: rgba(0,0,0,0.3) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 15px !important;
        color: white !important;
        font-size: 1.1rem !important;
        padding: 1rem !important;
    }
    
    .stButton button {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4) !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 1rem 2rem !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        color: white !important;
        box-shadow: 0 0 20px rgba(255, 107, 107, 0.5) !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        font-family: 'Orbitron', monospace !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 10px 25px rgba(255, 107, 107, 0.6) !important;
    }
    
    .stFileUploader {
        background: rgba(0,0,0,0.3) !important;
        border-radius: 15px !important;
        padding: 1rem !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
    }
    
    .stDataFrame {
        background: rgba(0,0,0,0.3) !important;
        border-radius: 15px !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
    }
    
    .stSuccess {
        background: rgba(78, 205, 196, 0.2) !important;
        border-left: 4px solid #4ecdc4 !important;
        border-radius: 10px !important;
    }
    
    .stWarning {
        background: rgba(255, 193, 7, 0.2) !important;
        border-left: 4px solid #ffc107 !important;
        border-radius: 10px !important;
    }
    
    .stError {
        background: rgba(255, 107, 107, 0.2) !important;
        border-left: 4px solid #ff6b6b !important;
        border-radius: 10px !important;
    }
    </style>
""", unsafe_allow_html=True)

# Floating particles effect
st.markdown("""
    <div class="particle" style="top: 10%; left: 10%; animation-delay: 0s;"></div>
    <div class="particle" style="top: 20%; left: 80%; animation-delay: 1s;"></div>
    <div class="particle" style="top: 60%; left: 20%; animation-delay: 2s;"></div>
    <div class="particle" style="top: 80%; left: 90%; animation-delay: 3s;"></div>
    <div class="particle" style="top: 40%; left: 60%; animation-delay: 4s;"></div>
""", unsafe_allow_html=True)

# Main Header
st.markdown("""
    <div class="main-header">
        <h1 class="hero-title">DETEKSI KOMENTAR NEGATIF</h1>
        <p class="hero-subtitle">🔮 Dengan Menggunakan Algoritma Naive Bayes 🔮</p>
    </div>
""", unsafe_allow_html=True)

# Create columns for better layout
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("""
        <div class="input-section">
            <h2 class="section-title">✨ Enter Your Text</h2>
        </div>
    """, unsafe_allow_html=True)
    
    # Input text area
    user_input = st.text_area(
        "📝 Paste your text here and let the magic begin...",
        height=150,
        placeholder="Type something amazing here... 🌟"
    )
    
    # Analysis button
    if st.button("🚀 ANALYZE NOW!", use_container_width=True):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some text to analyze!")
        else:
            # Loading animation
            with st.spinner('🔮 The Oracle is reading your text...'):
                time.sleep(1)  # Add dramatic pause
                
                cleaned = clean_text(user_input)
                prediction = model.predict([cleaned])[0]
                probs = model.predict_proba([cleaned])[0]
                
                label = "POSITIVE" if prediction == 1 else "NEGATIVE"
                emoji = "🌟" if prediction == 1 else "💥"
                card_class = "positive" if prediction == 1 else ""
                conf = probs[prediction] * 100
                
                # Display result with dramatic effect
                st.markdown(f"""
                    <div class="result-card {card_class}">
                        <div class="sentiment-emoji">{emoji}</div>
                        <div class="sentiment-label">SENTIMENT: {label}</div>
                        <div class="confidence-score">Confidence: {conf:.2f}%</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Statistics grid
                st.markdown("""
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="stat-number">{:.1f}%</div>
                            <div class="stat-label">Positive</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">{:.1f}%</div>
                            <div class="stat-label">Negative</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">{}</div>
                            <div class="stat-label">Words</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">{}</div>
                            <div class="stat-label">Characters</div>
                        </div>
                    </div>
                """.format(
                    probs[1] * 100,
                    probs[0] * 100,
                    len(user_input.split()),
                    len(user_input)
                ), unsafe_allow_html=True)
                
                # Enhanced visualization
                fig = go.Figure(data=[
                    go.Pie(
                        labels=["Negative", "Positive"],
                        values=probs,
                        hole=0.6,
                        textinfo='label+percent',
                        textfont=dict(size=16, color='white'),
                        marker=dict(
                            colors=['#ff6b6b', '#4ecdc4'],
                            line=dict(color='white', width=2)
                        ),
                        hovertemplate='<b>%{label}</b><br>Confidence: %{percent}<extra></extra>'
                    )
                ])
                
                fig.update_layout(
                    showlegend=True,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white', size=14),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    margin=dict(t=50, b=50, l=50, r=50),
                    height=400
                )
                
                st.markdown("""
                    <div class="glow-container">
                        <h3 class="section-title">📊 Probability Distribution</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                st.plotly_chart(fig, use_container_width=True)



# Footer
st.markdown("""
    <div class="footer">
        <h3 style="color: #4ecdc4; margin-bottom: 1rem;">✨ Sentiment Oracle ✨</h3>
        <p style="color: #ffffff; font-size: 1.1rem;">
            Powered by AI Magic | Built with ❤️ and 🔮
        </p>
        <p style="color: #999; font-size: 0.9rem; margin-top: 1rem;">
            © 2025 | Transforming text into insights, one sentiment at a time
        </p>
    </div>
""", unsafe_allow_html=True)