"""
Duova AI — Next-Generation Streamlit Front-End
------------------------------------------------
A state-of-the-art UI for the Duova AI RAG & Generative AI platform.

Features:
  • RAG Mode: Precision PDF Question Answering with ChromaDB Vector Search,
    live citations, source inspection drawer, and animated 0-100% ingestion.
  • AI Mode: Multi-Persona Conversational Intelligence (Normal, Expert, Quick)
    with persistent sub-mode histories.
  • Dynamic Dark Glassmorphic Design with Sora & Plus Jakarta Sans typography.

Run with:
    streamlit run duova_app.py
"""

import os
import re
import sqlite3
import tempfile
import time

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from create_DB import index_pdf
from preindex import run_preindex

# --------------------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Duova AI — Hybrid RAG & Intelligence",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------------------
# GLOBAL ULTRA-PREMIUM STYLE SYSTEM
# --------------------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@700;800;900&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Sora:wght@600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global Typography & Reset */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #E2E8F0;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Sora', sans-serif;
        letter-spacing: -0.02em;
    }
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Background with Subtle Dynamic Radial Gradients */
    .stApp {
        background-color: #07090E;
        background-image: 
            radial-gradient(circle at 10% 0%, rgba(0, 217, 192, 0.09) 0%, transparent 40%),
            radial-gradient(circle at 90% 10%, rgba(177, 74, 255, 0.09) 0%, transparent 40%),
            radial-gradient(circle at 50% 60%, rgba(15, 23, 42, 0.5) 0%, transparent 80%);
        background-attachment: fixed;
    }

    /* Sidebar Glassmorphism */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0B0E17 0%, #110E22 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.07);
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.4);
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    /* Refined Ultra-Spacious Hero Section */
    .duova-hero {
        position: relative;
        padding: 1.4rem 2rem;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(16, 24, 39, 0.7) 0%, rgba(20, 16, 36, 0.7) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(20px);
        margin-bottom: 2rem; /* Generous breathing room between hero and mode cards */
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
        overflow: hidden;
    }
    .duova-hero::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #00D9C0, #B14AFF, #3D8BFD);
    }
    .hero-top-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.5rem;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        background: rgba(0, 217, 192, 0.08);
        border: 1px solid rgba(0, 217, 192, 0.28);
        color: #00D9C0;
        font-size: 0.74rem;
        font-weight: 700;
        padding: 0.22rem 0.7rem;
        border-radius: 999px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .hero-status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.74rem;
        color: #94A3B8;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    .status-beacon {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #10B981;
        box-shadow: 0 0 8px #10B981;
        animation: pulse 2s infinite;
    }
    .pulsing-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #00D9C0;
        box-shadow: 0 0 8px #00D9C0;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(0.85); }
    }

    /* Grand Animated Title */
    @keyframes titleGradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes titleAura {
        0%, 100% { filter: drop-shadow(0 0 16px rgba(0, 217, 192, 0.45)) drop-shadow(0 0 30px rgba(177, 74, 255, 0.25)); }
        50% { filter: drop-shadow(0 0 26px rgba(177, 74, 255, 0.55)) drop-shadow(0 0 40px rgba(0, 217, 192, 0.35)); }
    }
    .duova-title {
        font-family: 'Outfit', sans-serif !important;
        font-size: 3.1rem !important;
        font-weight: 900 !important;
        letter-spacing: -0.03em !important;
        margin: 0.15rem 0 !important;
        background: linear-gradient(135deg, #FFFFFF 0%, #00F5D4 25%, #C084FC 55%, #38BDF8 80%, #FFFFFF 100%) !important;
        background-size: 250% 250% !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        line-height: 1.1 !important;
        display: inline-block !important;
        animation: titleGradient 6s ease-in-out infinite, titleAura 4s ease-in-out infinite !important;
    }
    .duova-sub {
        color: #94A3B8;
        font-size: 0.95rem;
        margin-top: 0.35rem;
        margin-bottom: 0.7rem;
        letter-spacing: 0.2px;
    }
    .hero-strip {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-top: 0.6rem;
        padding-top: 0.7rem;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
    }
    .hero-strip-item {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        color: #CBD5E1;
        font-size: 0.78rem;
        font-weight: 500;
    }
    .hero-strip-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
    }
    .hero-strip-dot.cyan { background-color: #00D9C0; box-shadow: 0 0 6px #00D9C0; }
    .hero-strip-dot.purple { background-color: #B14AFF; box-shadow: 0 0 6px #B14AFF; }
    .hero-strip-divider {
        color: rgba(255, 255, 255, 0.15);
        font-size: 0.75rem;
    }

    /* Mode Selection Showcase Cards (Home View) */
    .showcase-card {
        background: linear-gradient(145deg, rgba(17, 24, 39, 0.7) 0%, rgba(15, 23, 42, 0.5) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 1.8rem;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        backdrop-filter: blur(12px);
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
        position: relative;
        overflow: hidden;
    }
    .showcase-card:hover {
        transform: translateY(-4px);
        border-color: rgba(255, 255, 255, 0.16);
    }
    .showcase-card.rag:hover {
        box-shadow: 0 12px 30px rgba(0, 217, 192, 0.12);
        border-color: rgba(0, 217, 192, 0.35);
    }
    .showcase-card.ai:hover {
        box-shadow: 0 12px 30px rgba(177, 74, 255, 0.14);
        border-color: rgba(177, 74, 255, 0.35);
    }
    .showcase-card .icon-box {
        width: 52px;
        height: 52px;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.7rem;
        margin-bottom: 1.2rem;
    }
    .showcase-card.rag .icon-box {
        background: linear-gradient(135deg, rgba(0, 217, 192, 0.2), rgba(0, 168, 150, 0.05));
        border: 1px solid rgba(0, 217, 192, 0.3);
    }
    .showcase-card.ai .icon-box {
        background: linear-gradient(135deg, rgba(177, 74, 255, 0.2), rgba(124, 42, 232, 0.05));
        border: 1px solid rgba(177, 74, 255, 0.3);
    }
    .showcase-card h3 {
        color: #F8FAFC;
        font-size: 1.4rem;
        margin-top: 0;
        margin-bottom: 0.5rem;
    }
    .showcase-card p {
        color: #94A3B8;
        font-size: 0.95rem;
        line-height: 1.5;
        margin-bottom: 1.2rem;
    }
    .feature-list {
        list-style: none;
        padding: 0;
        margin: 0 0 1.5rem 0;
    }
    .feature-list li {
        color: #CBD5E1;
        font-size: 0.88rem;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 0.45rem;
    }

    /* Active Mode Status Ribbon */
    .mode-ribbon {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.85rem 1.3rem;
        border-radius: 14px;
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.07);
        backdrop-filter: blur(12px);
        margin-bottom: 1.2rem;
    }
    .ribbon-left {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .ribbon-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #F8FAFC;
        margin: 0;
    }
    .ribbon-tag {
        font-size: 0.78rem;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }
    .ribbon-tag.rag {
        background: rgba(0, 217, 192, 0.12);
        color: #00D9C0;
        border: 1px solid rgba(0, 217, 192, 0.3);
    }
    .ribbon-tag.ai {
        background: rgba(177, 74, 255, 0.12);
        color: #D8B4FE;
        border: 1px solid rgba(177, 74, 255, 0.3);
    }

    /* Starter Prompt Chips */
    .prompt-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 0.8rem;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }
    .prompt-chip {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 0.85rem 1rem;
        color: #CBD5E1;
        font-size: 0.88rem;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .prompt-chip:hover {
        background: rgba(0, 217, 192, 0.06);
        border-color: rgba(0, 217, 192, 0.3);
        color: #F8FAFC;
        transform: translateY(-2px);
    }

    /* Chat Messages Elevation */
    div[data-testid="stChatMessage"] {
        background: rgba(15, 23, 42, 0.45) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 16px !important;
        padding: 1rem 1.2rem !important;
        margin-bottom: 0.9rem !important;
        backdrop-filter: blur(10px) !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15) !important;
    }
    div[data-testid="stChatMessage"][data-test-assistant="true"],
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        border-left: 3px solid #00D9C0 !important;
        background: linear-gradient(135deg, rgba(16, 24, 39, 0.6) 0%, rgba(13, 20, 32, 0.6) 100%) !important;
    }

    /* Ultra-Visible Neon Progress Bar for PDF Ingestion */
    div[data-testid="stProgressBar"],
    .stProgress {
        background: rgba(255, 255, 255, 0.08) !important;
        border: 1.5px solid rgba(0, 217, 192, 0.6) !important;
        border-radius: 999px !important;
        height: 14px !important;
        overflow: hidden !important;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.6), 0 0 12px rgba(0, 217, 192, 0.4) !important;
        margin: 0.5rem 0 !important;
    }
    div[data-testid="stProgressBar"] > div,
    .stProgress > div {
        height: 100% !important;
        background-color: transparent !important;
    }
    div[data-testid="stProgressBar"] > div > div,
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #00D9C0 0%, #3D8BFD 50%, #B14AFF 100%) !important;
        border-radius: 999px !important;
        height: 100% !important;
        box-shadow: 0 0 16px #00D9C0, 0 0 28px rgba(0, 217, 192, 0.8) !important;
        transition: width 0.3s ease-in-out !important;
    }

    /* Sidebar Document Status Card */
    .sidebar-doc-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 0.8rem;
        margin-top: 0.5rem;
        margin-bottom: 0.8rem;
    }
    .sidebar-doc-card .doc-title {
        color: #F8FAFC;
        font-weight: 700;
        font-size: 0.88rem;
        word-break: break-all;
    }
    .sidebar-doc-card .doc-meta {
        color: #94A3B8;
        font-size: 0.78rem;
        margin-top: 0.2rem;
    }

    /* Comprehensive Attractive Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #090C15 0%, #0F0E20 50%, #080B14 100%) !important;
        border-right: 1px solid rgba(0, 217, 192, 0.15) !important;
        box-shadow: 6px 0 32px rgba(0, 0, 0, 0.6) !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    /* All Sidebar Buttons Base Styling & Dynamic Hover Effects */
    section[data-testid="stSidebar"] div[data-testid="stButton"] button {
        width: 100%;
        border-radius: 14px !important;
        padding: 0.65rem 1rem !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
        letter-spacing: 0.3px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        background: linear-gradient(145deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.015) 100%) !important;
        color: #E2E8F0 !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        position: relative !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover {
        transform: translateY(-3px) scale(1.02) !important;
        border-color: #00D9C0 !important;
        background: linear-gradient(145deg, rgba(0, 217, 192, 0.18) 0%, rgba(177, 74, 255, 0.12) 100%) !important;
        color: #FFFFFF !important;
        box-shadow: 0 8px 24px rgba(0, 217, 192, 0.4), 0 0 16px rgba(0, 217, 192, 0.2) !important;
        filter: brightness(1.15) !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stButton"] button:active {
        transform: translateY(0px) scale(0.99) !important;
    }

    /* Sidebar File Uploader — only style the dropzone Upload button, NOT the ✕ chip button */
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
        background: linear-gradient(135deg, #00D9C0, #009688) !important;
        color: #042420 !important;
        font-weight: 800 !important;
        border: none !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 14px rgba(0, 217, 192, 0.35) !important;
        transition: all 0.2s ease !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button:hover {
        transform: scale(1.04) !important;
        filter: brightness(1.15) !important;
        box-shadow: 0 6px 20px rgba(0, 217, 192, 0.5) !important;
    }
    /* Multi-document mode: the "+" add-more-files button is intentionally visible */
    /* Doc list scrollable area */
    .doc-list-scroll {
        max-height: 260px;
        overflow-y: auto;
        padding-right: 2px;
    }
    .doc-list-scroll::-webkit-scrollbar { width: 4px; }
    .doc-list-scroll::-webkit-scrollbar-track { background: transparent; }
    .doc-list-scroll::-webkit-scrollbar-thumb { background: rgba(0, 217, 192, 0.35); border-radius: 99px; }

    /* Sidebar Delete Buttons */
    div[class*="st-key-del_doc_"] button {
        background: rgba(239, 68, 68, 0.1) !important;
        border: 1px solid rgba(239, 68, 68, 0.3) !important;
        border-radius: 10px !important;
        color: #F87171 !important;
        padding: 0.35rem 0.5rem !important;
        transition: all 0.2s ease !important;
    }
    div[class*="st-key-del_doc_"] button:hover {
        background: rgba(239, 68, 68, 0.25) !important;
        border-color: #EF4444 !important;
        transform: scale(1.12) !important;
        box-shadow: 0 0 14px rgba(239, 68, 68, 0.6) !important;
    }

    /* Sidebar Selectbox Customization */
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
        background: rgba(15, 23, 42, 0.85) !important;
        border: 1px solid rgba(0, 217, 192, 0.3) !important;
        border-radius: 12px !important;
        color: #F8FAFC !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3) !important;
        transition: all 0.2s ease !important;
    }
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div:hover {
        border-color: #00D9C0 !important;
        box-shadow: 0 0 16px rgba(0, 217, 192, 0.3) !important;
    }

    /* Sleek Single Floating Chat Input (Zero Double Bar) */
    [data-testid="stBottomBlockContainer"] {
        background: transparent !important;
    }
    div[data-testid="stChatInput"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding-bottom: 0.8rem !important;
    }
    div[data-testid="stChatInput"] > div {
        background: rgba(15, 23, 42, 0.85) !important;
        border: 1.5px solid rgba(0, 217, 192, 0.3) !important;
        border-radius: 999px !important;
        backdrop-filter: blur(16px) !important;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.45), 0 0 14px rgba(0, 217, 192, 0.12) !important;
        transition: all 0.25s ease !important;
        padding: 0.1rem 0.5rem !important;
    }
    div[data-testid="stChatInput"] > div:focus-within {
        border-color: #00D9C0 !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.55), 0 0 22px rgba(0, 217, 192, 0.38) !important;
    }
    div[data-testid="stChatInput"] textarea {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
        color: #F8FAFC !important;
        font-size: 0.92rem !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        padding: 0.55rem 0.8rem !important;
        line-height: 1.4 !important;
    }
    div[data-testid="stChatInput"] textarea::placeholder {
        color: #64748B !important;
    }
    div[data-testid="stChatInput"] button {
        background: linear-gradient(135deg, #00D9C0, #00A896) !important;
        border: none !important;
        border-radius: 50% !important;
        color: #052622 !important;
        box-shadow: 0 0 10px rgba(0, 217, 192, 0.4) !important;
        transition: transform 0.15s ease, filter 0.15s ease !important;
    }
    div[data-testid="stChatInput"] button:hover {
        transform: scale(1.08) !important;
        filter: brightness(1.15) !important;
    }

    .sidebar-caption {
        color: #94A3B8;
        font-size: 0.72rem;
        text-transform: uppercase;
        margin: 1.1rem 0 0.45rem 0.15rem;
        font-weight: 800;
        letter-spacing: 0.8px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def colored_button(label: str, key: str, c1: str, c2: str, text_color: str = "#FFFFFF", active: bool = False) -> bool:
    """A styled button with vibrant gradient, luminous active glow, and rich 3D hover effects."""
    clicked = st.button(label, key=key, use_container_width=True)

    if active:
        btn_style = f"""
        div.st-key-{key} button,
        section[data-testid="stSidebar"] div.st-key-{key} button {{
            background: linear-gradient(135deg, {c1}, {c2}) !important;
            color: {text_color} !important;
            border: 1px solid rgba(255, 255, 255, 0.4) !important;
            border-radius: 14px !important;
            box-shadow: 0 4px 20px {c1}88, inset 0 0 10px rgba(255,255,255,0.2) !important;
            font-weight: 800 !important;
            letter-spacing: 0.3px !important;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }}
        div.st-key-{key} button:hover,
        section[data-testid="stSidebar"] div.st-key-{key} button:hover {{
            transform: translateY(-3px) scale(1.02) !important;
            box-shadow: 0 8px 28px {c1}AA, inset 0 0 14px rgba(255,255,255,0.3) !important;
            filter: brightness(1.15) !important;
            border-color: #FFFFFF !important;
        }}
        """
    else:
        btn_style = f"""
        div.st-key-{key} button,
        section[data-testid="stSidebar"] div.st-key-{key} button {{
            background: linear-gradient(145deg, rgba(255,255,255,0.05), rgba(255,255,255,0.015)) !important;
            color: #CBD5E1 !important;
            border: 1px solid {c1}45 !important;
            border-radius: 14px !important;
            font-weight: 700 !important;
            letter-spacing: 0.3px !important;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3) !important;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }}
        div.st-key-{key} button:hover,
        section[data-testid="stSidebar"] div.st-key-{key} button:hover {{
            transform: translateY(-3px) scale(1.02) !important;
            background: linear-gradient(135deg, {c1}30, {c2}20) !important;
            border-color: {c1} !important;
            color: #FFFFFF !important;
            box-shadow: 0 8px 24px {c1}66, 0 0 16px {c1}44 !important;
            filter: brightness(1.15) !important;
        }}
        """

    st.markdown(f"<style>{btn_style}</style>", unsafe_allow_html=True)
    return clicked


# --------------------------------------------------------------------------------------
# CACHED BACKEND RESOURCES & HELPERS
# --------------------------------------------------------------------------------------
def scroll_to_bottom():
    # Adding a timestamp ensures Streamlit sees this as a 'new' component every time
    # and actually executes the javascript instead of ignoring it.
    js = f"""
    <script id="scroll-{time.time()}">
        setTimeout(() => {{
            const parent = window.parent;
            parent.window.scrollTo({{top: parent.document.body.scrollHeight, behavior: 'smooth'}});
            
            const selectors = ['.stMainBlockContainer', '.main', '[data-testid="stMainBlockContainer"]', '.stApp'];
            selectors.forEach(selector => {{
                const els = parent.document.querySelectorAll(selector);
                els.forEach(el => {{
                    el.scrollTo({{top: el.scrollHeight, behavior: 'smooth'}});
                }});
            }});
        }}, 100);
    </script>
    """
    components.html(js, height=0)


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


@st.cache_resource(show_spinner=False)
def get_vectorstore():
    return Chroma(persist_directory="ChromaDB", embedding_function=get_embedding_model())


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGroq(model="openai/gpt-oss-120b")


def get_indexed_documents():
    """Returns a list of unique document source_names indexed in ChromaDB."""
    try:
        conn = sqlite3.connect("ChromaDB/chroma.sqlite3")
        cur = conn.cursor()
        cur.execute(
            "SELECT DISTINCT string_value FROM embedding_metadata WHERE key = 'source_name' ORDER BY string_value"
        )
        docs = [r[0] for r in cur.fetchall() if r[0]]
        conn.close()
        return docs
    except Exception:
        return []


def delete_document(filename: str) -> int:
    """Instantly deletes all chunks for a source_name using direct SQLite lookup."""
    try:
        # Step 1: Fast SQLite lookup — find all chroma embedding_id strings
        conn = sqlite3.connect("ChromaDB/chroma.sqlite3")
        cur = conn.cursor()
        cur.execute(
            "SELECT e.embedding_id FROM embeddings e "
            "INNER JOIN embedding_metadata m ON e.id = m.id "
            "WHERE m.key = 'source_name' AND m.string_value = ?",
            (filename,)
        )
        chroma_ids = [r[0] for r in cur.fetchall()]
        conn.close()

        if not chroma_ids:
            return 0

        # Step 2: Delete using cached vectorstore (no model reload needed)
        vs = get_vectorstore()
        vs.delete(ids=chroma_ids)
        get_vectorstore.clear()
        return len(chroma_ids)
    except Exception:
        return 0


def get_total_embeddings_count():
    """Returns total chunk count in ChromaDB."""
    try:
        conn = sqlite3.connect("ChromaDB/chroma.sqlite3")
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM embeddings")
        count = cur.fetchone()[0]
        conn.close()
        return count
    except Exception:
        return 0


RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are Duova AI, an intelligent and helpful document assistant.
Your goal is to answer the user's questions accurately, clearly, and helpfully using the provided Document Context.

Instructions:
1. Thoroughly ground your answer in the provided context.
2. Synthesize information across the excerpts to provide a clear, coherent, and comprehensive explanation.
3. If the user asks for a summary, key takeaways, main topics, or what the document is about, summarize the key concepts based on the excerpts.
4. If the user asks a greeting (e.g., 'hi', 'hello') or introductory question, respond warmly and let them know what document is loaded and how you can assist.
5. If the context has partial or related information, provide what is known and clarify what is not explicitly detailed.
6. Only if the provided context contains zero relevant information to answer the question, politely explain: "I could not find specific details about this in the document.", and briefly mention what related topics are covered in the excerpts.
7. Format with clean headings, bullet points, bold keywords, or tables when appropriate.""",
        ),
        (
            "human",
            """Document Context:
{context}

User Question: {question}

Helpful Answer:""",
        ),
    ]
)

AI_SUBMODE_PROMPTS = {
    "Normal": "You are Duova AI, a helpful, friendly, and natural AI assistant. Give clear, well-structured, and easy-to-understand answers.",
    "Expert": """You are Duova AI in Expert Mode.
Give rigorous, technically precise, and detailed answers using professional terminology and strong reasoning.

Structure your answers clearly:
- Use clean headings and subheadings.
- Use bullet points for key concepts.
- Use code blocks with comments for code snippets.
- Use markdown tables when comparing information or presenting structured data.
- Include concrete examples and walk through reasoning steps.""",
    "Quick": "You are Duova AI in Quick Mode. Give ultra-concise, direct, bulleted, and to-the-point answers. No unnecessary fluff or long preambles.",
}

MODE_ICON = {"Normal": "💬", "Expert": "🧠", "Quick": "⚡"}

# --------------------------------------------------------------------------------------
# SESSION STATE
# --------------------------------------------------------------------------------------
defaults = {
    "mode": None,                   # "rag" | "ai"
    "ai_submode": "Normal",         # Normal | Expert | Quick
    "rag_messages": [],             # [{"role": "user"/"assistant", "content": str, "sources": list}]
    "ai_messages": [],              # [{"role": ..., "content": ..., "submode": ...}]
    "show_uploader": False,
    "show_history": False,
    "active_document": "__all__",   # filename to scope RAG, or "__all__" for all docs
    "pending_query": None,          # For starter prompt chips
    "indexed_signatures": {},       # {filename: size} — tracks already-indexed uploads
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Fetch current indexed docs list
indexed_docs = get_indexed_documents()

# --------------------------------------------------------------------------------------
# AUTO PRE-INDEX: silently index any default PDFs in preload_data/ on first launch
# --------------------------------------------------------------------------------------
if "preindex_done" not in st.session_state:
    st.session_state.preindex_done = False

if not st.session_state.preindex_done:
    from preindex import get_already_indexed_names
    import os
    _preload_dir = os.path.join(os.path.dirname(__file__), "preload_data")
    _pdf_files = (
        [f for f in os.listdir(_preload_dir) if f.lower().endswith(".pdf")]
        if os.path.isdir(_preload_dir) else []
    )
    _already = get_already_indexed_names()
    _pending = [f for f in _pdf_files if f not in _already]

    if _pending:
        _preindex_banner = st.empty()
        _preindex_banner.markdown(
            """
            <div style="background: linear-gradient(135deg, rgba(0,217,192,0.12), rgba(61,139,253,0.10));
                        border: 1px solid rgba(0,217,192,0.4); border-radius: 14px;
                        padding: 0.9rem 1.2rem; margin-bottom: 1rem;
                        display:flex; align-items:center; gap:10px;">
                <span style="font-size:1.3rem;">⚡</span>
                <div>
                    <div style="color:#00D9C0; font-weight:800; font-size:0.9rem;">Setting up your Knowledge Base...</div>
                    <div style="color:#94A3B8; font-size:0.8rem; margin-top:2px;">
                        Indexing default document(s) — this only happens once.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        try:
            run_preindex()
            get_vectorstore.clear()
        except Exception as _e:
            pass
        st.session_state.preindex_done = True
        _preindex_banner.empty()
        st.rerun()
    else:
        st.session_state.preindex_done = True

total_chunks_in_db = get_total_embeddings_count()

# --------------------------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; padding: 1.1rem 0.8rem; margin-bottom: 1.1rem;
                    background: linear-gradient(145deg, rgba(255, 255, 255, 0.04) 0%, rgba(255, 255, 255, 0.01) 100%);
                    border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 18px;
                    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);">
            <div style="font-size: 2.3rem; filter: drop-shadow(0 0 16px rgba(0, 217, 192, 0.5)); animation: pulse 3s infinite;">🌀</div>
            <div style="font-family: 'Outfit', 'Sora', sans-serif; font-weight: 900; font-size: 1.55rem; letter-spacing: -0.02em;
                        background: linear-gradient(120deg, #FFFFFF, #00D9C0 50%, #B14AFF 100%);
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-top: 0.2rem;">
                Duova AI
            </div>
            <div style="display: inline-block; margin-top: 0.35rem; padding: 0.18rem 0.65rem; border-radius: 999px;
                        background: rgba(0, 217, 192, 0.08); border: 1px solid rgba(0, 217, 192, 0.25);
                        color: #00D9C0; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.6px;">
                HYBRID INTELLIGENCE
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-caption">⚡ SELECT SYSTEM MODE</div>', unsafe_allow_html=True)
    is_rag = st.session_state.mode == "rag"
    is_ai = st.session_state.mode == "ai"

    if colored_button(
        "📚  RAG Mode" + ("  ●" if is_rag else ""),
        "mode_rag_btn",
        "#00D9C0", "#00A896",
        text_color="#062622",
        active=is_rag
    ):
        st.session_state.mode = "rag"
        st.session_state.show_uploader = False
        st.rerun()

    if colored_button(
        "🤖  AI Mode" + ("  ●" if is_ai else ""),
        "mode_ai_btn",
        "#B14AFF", "#7C2AE8",
        text_color="#FFFFFF",
        active=is_ai
    ):
        st.session_state.mode = "ai"
        st.session_state.show_history = False
        st.rerun()

    st.markdown("<hr style='border-color: rgba(255,255,255,0.07); margin: 1rem 0;'>", unsafe_allow_html=True)

    # ---------------- RAG MODE SIDEBAR ----------------
    if st.session_state.mode == "rag":

        # ── Show existing docs FIRST if any are indexed ──
        indexed_docs = get_indexed_documents()

        if indexed_docs:
            # ---- Knowledge Base Panel (always visible when docs exist) ----
            st.markdown('<div class="sidebar-caption">📚 KNOWLEDGE BASE</div>', unsafe_allow_html=True)

            # Scope selector
            scope_options = ["🌐 All Documents"] + indexed_docs
            active_doc_state = st.session_state.active_document
            if active_doc_state == "__all__" or active_doc_state not in indexed_docs:
                current_scope_idx = 0
            else:
                current_scope_idx = indexed_docs.index(active_doc_state) + 1

            selected_scope = st.selectbox(
                "Query scope",
                scope_options,
                index=current_scope_idx,
                key="doc_scope_selector",
                help="Choose a specific PDF to query, or query all documents at once."
            )
            if selected_scope == "🌐 All Documents":
                st.session_state.active_document = "__all__"
            else:
                st.session_state.active_document = selected_scope

            # Stats line
            st.markdown(
                f"""
                <div style="margin-top:0.4rem; margin-bottom:0.3rem; color:#94A3B8;
                            font-size:0.75rem; font-weight:700; letter-spacing:0.5px;">
                    {len(indexed_docs)} PDF(s) · {total_chunks_in_db:,} total chunks
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Detect which PDFs are "default" (from preload_data/) — these are permanent
            _preload_dir = os.path.join(os.path.dirname(__file__), "preload_data")
            _default_pdfs = (
                set(os.listdir(_preload_dir)) if os.path.isdir(_preload_dir) else set()
            )

            # Global style for all delete (X) buttons in the doc list
            st.markdown(
                """
                <style>
                div[class*="st-key-del_doc_"] button {
                    background: rgba(239,68,68,0.13) !important;
                    border: 1.5px solid rgba(239,68,68,0.55) !important;
                    border-radius: 8px !important;
                    color: #F87171 !important;
                    padding: 0 !important;
                    min-width: 0 !important;
                    width: 2.1rem !important;
                    height: 2.1rem !important;
                    display: flex !important;
                    align-items: center !important;
                    justify-content: center !important;
                    transition: all 0.2s ease !important;
                }
                /* Force any inner Streamlit wrapper to center its contents */
                div[class*="st-key-del_doc_"] button > div,
                div[class*="st-key-del_doc_"] button > span {
                    display: flex !important;
                    align-items: center !important;
                    justify-content: center !important;
                    margin: 0 !important;
                    padding: 0 !important;
                }
                div[class*="st-key-del_doc_"] button:hover {
                    background: rgba(239,68,68,0.3) !important;
                    border-color: #EF4444 !important;
                    color: #FFFFFF !important;
                    box-shadow: 0 0 12px rgba(239,68,68,0.55) !important;
                    transform: scale(1.12) !important;
                }
                /* Ensure equal row gap for every doc row */
                div[class*="st-key-doc_row_"] {
                    margin-bottom: 0.45rem !important;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            # Document list — one row per PDF
            for doc_name in indexed_docs:
                short_name = doc_name if len(doc_name) <= 21 else doc_name[:18] + "..."
                is_active_scope = st.session_state.active_document == doc_name
                border_color = "rgba(0, 217, 192, 0.6)" if is_active_scope else "rgba(255,255,255,0.09)"
                is_default_doc = doc_name in _default_pdfs
                safe_key = doc_name.replace(".", "_").replace(" ", "_")[:20]

                # Each row = doc name card + action icon, side by side
                doc_col, act_col = st.columns([5, 1], gap="small")

                with doc_col:
                    st.markdown(
                        f"""
                        <div style="background: rgba(255,255,255,0.03);
                                    border: 1px solid {border_color};
                                    border-radius: 10px;
                                    padding: 0.52rem 0.7rem;
                                    height: 2.1rem;
                                    display: flex;
                                    align-items: center;
                                    box-sizing: border-box;">
                            <span style="color: #F8FAFC; font-weight: 700; font-size: 0.82rem;
                                         white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                                📄 {short_name}
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with act_col:
                    if is_default_doc:
                        # Permanent doc — show lock, no delete
                        st.markdown(
                            """
                            <div style="height:2.1rem; display:flex;
                                        align-items:center; justify-content:center;"
                                 title="Default document — permanent">
                                <span style="font-size:1.05rem; opacity:0.4; line-height:1;">🔒</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        # Deletable — show styled X button
                        if st.button("", icon=":material/delete:", key=f"del_doc_{safe_key}",
                                     help=f"Remove '{doc_name}' from knowledge base"):
                            delete_document(doc_name)
                            get_vectorstore.clear()
                            sigs = st.session_state.get("indexed_signatures", {})
                            sigs.pop(doc_name, None)
                            st.session_state.indexed_signatures = sigs
                            if st.session_state.active_document == doc_name:
                                remaining = [d for d in indexed_docs if d != doc_name]
                                st.session_state.active_document = remaining[0] if remaining else "__all__"
                            st.rerun()

            # Clear All — only removes user-uploaded PDFs, keeps default ones
            user_docs = [d for d in indexed_docs if d not in _default_pdfs]
            if user_docs:
                st.markdown("<div style='margin-top:0.5rem;'></div>", unsafe_allow_html=True)
                if colored_button(":material/delete: Clear Uploaded Docs", "clear_all_docs", "#EF4444", "#B91C1C"):
                    for _d in user_docs:
                        delete_document(_d)
                    get_vectorstore.clear()
                    st.session_state.indexed_signatures = {}
                    st.session_state.active_document = "__all__"
                    st.rerun()

            st.markdown("<hr style='border-color: rgba(255,255,255,0.06); margin: 0.8rem 0;'>", unsafe_allow_html=True)

        # ── Upload section (always available, but secondary when docs exist) ──
        st.markdown(
            '<div class="sidebar-caption">ADD MORE DOCUMENTS</div>' if indexed_docs
            else '<div class="sidebar-caption">KNOWLEDGE BASE CONTROLS</div>',
            unsafe_allow_html=True,
        )

        if colored_button(
            "📁  " + ("Hide Uploader" if st.session_state.show_uploader else "Upload PDF(s)"),
            "upload_toggle_btn",
            "#0891B2", "#22D3EE",
            text_color="#04222A"
        ):
            st.session_state.show_uploader = not st.session_state.show_uploader
            st.rerun()

        if st.session_state.show_uploader:
            uploaded_files = st.file_uploader(
                "Upload one or more PDFs",
                type=["pdf"],
                accept_multiple_files=True,
                key="pdf_uploader",
                help="Upload multiple PDFs to build a shared knowledge base."
            )

            if uploaded_files:
                sigs = st.session_state.get("indexed_signatures", {})
                new_files = [f for f in uploaded_files if sigs.get(f.name) != f.size]
                already_done = [f for f in uploaded_files if sigs.get(f.name) == f.size]

                if already_done:
                    st.success(f"Already indexed: {', '.join(f.name for f in already_done)}")

                for uploaded_file in new_files:
                    upload_box = st.container()
                    with upload_box:
                        st.markdown(
                            f"""
                            <div style="background: rgba(0, 217, 192, 0.08); border: 1px solid rgba(0, 217, 192, 0.3);
                                        border-radius: 14px; padding: 0.85rem; margin-top: 0.6rem; margin-bottom: 0.6rem;">
                                <div style="display: flex; align-items: center; gap: 8px;">
                                    <span class="pulsing-dot"></span>
                                    <span style="font-weight: 700; color: #00D9C0; font-size: 0.88rem;">
                                        Ingesting: {uploaded_file.name}
                                    </span>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        status_text = st.empty()

                        def update_progress(pct: int, msg: str, _st=status_text):
                            _st.markdown(
                                f"""
                                <div style="margin-top: 0.3rem; margin-bottom: 0.5rem;">
                                    <div style="width: 100%; height: 14px; background: rgba(255, 255, 255, 0.08);
                                                border: 1.5px solid rgba(0, 217, 192, 0.6); border-radius: 999px;
                                                overflow: hidden; box-shadow: inset 0 2px 4px rgba(0,0,0,0.6), 0 0 12px rgba(0, 217, 192, 0.35);">
                                        <div style="width: {pct}%; height: 100%;
                                                    background: linear-gradient(90deg, #00D9C0 0%, #3D8BFD 50%, #B14AFF 100%);
                                                    box-shadow: 0 0 14px #00D9C0, 0 0 22px rgba(0, 217, 192, 0.7);
                                                    border-radius: 999px; transition: width 0.25s ease;"></div>
                                    </div>
                                    <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 0.3rem;">
                                        <span style="color: #CBD5E1; font-size: 0.8rem; font-weight: 500;">{msg}</span>
                                        <span style="color: #00D9C0; font-weight: 800; font-size: 0.9rem; font-family: 'JetBrains Mono', monospace; text-shadow: 0 0 8px rgba(0,217,192,0.6);">{pct}%</span>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        tmp_dir = tempfile.mkdtemp()
                        tmp_path = os.path.join(tmp_dir, uploaded_file.name)
                        try:
                            update_progress(0, "Saving uploaded file to buffer...")
                            with open(tmp_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())

                            emb_model = get_embedding_model()
                            index_pdf(
                                tmp_path,
                                progress_callback=update_progress,
                                embedding_model=emb_model
                            )
                            get_vectorstore.clear()

                            sigs[uploaded_file.name] = uploaded_file.size
                            st.session_state.indexed_signatures = sigs
                            time.sleep(0.3)
                            status_text.markdown(
                                f"""
                                <div style="color: #00D9C0; font-weight: 700; font-size: 0.86rem; margin-top: 0.4rem;">
                                    ✅ 100% — '{uploaded_file.name}' indexed successfully!
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        finally:
                            if os.path.exists(tmp_path):
                                os.remove(tmp_path)
                            if os.path.exists(tmp_dir):
                                os.rmdir(tmp_dir)

                if new_files:
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        if colored_button("🗑️  Reset Chat", "reset_rag_btn", "#FF4D6D", "#D90429"):
            st.session_state.rag_messages = []
            st.session_state.pending_query = None
            st.rerun()

    # ---------------- AI MODE SIDEBAR ----------------
    elif st.session_state.mode == "ai":
        st.markdown('<div class="sidebar-caption">AI PERSONA SELECTOR</div>', unsafe_allow_html=True)
        sub = st.session_state.ai_submode

        if colored_button(
            "💬  Normal Mode" + ("  ●" if sub == "Normal" else ""),
            "sub_normal_btn",
            "#3D8BFD", "#1E5FCC",
            active=(sub == "Normal")
        ):
            st.session_state.ai_submode = "Normal"
            st.rerun()

        if colored_button(
            "🧠  Expert Mode" + ("  ●" if sub == "Expert" else ""),
            "sub_expert_btn",
            "#FF8A3D", "#E85D04",
            active=(sub == "Expert")
        ):
            st.session_state.ai_submode = "Expert"
            st.rerun()

        if colored_button(
            "⚡  Quick Mode" + ("  ●" if sub == "Quick" else ""),
            "sub_quick_btn",
            "#3DDC84", "#1DB954",
            text_color="#062E17",
            active=(sub == "Quick")
        ):
            st.session_state.ai_submode = "Quick"
            st.rerun()

        # Persona Specs Card
        persona_desc = {
            "Normal": "Clear, friendly, well-balanced explanations for general questions.",
            "Expert": "In-depth technical reasoning, code snippets, and structured tables.",
            "Quick": "Ultra-fast direct answers with minimal fluff for rapid workflows.",
        }
        st.markdown(
            f"""
            <div class="sidebar-doc-card">
                <div style="font-weight:700; color:#F8FAFC; font-size:0.85rem;">
                    {MODE_ICON[sub]} {sub} Persona Active
                </div>
                <div class="doc-meta" style="margin-top:0.3rem;">
                    {persona_desc[sub]}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<hr style='border-color: rgba(255,255,255,0.07); margin: 0.8rem 0;'>", unsafe_allow_html=True)
        st.markdown('<div class="sidebar-caption">SESSION TOOLS</div>', unsafe_allow_html=True)

        if colored_button("🕓  Toggle History", "history_btn", "#FFC93D", "#E8A000", text_color="#2E2200"):
            st.session_state.show_history = not st.session_state.show_history
            st.rerun()

        if colored_button("🗑️  Reset Chat", "reset_ai_btn", "#FF4D6D", "#D90429"):
            st.session_state.ai_messages = [
                m for m in st.session_state.ai_messages if m["submode"] != st.session_state.ai_submode
            ]
            st.session_state.pending_query = None
            st.rerun()

# --------------------------------------------------------------------------------------
# MAIN HERO BANNER (REFINED, SPACIOUS & MINIMALIST)
# --------------------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="duova-hero">
        <div class="hero-top-row">
            <div class="hero-badge">
                <span class="pulsing-dot"></span>
                DUOVA HYBRID INTELLIGENCE
            </div>
            <div class="hero-status-pill">
                <span class="status-beacon"></span> Neural Engine Online
            </div>
        </div>
        <div>
            <span class="duova-title">Duova AI</span>
        </div>
        <p class="duova-sub">Precision document analysis grounded in vector embeddings, paired with multi-persona LLM reasoning.</p>
        <div class="hero-strip">
            <div class="hero-strip-item">
                <span class="hero-strip-dot cyan"></span> RAG Vector Search
            </div>
            <span class="hero-strip-divider">•</span>
            <div class="hero-strip-item">
                <span class="hero-strip-dot purple"></span> Multi-Persona LLM
            </div>
            <span class="hero-strip-divider">•</span>
            <div class="hero-strip-item">
                <span style="color: #94A3B8;">Knowledge Base:</span>
                <strong style="color: #00D9C0; font-family: 'JetBrains Mono', monospace;">{len(indexed_docs)} Docs ({total_chunks_in_db:,} chunks)</strong>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------------
# HOME SCREEN: WHEN NO MODE IS SELECTED YET (SINGLE COMPACT SCREEN-FITTING BOX)
# --------------------------------------------------------------------------------------
if st.session_state.mode is None:
    st.markdown(
        f"""
        <style>
        @keyframes bounce-left {{
            0%, 100% {{ transform: translateX(0px); }}
            50%  {{ transform: translateX(-7px); }}
        }}
        @keyframes glow-rag {{
            0%, 100% {{ box-shadow: 0 0 14px rgba(0,217,192,0.22); }}
            50%        {{ box-shadow: 0 0 28px rgba(0,217,192,0.45); }}
        }}
        @keyframes glow-ai {{
            0%, 100% {{ box-shadow: 0 0 14px rgba(177,74,255,0.22); }}
            50%        {{ box-shadow: 0 0 28px rgba(177,74,255,0.45); }}
        }}
        .home-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.1rem;
            margin-top: 0.2rem;
        }}
        .home-card {{
            border-radius: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 1.4rem 1.4rem;
            border: 1px solid rgba(255,255,255,0.1);
            backdrop-filter: blur(14px);
            transition: transform 0.25s, border-color 0.25s;
            position: relative;
            overflow: hidden;
        }}
        .home-card:hover {{ transform: translateY(-4px); }}
        .home-card.rag {{
            background: linear-gradient(145deg, rgba(0,217,192,0.12) 0%, rgba(0,168,150,0.04) 100%);
            border-color: rgba(0,217,192,0.32);
            animation: glow-rag 3.5s ease-in-out infinite;
        }}
        .home-card.ai {{
            background: linear-gradient(145deg, rgba(177,74,255,0.14) 0%, rgba(124,42,232,0.04) 100%);
            border-color: rgba(177,74,255,0.32);
            animation: glow-ai 3.5s ease-in-out infinite;
        }}
        .home-card .big-icon {{
            font-size: 2.9rem;
            margin-bottom: 0.6rem;
            line-height: 1;
            filter: drop-shadow(0 0 14px currentColor);
        }}
        .home-card h2 {{
            font-family: 'Sora', sans-serif;
            font-size: 1.35rem;
            font-weight: 800;
            margin: 0 0 0.35rem 0;
            color: #F8FAFC;
        }}
        .home-card p {{
            font-size: 0.85rem;
            color: #94A3B8;
            margin: 0 0 1rem 0;
            line-height: 1.45;
        }}
        .home-arrow {{
            display: inline-flex;
            align-items: center;
            gap: 7px;
            font-weight: 700;
            font-size: 0.82rem;
            padding: 0.38rem 0.95rem;
            border-radius: 999px;
            border: 1.5px solid;
            letter-spacing: 0.2px;
        }}
        .home-arrow.rag {{
            color: #00D9C0;
            background: rgba(0, 217, 192, 0.1);
            border-color: rgba(0,217,192,0.45);
        }}
        .home-arrow.ai {{
            color: #D8B4FE;
            background: rgba(177, 74, 255, 0.1);
            border-color: rgba(177,74,255,0.45);
        }}
        .arrow-icon {{
            animation: bounce-left 1.2s ease-in-out infinite;
            display: inline-block;
        }}
        </style>

        <div class="home-grid">
            <div class="home-card rag">
                <div class="big-icon">📚</div>
                <h2>RAG Mode</h2>
                <p>Ask questions and get answers<br>grounded in your own PDFs.</p>
                <div class="home-arrow rag">
                    <span class="arrow-icon">👈</span> Select <strong>RAG Mode</strong> from sidebar
                </div>
            </div>
            <div class="home-card ai">
                <div class="big-icon">🤖</div>
                <h2>AI Mode</h2>
                <p>Free-form chat with Normal,<br>Expert &amp; Quick personas.</p>
                <div class="home-arrow ai">
                    <span class="arrow-icon">👈</span> Select <strong>AI Mode</strong> from sidebar
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------------------
# RAG MODE MAIN VIEW
# --------------------------------------------------------------------------------------
elif st.session_state.mode == "rag":
    active_doc = st.session_state.active_document  # "__all__" or a specific filename
    indexed_docs = get_indexed_documents()
    if active_doc == "__all__" or active_doc not in indexed_docs:
        scope_str = f"All {len(indexed_docs)} Document(s)" if indexed_docs else "No Documents Loaded"
    else:
        scope_str = active_doc

    st.markdown(
        f"""
        <div class="mode-ribbon">
            <div class="ribbon-left">
                <span style="font-size: 1.4rem;">📚</span>
                <div>
                    <p class="ribbon-title">RAG Document Intelligence</p>
                    <span style="color: #94A3B8; font-size: 0.8rem;">Grounded semantic search with page-level attribution</span>
                </div>
            </div>
            <div class="ribbon-tag rag">
                Scope: {scope_str}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Empty State: Starter Prompt Suggestions
    if not st.session_state.rag_messages:
        st.markdown(
            """
            <div style="margin-bottom: 0.6rem; color: #94A3B8; font-size: 0.9rem; font-weight: 600;">
                💡 Quick Starters — Click any prompt to explore:
            </div>
            """,
            unsafe_allow_html=True,
        )
        starter_cols = st.columns(2)
        starters = [
            ("📋 Summarize Document", "Give me a thorough summary of the main topics, purpose, and key takeaways from this document."),
            ("🔍 Key Concepts & Terminology", "What are the most important concepts, frameworks, or methodologies introduced in this document?"),
            ("📊 Methodologies & Findings", "Detail the methodologies, key findings, and compare the results in a structured bulleted format."),
            ("❓ FAQs & Applications", "What practical questions and real-world problems does this document solve?"),
        ]
        for idx, (label, prompt_text) in enumerate(starters):
            with starter_cols[idx % 2]:
                if colored_button(label, f"rag_starter_{idx}", "#1E293B", "#0F172A", text_color="#E2E8F0"):
                    st.session_state.pending_query = prompt_text
                    st.rerun()

    # Render Conversation Messages
    for msg in st.session_state.rag_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📑 View Document Citations ({len(msg['sources'])} Excerpts)", expanded=False):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(
                            f"""
                            <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.06);
                                        border-radius: 8px; padding: 0.7rem; margin-bottom: 0.5rem; font-size: 0.84rem;">
                                <div style="color: #00D9C0; font-weight: 700; margin-bottom: 0.2rem;">
                                    Excerpt {i}: {src['source']}{src['page']}
                                </div>
                                <div style="color: #CBD5E1; line-height: 1.45;">
                                    {src['text']}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    # Chat Input & Processing
    user_query = st.chat_input("Ask anything about your document...")
    if st.session_state.pending_query:
        user_query = st.session_state.pending_query
        st.session_state.pending_query = None

    if user_query:
        st.session_state.rag_messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing document context & synthesizing answer..."):
                docs = []
                scope_is_all = (active_doc == "__all__" or active_doc not in indexed_docs)

                # Step 1: Scoped retrieval for a specific document
                if not scope_is_all and active_doc:
                    try:
                        retriever = get_vectorstore().as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": 10, "filter": {"source_name": active_doc}}
                        )
                        docs = retriever.invoke(user_query)
                    except Exception:
                        docs = []

                # Step 2: Global retrieval across all documents (or fallback)
                if not docs:
                    try:
                        retriever = get_vectorstore().as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": 12}
                        )
                        all_docs = retriever.invoke(user_query)
                        if not scope_is_all and active_doc:
                            # Strict scope: filter to only the active doc
                            matched = [
                                d for d in all_docs
                                if d.metadata.get("source_name") == active_doc or
                                   active_doc in d.metadata.get("source", "")
                            ]
                            docs = matched if matched else all_docs
                        else:
                            # All docs: return as-is for cross-document synthesis
                            docs = all_docs
                    except Exception:
                        docs = []

                if not docs:
                    response_text = (
                        "I couldn't find any documents indexed in the knowledge base yet. "
                        "Please use the **📁 Upload New PDF** button in the sidebar to add a document."
                    )
                    sources_data = []
                else:
                    context_blocks = []
                    sources_data = []
                    for i, doc in enumerate(docs, 1):
                        src_name = doc.metadata.get("source_name") or os.path.basename(doc.metadata.get("source", "Document"))
                        page = doc.metadata.get("page")
                        if page is None:
                            page = doc.metadata.get("page_label")
                        page_str = f", Page {int(page) + 1}" if page is not None and str(page).isdigit() else ""
                        context_blocks.append(f"--- [Excerpt {i} | Document: {src_name}{page_str}] ---\n{doc.page_content.strip()}")
                        sources_data.append({
                            "source": src_name,
                            "page": page_str,
                            "text": doc.page_content.strip()[:300] + ("..." if len(doc.page_content) > 300 else "")
                        })

                    context = "\n\n".join(context_blocks)
                    final_prompt = RAG_PROMPT.invoke({"context": context, "question": user_query})
                    response = get_llm().invoke(final_prompt)
                    # Clean up any literal <br> tags the LLM may output
                    response_text = re.sub(r'<br\s*/?>', '\n\n', response.content)

            st.markdown(response_text)
            if sources_data:
                with st.expander(f"📑 View Document Citations ({len(sources_data)} Excerpts)", expanded=False):
                    for i, src in enumerate(sources_data, 1):
                        st.markdown(
                            f"""
                            <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.06);
                                        border-radius: 8px; padding: 0.7rem; margin-bottom: 0.5rem; font-size: 0.84rem;">
                                <div style="color: #00D9C0; font-weight: 700; margin-bottom: 0.2rem;">
                                    Excerpt {i}: {src['source']}{src['page']}
                                </div>
                                <div style="color: #CBD5E1; line-height: 1.45;">
                                    {src['text']}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

        st.session_state.rag_messages.append({
            "role": "assistant",
            "content": response_text,
            "sources": sources_data
        })
        scroll_to_bottom()

# --------------------------------------------------------------------------------------
# AI MODE MAIN VIEW
# --------------------------------------------------------------------------------------
elif st.session_state.mode == "ai":
    submode = st.session_state.ai_submode

    st.markdown(
        f"""
        <div class="mode-ribbon">
            <div class="ribbon-left">
                <span style="font-size: 1.4rem;">{MODE_ICON[submode]}</span>
                <div>
                    <p class="ribbon-title">AI Reasoning Hub — {submode} Mode</p>
                    <span style="color: #94A3B8; font-size: 0.8rem;">Groq LPU High-Speed Multi-Persona Generation</span>
                </div>
            </div>
            <div class="ribbon-tag ai">
                Persona: {submode}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.show_history:
        with st.expander("🕓 Multi-Persona Historical Transcripts", expanded=True):
            if not st.session_state.ai_messages:
                st.caption("No historical messages recorded in this session yet.")
            for msg in st.session_state.ai_messages:
                icon = MODE_ICON.get(msg["submode"], "💬")
                who = "You" if msg["role"] == "user" else f"{icon} {msg['submode']} AI"
                st.markdown(f"**{who}:** {msg['content']}")
        st.markdown("<hr style='border-color: rgba(255,255,255,0.07); margin: 1rem 0;'>", unsafe_allow_html=True)

    current_thread = [m for m in st.session_state.ai_messages if m["submode"] == submode]

    # Empty State: AI Prompt Suggestions
    if not current_thread:
        st.markdown(
            """
            <div style="margin-bottom: 0.6rem; color: #94A3B8; font-size: 0.9rem; font-weight: 600;">
                💡 Suggested Topics for this Persona:
            </div>
            """,
            unsafe_allow_html=True,
        )
        ai_starters = {
            "Normal": [
                ("💬 Explain Quantum Computing Simply", "Explain the concept of quantum computing and qubits as if I am an interested beginner."),
                ("🚀 Brainstorm Startup Ideas", "Give me 4 creative AI-powered SaaS product ideas for data analytics workflows in 2026."),
            ],
            "Expert": [
                ("🧠 Transformers vs Mamba & SSMs", "Provide an in-depth mathematical and architectural comparison between Transformer Attention and State Space Models (Mamba)."),
                ("💻 Production Python Pipeline", "Write a complete, type-hinted Python script demonstrating a resilient asynchronous batch processing pipeline."),
            ],
            "Quick": [
                ("⚡ RAG vs Fine-Tuning in 5 Bullets", "Compare RAG vs Fine-Tuning in exactly 5 high-impact bullet points covering cost, latency, and recency."),
                ("⚡ Explain Docker Containers", "Give me the fastest 2-sentence explanation of how Docker containers work."),
            ],
        }
        starter_cols = st.columns(2)
        for idx, (label, prompt_text) in enumerate(ai_starters.get(submode, [])):
            with starter_cols[idx]:
                if colored_button(label, f"ai_starter_{submode}_{idx}", "#1E293B", "#0F172A", text_color="#E2E8F0"):
                    st.session_state.pending_query = prompt_text
                    st.rerun()

    for msg in current_thread:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_prompt = st.chat_input(f"Message Duova AI in {submode} mode...")
    if st.session_state.pending_query:
        user_prompt = st.session_state.pending_query
        st.session_state.pending_query = None

    if user_prompt:
        st.session_state.ai_messages.append({"role": "user", "content": user_prompt, "submode": submode})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Generating reasoning response..."):
                history = [SystemMessage(content=AI_SUBMODE_PROMPTS[submode])]
                for m in current_thread + [{"role": "user", "content": user_prompt}]:
                    if m["role"] == "user":
                        history.append(HumanMessage(content=m["content"]))
                    else:
                        history.append(AIMessage(content=m["content"]))
                response = get_llm().invoke(history)
            clean_content = re.sub(r'<br\s*/?>', '\n\n', response.content)
            st.markdown(clean_content)

        st.session_state.ai_messages.append(
            {"role": "assistant", "content": clean_content, "submode": submode}
        )
        scroll_to_bottom()