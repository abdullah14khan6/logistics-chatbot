from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from uuid import uuid4

import streamlit as st

from backend.chatbot.service import ChatbotService
from backend.config.settings import get_settings
from backend.utils.logging import configure_logging


configure_logging()
CHAT_TIMEOUT_SECONDS = 45


@st.cache_resource
def chatbot_service() -> ChatbotService:
    return ChatbotService(get_settings())


@st.cache_resource
def chat_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=2)


def reset_chat() -> None:
    session_id = str(uuid4())
    st.session_state.session_id = session_id
    st.session_state.messages = []


def bootstrap_state() -> None:
    if "session_id" not in st.session_state:
        reset_chat()


def render_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --app-bg: #edf1f5;
            --panel: #ffffff;
            --ink: #111827;
            --muted: #5b6472;
            --line: #d9e0e8;
            --brand: #0f766e;
            --brand-strong: #0b5f59;
            --sidebar: #101826;
            --sidebar-soft: #172235;
        }
        header[data-testid="stHeader"] {
            background: var(--sidebar);
        }
        .stApp {
            background:
                linear-gradient(180deg, rgba(15, 118, 110, 0.08), transparent 280px),
                var(--app-bg);
        }
        .main .block-container {
            max-width: 920px;
            padding-top: 2rem;
            padding-bottom: 6.5rem;
        }
        [data-testid="stSidebar"] {
            background: var(--sidebar);
        }
        [data-testid="stSidebar"] * {
            color: #f9fafb;
        }
        [data-testid="stSidebar"] > div:first-child {
            padding-top: 2rem;
        }
        .logo-mark {
            width: 52px;
            height: 52px;
            border-radius: 8px;
            display: grid;
            place-items: center;
            background: #ffffff;
            color: var(--brand);
            font-weight: 800;
            font-size: 1.05rem;
            border: 1px solid #d1d5db;
            margin-bottom: 1rem;
        }
        .sidebar-note {
            margin-top: 1rem;
            padding: 0.85rem 0.9rem;
            background: var(--sidebar-soft);
            border: 1px solid #26364d;
            border-radius: 6px;
            color: #d7dee8;
            font-size: 0.88rem;
            line-height: 1.45;
        }
        .chat-title {
            color: var(--ink);
            font-size: 1.9rem;
            font-weight: 750;
            margin-bottom: 0.25rem;
        }
        .chat-subtitle {
            color: var(--muted);
            margin-bottom: 1.35rem;
        }
        .welcome-panel {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1.15rem 1.25rem;
            margin-bottom: 1.1rem;
            box-shadow: 0 14px 32px rgba(17, 24, 39, 0.06);
        }
        .welcome-panel strong {
            color: var(--ink);
        }
        .welcome-panel span {
            display: block;
            color: var(--muted);
            margin-top: 0.25rem;
        }
        [data-testid="stChatMessage"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.85rem 1rem;
            margin-bottom: 0.8rem;
            box-shadow: 0 8px 24px rgba(17, 24, 39, 0.05);
        }
        [data-testid="stChatMessage"] p,
        [data-testid="stChatMessage"] li {
            color: var(--ink);
            line-height: 1.55;
        }
        [data-testid="stChatMessage"] ul {
            padding-left: 1.2rem;
        }
        .stButton > button {
            border-radius: 6px;
            border: 1px solid var(--line);
            background: #ffffff;
            color: var(--ink);
            min-height: 2.6rem;
        }
        .stButton > button:hover {
            border-color: var(--brand);
            color: var(--brand-strong);
        }
        [data-testid="stSidebar"] .stButton > button {
            background: #253044;
            border-color: #39475f;
            color: #ffffff;
        }
        [data-testid="stChatInput"] {
            background: rgba(237, 241, 245, 0.92);
            border-top: 1px solid var(--line);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown('<div class="logo-mark">PL</div>', unsafe_allow_html=True)
        st.markdown("### Paramount Logistics")
        st.markdown(
            '<div class="sidebar-note">Support chat is ready. Ask about services, quotes, tracking, or logistics guidance.</div>',
            unsafe_allow_html=True,
        )
        st.divider()
        if st.button("Clear chat", use_container_width=True):
            reset_chat()
            st.rerun()


def render_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def handle_prompt(prompt: str) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Typing..."):
            future = chat_executor().submit(
                chatbot_service().chat,
                prompt,
                st.session_state.session_id,
            )
            try:
                result = future.result(timeout=CHAT_TIMEOUT_SECONDS)
                response = result.response
            except TimeoutError:
                response = (
                    "The request is taking longer than expected. Please try again with a "
                    "more specific question, or restart the app if the connection is stuck."
                )
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})


def main() -> None:
    st.set_page_config(
        page_title="Logistics Chatbot",
        page_icon="PL",
        layout="centered",
    )
    bootstrap_state()
    render_styles()
    render_sidebar()

    st.markdown('<div class="chat-title">Paramount Logistics Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="chat-subtitle">Fast answers for company services, shipment tracking, and logistics support.</div>',
        unsafe_allow_html=True,
    )
    starter_prompt = None
    if not st.session_state.messages:
        st.markdown(
            '<div class="welcome-panel"><strong>How can I help today?</strong><span>Choose a starter or type your question below.</span></div>',
            unsafe_allow_html=True,
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Company services", use_container_width=True):
                starter_prompt = "What services does the company offer?"
        with col2:
            if st.button("Track shipment", use_container_width=True):
                starter_prompt = "I want to track my shipment"
        with col3:
            if st.button("Get a quote", use_container_width=True):
                starter_prompt = "I need a shipping quote"
    render_history()

    prompt = st.chat_input("Message the logistics assistant")
    if starter_prompt or prompt:
        handle_prompt(starter_prompt or prompt)


if __name__ == "__main__":
    main()
