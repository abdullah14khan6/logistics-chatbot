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
        .stApp {
            background: #f6f7f9;
        }
        .main .block-container {
            max-width: 900px;
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        [data-testid="stSidebar"] {
            background: #111827;
        }
        [data-testid="stSidebar"] * {
            color: #f9fafb;
        }
        .logo-mark {
            width: 52px;
            height: 52px;
            border-radius: 8px;
            display: grid;
            place-items: center;
            background: #ffffff;
            color: #0f766e;
            font-weight: 800;
            font-size: 1.05rem;
            border: 1px solid #d1d5db;
            margin-bottom: 0.75rem;
        }
        .session-pill {
            font-size: 0.78rem;
            padding: 0.35rem 0.55rem;
            border: 1px solid #374151;
            border-radius: 6px;
            overflow-wrap: anywhere;
        }
        .chat-title {
            color: #111827;
            font-size: 1.65rem;
            font-weight: 750;
            margin-bottom: 0.15rem;
        }
        .chat-subtitle {
            color: #4b5563;
            margin-bottom: 1.25rem;
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
            f'<div class="session-pill">{st.session_state.session_id}</div>',
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

    st.markdown('<div class="chat-title">Logistics Chatbot</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="chat-subtitle">Ask about services, shipping options, pricing, or tracking.</div>',
        unsafe_allow_html=True,
    )
    render_history()

    prompt = st.chat_input("Message the logistics assistant")
    if prompt:
        handle_prompt(prompt)


if __name__ == "__main__":
    main()
