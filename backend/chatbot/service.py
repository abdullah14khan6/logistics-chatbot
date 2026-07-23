import logging
from dataclasses import dataclass
from typing import TypedDict
from uuid import uuid4

from backend.chatbot.intents import Intent, IntentRouter
from backend.chatbot.memory import MemoryStore
from backend.config.settings import Settings
from backend.rag.generator import GroqAnswerGenerator
from backend.rag.retriever import PineconeRetriever, format_context

logger = logging.getLogger(__name__)


class ChatState(TypedDict, total=False):
    message: str
    session_id: str
    intent: Intent
    answer: str
    context: str
    history: str


@dataclass(frozen=True)
class ChatResponse:
    response: str
    session_id: str
    intent: str


class ChatbotService:
    def __init__(
        self,
        settings: Settings,
        retriever: PineconeRetriever | None = None,
        generator: GroqAnswerGenerator | None = None,
        memory_store: MemoryStore | None = None,
    ) -> None:
        self.settings = settings
        self.intent_router = IntentRouter(settings)
        self.retriever = retriever or PineconeRetriever(settings)
        self.generator = generator or GroqAnswerGenerator(settings)
        self.memory_store = memory_store or MemoryStore()
        self.graph = self._build_graph()

    def chat(self, message: str, session_id: str | None = None) -> ChatResponse:
        active_session_id = session_id or str(uuid4())
        initial_state: ChatState = {
            "message": message,
            "session_id": active_session_id,
        }
        final_state = self.graph.invoke(initial_state)
        return ChatResponse(
            response=final_state["answer"],
            session_id=active_session_id,
            intent=str(final_state["intent"]),
        )

    def _build_graph(self):
        from langgraph.graph import END, StateGraph

        graph = StateGraph(ChatState)
        graph.add_node("detect_intent", self._detect_intent)
        graph.add_node("special_intent", self._special_intent)
        graph.add_node("rag_answer", self._rag_answer)
        graph.set_entry_point("detect_intent")
        graph.add_conditional_edges(
            "detect_intent",
            self._route_after_intent,
            {
                "special_intent": "special_intent",
                "rag_answer": "rag_answer",
            },
        )
        graph.add_edge("special_intent", END)
        graph.add_edge("rag_answer", END)
        return graph.compile()

    def _detect_intent(self, state: ChatState) -> ChatState:
        result = self.intent_router.detect(state["message"])
        updates: ChatState = {"intent": result.intent}
        if result.response:
            updates["answer"] = result.response
        return updates

    def _route_after_intent(self, state: ChatState) -> str:
        if state["intent"] == Intent.RAG:
            return "rag_answer"
        return "special_intent"

    def _special_intent(self, state: ChatState) -> ChatState:
        self._remember(state["session_id"], state["message"], state["answer"])
        return state

    def _rag_answer(self, state: ChatState) -> ChatState:
        memory = self.memory_store.get(state["session_id"])
        history = memory.as_text()
        is_general = self.intent_router.is_general_logistics_question(state["message"])

        chunks = self.retriever.retrieve(state["message"])
        if not chunks and not is_general:
            answer = (
                "I am sorry, but that information is not available in the company "
                "knowledge base right now."
            )
        else:
            context = format_context(chunks)
            if is_general and not chunks:
                context = (
                    "No relevant company context was retrieved. Answer using general "
                    "logistics knowledge and label it as General logistics knowledge."
                )
            answer = self.generator.answer(
                question=state["message"],
                context=context,
                history=history,
            )

        self._remember(state["session_id"], state["message"], answer)
        return {
            "answer": answer,
            "context": format_context(chunks),
            "history": history,
            "intent": Intent.RAG,
        }

    def _remember(self, session_id: str, user_message: str, assistant_message: str) -> None:
        memory = self.memory_store.get(session_id)
        memory.add_user_message(user_message)
        memory.add_ai_message(assistant_message)

