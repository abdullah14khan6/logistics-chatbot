import json
import logging
from dataclasses import dataclass
from uuid import uuid4

from backend.chatbot.intents import IntentAnalysis, IntentAnalyzer
from backend.chatbot.memory import ConversationMemory, MemoryStore
from backend.config.settings import Settings
from backend.rag.generator import GroqAnswerGenerator
from backend.rag.retriever import PineconeRetriever, RetrievedChunk, format_context

logger = logging.getLogger(__name__)


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
        intent_analyzer: IntentAnalyzer | None = None,
    ) -> None:
        self.settings = settings
        self.retriever = retriever or PineconeRetriever(settings)
        self.generator = generator or GroqAnswerGenerator(settings)
        self.memory_store = memory_store or MemoryStore()
        self.intent_analyzer = intent_analyzer or IntentAnalyzer(settings)

    def chat(self, message: str, session_id: str | None = None) -> ChatResponse:
        active_session_id = session_id or str(uuid4())
        memory = self.memory_store.get(active_session_id)
        history = memory.as_text()
        analysis = self.intent_analyzer.analyze(message, history)
        logger.info("Intent analysis: %s", analysis)

        answer = self._sanitize_response(self._handle_message(message, analysis, memory))
        self._remember(memory, message, answer, analysis)
        return ChatResponse(
            response=answer,
            session_id=active_session_id,
            intent=analysis.primary_label(),
        )

    def clear_memory(self, session_id: str) -> None:
        self.memory_store.clear(session_id)

    def _handle_message(
        self,
        message: str,
        analysis: IntentAnalysis,
        memory: ConversationMemory,
    ) -> str:
        if analysis.prompt_injection:
            return (
                "I can't help with hidden instructions, internal prompts, environment "
                "variables, API keys, or invented company information. I can help with "
                "Paramount Logistics services, shipments, quotations, and logistics guidance."
            )

        if analysis.unrelated:
            return (
                "I'm here to help with Paramount Logistics and logistics-related questions. "
                "For shipments, customs, warehousing, freight, or service guidance, I'll be happy to help."
            )

        if analysis.gratitude:
            return _gratitude_response(memory.user_message_count)

        if analysis.acknowledgement and not self._needs_action(analysis):
            return (
                "Sure. Tell me what you're shipping, where it's going, or which logistics "
                "service you'd like to understand."
            )

        if analysis.unclear and not self._needs_action(analysis):
            return _fallback_response(memory.user_message_count)

        chunks = self._retrieve_company_chunks(message, analysis, memory)
        response_parts: list[str] = []

        if chunks or analysis.general_logistics:
            response_parts.append(self._generate_answer(message, analysis, memory, chunks))
        elif analysis.needs_rag or analysis.company_specific:
            response_parts.append(self._unavailable_company_info(analysis))

        if analysis.needs_tracking:
            response_parts.append(self._tracking_block())

        if analysis.needs_pricing or analysis.needs_handoff:
            response_parts.append(self._handoff_block(memory, analysis))
        elif self._should_soft_handoff(memory, analysis):
            memory.mark_handoff_suggested()
            response_parts.append(
                "If you'd like to discuss your shipment requirements in more detail, "
                "I can also connect you with our Head of Services."
            )

        if not response_parts:
            response_parts.append(self._ask_follow_up(analysis))

        return "\n\n".join(part for part in response_parts if part.strip())

    def _retrieve_company_chunks(
        self,
        message: str,
        analysis: IntentAnalysis,
        memory: ConversationMemory,
    ) -> list[RetrievedChunk]:
        if not (analysis.needs_rag or analysis.company_specific):
            return []
        query = analysis.query_for_rag or message
        if analysis.follow_up and memory.last_topic and memory.last_topic not in query.lower():
            query = f"{memory.last_topic}: {query}"
        chunks = self.retriever.retrieve(query)
        confident = [
            chunk for chunk in chunks if chunk.score >= self.settings.retrieval_min_score
        ]
        if chunks and not confident:
            logger.info(
                "Dropped %s low-confidence retrieval results below %.2f",
                len(chunks),
                self.settings.retrieval_min_score,
            )
        return confident

    def _generate_answer(
        self,
        message: str,
        analysis: IntentAnalysis,
        memory: ConversationMemory,
        chunks: list[RetrievedChunk],
    ) -> str:
        context = format_context(chunks) if chunks else "No company information was found."
        instructions = [
            "Write one coherent customer-support response.",
            "Do not mention sources, PDFs, retrieval, context, chunks, pages, or tools.",
            "Do not invent company information.",
            "Use bullets only when they improve readability.",
            "Do not repeat topics already explained unless the user asks for repetition.",
        ]
        repeated = sorted(set(analysis.intents).intersection(memory.explained_topics))
        if repeated:
            instructions.append(
                "The customer has already seen an explanation for: "
                f"{', '.join(repeated)}. Summarize briefly or build on it."
            )
        if analysis.user_situation:
            instructions.append(
                "Personalize the response to this customer situation: "
                f"{analysis.user_situation}"
            )
        if analysis.needs_pricing or analysis.needs_handoff:
            instructions.append(
                "Answer the service question first. Mention that pricing or detailed planning "
                "depends on shipment requirements; the controller will add contact details."
            )
        if analysis.general_logistics and not chunks:
            instructions.append(
                "This is general logistics guidance. Do not claim it is a Paramount-specific service."
            )

        return self.generator.answer(
            question=message,
            context=context,
            history=memory.as_text(),
            intent_analysis=json.dumps(analysis.__dict__, ensure_ascii=True),
            controller_instructions="\n".join(f"- {item}" for item in instructions),
        )

    def _tracking_block(self) -> str:
        return f"**Shipment tracking**\n\nYou can track your shipment here:\n{self.settings.tracking_url}"

    def _handoff_block(self, memory: ConversationMemory, analysis: IntentAnalysis) -> str:
        if memory.contact_card_shown and not analysis.show_contact_details:
            return (
                "As mentioned earlier, our Head of Services can help with pricing, "
                "custom quotations, or a detailed logistics consultation."
            )

        memory.contact_card_shown = True
        memory.mark_handoff_suggested()
        return (
            "Need a customised quotation or logistics consultation?\n\n"
            "Please contact our Head of Services:\n\n"
            "Name:\n"
            f"{self.settings.head_of_services_name}\n\n"
            "Email:\n"
            f"{self.settings.head_of_services_email}\n\n"
            "Phone:\n"
            f"{self.settings.head_of_services_phone}"
        )

    def _unavailable_company_info(self, analysis: IntentAnalysis) -> str:
        topic = ", ".join(analysis.intents) if analysis.intents else "that"
        return (
            f"I couldn't find reliable information about {topic} within our available "
            "company information.\n\n"
            "If you'd like, I can connect you with our Head of Services who can assist you further."
        )

    def _ask_follow_up(self, analysis: IntentAnalysis) -> str:
        if analysis.needs_pricing or "shipping" in analysis.intents:
            return (
                "I can help narrow that down. Is the shipment domestic or international, "
                "what is the approximate weight or volume, and where is it going?"
            )
        return _fallback_response(0)

    def _should_soft_handoff(
        self,
        memory: ConversationMemory,
        analysis: IntentAnalysis,
    ) -> bool:
        if memory.user_message_count < 4 or memory.recently_suggested_handoff():
            return False
        return analysis.company_specific or analysis.general_logistics or analysis.needs_rag

    def _needs_action(self, analysis: IntentAnalysis) -> bool:
        return (
            analysis.needs_tracking
            or analysis.needs_pricing
            or analysis.needs_handoff
            or analysis.needs_rag
            or analysis.company_specific
            or analysis.general_logistics
        )

    def _remember(
        self,
        memory: ConversationMemory,
        user_message: str,
        assistant_message: str,
        analysis: IntentAnalysis,
    ) -> None:
        memory.add_user_message(user_message)
        memory.add_ai_message(assistant_message)
        memory.remember_topics(
            [
                intent
                for intent in analysis.intents
                if intent
                not in {
                    "tracking",
                    "pricing",
                    "human_handoff",
                    "gratitude",
                    "acknowledgement",
                    "unclear",
                    "unrelated",
                    "prompt_injection",
                }
            ]
        )

    def _sanitize_response(self, response: str) -> str:
        replacements = {
            "According to the retrieved company context, ": "",
            "According to the retrieved context, ": "",
            "According to the company data, ": "",
            "According to page": "In our available information",
            "Based on the PDF, ": "",
            "Retrieved company context": "Our available company information",
            "retrieved company context": "available company information",
            "retrieved context": "available information",
            "Pinecone": "internal systems",
            "RAG": "internal systems",
        }
        cleaned = response
        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)
        return cleaned.strip()


def _gratitude_response(turn_count: int) -> str:
    options = [
        "You're very welcome. If you need help with a shipment or logistics service, I'm here to help.",
        "Happy to help. Let me know if you'd like to understand any service in more detail.",
        "You're welcome. If you're planning a shipment, I can also help you identify the right service.",
    ]
    return options[turn_count % len(options)]


def _fallback_response(turn_count: int) -> str:
    options = [
        "I'm sorry, I didn't quite understand your question. Could you rephrase it or tell me how I can help with your logistics requirements?",
        "I want to make sure I understand correctly. Could you provide a little more detail about what you need?",
        "Could you clarify what you're trying to ship or which logistics service you're asking about?",
    ]
    return options[turn_count % len(options)]
