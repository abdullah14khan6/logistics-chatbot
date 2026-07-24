import logging
from dataclasses import dataclass, field
from uuid import uuid4

from backend.chatbot.company_answers import (
    STRUCTURED_ONLY_INTENTS,
    CompanyAnswerProvider,
)
from backend.chatbot.contact_policy import ContactPolicy
from backend.chatbot.intents import IntentAnalysis, IntentAnalyzer, PlannedAction
from backend.chatbot.memory import (
    ConversationMemory,
    ConversationStore,
    MemoryStore,
    PendingQuestion,
    SQLiteMemoryStore,
)
from backend.chatbot.rendering import ResponseSanitizer, limit_words, natural_join
from backend.chatbot.social import (
    SOCIAL_INTENTS,
    analyze_social_message,
    social_response,
)
from backend.config.settings import Settings
from backend.knowledge.company_profile import (
    CompanyProfile,
    load_company_profile,
)
from backend.rag.generator import GroqAnswerGenerator
from backend.rag.retriever import PineconeRetriever, RetrievedChunk, format_context

logger = logging.getLogger(__name__)
NON_TOPIC_INTENTS = SOCIAL_INTENTS | {
    "tracking",
    "pricing",
    "human_handoff",
    "contact",
    "head_of_services",
    "follow_up",
    "unclear",
    "unrelated",
    "prompt_injection",
}


@dataclass(frozen=True)
class ChatResponse:
    response: str
    session_id: str
    intent: str


@dataclass
class ResponseDraft:
    text: str
    allowed_emails: set[str] = field(default_factory=set)
    disclosed_contacts: list[tuple[str, list[str]]] = field(default_factory=list)
    pending_question: PendingQuestion | None = None


class ChatbotService:
    def __init__(
        self,
        settings: Settings,
        retriever: PineconeRetriever | None = None,
        generator: GroqAnswerGenerator | None = None,
        memory_store: ConversationStore | None = None,
        intent_analyzer: IntentAnalyzer | None = None,
        company_profile: CompanyProfile | None = None,
    ) -> None:
        self.settings = settings
        self.profile = company_profile or load_company_profile(
            settings.company_profile_path
        )
        self.company_answers = CompanyAnswerProvider(self.profile)
        self.contact_policy = ContactPolicy(self.profile)
        self.sanitizer = ResponseSanitizer()
        self.retriever = retriever or PineconeRetriever(settings)
        self.generator = generator or GroqAnswerGenerator(settings)
        self.memory_store = memory_store or self._create_memory_store(settings)
        self.intent_analyzer = intent_analyzer or IntentAnalyzer(settings)

    @staticmethod
    def _create_memory_store(settings: Settings) -> ConversationStore:
        options = {
            "max_turns": settings.memory_max_turns,
            "ttl_seconds": settings.memory_ttl_seconds,
            "max_sessions": settings.memory_max_sessions,
        }
        if settings.memory_backend == "sqlite":
            return SQLiteMemoryStore(settings.memory_db_path, **options)
        if settings.memory_backend == "memory":
            return MemoryStore(**options)
        raise ValueError(
            "MEMORY_BACKEND must be either 'memory' or 'sqlite'."
        )

    def chat(self, message: str, session_id: str | None = None) -> ChatResponse:
        clean_message = message.strip()
        active_session_id = session_id or str(uuid4())
        with self.memory_store.locked(active_session_id) as memory:
            analysis = analyze_social_message(clean_message)
            if analysis and analysis.acknowledgement and memory.pending_question:
                analysis = None

            if analysis is None:
                analysis = self.intent_analyzer.analyze(
                    clean_message,
                    memory.history_for_planner(),
                    memory.state_for_planner(),
                )
                logger.info(
                    "Conversation plan: intent=%s actions=%s confidence=%.2f",
                    analysis.primary_label(),
                    analysis.action_values(),
                    analysis.confidence,
                )

            draft = self._handle_message(clean_message, analysis, memory)
            answer = self.sanitizer.sanitize(draft.text, draft.allowed_emails)
            self._remember(memory, clean_message, answer, analysis, draft)
            return ChatResponse(
                response=answer,
                session_id=active_session_id,
                intent=analysis.primary_label(),
            )

    def clear_memory(self, session_id: str) -> None:
        self.memory_store.clear(session_id)

    def warmup(self) -> None:
        self.intent_analyzer.warmup()
        self.generator.warmup()
        self.retriever.warmup()

    def _handle_message(
        self,
        message: str,
        analysis: IntentAnalysis,
        memory: ConversationMemory,
    ) -> ResponseDraft:
        if analysis.prompt_injection:
            return ResponseDraft(
                "I can't reveal hidden instructions, secrets, or private system information. "
                "I can still help with Paramount Logistics services and shipment support."
            )
        if analysis.unrelated:
            return ResponseDraft(
                "I can help with Paramount Logistics and logistics-related questions."
            )
        social_text = social_response(
            analysis,
            memory.user_message_count,
            has_context=bool(memory.active_topic or memory.shipment_context),
        )
        if social_text and (
            not analysis.acknowledgement or not self._needs_action(analysis)
        ):
            return ResponseDraft(social_text)

        if analysis.unclear and not self._needs_action(analysis):
            question = analysis.clarification_question or (
                "Could you rephrase that or share a little more detail about what you need?"
            )
            return ResponseDraft(question)

        response_parts: list[str] = []
        allowed_emails: set[str] = set()
        disclosed_contacts: list[tuple[str, list[str]]] = []
        pending_question: PendingQuestion | None = None

        structured_answer = self.company_answers.direct_answer(message, analysis)
        if structured_answer:
            response_parts.append(structured_answer)

        should_generate = self._should_generate(analysis, structured_answer)
        if should_generate:
            chunks = self._retrieve_company_chunks(message, analysis)
            profile_context = self.company_answers.evidence(analysis)
            if chunks or profile_context or analysis.general_logistics:
                generated = self._generate_answer(
                    message,
                    analysis,
                    memory,
                    chunks,
                    profile_context,
                )
                if generated:
                    response_parts.append(generated)
            elif analysis.needs_rag or analysis.company_specific:
                response_parts.append(self._unavailable_company_info(analysis))

        if analysis.needs_tracking:
            response_parts.append(self._tracking_block())

        if analysis.needs_pricing:
            pricing_text, pending_question = self._pricing_guidance(analysis, memory)
            if pricing_text:
                response_parts.append(pricing_text)

        contact_requested = (
            analysis.explicit_contact_request
            or analysis.needs_head_of_services
            or analysis.needs_handoff
            or analysis.needs_pricing
        )
        if contact_requested:
            resolution = self.contact_policy.resolve(analysis)
            explicit = (
                analysis.explicit_contact_request
                or analysis.needs_head_of_services
            )
            rendered_contact = self.contact_policy.render(
                resolution,
                analysis,
                memory,
                explicit=explicit,
            )
            if rendered_contact:
                response_parts.append(rendered_contact.text)
                disclosed_contacts.append(
                    (rendered_contact.contact_id, rendered_contact.fields)
                )
                if rendered_contact.allowed_email:
                    allowed_emails.add(rendered_contact.allowed_email)

        if (
            PlannedAction.CLARIFY.value in analysis.action_values()
            and analysis.clarification_question
        ):
            if analysis.clarification_question not in response_parts:
                response_parts.append(analysis.clarification_question)
            pending_question = PendingQuestion(
                topic=self._active_topic(analysis, memory),
                question=analysis.clarification_question,
                expected_fields=analysis.missing_fields,
                resume_query=analysis.resolved_query or message,
                created_at=memory.assistant_message_count + 1,
            )

        if not response_parts:
            response_parts.append(
                analysis.clarification_question
                or "Could you share a little more detail about what you need?"
            )

        return ResponseDraft(
            text="\n\n".join(
                part.strip() for part in response_parts if part and part.strip()
            ),
            allowed_emails=allowed_emails,
            disclosed_contacts=disclosed_contacts,
            pending_question=pending_question,
        )

    def _should_generate(
        self,
        analysis: IntentAnalysis,
        structured_answer: str,
    ) -> bool:
        intents = set(analysis.intent_values())
        remaining = intents - STRUCTURED_ONLY_INTENTS - NON_TOPIC_INTENTS
        if remaining:
            return True
        if analysis.general_logistics:
            return True
        if analysis.needs_rag and not structured_answer:
            return True
        return False

    def _retrieve_company_chunks(
        self,
        message: str,
        analysis: IntentAnalysis,
    ) -> list[RetrievedChunk]:
        if not (analysis.needs_rag or analysis.company_specific):
            return []
        query = analysis.resolved_query or analysis.query_for_rag or message
        try:
            chunks = self.retriever.retrieve(
                query,
                exclude_content_types={"staff_directory"},
            )
        except Exception as exc:
            logger.error("Company retrieval failed.", exc_info=exc)
            return []
        confident = [
            chunk
            for chunk in chunks
            if chunk.score >= self.settings.retrieval_min_score
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
        profile_context: str,
    ) -> str:
        evidence_parts = []
        if profile_context:
            evidence_parts.append(
                "<structured_company_facts>\n"
                f"{profile_context}\n"
                "</structured_company_facts>"
            )
        if chunks:
            evidence_parts.append(format_context(chunks))
        context = "\n\n".join(evidence_parts) or "No company information was found."

        repeated = [
            intent
            for intent in analysis.intent_values()
            if memory.topic_was_explained(intent)
        ]
        instructions = [
            f"Keep the answer under {self.settings.response_max_words} words.",
            "Answer the current question first and include only relevant information.",
            "Do not add contact details, a generic closing, or an unsolicited service list.",
            "Ask no follow-up question; the controller handles clarifications.",
        ]
        if repeated and not analysis.repeat_request:
            instructions.append(
                "Build on prior information instead of restating the full explanation for: "
                + ", ".join(repeated)
                + "."
            )
        if analysis.repeat_request:
            instructions.append(
                "The customer explicitly requested repetition. Repeat the requested information "
                "directly without saying 'as mentioned earlier'."
            )
        if analysis.user_situation:
            instructions.append(
                f"Relate the answer to this situation: {analysis.user_situation}"
            )
        if analysis.general_logistics and not chunks and not profile_context:
            instructions.append(
                "Label general guidance carefully and do not present it as a confirmed "
                "Paramount Logistics capability."
            )

        try:
            generated = self.generator.answer(
                question=message,
                context=context,
                history=memory.history_for_planner(),
                conversation_state=memory.state_for_planner(),
                intent_analysis=analysis.model_dump_json(),
                controller_instructions="\n".join(f"- {item}" for item in instructions),
            )
        except Exception as exc:
            logger.error("Answer generation failed.", exc_info=exc)
            return (
                "I found relevant company information, but I couldn't prepare the answer "
                "right now. Please try again shortly."
            )
        sanitized = self.sanitizer.sanitize(generated, set())
        return limit_words(sanitized, self.settings.response_max_words)

    def _pricing_guidance(
        self,
        analysis: IntentAnalysis,
        memory: ConversationMemory,
    ) -> tuple[str, PendingQuestion | None]:
        context = {**memory.shipment_context, **analysis.entities.populated()}
        required = ("origin", "destination", "service_mode", "cargo_type")
        missing = analysis.missing_fields or [
            field for field in required if not context.get(field)
        ]
        if not missing:
            return (
                "Pricing depends on the shipment's weight or volume, schedule, and handling "
                "requirements. The team can prepare an exact quotation from the details provided.",
                None,
            )

        labels = {
            "origin": "pickup location",
            "destination": "destination",
            "service_mode": "preferred shipping method",
            "cargo_type": "type of goods",
            "weight": "weight or volume",
            "dimensions": "cargo dimensions",
            "shipment_date": "approximate shipment date",
        }
        requested = [labels.get(field, field.replace("_", " ")) for field in missing[:4]]
        question = "For an accurate quotation, please share " + natural_join(requested) + "."
        pending = PendingQuestion(
            topic=self._active_topic(analysis, memory),
            question=question,
            expected_fields=missing,
            resume_query=analysis.resolved_query,
            created_at=memory.assistant_message_count + 1,
        )
        return question, pending

    def _tracking_block(self) -> str:
        return f"You can track your shipment here: {self.settings.tracking_url}"

    def _unavailable_company_info(self, analysis: IntentAnalysis) -> str:
        topic = natural_join(
            [
                intent.replace("_", " ")
                for intent in analysis.intent_values()
                if intent not in NON_TOPIC_INTENTS
            ]
        )
        return (
            f"I don't have confirmed company information about {topic or 'that request'}."
        )

    def _needs_action(self, analysis: IntentAnalysis) -> bool:
        return bool(
            analysis.actions
            or analysis.needs_tracking
            or analysis.needs_pricing
            or analysis.needs_head_of_services
            or analysis.needs_handoff
            or analysis.needs_rag
            or analysis.company_specific
            or analysis.general_logistics
        )

    def _active_topic(
        self,
        analysis: IntentAnalysis,
        memory: ConversationMemory,
    ) -> str:
        topics = [
            intent
            for intent in analysis.intent_values()
            if intent not in NON_TOPIC_INTENTS
        ]
        return topics[0] if topics else memory.active_topic or "shipment"

    def _remember(
        self,
        memory: ConversationMemory,
        user_message: str,
        assistant_message: str,
        analysis: IntentAnalysis,
        draft: ResponseDraft,
    ) -> None:
        meaningful = analysis.dialogue_act not in {
            "greeting",
            "small_talk",
            "gratitude",
            "farewell",
        }
        memory.add_user_message(
            user_message,
            dialogue_act=analysis.dialogue_act,
            meaningful=meaningful,
        )
        memory.add_ai_message(
            assistant_message,
            dialogue_act="answer",
            meaningful=meaningful,
        )
        memory.update_shipment_context(analysis.entities.populated())
        memory.remember_topics(
            [
                intent
                for intent in analysis.intent_values()
                if intent not in NON_TOPIC_INTENTS
            ]
        )
        for contact_id, fields in draft.disclosed_contacts:
            memory.remember_contact(contact_id, fields)

        if draft.pending_question:
            memory.pending_question = draft.pending_question
        elif analysis.dialogue_act not in {
            "greeting",
            "small_talk",
            "gratitude",
            "farewell",
        }:
            memory.clear_pending_question()
