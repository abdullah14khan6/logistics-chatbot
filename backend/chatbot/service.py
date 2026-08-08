import logging
import re
from dataclasses import dataclass, field
from uuid import uuid4

from backend.chatbot.company_answers import (
    STRUCTURED_ONLY_INTENTS,
    CompanyAnswerProvider,
)
from backend.chatbot.contact_policy import ContactPolicy
from backend.chatbot.intents import (
    DomainIntent,
    IntentAnalysis,
    IntentAnalyzer,
    PlannedAction,
)
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
PRICING_CONTEXT_INTENTS = {
    "air_freight",
    "sea_freight",
    "ocean_freight",
    "express_shipping",
    "warehousing",
    "door_to_door",
    "transportation",
    "freight_forwarding",
}
CONSULTATIVE_SERVICE_INTENTS = {
    "supplier_pickup",
    "dangerous_goods",
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
            analysis = self._recover_or_normalize_plan(
                clean_message,
                analysis,
                memory,
            )
            self._supplement_shipment_entities(clean_message, analysis)

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
            allowed_emails.update(
                self.company_answers.authorized_emails(message, analysis)
            )

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

        consultative_service_inquiry = bool(
            set(analysis.intent_values()) & CONSULTATIVE_SERVICE_INTENTS
        )
        leadership_request = "leadership" in analysis.intent_values()
        leadership_contact_query = (
            analysis.requested_contact_role
            or analysis.entities.contact_role
            or analysis.entities.person_name
        )
        leadership_contact_only = bool(
            leadership_request
            and leadership_contact_query
            and self.profile.resolve_leader(leadership_contact_query)
            and not analysis.needs_head_of_services
            and not analysis.needs_handoff
            and not analysis.needs_pricing
            and not consultative_service_inquiry
        )
        contact_requested = not leadership_contact_only and (
            analysis.explicit_contact_request
            or analysis.needs_head_of_services
            or analysis.needs_handoff
            or analysis.needs_pricing
            or consultative_service_inquiry
        )
        if contact_requested:
            if consultative_service_inquiry and not analysis.explicit_contact_request:
                resolution = self.profile.resolve_contact(
                    self.profile.default_contact_role
                )
            else:
                resolution = self.contact_policy.resolve(analysis)
            explicit = (
                analysis.explicit_contact_request
                or analysis.needs_head_of_services
                or consultative_service_inquiry
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

    def _recover_or_normalize_plan(
        self,
        message: str,
        analysis: IntentAnalysis,
        memory: ConversationMemory,
    ) -> IntentAnalysis:
        if (
            analyze_social_message(message) is None
            and set(analysis.intent_values())
            and set(analysis.intent_values()) <= SOCIAL_INTENTS
        ):
            analysis = IntentAnalysis(
                dialogue_act="request",
                intents=[DomainIntent.GENERAL_LOGISTICS],
                actions=[PlannedAction.GENERAL_ANSWER],
                general_logistics=True,
                resolved_query=message,
                query_for_rag=message,
                confidence=min(analysis.confidence, 0.5),
            )

        if (
            analysis.confidence == 0.0
            and analysis.unclear
            and memory.pending_question
        ):
            pending = memory.pending_question
            expected = list(pending.expected_fields)
            entities: dict[str, str] = {}
            if len(expected) == 1:
                entities[expected[0]] = message
            elif "cargo_type" in expected:
                entities["cargo_type"] = message

            if entities:
                remaining = [field for field in expected if field not in entities]
                intents: list[DomainIntent] = [
                    DomainIntent.PRICING,
                    DomainIntent.FOLLOW_UP,
                ]
                try:
                    topic_intent = DomainIntent(pending.topic)
                except ValueError:
                    topic_intent = None
                if topic_intent and topic_intent not in intents:
                    intents.insert(0, topic_intent)
                analysis = IntentAnalysis(
                    dialogue_act="follow_up",
                    intents=intents,
                    actions=[PlannedAction.QUOTE],
                    follow_up=True,
                    needs_pricing=True,
                    pricing_request="quotation",
                    entities=entities,
                    missing_fields=remaining,
                    resolved_query=pending.resume_query,
                    confidence=0.4,
                )

        if analysis.needs_pricing and self._asks_for_exact_current_rate(message):
            analysis.pricing_request = "current_exact_rate"

        leader_query = " ".join(
            value
            for value in (
                analysis.entities.person_name,
                analysis.requested_contact_role,
                message,
            )
            if value
        )
        if self.profile.resolve_leader(leader_query):
            if DomainIntent.LEADERSHIP not in analysis.intents:
                analysis.intents.insert(0, DomainIntent.LEADERSHIP)
            if PlannedAction.COMPANY_LOOKUP not in analysis.actions:
                analysis.actions.insert(0, PlannedAction.COMPANY_LOOKUP)
            analysis.company_specific = True
        return analysis

    @staticmethod
    def _asks_for_exact_current_rate(message: str) -> bool:
        normalized = " ".join(message.casefold().split())
        has_rate = bool(re.search(r"\b(rate|rates|price|pricing|cost)\b", normalized))
        has_live_qualifier = bool(
            re.search(r"\b(today(?:'s)?|current|live|exact|right now)\b", normalized)
        )
        return has_rate and has_live_qualifier

    def _should_generate(
        self,
        analysis: IntentAnalysis,
        structured_answer: str,
    ) -> bool:
        intents = set(analysis.intent_values())
        remaining = intents - STRUCTURED_ONLY_INTENTS - NON_TOPIC_INTENTS
        if (
            structured_answer
            and "leadership" in intents
            and analysis.explicit_contact_request
            and analysis.question_complexity == "simple"
            and not analysis.needs_rag
        ):
            return False
        if (
            analysis.pricing_request == "current_exact_rate"
            and remaining <= PRICING_CONTEXT_INTENTS
        ):
            return False
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
        word_limit = self._response_word_limit(analysis)
        instructions = [
            (
                f"Use up to {word_limit} words when needed. Do not pad the answer merely "
                "to approach this limit."
            ),
            "Answer the current question first and include only relevant information.",
            "Do not add contact details, a generic closing, or an unsolicited service list.",
            "Ask no follow-up question; the controller handles clarifications.",
        ]
        if set(analysis.intent_values()) & CONSULTATIVE_SERVICE_INTENTS:
            instructions.extend(
                [
                    (
                        "Treat this as a qualified service inquiry. Give a consultative answer "
                        "of roughly 2 to 4 useful sentences when the evidence supports it."
                    ),
                    (
                        "Explain the relevant service scope, important shipment or compliance "
                        "considerations, and how Paramount Logistics can help. Be professionally "
                        "sales-oriented without hype or unsupported promises."
                    ),
                    (
                        "For a yes-or-no capability question, begin with a direct 'Yes' when "
                        "supported. Do not restate the question, begin with 'Paramount Logistics "
                        "offers', or explain an obvious service definition before giving the "
                        "customer the useful next step."
                    ),
                    (
                        "End the generated portion with a concise invitation to discuss the "
                        "shipment requirements. The controller will append the authorized "
                        "Head of Services contact details."
                    ),
                ]
            )
        if analysis.needs_pricing:
            instructions.extend(
                [
                    (
                        "If a shipment solution is requested, recommend only the relevant "
                        "service structure in no more than two concise sentences."
                    ),
                    (
                        "Do not state or estimate any price, duty, tax, transit time, or "
                        "delivery commitment. Do not promise to prepare, provide, send, or "
                        "follow up with a quotation. The controller handles quotation limits "
                        "and the human handoff."
                    ),
                ]
            )
        if "supplier_pickup" in analysis.intent_values():
            instructions.append(
                "For supplier pickup, confirm the collection capability directly, then invite "
                "the customer to provide the exact pickup location, cargo type, volume or "
                "weight, and destination so operations can confirm the arrangement. Mention "
                "onward services only when they materially answer the question. Do not narrate "
                "routine coordination with suppliers or partners. Never mention evidence, "
                "sources, or missing data."
            )
        if "dangerous_goods" in analysis.intent_values():
            instructions.append(
                "For lithium batteries or other dangerous goods, explain that acceptance is "
                "subject to classification, applicable regulations, carrier approval, suitable "
                "packaging, and required documents. Invite the customer to share battery type, "
                "quantity, packing method, origin, and destination for an eligibility review."
            )
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
        if analysis.needs_pricing:
            sanitized = self.sanitizer.remove_quotation_promises(sanitized)
        return limit_words(sanitized, word_limit)

    @staticmethod
    def _supplement_shipment_entities(
        message: str,
        analysis: IntentAnalysis,
    ) -> None:
        labels = {
            "pickup location": "origin",
            "origin": "origin",
            "destination": "destination",
            "total weight": "weight",
            "weight": "weight",
            "cargo": "cargo_type",
            "type of goods": "cargo_type",
            "number of cartons": "package_count",
            "package count": "package_count",
            "preferred shipping method": "service_mode",
            "shipping method": "service_mode",
            "cargo value": "cargo_value",
            "pickup required": "pickup_required",
            "delivery required": "delivery_required",
            "shipment ready date": "shipment_date",
            "ready date": "shipment_date",
        }
        label_pattern = "|".join(
            re.escape(label) for label in sorted(labels, key=len, reverse=True)
        )
        pattern = re.compile(
            rf"(?i)\b({label_pattern})\s*:\s*(.*?)"
            rf"(?=\s+\b(?:{label_pattern})\s*:|$)"
        )
        for match in pattern.finditer(message):
            field = labels[match.group(1).casefold()]
            value = match.group(2).strip(" ,.;")
            if value and not getattr(analysis.entities, field):
                setattr(analysis.entities, field, value)

        if not analysis.entities.volume:
            volume = re.search(r"\b\d+(?:\.\d+)?\s*CBM\b", message, re.IGNORECASE)
            if volume:
                analysis.entities.volume = volume.group(0)
        if not analysis.entities.weight:
            weight = re.search(
                r"\b\d[\d,]*(?:\.\d+)?\s*(?:kg|kilograms?|tons?)\b",
                message,
                re.IGNORECASE,
            )
            if weight:
                analysis.entities.weight = weight.group(0)
        if re.search(r"\bnon[- ]?hazardous\b", message, re.IGNORECASE):
            analysis.entities.hazardous_status = "non-hazardous"
        elif re.search(r"\bhazardous\b", message, re.IGNORECASE):
            analysis.entities.hazardous_status = "hazardous"
        if re.search(r"\b(?:does not require|no) refrigeration\b", message, re.IGNORECASE):
            analysis.entities.temperature_control_status = "not required"

    def _response_word_limit(self, analysis: IntentAnalysis) -> int:
        if analysis.response_detail == "detailed":
            return self.settings.response_detailed_max_words
        if analysis.response_detail == "brief":
            return self.settings.response_brief_max_words
        if analysis.question_complexity == "complex":
            return self.settings.response_complex_max_words
        return self.settings.response_max_words

    def _pricing_guidance(
        self,
        analysis: IntentAnalysis,
        memory: ConversationMemory,
    ) -> tuple[str, PendingQuestion | None]:
        context = {**memory.shipment_context, **analysis.entities.populated()}
        required = ("origin", "destination", "service_mode", "cargo_type")
        inferred_missing = [
            field for field in required if not context.get(field)
        ]
        aliases = {
            "pickup_location": "origin",
            "preferred_shipping_method": "service_mode",
            "type_of_goods": "cargo_type",
            "shipment_ready_date": "shipment_date",
            "number_of_cartons": "package_count",
        }
        planner_missing = [
            aliases.get(field, field)
            for field in analysis.missing_fields
            if not context.get(aliases.get(field, field))
        ]
        missing = list(dict.fromkeys([*inferred_missing, *planner_missing]))
        explanation = (
            "I can't provide an exact quotation, confirmed transit time, or duties and taxes "
            "in chat. These require current carrier rates, shipment classification, and "
            "customs assessment."
        )

        if not missing:
            return (
                explanation
                + " You've provided enough shipment information for our team to prepare "
                "the quotation.",
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
        question = (
            explanation
            + "\n\nTo send this inquiry for pricing, please share "
            + natural_join(requested)
            + "."
        )
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
