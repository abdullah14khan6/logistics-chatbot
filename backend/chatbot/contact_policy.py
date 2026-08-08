from dataclasses import dataclass

from backend.chatbot.intents import IntentAnalysis
from backend.chatbot.memory import ConversationMemory
from backend.knowledge.company_profile import (
    CompanyContact,
    CompanyProfile,
    ContactResolution,
)

NON_CONTACT_TOPICS = {
    "gratitude",
    "acknowledgement",
    "greeting",
    "small_talk",
    "farewell",
    "tracking",
    "pricing",
    "human_handoff",
    "contact",
    "head_of_services",
    "follow_up",
    "unclear",
    "unrelated",
    "prompt_injection",
    "company_information",
    "company_services",
    "general_logistics",
    "office_locations",
    "office_hours",
}


@dataclass(frozen=True)
class RenderedContact:
    text: str
    contact_id: str
    fields: list[str]
    allowed_email: str = ""


class ContactPolicy:
    def __init__(self, profile: CompanyProfile) -> None:
        self.profile = profile

    def resolve(self, analysis: IntentAnalysis) -> ContactResolution:
        requested = analysis.requested_contact_role or analysis.entities.contact_role
        if not requested and analysis.needs_pricing:
            requested = "pricing"
        if not requested:
            requested = (
                self._role_from_intents(analysis)
                or self.profile.default_contact_role
            )
        return self.profile.resolve_contact(requested)

    def render(
        self,
        resolution: ContactResolution,
        analysis: IntentAnalysis,
        memory: ConversationMemory,
        explicit: bool,
    ) -> RenderedContact | None:
        age = memory.contact_age(resolution.contact.id)
        if not explicit and age == 0:
            return None

        fields = self.fields_for(resolution.contact, analysis)
        contact = resolution.contact
        if not resolution.exact_match and resolution.requested_role:
            introduction = (
                f"A separate {resolution.requested_role.replace('_', ' ')} contact is not "
                f"listed, so {contact.name}, our {contact.title}, is the appropriate contact."
            )
        elif analysis.repeat_request:
            introduction = "Certainly - here are the requested contact details again."
        else:
            introduction = f"For assistance, please contact {contact.name}, {contact.title}."

        lines = [introduction]
        if "name" in fields:
            lines.append(f"- **Name:** {contact.name}")
        if "title" in fields:
            lines.append(f"- **Role:** {contact.title}")
        if "email" in fields and contact.email:
            lines.append(f"- **Email:** {contact.email}")
        elif "email" in fields:
            lines.append("- **Email:** Not publicly listed")
        if "phone" in fields and contact.phone:
            lines.append(f"- **Phone:** {contact.phone}")
        elif "phone" in fields:
            lines.append("- **Phone:** Not publicly listed")
        return RenderedContact(
            text="\n".join(lines),
            contact_id=contact.id,
            fields=fields,
            allowed_email=contact.email.lower(),
        )

    @staticmethod
    def fields_for(
        contact: CompanyContact,
        analysis: IntentAnalysis,
    ) -> list[str]:
        if analysis.contact_fields:
            requested = list(analysis.contact_fields)
            if "name" not in requested:
                requested.insert(0, "name")
            return requested
        fields = ["name", "title"]
        if contact.email:
            fields.append("email")
        if contact.phone:
            fields.append("phone")
        return fields

    @staticmethod
    def _role_from_intents(analysis: IntentAnalysis) -> str:
        candidates = [
            intent
            for intent in analysis.intent_values()
            if intent not in NON_CONTACT_TOPICS
        ]
        return candidates[0] if candidates else ""
