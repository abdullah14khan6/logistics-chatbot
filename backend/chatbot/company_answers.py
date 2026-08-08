import re

from backend.chatbot.intents import IntentAnalysis
from backend.knowledge.company_profile import CompanyProfile, normalize_lookup

STRUCTURED_ONLY_INTENTS = {
    "office_locations",
    "office_hours",
    "payment_terms",
    "prohibited_items",
    "shipment_delay",
    "company_services",
    "amazon_fba",
    "leadership",
    "subsidiaries",
}


class CompanyAnswerProvider:
    def __init__(self, profile: CompanyProfile) -> None:
        self.profile = profile

    def direct_answer(self, message: str, analysis: IntentAnalysis) -> str:
        intents = set(analysis.intent_values())
        parts: list[str] = []

        if "office_locations" in intents:
            lines = ["We operate offices in Sialkot and Karachi:"]
            for office in self.profile.offices:
                phones = ", ".join(office.phones)
                lines.append(
                    f"- **{office.name}:** {office.address}"
                    + (f" Phone: {phones}." if phones else "")
                )
            if "office_hours" in intents:
                lines.append(f"- **Office hours:** {self.profile.office_hours.display}.")
            parts.append("\n".join(lines))
        elif "office_hours" in intents:
            parts.append(f"Our office hours are {self.profile.office_hours.display}.")

        if "payment_terms" in intents:
            parts.append(self.profile.policies.payment_terms)
        if "prohibited_items" in intents:
            parts.append(self.profile.policies.prohibited_items)
        if "shipment_delay" in intents:
            parts.append(self.profile.policies.shipment_delays)
        if "amazon_fba" in intents:
            parts.append(self.profile.capabilities.amazon_fba)
        if "company_services" in intents:
            full_list_requested = bool(
                re.search(r"\b(all|complete|full|every)\b", message, re.IGNORECASE)
            )
            if full_list_requested:
                services = "\n".join(
                    f"- {service}" for service in self.profile.services_offered
                )
                parts.append(f"Our services include:\n{services}")
            else:
                parts.append(
                    "We provide air and sea freight, imports and exports, customs clearance, "
                    "door-to-door delivery, warehousing, consolidation, cargo insurance, "
                    "project cargo, express shipping, e-commerce, and Amazon FBA logistics."
                )

        if "leadership" in intents:
            parts.append(self._leadership_answer(message, analysis))

        if "subsidiaries" in intents:
            lines = ["Paramount Logistics International's group companies include:"]
            for company in self.profile.subsidiaries:
                lines.append(f"- **{company.name}:** {company.description}")
            parts.append("\n".join(lines))

        return "\n\n".join(parts)

    def _leadership_answer(self, message: str, analysis: IntentAnalysis) -> str:
        query = " ".join(
            value
            for value in (
                analysis.entities.person_name,
                analysis.requested_contact_role,
                message,
            )
            if value
        )
        leader = self.profile.resolve_leader(query)
        phone_requested = "phone" in analysis.contact_fields or bool(
            re.search(r"\b(phone|number|mobile|call)\b", message, re.IGNORECASE)
        )
        email_requested = "email" in analysis.contact_fields or bool(
            re.search(r"\b(email|e-mail|contact)\b", message, re.IGNORECASE)
        )

        if leader:
            lines = [
                f"**{leader.name}**",
                f"{leader.title}, {leader.company}.",
            ]
            if leader.department and leader.department != "Executive Leadership":
                lines.append(f"- **Department:** {leader.department}")
            if email_requested and leader.email:
                lines.append(f"- **Email:** {leader.email}")
            if phone_requested:
                lines.append("Leadership phone numbers are not shared.")
            return "\n".join(lines)

        if analysis.entities.person_name and not self._is_generic_leadership_query(
            analysis.entities.person_name
        ):
            return (
                "I don't have confirmed leadership information about "
                f"{analysis.entities.person_name}."
            )

        if phone_requested:
            return "Leadership phone numbers are not shared."

        primary_leaders = [
            leader
            for leader in self.profile.leadership
            if any(
                title in leader.title.casefold()
                for title in ("chief executive", "director", "general manager")
            )
        ]
        lines = ["PLI's leadership team includes:"]
        lines.extend(
            f"- **{leader.name}:** {leader.title}"
            for leader in primary_leaders
        )
        return "\n".join(lines)

    @staticmethod
    def _is_generic_leadership_query(value: str) -> bool:
        tokens = set(normalize_lookup(value).split())
        generic_tokens = {
            "leadership",
            "leaders",
            "management",
            "managers",
            "team",
            "paramount",
            "pli",
            "company",
            "group",
            "of",
            "the",
        }
        return bool(tokens) and tokens <= generic_tokens

    def authorized_emails(
        self,
        message: str,
        analysis: IntentAnalysis,
    ) -> set[str]:
        if "leadership" not in analysis.intent_values():
            return set()
        email_requested = "email" in analysis.contact_fields or bool(
            re.search(r"\b(email|e-mail|contact)\b", message, re.IGNORECASE)
        )
        if not email_requested:
            return set()
        query = " ".join(
            value
            for value in (
                analysis.entities.person_name,
                analysis.requested_contact_role,
                message,
            )
            if value
        )
        leader = self.profile.resolve_leader(query)
        return {leader.email.lower()} if leader and leader.email else set()

    def evidence(self, analysis: IntentAnalysis) -> str:
        intents = set(analysis.intent_values())
        lines: list[str] = []

        if "company_information" in intents:
            lines.extend(
                [
                    f"- Company: {self.profile.company_name}",
                    f"- Office hours: {self.profile.office_hours.display}",
                    f"- Payment terms: {self.profile.policies.payment_terms}",
                    f"- Prohibited items: {self.profile.policies.prohibited_items}",
                    f"- Shipment delays: {self.profile.policies.shipment_delays}",
                    f"- Amazon FBA: {self.profile.capabilities.amazon_fba}",
                    "- Services: " + ", ".join(self.profile.services_offered),
                    "- Value-added services: "
                    + ", ".join(self.profile.value_added_services),
                ]
            )

        if "subsidiaries" in intents:
            lines.extend(
                f"- Group company: {company.name} - {company.description}"
                for company in self.profile.subsidiaries
            )

        service_terms = {
            "air_freight": ("air freight",),
            "sea_freight": ("sea freight",),
            "ocean_freight": ("sea freight",),
            "imports": ("import",),
            "exports": ("export",),
            "door_to_door": ("door-to-door",),
            "customs": ("customs",),
            "freight_forwarding": ("freight forwarding",),
            "supplier_pickup": ("supplier pickup",),
            "warehousing": ("warehousing",),
            "cargo_insurance": ("cargo insurance",),
            "project_cargo": ("project cargo",),
            "dangerous_goods": ("dangerous goods",),
            "express_shipping": ("express shipping",),
            "temperature_controlled": ("temperature-controlled",),
            "cargo_consolidation": ("consolidation",),
            "amazon_fba": ("amazon fba",),
            "documentation": ("documentation", "invoice", "packing list"),
        }
        selected_terms = [
            term
            for intent, terms in service_terms.items()
            if intent in intents
            for term in terms
        ]
        for service in [
            *self.profile.services_offered,
            *self.profile.value_added_services,
        ]:
            normalized = normalize_lookup(service)
            if any(normalize_lookup(term) in normalized for term in selected_terms):
                lines.append(f"- Supported capability: {service}")
        return "\n".join(lines)
