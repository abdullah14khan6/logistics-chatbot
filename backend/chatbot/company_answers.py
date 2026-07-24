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

        return "\n\n".join(parts)

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
