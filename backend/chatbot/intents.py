from dataclasses import dataclass
from enum import StrEnum

from backend.config.settings import Settings


class Intent(StrEnum):
    ACKNOWLEDGEMENT = "acknowledgement"
    SHIPMENT_TRACKING = "shipment_tracking"
    PRICING = "pricing"
    CUSTOM_SOLUTION = "custom_solution"
    RAG = "rag"


@dataclass(frozen=True)
class IntentResult:
    intent: Intent
    response: str | None = None


TRACKING_KEYWORDS = (
    "tracking",
    "shipment status",
    "track package",
    "track my package",
    "parcel",
    "awb",
    "shipment",
)

PRICING_KEYWORDS = (
    "pricing",
    "rates",
    "quotation",
    "quote",
    "shipping cost",
    "freight cost",
)

CUSTOM_SOLUTION_KEYWORDS = (
    "enterprise logistics",
    "consultation",
    "custom logistics",
    "tailored solution",
)

GENERAL_LOGISTICS_KEYWORDS = (
    "freight forwarding",
    "customs clearance",
    "warehousing",
    "incoterms",
    "ocean freight",
    "air freight",
)

ACKNOWLEDGEMENTS = (
    "yes",
    "yeah",
    "yep",
    "ok",
    "okay",
    "sure",
    "thanks",
    "thank you",
    "hi",
    "hello",
    "hey",
)


class IntentRouter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def detect(self, message: str) -> IntentResult:
        normalized = message.lower().strip()
        if normalized in ACKNOWLEDGEMENTS:
            return IntentResult(
                intent=Intent.ACKNOWLEDGEMENT,
                response=(
                    "Sure. Please ask me about Paramount Logistics services, shipment "
                    "tracking, pricing, or a custom logistics requirement."
                ),
            )

        if self._contains(normalized, TRACKING_KEYWORDS):
            return IntentResult(
                intent=Intent.SHIPMENT_TRACKING,
                response=f"You can track your shipment here:\n\n{self.settings.tracking_url}",
            )

        if self._contains(normalized, PRICING_KEYWORDS):
            return IntentResult(
                intent=Intent.PRICING,
                response=self._service_head_response(
                    "Our pricing depends on your shipment requirements."
                ),
            )

        if self._contains(normalized, CUSTOM_SOLUTION_KEYWORDS):
            return IntentResult(
                intent=Intent.CUSTOM_SOLUTION,
                response=self._service_head_response(
                    "For custom logistics solutions, please contact our Head of Services."
                ),
            )

        return IntentResult(intent=Intent.RAG)

    def is_general_logistics_question(self, message: str) -> bool:
        return self._contains(message.lower(), GENERAL_LOGISTICS_KEYWORDS)

    def _service_head_response(self, opening: str) -> str:
        return (
            f"{opening}\n\n"
            "Please contact our Head of Services.\n\n"
            f"Name: {self.settings.head_of_services_name}\n"
            f"Email: {self.settings.head_of_services_email}\n"
            f"Phone: {self.settings.head_of_services_phone}"
        )

    @staticmethod
    def _contains(message: str, keywords: tuple[str, ...]) -> bool:
        return any(keyword in message for keyword in keywords)
