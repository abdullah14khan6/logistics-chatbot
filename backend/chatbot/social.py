import re

from backend.chatbot.intents import DomainIntent, IntentAnalysis

FAST_ACKNOWLEDGEMENTS = {"yes", "yeah", "yep", "ok", "okay", "sure"}
FAST_GREETINGS = {
    "hi",
    "hii",
    "hello",
    "hey",
    "hi there",
    "good morning",
    "good afternoon",
    "good evening",
}
FAST_SMALL_TALK = {
    "hru",
    "how are you",
    "how are you?",
    "how's it going",
    "hows it going",
    "how is it going",
}
FAST_FAREWELLS = {"bye", "goodbye", "see you", "take care", "have a nice day"}
FAST_GRATITUDE = {"thanks", "thank you", "thx"}

SOCIAL_INTENTS = {
    "gratitude",
    "acknowledgement",
    "greeting",
    "small_talk",
    "farewell",
}


def analyze_social_message(message: str) -> IntentAnalysis | None:
    normalized = re.sub(r"\s+", " ", message.strip().lower()).strip(" .!?")
    if normalized in FAST_GRATITUDE:
        return IntentAnalysis(
            dialogue_act="gratitude",
            intents=[DomainIntent.GRATITUDE],
            gratitude=True,
            confidence=1.0,
        )
    if normalized in FAST_FAREWELLS:
        return IntentAnalysis(
            dialogue_act="farewell",
            intents=[DomainIntent.FAREWELL],
            farewell=True,
            confidence=1.0,
        )
    if normalized in FAST_SMALL_TALK:
        return IntentAnalysis(
            dialogue_act="small_talk",
            intents=[DomainIntent.SMALL_TALK],
            small_talk=True,
            confidence=1.0,
        )
    if normalized in FAST_GREETINGS:
        return IntentAnalysis(
            dialogue_act="greeting",
            intents=[DomainIntent.GREETING],
            greeting=True,
            confidence=1.0,
        )
    if normalized in FAST_ACKNOWLEDGEMENTS:
        return IntentAnalysis(
            dialogue_act="acknowledgement",
            intents=[DomainIntent.ACKNOWLEDGEMENT],
            acknowledgement=True,
            confidence=1.0,
        )
    return None


def social_response(
    analysis: IntentAnalysis,
    turn_count: int,
    has_context: bool,
) -> str | None:
    if analysis.gratitude:
        return _option(
            turn_count,
            [
                "You're very welcome! I'm happy to help.",
                "My pleasure! Let me know what else you need.",
                "Happy to help!",
                "You're welcome! Have a wonderful day.",
            ],
        )
    if analysis.farewell:
        return _option(
            turn_count,
            [
                "Thank you for contacting Paramount Logistics. Have a wonderful day!",
                "Take care! We're here whenever you need assistance.",
                "Goodbye, and thank you for choosing Paramount Logistics!",
            ],
        )
    if analysis.greeting:
        if has_context:
            options = [
                "Hi again! How can I help you further?",
                "Hello again! What would you like to continue with?",
                "Welcome back. How may I assist you?",
            ]
        else:
            options = [
                "Hello! Welcome to Paramount Logistics. How can I assist you today?",
                "Hi! It's great to have you here. How can I help you today?",
                "Hello! How may I assist you today?",
                "Hi there! How can I help?",
                "Welcome to Paramount Logistics! What can I help you with today?",
            ]
        return _option(turn_count, options)
    if analysis.small_talk:
        return _option(
            turn_count,
            [
                "I'm doing well, thank you for asking! How can I assist you today?",
                "I'm doing great, thanks! How may I help you today?",
                "I'm well, thank you. What can I help you with today?",
            ],
        )
    if analysis.acknowledgement:
        return _option(
            turn_count,
            [
                "Of course! How can I help?",
                "Certainly. What would you like to know?",
                "Absolutely! How may I assist you today?",
            ],
        )
    return None


def _option(turn_count: int, options: list[str]) -> str:
    return options[turn_count % len(options)]
