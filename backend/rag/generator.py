from backend.config.settings import Settings
from backend.prompts.rag import RAG_SYSTEM_PROMPT, RAG_USER_PROMPT


class GroqAnswerGenerator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._llm = None

    def answer(
        self,
        question: str,
        context: str,
        history: str,
        intent_analysis: str = "",
        controller_instructions: str = "",
    ) -> str:
        messages = [
            ("system", RAG_SYSTEM_PROMPT),
            (
                "user",
                RAG_USER_PROMPT.format(
                    history=history or "No previous conversation.",
                    context=context,
                    intent_analysis=intent_analysis or "No structured analysis provided.",
                    controller_instructions=(
                        controller_instructions
                        or "Answer the user's logistics question professionally."
                    ),
                    question=question,
                ),
            ),
        ]
        response = self._client().invoke(messages)
        return str(response.content).strip()

    def _client(self):
        if self._llm is None:
            from langchain_groq import ChatGroq

            self._llm = ChatGroq(
                api_key=self.settings.groq_api_key,
                model=self.settings.groq_model_name,
                temperature=self.settings.groq_temperature,
            )
        return self._llm
