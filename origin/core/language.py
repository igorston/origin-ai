"""Tiny stopword-based language guess, good enough to tell the model which language to use."""

import re

# Written in the target language itself: an English "reply in Portuguese" still primed
# English words ("Okay, note salva!").
REPLY_INSTRUCTIONS = {
    "pt": "[Responda ao usuário em português do Brasil, em uma ou duas frases diretas.]",
    "en": "[Reply to the user in English, in one or two direct sentences.]",
    "es": "[Responde al usuario en español, en una o dos frases directas.]",
}
FALLBACK_INSTRUCTION = "[Reply in the same language the user wrote in.]"

STOPWORDS = {
    "pt": {
        "que", "não", "nao", "você", "voce", "meu", "minha", "eu", "é", "pra", "para", "com",
        "um", "uma", "do", "da", "no", "na", "os", "as", "se", "mas", "isso", "aí", "também",
        "lembra", "lembre", "guarda", "anota", "esquece", "qual", "quando", "onde", "hoje",
        "tenho", "sou", "estou", "quantos", "faltam", "dia", "agora", "tudo", "bem",
    },
    "en": {
        "the", "is", "are", "my", "you", "your", "and", "what", "to", "of", "in", "it", "that",
        "i", "me", "please", "remember", "how", "do", "does", "when", "where", "today", "have",
        "am", "many", "days", "until", "now", "hello", "hi", "thanks", "about", "can",
    },
    "es": {
        "el", "la", "los", "las", "mi", "es", "y", "qué", "cómo", "cuándo", "dónde", "hoy",
        "también", "recuerda", "tengo", "soy", "estoy", "cuántos", "faltan", "ahora", "hola",
        "gracias", "por", "favor", "usted", "tú", "muy",
    },
}  # fmt: skip

# Letters that only one of the candidate languages uses.
MARKERS = {"pt": re.compile(r"[ãõç]"), "es": re.compile(r"[ñ¿¡]")}


def detect_language(text: str) -> str | None:
    """Return "pt", "en" or "es" for the most likely language, or None if unsure."""
    lowered = text.lower()
    words = re.findall(r"[^\W\d_]+", lowered)
    scores = {lang: sum(word in stop for word in words) for lang, stop in STOPWORDS.items()}
    for lang, marker in MARKERS.items():
        if marker.search(lowered):
            scores[lang] += 2
    best = max(scores, key=scores.__getitem__)
    ranked = sorted(scores.values(), reverse=True)
    return best if ranked[0] > 0 and ranked[0] > ranked[1] else None


def reply_instruction(message: str) -> str:
    return REPLY_INSTRUCTIONS.get(detect_language(message) or "", FALLBACK_INSTRUCTION)
