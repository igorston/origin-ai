"""Tiny stopword-based language guess, good enough to tell the model which language to use."""

import re

# Written in the target language itself: an English "reply in Portuguese" still primed
# English words ("Okay, note salva!").
# After actions and lookups (saving a fact, a date): short. Longer wording made the model
# pad confirmations ("posso ajudar com algo else?"), and English leaked into the padding.
REPLY_INSTRUCTIONS = {
    # No example sentences: the model copied the example's facts into real replies.
    "pt": (
        "[Responda ao usuário em português do Brasil, em uma ou duas frases diretas, "
        'falando com ele por "você" — nunca como se você fosse o usuário. '
        "Não misture palavras em inglês.]"
    ),
    "en": (
        "[Reply to the user in English, in one or two direct sentences, addressing them "
        'as "you" — never as if you were the user.]'
    ),
    "es": (
        "[Responde al usuario en español, en una o dos frases directas, hablándole de "
        '"tú", nunca como si fueras el usuario. No mezcles palabras en inglés.]'
    ),
}
# After web results: the length follows the request. "One or two direct sentences" turned
# "explique a MP das Bets" into two lines.
DETAILED_INSTRUCTIONS = {
    "pt": (
        '[Responda ao usuário em português do Brasil, falando com ele por "você" — nunca '
        "como se você fosse o usuário. Use a extensão que o pedido pede: uma frase para "
        "confirmar uma ação; uma explicação completa e organizada quando ele pede para "
        "explicar um assunto. Não misture palavras em inglês.]"
    ),
    "en": (
        '[Reply to the user in English, addressing them as "you" — never as if you were '
        "the user. Match the length to the request: one sentence to confirm an action; a "
        "complete, organized explanation when they ask you to explain something.]"
    ),
    "es": (
        '[Responde al usuario en español, hablándole de "tú", nunca como si fueras el '
        "usuario. Usa la extensión que pide la solicitud: una frase para confirmar una "
        "acción; una explicación completa y organizada cuando pide explicar un tema. No "
        "mezcles palabras en inglés.]"
    ),
}
FALLBACK_INSTRUCTION = "[Reply in the same language the user wrote in.]"
LANGUAGE_NAMES = {"pt": "Brazilian Portuguese", "en": "English", "es": "Spanish"}

STOPWORDS = {
    "pt": {
        "que", "não", "nao", "você", "voce", "meu", "minha", "eu", "é", "pra", "para", "com",
        "um", "uma", "do", "da", "no", "na", "os", "as", "se", "mas", "isso", "aí", "também",
        "lembra", "lembre", "guarda", "anota", "esquece", "qual", "quando", "onde", "hoje",
        "tenho", "sou", "estou", "quantos", "faltam", "dia", "agora", "tudo", "bem",
        "me", "te", "de", "das", "dos", "o", "e", "em", "sobre", "como", "porque", "explique",
        "explica", "entenda", "muda", "está", "são", "foi", "ser", "mais", "ele", "ela",
        "vai", "pode", "quero", "preciso", "seu", "sua", "nova", "novo", "nas", "nos", "ao",
    },
    "en": {
        "the", "is", "are", "my", "you", "your", "and", "what", "to", "of", "in", "it", "that",
        "i", "please", "remember", "how", "do", "does", "when", "where", "today", "have",
        "am", "many", "days", "until", "now", "hello", "hi", "thanks", "about", "can",
    },
    "es": {
        "el", "la", "los", "las", "mi", "es", "y", "qué", "cómo", "cuándo", "dónde", "hoy",
        "también", "recuerda", "tengo", "soy", "estoy", "cuántos", "faltan", "ahora", "hola",
        "gracias", "por", "favor", "usted", "tú", "muy", "me", "explica", "sobre", "como",
    },
}  # fmt: skip

# Letters that only one of the candidate languages uses.
MARKERS = {"pt": re.compile(r"[ãõçâêôà]"), "es": re.compile(r"[ñ¿¡]")}


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


def reply_instruction(message: str, locale: str | None = None, detailed: bool = False) -> str:
    """Undetected, the configured locale's language (an English "same language as the
    user" after English tool results produced English replies)."""
    language = detect_language(message) or (locale or "")[:2].lower()
    table = DETAILED_INSTRUCTIONS if detailed else REPLY_INSTRUCTIONS
    return table.get(language, FALLBACK_INSTRUCTION)
