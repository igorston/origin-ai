import logging
import re
from collections.abc import Mapping, Sequence

from langchain_core.messages import ToolCall, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from origin.i18n.packs import rule

logger = logging.getLogger(__name__)

# Phrases in a final reply that claim a side effect only a given tool can produce. Small
# models sometimes say "anotei!" without calling the tool; such replies get verified.
CLAIMS: dict[str, re.Pattern[str]] = {
    "remember": rule("claims_remember"),
    "forget": rule("claims_forget"),
}


def unbacked_claims(
    text: str, executed: Sequence["ToolCallRecord"], available: set[str]
) -> set[str]:
    """Tools whose effect the reply claims although they were not called this turn."""
    called = {record.name for record in executed}
    return {
        name
        for name, pattern in CLAIMS.items()
        if name in available and name not in called and pattern.search(text)
    }


class ToolCallRecord(BaseModel):
    name: str
    args: dict
    output: str


SENTENCE = re.compile(r"[^.!?\n]+[.!?]*")


def is_pure_question(message: str) -> bool:
    """Every sentence is a question ("Onde eu moro?"), with no statement mixed in."""
    sentences = [s.strip() for s in SENTENCE.findall(message) if s.strip()]
    return bool(sentences) and all(s.endswith("?") for s in sentences)


def blocked_for_question(tool: BaseTool, message: str) -> bool:
    """Tools can opt out of pure questions via metadata, unless the message explicitly asks
    for them. Small models sometimes answer "O que eu levo de presente pra ela?" by saving
    a fact from the history, or save "A capital da França é Paris." from a question; prompt
    rules alone did not stop it."""
    meta = tool.metadata or {}
    if not meta.get("not_for_questions") or not is_pure_question(message):
        return False
    intent = meta.get("explicit_intent")
    return not (intent and re.search(intent, message, re.IGNORECASE))


async def execute_tool_calls(
    tool_calls: Sequence[ToolCall], tools: Mapping[str, BaseTool], message: str = ""
) -> tuple[list[ToolMessage], list[ToolCallRecord]]:
    """Run the model's tool calls. Errors are reported back to the model, never raised.
    Calls blocked for being made on a pure question get a ToolMessage (the protocol needs
    one per call) but no record, since nothing ran."""
    messages: list[ToolMessage] = []
    records: list[ToolCallRecord] = []
    for call in tool_calls:
        tool = tools.get(call["name"])
        if tool is not None and blocked_for_question(tool, message):
            logger.info("Blocked %s(%s) on a pure question", call["name"], call["args"])
            skipped = "Not executed: the user only asked a question. Answer it; nothing was saved."
            messages.append(ToolMessage(skipped, tool_call_id=call["id"], name=call["name"]))
            continue
        if tool is None:
            output = f"Error: unknown tool {call['name']!r}. Available: {', '.join(tools)}."
        else:
            try:
                output = str(await tool.ainvoke(call["args"]))
            except Exception as exc:
                logger.warning("Tool %s failed", call["name"], exc_info=True)
                output = f"Error: {type(exc).__name__}: {exc}"
        logger.info("Tool call %s(%s) -> %s", call["name"], call["args"], output[:200])
        messages.append(ToolMessage(output, tool_call_id=call["id"], name=call["name"]))
        records.append(ToolCallRecord(name=call["name"], args=call["args"], output=output))
    return messages, records
