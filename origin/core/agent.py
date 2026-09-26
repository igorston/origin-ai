import logging
import re
from collections.abc import Mapping, Sequence

from langchain_core.messages import ToolCall, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Phrases in a final reply that claim a side effect only a given tool can produce. Small
# models sometimes say "anotei!" without calling the tool; such replies get verified.
CLAIMS: dict[str, re.Pattern[str]] = {
    "remember": re.compile(
        r"\b(salve[i]?|salv[oa]s?|anotei|anotad[oa]s?|guardei|guardad[oa]s?|registrei|"
        r"registrad[oa]s?|memorizei|lembrete|vou lembrar|lembrarei|saved|noted|"
        r"i'?ll remember|i will remember|remembered)\b",
        re.IGNORECASE,
    ),
    "forget": re.compile(
        r"\b(apaguei|apagad[oa]s?|esqueci|removi|removid[oa]s?|exclu[íi]|exclu[íi]d[oa]s?|"
        r"deleted|removed|forgot(ten)?)\b",
        re.IGNORECASE,
    ),
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


async def execute_tool_calls(
    tool_calls: Sequence[ToolCall], tools: Mapping[str, BaseTool]
) -> tuple[list[ToolMessage], list[ToolCallRecord]]:
    """Run the model's tool calls. Errors are reported back to the model, never raised."""
    messages: list[ToolMessage] = []
    records: list[ToolCallRecord] = []
    for call in tool_calls:
        tool = tools.get(call["name"])
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
