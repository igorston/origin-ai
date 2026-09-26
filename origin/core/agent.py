import logging
from collections.abc import Mapping, Sequence

from langchain_core.messages import ToolCall, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

logger = logging.getLogger(__name__)


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
