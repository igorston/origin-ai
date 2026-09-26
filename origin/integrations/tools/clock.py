from datetime import date, datetime
from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool, tool

if TYPE_CHECKING:
    from origin.integrations.registry import ToolContext

# Small models mistranslate English weekday names, so return them already localized.
WEEKDAYS = {
    "en": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    "pt": [
        "segunda-feira",
        "terça-feira",
        "quarta-feira",
        "quinta-feira",
        "sexta-feira",
        "sábado",
        "domingo",
    ],
    "es": ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"],
}


def _now() -> datetime:
    return datetime.now().astimezone()


def _parse_target(target_date: str, today: date) -> date:
    """Accept YYYY-MM-DD, or MM-DD meaning the next occurrence (today included)."""
    if len(target_date) == 5:
        month, day = (int(part) for part in target_date.split("-"))
        target = date(today.year, month, day)
        return target if target >= today else date(today.year + 1, month, day)
    return date.fromisoformat(target_date)


def get_tools(ctx: "ToolContext") -> list[BaseTool]:
    weekdays = WEEKDAYS.get(ctx.settings.origin_locale.split("-")[0].lower(), WEEKDAYS["en"])

    @tool
    def get_current_datetime() -> str:
        """Return the current local date, time, weekday and UTC offset.

        Use when the user asks about today's date, the current time or the weekday.
        """
        now = _now()
        return (
            f"weekday: {weekdays[now.weekday()]}\n"
            f"date: {now:%Y-%m-%d}\n"
            f"time: {now:%H:%M}\n"
            f"utc_offset: {now:%z}"
        )

    @tool
    def days_until(target_date: str) -> str:
        """Count the days from today until a calendar date (negative if in the past).

        Use for questions like "how many days until Christmas?".
        `target_date` is "MM-DD" for the next occurrence of a recurring date
        (e.g. Christmas = "12-25"), or "YYYY-MM-DD" for a specific year.
        """
        today = _now().date()
        target = _parse_target(target_date, today)
        return (
            f"days: {(target - today).days}\n"
            f"from: {today.isoformat()} ({weekdays[today.weekday()]})\n"
            f"to: {target.isoformat()} ({weekdays[target.weekday()]})"
        )

    return [get_current_datetime, days_until]
