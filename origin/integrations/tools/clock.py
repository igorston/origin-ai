import calendar
from datetime import date, datetime, timedelta
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


def shift_date(start: date, days: int = 0, weeks: int = 0, months: int = 0, years: int = 0) -> date:
    """Calendar arithmetic; month/year steps clamp the day (Jan 31 + 1 month = Feb 28/29)."""
    month_index = start.month - 1 + months + 12 * years
    year, month = start.year + month_index // 12, month_index % 12 + 1
    day = min(start.day, calendar.monthrange(year, month)[1])
    return date(year, month, day) + timedelta(days=days, weeks=weeks)


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

        Use when the user asks about TODAY's date, the current time or today's weekday.
        For any other day ("ontem", "amanhã", "daqui a..."), use `date_offset` instead.
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
        """Answer "HOW MANY DAYS until <a known date>?" (negative if in the past).

        Use for questions like "quantos dias faltam pro Natal / pro meu aniversário?".
        `target_date` is "MM-DD" for the next occurrence of a recurring date
        (Natal = "12-25", Ano Novo = "01-01", Réveillon = "12-31"), or "YYYY-MM-DD" for a
        specific year.
        For dates relative to today ("daqui a um ano", "ontem", "há 3 meses") use
        `date_offset` instead — never guess the target date.
        """
        today = _now().date()
        target = _parse_target(target_date, today)
        return (
            f"days: {(target - today).days}\n"
            f"from: {today.isoformat()} ({weekdays[today.weekday()]})\n"
            f"to: {target.isoformat()} ({weekdays[target.weekday()]})"
        )

    @tool
    def date_offset(days: int = 0, weeks: int = 0, months: int = 0, years: int = 0) -> str:
        """Get the date (and weekday) of ANY day other than today, relative to today.

        Always use it — never compute dates or weekdays yourself — for:
        "ontem" (days=-1), "amanhã" (days=1), "anteontem" (days=-2),
        "daqui a duas semanas" (weeks=2), "daqui a 100 dias" (days=100),
        "há 3 meses" (months=-3), "daqui a um ano" (years=1).
        Negative values go into the past.
        Not for "quantos dias faltam para <data ou feriado>" — that is `days_until`.
        """
        today = _now().date()
        target = shift_date(today, days=days, weeks=weeks, months=months, years=years)
        return (
            f"date: {target.isoformat()} ({weekdays[target.weekday()]})\n"
            f"today: {today.isoformat()} ({weekdays[today.weekday()]})\n"
            f"days_from_today: {(target - today).days}"
        )

    return [get_current_datetime, days_until, date_offset]
