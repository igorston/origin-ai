from fastapi import APIRouter
from pydantic import BaseModel

from origin.api.routes.chat import Engine

router = APIRouter(prefix="/tools", tags=["tools"])


class ToolInfo(BaseModel):
    name: str
    description: str
    args: dict


@router.get("", response_model=list[ToolInfo])
async def list_tools(engine: Engine) -> list[ToolInfo]:
    return [
        ToolInfo(name=tool.name, description=tool.description, args=tool.args)
        for tool in engine.tools.values()
    ]
