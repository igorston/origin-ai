from typing import Annotated

from fastapi import APIRouter, Depends, Response, status
from pydantic import BaseModel

from origin import __version__
from origin.config import Settings, get_settings
from origin.core.ollama import OllamaStatus, check_ollama

router = APIRouter(tags=["system"])


class Health(BaseModel):
    status: str
    version: str
    ollama: OllamaStatus


@router.get("/health", response_model=Health)
async def health(
    response: Response, settings: Annotated[Settings, Depends(get_settings)]
) -> Health:
    ollama = await check_ollama(settings)
    if not ollama.healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return Health(status="ok" if ollama.healthy else "degraded", version=__version__, ollama=ollama)
