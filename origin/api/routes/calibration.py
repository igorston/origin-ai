from typing import Annotated

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from origin.api.routes.chat import get_workspace
from origin.memory.calibration import (
    CalibrationReport,
    IndexStatus,
    MemoryCalibrator,
    ReindexResult,
    Thresholds,
)

router = APIRouter(prefix="/memory", tags=["memory calibration"])


def get_calibrator(request: Request) -> MemoryCalibrator:
    return get_workspace(request).calibrator


Calibrator = Annotated[MemoryCalibrator, Depends(get_calibrator)]


class CalibrationRun(BaseModel):
    apply: bool = False  # apply the suggested thresholds right away


@router.get("/calibration", response_model=IndexStatus)
async def calibration_status(calibrator: Calibrator) -> IndexStatus:
    """Which embedding model indexed the memories, whether a reindex is needed, and the
    thresholds in use (calibrated for the current model or defaults from settings)."""
    return await calibrator.status()


@router.post("/calibration/run", response_model=CalibrationReport)
async def run_calibration(
    calibrator: Calibrator, body: CalibrationRun | None = None
) -> CalibrationReport:
    """Measure the current embedding and chat models on a built-in labeled set and
    suggest thresholds; with `apply`, use them from now on."""
    return await calibrator.calibrate(apply=(body or CalibrationRun()).apply)


@router.put("/calibration/thresholds", response_model=Thresholds)
async def set_thresholds(thresholds: Thresholds, calibrator: Calibrator) -> Thresholds:
    return calibrator.apply(thresholds)


@router.delete("/calibration/thresholds", response_model=Thresholds)
async def reset_thresholds(calibrator: Calibrator) -> Thresholds:
    """Go back to the thresholds from the settings (.env)."""
    return calibrator.reset()


@router.post("/reindex", response_model=ReindexResult)
async def reindex(calibrator: Calibrator) -> ReindexResult:
    """Re-embed every memory with the current embedding model (after changing it). A JSON
    backup is written first, and the old vectors are only dropped once all new ones exist."""
    return await calibrator.reindex()
