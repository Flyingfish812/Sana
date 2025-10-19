"""FastAPI application exposing a lightweight interface layer."""
from __future__ import annotations

import asyncio
import copy
import json
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.dataio.api import run as build_dataset_pipeline
from backend.train.runner import run_training

from .config import ConfigPayload
from .events import stream_function
from .state import STATE
from .viz import run_one_click_check


app = FastAPI(title="Sana Backend Interface", version="0.1.0")


class DataRunRequest(ConfigPayload):
    """Request payload for the data preparation endpoint."""


class DataRunResponse(BaseModel):
    summary: Dict[str, Any]
    dataloader_type: str
    split_keys: Optional[List[str]] = None
    config_source: str


@app.post("/data/run", response_model=DataRunResponse)
def run_data_pipeline(payload: DataRunRequest) -> DataRunResponse:
    cfg = copy.deepcopy(payload.load())
    dataset, dataloaders, summary = build_dataset_pipeline(cfg)

    STATE.data.config = cfg
    STATE.data.dataset = dataset
    STATE.data.dataloaders = dataloaders
    STATE.data.summary = summary

    if isinstance(dataloaders, dict):
        dataloader_type = "dict"
        split_keys = list(dataloaders.keys())
    else:
        dataloader_type = "single"
        split_keys = None

    return DataRunResponse(
        summary=summary,
        dataloader_type=dataloader_type,
        split_keys=split_keys,
        config_source=payload.describe_source(),
    )


class VisualizationRequest(BaseModel):
    channel: Optional[Union[int, str]] = None
    n_batches: int = 2
    with_sizecheck: bool = True


class VisualizationResponse(BaseModel):
    stdout: List[str]
    stderr: List[str]
    figures: List[str]


@app.post("/viz/one-click", response_model=VisualizationResponse)
def visualize_data(payload: VisualizationRequest) -> VisualizationResponse:
    if STATE.data.dataset is None or STATE.data.dataloaders is None:
        raise HTTPException(status_code=400, detail="dataset and dataloaders are not initialised; run /data/run first")

    result = run_one_click_check(
        STATE.data.dataset,
        STATE.data.dataloaders,
        channel=payload.channel,
        n_batches=payload.n_batches,
        with_sizecheck=payload.with_sizecheck,
    )

    return VisualizationResponse(**result)


class TrainingRequest(ConfigPayload):
    disable_progress_bar: bool = True


async def _training_event_stream(cfg: Dict[str, Any]) -> AsyncIterator[bytes]:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict] = asyncio.Queue()

    def emit(event: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    consumer_task = asyncio.create_task(stream_function(lambda: run_training(cfg), emit))

    try:
        while True:
            event = await queue.get()
            if event.get("event") == "completed" and isinstance(event.get("result"), (list, tuple)):
                result = event["result"]
                if len(result) == 2:
                    model, artefacts = result
                    STATE.training.model = model
                    STATE.training.artefacts = artefacts
                    STATE.training.config = cfg
                    event = {"event": "completed", "artefacts": artefacts}
            data = (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")
            yield data
            if event.get("event") in {"completed", "error"}:
                break
    finally:
        await consumer_task


@app.post("/train/run")
async def run_training_pipeline(payload: TrainingRequest) -> StreamingResponse:
    raw_cfg = payload.load()
    cfg = copy.deepcopy(raw_cfg)

    if payload.disable_progress_bar:
        trainer_cfg = cfg.setdefault("trainer", {})
        trainer_cfg["enable_progress_bar"] = False

    stream = _training_event_stream(cfg)
    return StreamingResponse(stream, media_type="application/x-ndjson")


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}
