"""FastAPI application. Demo only: API-key protection is not fit for public deployment.

Run: uvicorn claimguard.api.main:app
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from claimguard.api.db import get_engine
from claimguard.api.routes import router

_CODES = {400: "bad_request", 401: "unauthorized", 404: "not_found", 409: "conflict"}


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    get_engine()  # creates tables
    yield


app = FastAPI(
    title="ClaimGuard",
    description="Pre-submission denial risk and documentation checker on synthetic data. "
    "Flags risk only; never approves or denies a claim.",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(router)


@app.exception_handler(HTTPException)
async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
    code = _CODES.get(exc.status_code, "error")
    return JSONResponse(
        status_code=exc.status_code, content={"error": {"code": code, "message": exc.detail}}
    )


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    details = [
        {"loc": list(e.get("loc", [])), "msg": e.get("msg", ""), "type": e.get("type", "")}
        for e in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "validation_error",
                "message": "request validation failed",
                "details": details,
            }
        },
    )
