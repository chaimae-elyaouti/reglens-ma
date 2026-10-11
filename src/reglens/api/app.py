"""API FastAPI.

    uvicorn reglens.api.app:create_app --factory --reload

Endpoints : POST /ask, GET /chunks/{id} (inspection d'une source), GET /health, GET /version.
Les clés API ne vivent QUE côté serveur : le frontend n'en voit jamais.
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from reglens import __version__
from reglens.api.ratelimit import DailyIPLimiter
from reglens.config import Settings, get_settings
from reglens.errors import InvalidRequest, LLMUnavailable
from reglens.logging_setup import log_event, request_id_var, setup_logging
from reglens.schemas import AskRequest, AskResponse
from reglens.service import RagService, build_service


def create_app(service: RagService | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.log_level)
    limiter = DailyIPLimiter(settings.rate_limit_per_ip_per_day)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if getattr(app.state, "service", None) is None:
            app.state.service = build_service(settings)  # charge l'index et le modèle une seule fois
        yield

    app = FastAPI(title="RegLens MA", version=__version__, lifespan=lifespan)
    app.state.service = service
    app.add_middleware(
        CORSMiddleware, allow_origins=settings.origins_list, allow_methods=["GET", "POST"], allow_headers=["*"]
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        token = request_id_var.set(rid)
        t0 = perf_counter()
        try:
            if request.method == "POST" and request.url.path == "/ask":
                fwd = request.headers.get("x-forwarded-for", "")
                ip = fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "?")
                if not limiter.allow(ip):
                    log_event("rate_limited", level=logging.WARNING, ip_hash=hash(ip) % 10**8)
                    return JSONResponse(
                        {"error": "rate_limited", "detail": "Limite quotidienne atteinte.", "request_id": rid},
                        status_code=429,
                    )
            response = await call_next(request)
        except Exception:
            logging.getLogger("reglens").exception("unhandled_error")
            return JSONResponse({"error": "internal_error", "request_id": rid}, status_code=500)
        finally:
            log_event("http", path=request.url.path, method=request.method,
                      ms=round((perf_counter() - t0) * 1000, 1))
            request_id_var.reset(token)
        response.headers["x-request-id"] = rid
        return response

    @app.exception_handler(LLMUnavailable)
    async def llm_unavailable(request: Request, exc: LLMUnavailable):
        return JSONResponse(
            {"error": "llm_unavailable", "detail": str(exc), "request_id": request_id_var.get()},
            status_code=503,
        )

    @app.exception_handler(InvalidRequest)
    async def invalid_request(request: Request, exc: InvalidRequest):
        return JSONResponse(
            {"error": "invalid_request", "detail": str(exc), "request_id": request_id_var.get()},
            status_code=422,
        )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/version")
    def version(request: Request):
        svc = request.app.state.service
        m = svc.manifest if svc else {}
        return {
            "app": __version__,
            "index_version": m.get("index_version"),
            "embedding_model": m.get("embedding_model"),
            "corpus_as_of": m.get("corpus_as_of"),
            "n_chunks": m.get("n_chunks"),
        }

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest, request: Request):
        if len(req.question) > settings.max_question_chars:
            raise InvalidRequest(f"Question trop longue (max {settings.max_question_chars} caractères).")
        return request.app.state.service.ask(req, request_id=request_id_var.get())

    @app.get("/chunks/{chunk_id}")
    def chunk(chunk_id: str, request: Request):
        found = request.app.state.service.get_chunk(chunk_id)
        if found is None:
            raise HTTPException(status_code=404, detail="Passage introuvable.")
        return found

    return app
