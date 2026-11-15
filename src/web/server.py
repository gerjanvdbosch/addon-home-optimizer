import logging
from contextlib import asynccontextmanager
from pathlib import Path

import plotly
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import JSONResponse

from app.settings import configure_logger, create_repositories, load_settings
from app.worker import Worker
from domain.config import Config
from domain.jobs import (
    CalibrateConfig,
    Job,
    JobType,
    OptimizeConfig,
    UpdateConfig,
    ValidateConfig,
    WorkerState,
)
from web.charts import dashboard_chart

BASE_DIR = Path(__file__).resolve().parent

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()

    configure_logger(settings.log_level)

    app.state.repositories = create_repositories(settings)

    app.state.worker = Worker()

    app.state.worker.start()

    yield

    app.state.worker.stop()


app = FastAPI(
    lifespan=lifespan,
)

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static",
)

# plotly.js from the installed plotly package: no CDN needed, nothing vendored.
app.mount(
    "/plotly",
    StaticFiles(directory=Path(plotly.__file__).parent / "package_data"),
    name="plotly",
)

templates = Jinja2Templates(
    directory=BASE_DIR / "templates",
)
# The config page's form follows the schema's field order, which tojson
# would otherwise sort.
templates.env.policies["json.dumps_kwargs"] = {"sort_keys": False}


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, error: RequestValidationError):
    logger.error("%s %s: %s", request.method, request.url.path, error)

    return JSONResponse(
        status_code=422,
        content={"detail": error.errors()},
    )


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    repositories = request.app.state.repositories

    try:
        config = repositories.config.load()
    except Exception:
        return config_page(request)

    state = repositories.state.load()

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "dashboard_chart": dashboard_chart(state, config.heat_pump.states),
        },
    )


@app.get("/config", response_class=HTMLResponse)
async def config_form(request: Request):
    return config_page(request)


def config_page(request: Request) -> HTMLResponse:
    """The form is built from the config's own schema, so it follows every
    field the model has. It is filled with the file as stored, not as
    validated: a config that no longer validates is shown to be fixed."""

    return templates.TemplateResponse(
        request=request,
        name="config.html",
        context={
            "schema": Config.model_json_schema(),
            "config": request.app.state.repositories.config.storage.load(),
        },
    )


@app.get("/api/status")
async def status(request: Request):
    return list(request.app.state.worker.jobs.values())


@app.get("/api/state")
async def state(request: Request):
    return request.app.state.repositories.state.load()


@app.get("/api/config")
async def get_config(request: Request) -> Config:
    """Validated, so with the defaults filled in: the form POST takes back."""
    data = request.app.state.repositories.config.storage.load()

    if not data:
        raise HTTPException(status_code=404, detail="No config found")

    return Config.model_validate(data)


@app.post("/api/config")
async def config(request: Request, config: Config):
    job = Job(type=JobType.CONFIG, config=config)

    request.app.state.worker.submit(job)

    return {
        "job_id": job.id,
        "state": WorkerState.QUEUED,
    }


@app.post("/api/update")
async def update(request: Request, config: UpdateConfig):
    job = Job(type=JobType.UPDATE, config=config)

    request.app.state.worker.submit(job)

    return {
        "job_id": job.id,
        "state": WorkerState.QUEUED,
    }


@app.post("/api/calibrate")
async def calibrate(request: Request, config: CalibrateConfig):
    job = Job(type=JobType.CALIBRATE, config=config)

    request.app.state.worker.submit(job)

    return {
        "job_id": job.id,
        "state": WorkerState.QUEUED,
    }


@app.post("/api/validate")
async def validate(request: Request, config: ValidateConfig):
    job = Job(type=JobType.VALIDATE, config=config)

    request.app.state.worker.submit(job)

    return {
        "job_id": job.id,
        "state": WorkerState.QUEUED,
    }


@app.post("/api/optimize")
async def optimize(request: Request, config: OptimizeConfig):
    job = Job(type=JobType.OPTIMIZE, config=config)

    request.app.state.worker.submit(job)

    return {
        "job_id": job.id,
        "state": WorkerState.QUEUED,
    }
