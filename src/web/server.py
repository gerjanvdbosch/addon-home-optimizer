import logging
from contextlib import asynccontextmanager
from multiprocessing import Manager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import JSONResponse

from app.bootstrap import create_container
from app.worker import Worker
from domain.types import (
    BacktestConfig,
    CalibrateConfig,
    Config,
    FitConfig,
    Job,
    JobType,
    OptimizeConfig,
    PredictConfig,
    TuneConfig,
    UpdateConfig,
    ValidateConfig,
)
from web.charts import backtest_chart, dashboard_chart

BASE_DIR = Path(__file__).resolve().parent

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.container = create_container()

    app.state.manager = Manager()

    app.state.worker = Worker(
        create_container,
        app.state.manager,
    )

    app.state.worker.start()

    yield

    app.state.worker.stop()
    app.state.manager.shutdown()


app = FastAPI(
    lifespan=lifespan,
)

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static",
)

templates = Jinja2Templates(
    directory=BASE_DIR / "templates",
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, error: RequestValidationError):
    logger.error("%s %s: %s", request.method, request.url.path, error)

    return JSONResponse(
        status_code=422,
        content={"detail": error.errors()},
    )


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    container = request.app.state.container

    try:
        container.config_repository.load()
    except Exception:
        return templates.TemplateResponse(request=request, name="setup.html")

    state = container.state_manager.load()
    backtest = container.backtest_repository.load()

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "dashboard_chart": dashboard_chart(state),
            "backtest_chart": backtest_chart(backtest),
        },
    )


@app.get("/api/status")
async def status(request: Request):
    return list(request.app.state.worker.jobs.values())


@app.get("/api/state")
async def state(request: Request):
    return request.app.state.container.state_manager.load()


@app.post("/api/config")
async def config(request: Request, config: Config):
    job = Job(type=JobType.CONFIG, config=config)

    request.app.state.worker.submit(job)

    return {
        "job_id": job.id,
        "state": "queued",
    }


@app.post("/api/update")
async def update(request: Request, config: UpdateConfig):
    job = Job(type=JobType.UPDATE, config=config)

    request.app.state.worker.submit(job)

    return {
        "job_id": job.id,
        "state": "queued",
    }


@app.post("/api/fit")
async def fit(request: Request, config: FitConfig):
    job = Job(type=JobType.FIT, config=config)

    request.app.state.worker.submit(job)

    return {
        "job_id": job.id,
        "state": "queued",
    }


@app.post("/api/predict")
async def predict(request: Request, config: PredictConfig):
    job = Job(type=JobType.PREDICT, config=config)

    request.app.state.worker.submit(job)

    return {
        "job_id": job.id,
        "state": "queued",
    }


@app.post("/api/backtest")
async def backtest(request: Request, config: BacktestConfig):
    job = Job(type=JobType.BACKTEST, config=config)

    request.app.state.worker.submit(job)

    return {
        "job_id": job.id,
        "state": "queued",
    }


@app.post("/api/tune")
async def tune(request: Request, config: TuneConfig):
    job = Job(type=JobType.TUNE, config=config)

    request.app.state.worker.submit(job)

    return {
        "job_id": job.id,
        "state": "queued",
    }


@app.post("/api/calibrate")
async def calibrate(request: Request, config: CalibrateConfig):
    job = Job(type=JobType.CALIBRATE, config=config)

    request.app.state.worker.submit(job)

    return {
        "job_id": job.id,
        "state": "queued",
    }


@app.post("/api/validate")
async def validate(request: Request, config: ValidateConfig):
    job = Job(type=JobType.VALIDATE, config=config)

    request.app.state.worker.submit(job)

    return {
        "job_id": job.id,
        "state": "queued",
    }


@app.post("/api/optimize")
async def optimize(request: Request, config: OptimizeConfig):
    job = Job(type=JobType.OPTIMIZE, config=config)

    request.app.state.worker.submit(job)

    return {
        "job_id": job.id,
        "state": "queued",
    }
