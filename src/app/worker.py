import logging
from datetime import UTC, datetime
from multiprocessing import Process, Queue

from joblib import parallel_backend

from app.bootstrap import Container
from app.identification import Identification
from domain.types import Job, JobType


class Worker:
    def __init__(self, container_factory, manager):
        self.queue: Queue = Queue()
        self.container_factory = container_factory
        self.process: Process | None = None
        self.jobs = manager.dict()

    def start(self):
        if self.process and self.process.is_alive():
            return

        self.process = Process(
            target=self._run,
            daemon=True,
        )

        self.process.start()

    def submit(self, job: Job):
        self.jobs[job.id] = {
            "id": job.id,
            "type": job.type.value,
            "state": "queued",
            "created_at": datetime.now(UTC).isoformat(),
        }

        logging.info(
            "Job queued: id=%s type=%s",
            job.id[:6],
            job.type.value,
        )

        self.queue.put(job)

    def stop(self):
        if not self.process:
            return

        self.queue.put(None)

        self.process.join(timeout=10)

        if self.process.is_alive():
            self.process.terminate()

    def _run(self):
        with parallel_backend("threading", n_jobs=1):
            container = self.container_factory()

            logging.info("Worker started")

            while True:
                try:
                    job = self.queue.get()
                except (KeyboardInterrupt, SystemExit):
                    logging.info("Worker interrupted by shutdown")
                    break

                if job is None:
                    logging.info("Worker stopped")
                    break

                self._update_job(
                    job.id,
                    state="running",
                )

                logging.info(
                    "Job started: id=%s type=%s",
                    job.id[:6],
                    job.type.value,
                )

                try:
                    self._execute(container, job)

                    logging.info(
                        "Job completed: id=%s type=%s",
                        job.id[:6],
                        job.type.value,
                    )

                except Exception:
                    logging.exception(
                        "Job failed: id=%s type=%s",
                        job.id[:6],
                        job.type.value,
                    )

                finally:
                    del self.jobs[job.id]

    def _update_job(self, job_id: str, **kwargs):
        job = dict(self.jobs[job_id])
        job.update(kwargs)
        self.jobs[job_id] = job

    def _execute(self, container, job: Job):
        match job.type:
            case JobType.CONFIG:
                container.config_repository.save(job.config)
            case JobType.UPDATE:
                container.state_manager.update()
                container.backtest_repository.clear()
            case JobType.FIT:
                container.forecasting.fit(job.config)
            case JobType.PREDICT:
                container.forecasting.predict(job.config)
            case JobType.TUNE:
                container.forecasting.tune(job.config)
            case JobType.BACKTEST:
                container.forecasting.backtest(job.config)
            case JobType.CALIBRATE:
                container.identification.calibrate(job.config)
            case JobType.VALIDATE:
                container.identification.validate(job.config)
            case JobType.OPTIMIZE:
                container.optimization.run()
            case _:
                raise NotImplementedError(f"Unknown job type={job.type}")
