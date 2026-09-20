import logging
import sys
from datetime import UTC, datetime
from multiprocessing import get_context
from multiprocessing.context import SpawnProcess
from queue import Queue
from threading import Thread

from domain.jobs import Job, JobType


class Worker:
    """Runs queued jobs one at a time, each in its own process.

    A job's process builds the container and exits when the job is done, so the
    modelling libraries (~250 MB) and whatever the job loaded are returned to
    the operating system - a long-lived worker kept all of it, since the
    allocator rarely hands freed memory back. Queueing is a thread of the web
    process (a daemon process may not start processes of its own).

    Jobs are spawned, not forked: the web process runs several threads, and a
    fork copies any lock another thread holds at that moment (e.g. logging's),
    which can hang the job - and with it the queue - for good.
    """

    def __init__(self):
        self.queue: Queue[Job | None] = Queue()
        self.thread: Thread | None = None
        self.process: SpawnProcess | None = None
        self.jobs: dict[str, dict] = {}

    def start(self):
        if self.thread and self.thread.is_alive():
            return

        self.thread = Thread(target=self._run, daemon=True)

        self.thread.start()

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
        if not self.thread:
            return

        self.queue.put(None)

        if self.process and self.process.is_alive():
            self.process.terminate()

        self.thread.join(timeout=10)

    def _run(self):
        logging.info("Worker started")

        while True:
            job = self.queue.get()

            if job is None:
                logging.info("Worker stopped")
                break

            self.jobs[job.id]["state"] = "running"

            logging.info(
                "Job started: id=%s type=%s",
                job.id[:6],
                job.type.value,
            )

            try:
                self.process = get_context("spawn").Process(
                    target=run_job, args=(job,), daemon=True
                )
                self.process.start()
                self.process.join()

                if self.process.exitcode == 0:
                    logging.info(
                        "Job completed: id=%s type=%s",
                        job.id[:6],
                        job.type.value,
                    )
                else:
                    logging.error(
                        "Job failed: id=%s type=%s exit code=%s",
                        job.id[:6],
                        job.type.value,
                        self.process.exitcode,
                    )

            except Exception:
                logging.exception(
                    "Job failed: id=%s type=%s",
                    job.id[:6],
                    job.type.value,
                )

            finally:
                del self.jobs[job.id]


def run_job(job: Job):
    # Imported here, in the job's process, so the web process never loads the
    # modelling libraries.
    from joblib import parallel_backend

    from app.bootstrap import create_container

    try:
        with parallel_backend("threading", n_jobs=1):
            execute(create_container(), job)
    except Exception:
        logging.exception("Job raised: id=%s type=%s", job.id[:6], job.type.value)
        sys.exit(1)


def execute(container, job: Job):
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
            container.optimization.run(job.config)
        case _:
            raise NotImplementedError(f"Unknown job type={job.type}")
