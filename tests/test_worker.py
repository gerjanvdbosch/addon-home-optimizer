import logging

from app.worker import Worker
from domain.jobs import Job, JobType, UpdateConfig


def test_a_failing_job_is_reported_and_leaves_the_worker_running(
    tmp_path, monkeypatch, caplog
):
    """A job runs in its own process: its exception must surface as a failed
    job, not take the queue down with it."""

    # An empty data directory: the update job fails on the missing config
    # before it touches the database.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LATITUDE", "53.2")
    monkeypatch.setenv("LONGITUDE", "6.5")

    worker = Worker()
    job = Job(type=JobType.UPDATE, config=UpdateConfig())
    worker.submit(job)
    worker.queue.put(None)

    with caplog.at_level(logging.INFO):
        worker._run()

    assert "Job failed" in caplog.text
    assert "exit code=1" in caplog.text
    assert worker.jobs == {}


def test_jobs_queued_when_stopping_are_not_started(caplog):
    """On shutdown the queue is dropped: a job started now would only be
    terminated again."""

    worker = Worker()
    job = Job(type=JobType.UPDATE, config=UpdateConfig())
    worker.submit(job)
    # As stop() leaves it, with the job still ahead of the stop in the queue.
    worker.stopping = True
    worker.queue.put(None)

    with caplog.at_level(logging.INFO):
        worker._run()

    assert worker.process is None
    assert "Job started" not in caplog.text
    assert "Worker stopped" in caplog.text
    assert worker.jobs == {}
