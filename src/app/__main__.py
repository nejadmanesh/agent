"""Command-line interface for the Agent project scaffold with built-in metrics support."""

from __future__ import annotations

import logging
import os
import random
import signal
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Lock, Thread
from typing import Dict, Iterable, Mapping, Tuple

Labels = Tuple[Tuple[str, str], ...]


@dataclass
class MetricBase:
    """Base class representing a Prometheus metric."""

    name: str
    description: str
    metric_type: str
    _values: Dict[Labels, float] = field(default_factory=dict)

    def snapshot(self) -> Iterable[Tuple[Labels, float]]:
        return tuple(self._values.items())


_METRICS: Dict[str, MetricBase] = {}
_METRICS_LOCK = Lock()


def _normalize_labels(labels: Mapping[str, str] | None) -> Labels:
    if not labels:
        return ()
    return tuple(sorted(labels.items()))


def _register_metric(metric: MetricBase) -> None:
    with _METRICS_LOCK:
        if metric.name in _METRICS:
            raise ValueError(f"Metric {metric.name!r} is already registered")
        _METRICS[metric.name] = metric


class CounterMetric(MetricBase):
    def __init__(self, name: str, description: str) -> None:
        super().__init__(name=name, description=description, metric_type="counter")
        _register_metric(self)

    def inc(self, amount: float = 1.0, labels: Mapping[str, str] | None = None) -> None:
        key = _normalize_labels(labels)
        with _METRICS_LOCK:
            self._values[key] = self._values.get(key, 0.0) + amount


class GaugeMetric(MetricBase):
    def __init__(self, name: str, description: str) -> None:
        super().__init__(name=name, description=description, metric_type="gauge")
        _register_metric(self)

    def set(self, value: float, labels: Mapping[str, str] | None = None) -> None:
        key = _normalize_labels(labels)
        with _METRICS_LOCK:
            self._values[key] = value


def _render_metrics() -> str:
    with _METRICS_LOCK:
        snapshot = [
            (
                metric.name,
                metric.description,
                metric.metric_type,
                tuple(sorted(metric.snapshot(), key=lambda item: item[0])),
            )
            for metric in _METRICS.values()
        ]

    lines: list[str] = []
    for name, description, metric_type, values in snapshot:
        lines.append(f"# HELP {name} {description}")
        lines.append(f"# TYPE {name} {metric_type}")
        for labels, value in values:
            label_suffix = ""
            if labels:
                label_pairs = ",".join(f'{k}="{v}"' for k, v in labels)
                label_suffix = f"{{{label_pairs}}}"
            lines.append(f"{name}{label_suffix} {value}")
    lines.append("")
    return "\n".join(lines)


class MetricsHandler(BaseHTTPRequestHandler):
    server_version = "AgentMetrics/1.0"

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        if self.path not in ("/metrics", "/metrics/"):
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
            return

        payload = _render_metrics().encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:  # noqa: D401, N802
        """Emit metrics server logs at debug level."""
        logging.getLogger(__name__).debug("Metrics server: " + format, *args)


def _start_metrics_server(port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("0.0.0.0", port), MetricsHandler)
    thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.5}, daemon=True)
    thread.start()
    return server


_LOOP_ITERATIONS = CounterMetric(
    "app_loop_iterations_total",
    "Total number of work loop iterations completed.",
)
_ITERATION_TIME_TOTAL = CounterMetric(
    "app_iteration_seconds_total",
    "Accumulated processing time spent handling work iterations in seconds.",
)
_TEMPERATURE = GaugeMetric(
    "app_temperature_celsius",
    "Simulated application temperature in Celsius.",
)
_APP_HEALTH = GaugeMetric(
    "app_health_status",
    "Health indicator for the long-running worker (1=healthy, 0=unhealthy).",
)
_LAST_ITERATION = GaugeMetric(
    "app_last_iteration_seconds",
    "Duration of the most recent work iteration in seconds.",
)
_APP_INFO = GaugeMetric(
    "app_build_info",
    "Build metadata for the application.",
)

DEFAULT_METRICS_PORT = 8000
_SLEEP_INTERVAL_SECONDS = 5.0


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def _get_metrics_port() -> int:
    raw_port = os.getenv("METRICS_PORT", str(DEFAULT_METRICS_PORT))
    try:
        return int(raw_port)
    except ValueError:
        logging.warning(
            "Invalid METRICS_PORT=%s received; falling back to default %s.",
            raw_port,
            DEFAULT_METRICS_PORT,
        )
        return DEFAULT_METRICS_PORT


def _register_signal_handlers(stop_event: Event) -> None:
    def _handler(signum: int, _frame: object) -> None:
        logging.info("Received signal %s. Shutting down gracefully.", signum)
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _handler)


def _simulate_iteration() -> None:
    start = time.perf_counter()
    # Simulate work by sleeping a random duration and producing telemetry.
    duration = random.uniform(0.25, 1.0)
    time.sleep(duration)

    _LOOP_ITERATIONS.inc()
    _TEMPERATURE.set(24.0 + random.uniform(-3.0, 3.0))

    elapsed = time.perf_counter() - start
    _ITERATION_TIME_TOTAL.inc(elapsed)
    _LAST_ITERATION.set(elapsed)


def _run_worker(stop_event: Event) -> None:
    while not stop_event.is_set():
        try:
            _simulate_iteration()
            _APP_HEALTH.set(1.0)
        except Exception:  # pragma: no cover - defensive safeguard
            _APP_HEALTH.set(0.0)
            logging.exception("Unexpected error while running worker iteration.")
            time.sleep(1.0)
        stop_event.wait(_SLEEP_INTERVAL_SECONDS)


def main() -> None:
    """Entry point for the application."""
    _configure_logging()

    stop_event = Event()
    _register_signal_handlers(stop_event)

    metrics_port = _get_metrics_port()
    metrics_server = _start_metrics_server(metrics_port)

    app_version = os.getenv("APP_VERSION", "dev")
    _APP_INFO.set(1.0, labels={"version": app_version})

    logging.info(
        "Agent application started. Metrics are exposed on port %s.",
        metrics_port,
    )

    try:
        _run_worker(stop_event)
    except KeyboardInterrupt:
        logging.info("Interrupted by user; shutting down.")
    finally:
        _APP_HEALTH.set(0.0)
        metrics_server.shutdown()
        metrics_server.server_close()
        logging.info("Application shutdown complete.")


if __name__ == "__main__":
    main()
