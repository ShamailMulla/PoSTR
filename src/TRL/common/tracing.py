"""Sampled MLflow tracing of BTRL update cycles.

Tracing is on only when MLFLOW_RUN_ID is set (see run_v3_1M.py) and only for one update
cycle in every TRACE_EVERY (default 20), so untraced cycles pay nothing. Inside a traced
cycle, functions decorated with @traced become child spans; tensors are summarised as
shape/mean/std/min/max rather than logged in full.
"""
import functools
import os
import time

import numpy as np
import torch

ENABLED = bool(os.environ.get("MLFLOW_RUN_ID"))
EVERY = int(os.environ.get("TRACE_EVERY", 20))

_active = False
_calls = {}
_cycles = 0

if ENABLED:
    import mlflow


def summarize(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().float()
        if x.numel() == 0:
            return {"shape": list(x.shape)}
        return {"shape": list(x.shape), "mean": x.mean().item(), "std": x.std().item() if x.numel() > 1 else 0.0,
                "min": x.min().item(), "max": x.max().item()}
    if isinstance(x, np.ndarray):
        return summarize(torch.as_tensor(x))
    if isinstance(x, (list, tuple)):
        return [summarize(v) for v in x[:8]]
    if isinstance(x, dict):
        return {str(k): summarize(v) for k, v in list(x.items())[:32]}
    if isinstance(x, (int, float, str, bool)) or x is None:
        return x
    return type(x).__name__


def traced(name=None, span_type="UNKNOWN", max_calls=None, capture_args=True):
    """Span for the wrapped function while a cycle is being traced.
    `max_calls` caps spans per cycle for functions called in inner loops."""
    def deco(fn):
        span_name = name or fn.__qualname__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not _active:
                return fn(*args, **kwargs)
            n = _calls.get(span_name, 0)
            _calls[span_name] = n + 1
            if max_calls is not None and n >= max_calls:
                return fn(*args, **kwargs)
            with mlflow.start_span(span_name, span_type=span_type) as span:
                if capture_args:
                    span.set_inputs({f"arg{i}": summarize(a) for i, a in enumerate(args[1:])}
                                    | {k: summarize(v) for k, v in kwargs.items()})
                out = fn(*args, **kwargs)
                span.set_outputs(summarize(out))
                return out
        return wrapper
    return deco


class cycle:
    """Root span for one update cycle; a no-op unless this cycle is sampled."""

    def __init__(self, timestep, **attrs):
        global _cycles
        _cycles += 1
        self.on = ENABLED and (_cycles == 1 or _cycles % EVERY == 0)
        self.timestep, self.attrs, self.outputs = timestep, attrs, {}

    def __enter__(self):
        global _active
        if self.on:
            self._cm = mlflow.start_span("BTRL.update_cycle", span_type="CHAIN")
            self.span = self._cm.__enter__()
            self.span.set_attributes({"timestep": self.timestep, "cycle": _cycles, **self.attrs})
            self.span.set_inputs({"timestep": self.timestep})
            try:  # searchable in the Traces tab
                mlflow.update_current_trace(tags={"timestep": str(self.timestep), "cycle": str(_cycles),
                                                  **{k: str(v) for k, v in self.attrs.items()}})
            except Exception:
                pass
            _active = True
            _calls.clear()
            self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        global _active
        if self.on:
            _active = False
            self.outputs["cycle_seconds"] = round(time.time() - self.t0, 3)
            self.span.set_outputs(self.outputs)
            self._cm.__exit__(*exc)
        return False
