"""Typed transport refusal, emitted only before a guarded HTTP dispatch.

This marker is not a reservation refund. A transport guard must only raise it
when it has refused the request before entering the HTTP implementation. All
other exceptions remain attempted requests for conservative deduplication.
"""
from __future__ import annotations


class ProviderPreDispatchBlocked(RuntimeError):
    """A trusted transport guard proved that it did not dispatch this call."""

    _REASONS = frozenset({"http_budget_exhausted", "http_budget_unavailable"})

    def __init__(self, reason: str) -> None:
        if reason not in self._REASONS:
            raise ValueError("unsupported_pre_dispatch_refusal")
        self.reason = reason
        super().__init__(reason)
