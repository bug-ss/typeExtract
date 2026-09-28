"""Error taxonomy. ``retryable`` errors are retried by the backend; the rest surface at once."""

from __future__ import annotations


class TypeExtractError(Exception):
    """Base class for all typeextract errors."""


class ConfigurationError(TypeExtractError):
    """Missing API key, bad option, ..."""


class BudgetExceededError(TypeExtractError):
    """The configured ``max_cost_usd`` would be exceeded by the next request."""


class ResponseValidationError(TypeExtractError):
    """The API answered with something that does not match the question asked."""


class ExtractionError(TypeExtractError):
    """A document failed with ``on_error="raise"``. ``document`` holds the partial result."""

    def __init__(self, message: str, document: object = None):
        super().__init__(message)
        self.document = document


class APIError(TypeExtractError):
    retryable = False

    def __init__(
        self,
        message: str,
        status: int | None = None,
        request_id: str | None = None,
        retry_after: float | None = None,
    ):
        detail = f"{status} {message}" if status else message
        if request_id:
            detail += f" (request_id={request_id})"
        super().__init__(detail)
        self.status = status
        self.request_id = request_id
        self.retry_after = retry_after
        self.retries = 0  # how many times the backend retried before giving up


class AuthenticationError(APIError):
    """401/403: the key is missing, invalid or not allowed. Never retried."""


class RequestTooLargeError(APIError):
    """The request exceeds a server limit (tokens, questions, payload). Split and retry."""


class InvalidRequestError(APIError):
    """400/422 for another reason: a malformed question. Not retried."""


class RateLimitError(APIError):
    """429 after all retries."""

    retryable = True


class ServerError(APIError):
    """5xx / 529 overloaded / 408 after all retries."""

    retryable = True


class TransportError(APIError):
    """Connection failure or timeout after all retries."""

    retryable = True


FATAL_ERRORS: tuple[type[Exception], ...] = (
    AuthenticationError,
    BudgetExceededError,
    ConfigurationError,
)
"""Errors that stop a whole run even with ``on_error="skip"`` (every window would fail the same way)."""
