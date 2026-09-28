"""typeextract: grounded extraction with typed-decision models.

Code proposes spans with exact offsets; Jev (TypeSafe's System One model) decides which of them
are mentions of your classes. Nothing is generated, so every extraction is a substring of the
source by construction.
"""

__version__ = "0.1.0"

from .asker import Limits
from .cache import MemoryCache, SQLiteCache
from .data import (
    AnnotatedDocument,
    CharInterval,
    ExampleData,
    Extraction,
    Metrics,
    load_jsonl,
    save_jsonl,
)
from .errors import (
    APIError,
    AuthenticationError,
    BudgetExceededError,
    ConfigurationError,
    ExtractionError,
    InvalidRequestError,
    RateLimitError,
    RequestTooLargeError,
    ResponseValidationError,
    ServerError,
    TransportError,
    TypeExtractError,
)
from .extractor import Extractor, aextract, extract, resolve_overlaps
from .jev import Backend, BackendResponse, JevBackend
from .schema import Attribute, Entity, Field, Schema, SchemaError, SentenceLabel
from .visualize import save_html, to_html

__all__ = [
    "__version__",
    "APIError",
    "AnnotatedDocument",
    "Attribute",
    "AuthenticationError",
    "Backend",
    "BackendResponse",
    "BudgetExceededError",
    "CharInterval",
    "ConfigurationError",
    "Entity",
    "ExampleData",
    "Extraction",
    "ExtractionError",
    "Extractor",
    "Field",
    "InvalidRequestError",
    "JevBackend",
    "Limits",
    "MemoryCache",
    "Metrics",
    "RateLimitError",
    "RequestTooLargeError",
    "ResponseValidationError",
    "SQLiteCache",
    "Schema",
    "SchemaError",
    "SentenceLabel",
    "ServerError",
    "TransportError",
    "TypeExtractError",
    "aextract",
    "extract",
    "load_jsonl",
    "resolve_overlaps",
    "save_html",
    "save_jsonl",
    "to_html",
]
