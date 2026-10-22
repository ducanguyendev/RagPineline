"""Safe, user-facing PDF indexing errors."""


class PdfInputError(ValueError):
    """The PDF is invalid or a required OCR input dependency is unavailable."""


class PdfRuntimeError(RuntimeError):
    """Docling or OCR failed because of a server-side runtime issue."""
