class SourceImportError(Exception):
    """Base error for import, parsing, and validation failures."""


class SourceParseError(SourceImportError):
    pass


class SourceValidationError(SourceImportError):
    pass

