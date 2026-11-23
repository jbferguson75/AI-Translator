class TranslationError(Exception):
    """Base exception for translation configuration and runtime errors."""


class TranslationConfigurationError(TranslationError):
    pass


class OllamaError(TranslationError):
    pass

