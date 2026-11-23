class EvaluationError(Exception):
    """Base error for evaluation execution and validation."""


class EvaluationConfigurationError(EvaluationError):
    pass


class EvaluationOutputError(EvaluationError):
    pass

