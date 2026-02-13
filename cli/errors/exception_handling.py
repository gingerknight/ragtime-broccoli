# File for custom exception handling
# Intent: To handle the exceptions at various levels of the cli, preprocessing, etc.

# search_cls.py
class SearchEngineError(Exception):
    pass


class DataLoadError(SearchEngineError):
    pass


class IndexBuildError(SearchEngineError):
    pass


class CacheIOError(SearchEngineError):
    pass


class InvalidTerm(SearchEngineError):
    pass
