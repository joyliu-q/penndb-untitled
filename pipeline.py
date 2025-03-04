"""Pipeline Construction Tools"""

import functools
import logging
import typing as t

from error import PipelineError
from edf import EDF
import pandas as pd

from abc import ABC

T = t.TypeVar("T")


class Pipeline:
    def __init__(self, name: str):
        self.name = name
        self.stages: t.List[ETLStage] = []

    def add_stage(self, stage: "ETLStage"):
        self.stages.append(stage)

    def execute(self):
        for stage in self.stages:
            stage.execute()

    # --- Stage Decorators: tuple level ---
    def extract(self, func: t.Callable[[], EDF]) -> t.Callable[[], EDF]:
        """Decorator for creating a ExtractStage."""
        stage = ExtractStage(func.__name__, func)
        self.add_stage(stage)

        def wrapper() -> EDF:
            return stage.execute()

        return wrapper

    def transform(self, func: t.Callable[[EDF], EDF]) -> t.Callable[[EDF], EDF]:
        """Decorator for creating a TransformStage."""
        stage = TransformStage(func.__name__, func)
        self.add_stage(stage)

        def wrapper(df: EDF) -> EDF:
            return stage.execute(df)

        return wrapper

    def fold(self, func: t.Callable[[EDF], T]) -> t.Callable[[EDF], T]:
        """Decorator for creating a FoldStage."""
        stage = FoldStage(func.__name__, func)
        self.add_stage(stage)

        def wrapper(df: EDF) -> T:
            return stage.execute(df)

        return wrapper

    def aggregate(self, func: t.Callable[[t.List[EDF]], EDF]) -> t.Callable[[t.List[EDF]], EDF]:
        """Decorator for creating an AggregateStage."""
        stage = AggregateStage(func.__name__, func)
        self.add_stage(stage)

        def wrapper(dfs: t.List[EDF]) -> EDF:
            return stage.execute(dfs)

        return wrapper


# TODO: All stages belong to a pipeline, which thye must be registered to
class ETLStage(ABC):
    """Base class for all ETL stages."""

    def __init__(self, name: str):
        self.name = name


class ExtractStage(ETLStage):
    def __init__(self, name: str, loader: t.Callable[[], EDF]):
        super().__init__(name)
        self._loader = loader

    def execute(self) -> EDF:
        return self._loader()


class TransformStage(ETLStage):
    def __init__(self, name: str, transformer: t.Callable[[EDF], EDF]):
        super().__init__(name)
        self._transformer = transformer

    def execute(self, df: EDF) -> EDF:
        return self._transformer(df)


class FoldStage(ETLStage, t.Generic[T]):
    """Stage that reduces a DataFrame to a single value."""

    def __init__(self, name: str, folder: t.Callable[[EDF], T]):
        super().__init__(name)
        self._folder = folder

    def execute(self, df: EDF) -> T:
        return self._folder(df)


class AggregateStage(ETLStage):
    """Stage that combines multiple DataFrames into one."""

    def __init__(self, name: str, aggregator: t.Callable[[t.List[EDF]], EDF]):
        super().__init__(name)
        self._aggregator = aggregator

    def execute(self, dfs: t.List[EDF]) -> EDF:
        return self._aggregator(dfs)


pipeline = Pipeline("My ETL Pipeline")


@pipeline.extract
def load_data() -> EDF:
    return EDF(pd.read_csv("data.csv"))


@pipeline.transform
def clean_data(df: EDF) -> EDF:
    return df.register_natural_error("age must be >= 0")


@pipeline.fold
def calc_average(df: EDF) -> float:
    return df.age.mean()


@pipeline.aggregate
def combine_dfs(dfs: t.List[EDF]) -> EDF:
    return pd.concat(dfs, ignore_index=True)


class RowLevelPipelineError(Exception):
    """
    Raise this exception when you want to register an error for specific row(s)
    rather than globally.
    """

    def __init__(
        self,
        row_idx: t.Union[int, t.List[int]],
        category: PipelineError,
        description: str,
        column: t.Optional[str] = None,
    ):
        super().__init__(description)
        self.row_idx = row_idx
        self.category = category
        self.column = column


def pipeline_error_handler(
    stage_name: str,
    error_classes: t.Union[type, t.Tuple[type, ...]],
    default_category: PipelineError = PipelineError.BAD_REQUEST,
):
    """
    A decorator that:
      - Expects the wrapped function's first argument to be an EDF (our target DF).
      - Catches ONLY exceptions of the specified types (error_classes).
      - If it's a RowLevelPipelineError, registers row-based error(s).
      - Otherwise, registers a global error with 'default_category'.
      - Returns the EDF (with errors) after catching, or re-raises if the exception type is not matched.

    :param stage_name: Identifies which pipeline stage is being decorated (for logging/errors).
    :param error_classes: Exception class or tuple of classes to catch.
    :param default_category: The fallback category for non-row-level errors.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(edf: EDF, *args, **kwargs) -> EDF:
            try:
                return func(edf, *args, **kwargs)
            except error_classes as e:
                logging.exception(f"Error in stage '{stage_name}': {str(e)}")

                if isinstance(e, RowLevelPipelineError):
                    # --- Row-level registration ---
                    edf_with_error = edf.register_error(
                        row_idx=e.row_idx,
                        category=e.category,
                        description=str(e),
                        column=e.column,
                    )
                    return edf_with_error
                else:
                    # --- Fallback global registration ---
                    edf_with_error = edf.register_global_error(
                        category=default_category,
                        description=f"Error in stage '{stage_name}': {str(e)}",
                    )
                    return edf_with_error

        return wrapper

    return decorator
