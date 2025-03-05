import typing as t
import pandas as pd
from edf import EDF
from abc import ABC

T = t.TypeVar("T")


class Pipeline:
    def __init__(self, name: str):
        self.name = name
        self.stages: t.List[ETLStage] = []

    def add_stage(self, stage: "ETLStage"):
        self.stages.append(stage)

    def execute(self, *args, **kwargs):
        result = None
        for stage in self.stages:
            if result is None:
                result = stage.execute(*args, **kwargs)
            else:
                result = stage.execute(result, **kwargs)
        return result

    # --- Stage Decorators ---
    def extract(self, func: t.Callable[..., EDF]) -> t.Callable[..., EDF]:
        """Decorator for creating a ExtractStage."""
        stage = ExtractStage(func.__name__, func)
        self.add_stage(stage)

        def wrapper(*args, **kwargs) -> EDF:
            return stage.execute(*args, **kwargs)

        return wrapper

    def transform(self, func: t.Callable[..., EDF]) -> t.Callable[..., EDF]:
        """Decorator for creating a TransformStage."""
        stage = TransformStage(func.__name__, func)
        self.add_stage(stage)

        def wrapper(*args, **kwargs) -> EDF:
            return stage.execute(*args, **kwargs)

        return wrapper

    def fold(self, func: t.Callable[..., T]) -> t.Callable[..., T]:
        """Decorator for creating a FoldStage."""
        stage = FoldStage(func.__name__, func)
        self.add_stage(stage)

        def wrapper(*args, **kwargs) -> T:
            return stage.execute(*args, **kwargs)

        return wrapper

    def aggregate(self, func: t.Callable[..., EDF]) -> t.Callable[..., EDF]:
        """Decorator for creating an AggregateStage."""
        stage = AggregateStage(func.__name__, func)
        self.add_stage(stage)

        def wrapper(*args, **kwargs) -> EDF:
            return stage.execute(*args, **kwargs)

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

    def execute(self, *args, **kwargs) -> EDF:
        return self._loader(*args, **kwargs)


class TransformStage(ETLStage):
    def __init__(self, name: str, transformer: t.Callable[[EDF], EDF]):
        super().__init__(name)
        self._transformer = transformer

    def execute(
        self,
        df: EDF,
    ) -> EDF:
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
