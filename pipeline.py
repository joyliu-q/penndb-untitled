"""Pipeline Construction Tools"""

import functools
import logging
import typing as t
from graphviz import Digraph

from error import PipelineError
from edf import EDF
import pandas as pd
from pydantic import BaseModel
from utils import get_llm
import os
import inspect

from abc import ABC
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA


class ErrorPattern(BaseModel):
    """A pattern in the error rows of a stage."""

    type: t.Literal["stage", "global"]
    stage_name: t.Optional[str] = None
    description: str


# TODO: All stages belong to a pipeline, which thye must be registered to
class ETLStage(ABC):
    """Base class for all ETL stages."""

    def __init__(self, name: str):
        self.name = name
        self.dependencies: t.List[ETLStage] = []
        self.result = None

    def add_dependency(self, stage: "ETLStage"):
        self.dependencies.append(stage)

    def get_dependencies(self) -> t.List["ETLStage"]:
        return self.dependencies

    @property
    def has_run(self) -> bool:
        return self.result is not None


class Pipeline:
    def __init__(self, name: str):
        self.name = name
        self.stages: t.Dict[str, ETLStage] = {}  # Changed to dict for name lookup
        self.last_stage: t.Optional[ETLStage] = None

    def add_stage(self, stage: ETLStage):
        """Add a stage to the pipeline without creating automatic dependencies"""
        self.stages[stage.name] = stage
        self.last_stage = stage

    def depends_on(self, *stage_names: str):
        """Decorator to specify stage dependencies"""

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            # Store dependencies to be resolved when the stage is created
            wrapper._dependencies = stage_names
            return wrapper

        return decorator

    def _create_stage(self, func: t.Callable, stage_class: t.Type[ETLStage]) -> ETLStage:
        """Helper to create a stage with dependencies"""
        stage = stage_class(func.__name__, func)

        # Add explicit dependencies if specified
        if hasattr(func, "_dependencies"):
            for dep_name in func._dependencies:
                if dep_name not in self.stages:
                    raise ValueError(
                        f"Dependency '{dep_name}' not found for stage '{func.__name__}'"
                    )
                stage.add_dependency(self.stages[dep_name])

        return stage

    def extract(self, func: t.Callable[[], EDF]) -> t.Callable[[], EDF]:
        """Decorator for creating an ExtractStage."""
        stage = self._create_stage(func, ExtractStage)
        self.add_stage(stage)

        def wrapper() -> EDF:
            return stage.execute()

        return wrapper

    def transform(self, func: t.Callable[[EDF], EDF]) -> t.Callable[[EDF], EDF]:
        """Decorator for creating a TransformStage."""
        stage = self._create_stage(func, TransformStage)
        self.add_stage(stage)

        def wrapper(df: EDF) -> EDF:
            return stage.execute(df)

        return wrapper

    def fold(self, func: t.Callable[[EDF], EDF]) -> t.Callable[[EDF], EDF]:
        """Decorator for creating a FoldStage."""
        stage = self._create_stage(func, FoldStage)
        self.add_stage(stage)

        def wrapper(df: EDF) -> EDF:
            return stage.execute(df)

        return wrapper

    def aggregate(self, func: t.Callable[[t.List[EDF]], EDF]) -> t.Callable[[t.List[EDF]], EDF]:
        """Decorator for creating an AggregateStage."""
        stage = self._create_stage(func, AggregateStage)
        self.add_stage(stage)

        def wrapper(dfs: t.List[EDF]) -> EDF:
            return stage.execute(dfs)

        return wrapper

    def run(self, debug: bool = True) -> t.Dict[str, t.Any]:
        """Execute the pipeline in dependency order

        If debug is True, writes the results of each stage to a separate CSV file, under the `debug` directory.
        """
        results = {}
        executed = set()

        def execute_stage(stage: ETLStage):
            if stage.name in executed:
                return stage.result

            dep_results = []
            for dep in stage.dependencies:
                dep_result = execute_stage(dep)
                dep_results.append(dep_result)

            if isinstance(stage, ExtractStage):
                if dep_results:
                    raise ValueError(f"Extract stage {stage.name} should not have dependencies")
                stage.result = stage.execute()
            elif isinstance(stage, TransformStage):
                if len(dep_results) != 1:
                    raise ValueError(
                        f"Transform stage {stage.name} expects exactly one dependency, got {len(dep_results)}"
                    )
                stage.result = stage.execute(dep_results[0])
            elif isinstance(stage, FoldStage):
                if len(dep_results) != 1:
                    raise ValueError(
                        f"Fold stage {stage.name} expects exactly one dependency, got {len(dep_results)}"
                    )
                stage.result = stage.execute(dep_results[0])
            elif isinstance(stage, AggregateStage):
                stage.result = stage.execute(
                    dep_results
                )  # Aggregate can take multiple dependencies

            if debug:
                if not os.path.exists(f"debug/{self.name}"):
                    os.makedirs(f"debug/{self.name}", exist_ok=True)
                stage.result.to_csv(f"debug/{self.name}/{stage.name}.csv")

            executed.add(stage.name)
            results[stage.name] = stage.result
            return stage.result

        for stage in self.stages.values():
            execute_stage(stage)

        return results

    def visualize(self, filename: t.Optional[str] = None, open: bool = False) -> None:
        """
        Visualize the pipeline as a DAG using graphviz.

        Args:
            filename: Name of the output file (without extension)
        """
        dot = Digraph(comment=f"Pipeline: {self.name}")
        dot.attr(rankdir="LR")
        if filename is None:
            filename = self.name

        for stage_name, stage in self.stages.items():
            color = {
                ExtractStage: "lightblue",
                TransformStage: "lightgreen",
                FoldStage: "lightyellow",
                AggregateStage: "lightpink",
            }.get(type(stage), "white")

            dot.node(stage_name, stage_name, style="filled", fillcolor=color)

        for stage_name, stage in self.stages.items():
            for dep in stage.get_dependencies():
                dot.edge(dep.name, stage_name)

        dot.render(filename, view=open, format="svg")

    def identify_error_patterns(self) -> t.Union[str, t.Dict[str, t.List[ErrorPattern]]]:
        """Find patterns in the pipeline that are likely to cause errors.

        For each stage in pipeline:
        1. Create a document with the stage's source code
        2. Create documents for error rows
        3. Use RAG to analyze patterns with both code and error context
        """
        documents = []

        for stage_name, stage in self.stages.items():
            source_code = ""
            if hasattr(stage, "_loader"):
                source_code = inspect.getsource(stage._loader)
            elif hasattr(stage, "_transformer"):
                source_code = inspect.getsource(stage._transformer)
            elif hasattr(stage, "_folder"):
                source_code = inspect.getsource(stage._folder)
            elif hasattr(stage, "_aggregator"):
                source_code = inspect.getsource(stage._aggregator)

            if source_code:
                documents.append(
                    Document(
                        page_content=f"Stage Implementation for {stage_name}:\n\n{source_code}",
                        metadata={"stage_name": stage_name, "doc_type": "source_code"},
                    )
                )

        for stage_name, stage in self.stages.items():
            error_df = stage.result.query_errors()
            if error_df.empty:
                continue

            for _, row in error_df.iterrows():
                documents.append(
                    Document(
                        page_content=f"Stage: {stage_name}\n\nError Row:\n{row.to_string(index=False)}",
                        metadata={"stage_name": stage_name, "doc_type": "error_data"},
                    )
                )

        if not documents:
            return "No errors found in any stage."

        embeddings = OpenAIEmbeddings()
        db = Chroma.from_documents(documents, embeddings)

        retriever = db.as_retriever(
            search_kwargs={"k": 20, "filter": {"doc_type": {"$in": ["source_code", "error_data"]}}}
        )

        chain = RetrievalQA.from_chain_type(
            llm=get_llm(0.7),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
        )

        query = """
Currently, the pipeline is encountering the errors. Given the implementation code and error patterns, please identify:
1. What errors we are currently encountering
2. Based on the encountered errors, what are the most common patterns or root causes of errors
3. Given these patterns, what are the most likely causes of the errors

Consider both the stage implementations and the actual errors encountered."""

        result = chain.run(query)
        return result


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


class FoldStage(ETLStage):
    """Stage that reduces a DataFrame to a single value (in a DataFrame)."""

    def __init__(self, name: str, folder: t.Callable[[EDF], EDF]):
        super().__init__(name)
        self._folder = folder

    def execute(self, df: EDF) -> EDF:
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
def calc_average(df: EDF) -> EDF:
    return EDF({"age": [df.age.mean()]})


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
