# Pipeline Demo (EDF and Provenance Store)


## Installation

Run example pipeline:
```bash
uv run python -m  examples.grocery
```


### TODO
- add actual lineage (add lineage to data, not just edfs across pipelines)
- make api endpoints flaky to see network errors (hard)
- using error registrations for healing / improve future runs
- introduce chaining with https://github.com/microsoft/autogen

feedback
- visualization of stages and expose info about sources at each stage to optimizer
- hard problem: knowing when model is right
- some errors may be correlated
- different sources have trust