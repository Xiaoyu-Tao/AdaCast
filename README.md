# adacast-lite

Independent command-line forecasting agents with durable LLM context, resumable execution and verifiable delivery packages. Source installation requires Bun >= 1.3.11 and Python >= 3.12. The compiled release requires Python and installs its own environment.

## Install and run

```bash
bun install --frozen-lockfile
uv sync --project python --locked
python3 examples/generate.py
bin/adacast-lite models --json
bin/adacast-lite validate --config examples/task.local.json
bin/adacast-lite run --config examples/task.local.json --workspace workspaces/demo
bin/adacast-lite resume --workspace workspaces/demo
bin/adacast-lite export --workspace workspaces/demo --output deliveries/demo
```

Use a fresh workspace for each task. `task.local.json` performs actual deterministic model comparison without LLM calls. The scientific stack is installed by default; neural and foundation model dependencies are optional.

For `dataset/ETTh1.csv`, use [the fixed DLinear configuration](examples/task.etth1.local.json) or [the Agent configuration](examples/task.etth1.agent.json). Commands, metric conventions and recovery are described in [the ETTh1 guide](docs/ETTH1.md).

The Agent examples use `https://api.kabuai.cn/v1` with `deepseek-v4.1-flash` and `temperature: 0.7`. Set `LLM-API-KEY` in the project-root `.env` (see `.env.example`). Bun loads it automatically when commands run from the project root; the configuration references the variable name without storing the key. The client appends `/chat/completions` to the base URL.

To run Planner, Forecaster and Critic from the project root:

```bash
bin/adacast-lite run --config examples/task.agent.json --workspace workspaces/agent --json
```

The task configuration replaces interactive decisions. Planner analyzes training targets, writes the report and reviews Skills. Forecaster searches the permitted models, records experiment reflections, and validates conditional adjustment rules using separate validation-front evidence. Critic reuses frozen fitted artifacts for testing and runs the original isolated evidence controller. Role prompts, conditional rule validation, calibration and the window state reducer are owned by this project.

## Behavioral contract

Version 0.3 preserves all numeric input columns, their order, float32 training normalization, the original model definitions, training initialization/defaults, statistical fitting and evaluation windows. Statistical forecasts continue from the training boundary. Model selection first freezes the lowest-validation-MSE representative **per model family**, then selects the lowest **raw test MSE** among these representatives before adjustment or window refinement. This intentionally follows the reference experiment protocol; those test scores are part of selection, rather than an independent estimate after model selection.

`model_families` accepts explicit model names or `statistical`, `deep_learning`, and `foundation`. Deterministic mode maps these family selectors to ARIMA, DLinear and Chronos respectively; agent mode chooses models within the permitted families. Neural defaults include seed 42, 5 epochs, batch size 32 and learning rate 0.0001. Parameter aliases `epochs`/`kernel` and ARIMA `p`/`d`/`q` are accepted. Holt uses optimized statsmodels fitting.

```bash
uv sync --project python --locked --extra neural       # DLinear, PatchTST and other torch models
uv sync --project python --locked --extra foundation   # Chronos integration
uv sync --project python --locked --extra timesfm      # TimesFM 2.5 wrapper dependencies
uv sync --project python --locked --extra moirai       # Moirai + reference stack overrides
uv sync --project python --locked --extra tirex        # TiRex
uv sync --project python --locked --extra transformers-models # Sundial / TimeMoE
```

The `models` command reports missing modules and version constraints without downloading weights. Optional compiled Mamba support uses `--extra mamba` and requires a compatible build environment. Availability of dependencies does not certify weights, hardware or every parameter combination. Foundation inference retains the reference repository, resolved revision, aggregation and device behavior; some wrappers require CUDA. These paths are not replaced with smaller surrogate models. See [current scoped alignment](docs/ALIGNMENT_0_3.md) and [numerical reference measurements](docs/BEHAVIOR_ALIGNMENT.md) for the tested scope and remaining verification limits.

## Recovery and delivery

Interrupt with SIGINT/SIGTERM and use `resume` with the same credentials. Completed experiments, tool results, semantic context summaries and immutable window states are retained. Input and model hashes are checked. SQLite is the authoritative workflow state; Markdown/JSON state files are projections. Request, token and time budgets survive recovery. Version 0.1/0.2 workspaces cannot silently resume with the 0.3 execution contract; start a fresh workspace.

Canonical test reflection tools also update the final output and report completion gate. Replaying a matching, audited completed reflection restores these host projections without model calls or prediction changes.

The workspace contains `.forecast/runtime/state.sqlite`, full LLM request/response audits, model runs, adjustment evidence and reflection checkpoints. `.forecast/best-model/` contains model/scaler metadata, raw and final predictions, the report, plot, audits and a relative-path SHA-256 manifest. Export verifies the package before copying it. Supplied UserPrior CSVs retain their provenance and skip model search.

Portable inference uses the installed Python JSON interface:

```json
{"protocol_version":1,"operation_id":"predict-1","operation":"predict","model_artifact":"/path/to/model/model.json","history":[1,2,3]}
```

Provide at least `seq_len` original-unit history values, then run `python -m adacast_lite_ml.cli --request request.json`. Multivariate models need a matrix in the saved `feature_columns` order, plus matching `history_timestamps` for calendar-aware models. `values` includes the forecast gap; `forecast_values` excludes it. Statistical models forecast from their saved training boundary. UserPrior is a supplied forecast source and cannot predict new history.

Explicit knowledge/instruction file paths resolve against the configuration file and are frozen with the task. Shell and network tools default to disabled. `--python` or `ADACAST_LITE_PYTHON` selects an interpreter; otherwise the installation's own environment is used. No original checkout, workspace package or original Python import is required.

## Build and verify

```bash
bun run typecheck
bun test --timeout 240000
uv run --project python --locked python -m unittest discover -s python/tests -v
bun run check:independence
bun run schemas
bun run build
bun run test:release
sh dist/install.sh /absolute/path/to/installation
```

The release contains a compiled CLI, an independent Python wheel and an installer. Release smoke tests install into a temporary directory and exercise validation, run, recovery, export and portable inference using the installed package. Agent integration tests use a marked local mock provider; computation and artifacts are real, while LLM quality is not measured by those tests.

`--json` emits JSONL events/results. Exit codes: 2 input/configuration, 3 workflow gate, 4 runtime, 5 budget, 130/143 interruption.


## Context and LLM controls (0.3)

The default output cap is 32,000 tokens, bounded by each declared model limit. Sampling defaults follow the reference model policy; default HTTP retries are zero. All options can be frozen in task configuration:

```json
{
  "llm": {
    "model": "my-model", "base_url": "https://example.com/v1",
    "max_output_tokens": 8192, "temperature": 0,
    "model_capabilities": {
      "my-model": {"context_window_tokens": 64000, "max_output_tokens": 8192, "supports_temperature": true, "supports_tools": true}
    }
  },
  "context": {"auto": true, "prune": true},
  "agents": {"forecaster": {"top_p": 0.95}, "compaction": {"max_output_tokens": 4096}}
}
```

This is a configuration fragment; include it in a complete task configuration. A compaction model can have its own entry in `model_capabilities`. `top_k` requires an explicit `supports_top_k` capability. Original messages remain in the audit database; projection pruning and compaction checkpoints survive recovery.

An explicit `model_capabilities[model].max_output_tokens` can declare a model limit above the default 32,000-token fallback. Window prediction generation uses that declared limit up to 65,536 tokens, and its timeout follows `execution.llm_timeout_seconds`. Failed provider requests remain incomplete checkpoints and are retried on recovery; genuine evidence exhaustion remains a valid terminal outcome.

Calendar frequencies and irregular observations are accepted. Windows always count rows; enable `task.strict_frequency` to require a continuous declared calendar grid. Use `min`/`t` for minutes and `m`/`M` for months.

Timestamp splits may specify `task.validation_start_timestamp` separately from the inclusive `train_end_timestamp`. Rows between training and the first validation label remain rolling history only. The fixed validation-front/validation-back halves start at this explicit label boundary. [The windy Chronos-2 prior configuration](examples/task.windy.chronos2-prior.agent.json) uses this to preserve complete daily 96→96 tasks.

UserPrior tasks may specify `task.prediction_target_start_timestamp` to anchor the global prediction grid explicitly. This is needed when the supplied prior also replaces gap rows: the first changed CSV value then precedes the first evaluated target. `task.exogenous_columns` limits adjustment diagnostics and blind exogenous context without changing either source CSV. `task.exogenous_history_steps` limits past exogenous rows while retaining the full `seq_len` target history and all future-known gap/horizon exogenous rows. [The Yaguo configuration](examples/task.yaguo.userprior.agent.json) preserves the source prompt's exact prior path, 711-step target history, 57-step gap and 96-step daily labels, and uses 288 recent exogenous rows.

Optional extras can also be selected during standalone release installation:

```bash
ADACAST_LITE_INSTALL_EXTRAS=neural sh dist/install.sh /absolute/path/to/installation
ADACAST_LITE_INSTALL_EXTRAS=moirai sh dist/install.sh /another/installation
```

Use uv for Moirai so the supplied scientific-stack overrides apply. The core release remains usable without optional torch/foundation packages. Task Skills and cross-task memory parity, Critic postprocessing parity and native provider adapters are explicitly deferred for this iteration.
