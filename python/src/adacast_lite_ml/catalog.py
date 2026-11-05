"""Owned model names and optional dependency profiles; no weight downloads."""
import importlib.util
import importlib.metadata
from packaging.specifiers import SpecifierSet
from pathlib import Path

from .engine.statistical import STATISTICAL_MODEL_NAMES, normalize_statistical_model_name
from .engine.models._foundation import FOUNDATION_MODEL_REPOS

BASELINES=("Naive","Mean","SeasonalNaive","Linear")
NEURAL_MODELS=tuple(sorted(p.stem for p in (Path(__file__).parent/"engine/models").glob("*.py") if not p.stem.startswith("_")))
NAMES=(*BASELINES,*STATISTICAL_MODEL_NAMES,*NEURAL_MODELS)
SPECIAL={"Chronos":("chronos","transformers","huggingface_hub"),"Chronos2":("chronos","transformers","huggingface_hub"),"TimesFM":("timesfm","huggingface_hub"),"Moirai":("uni2ts",),"Sundial":("transformers",),"TimeMoE":("transformers",),"TiRex":("tirex",),"Mamba":("mamba_ssm",),"Reformer":("reformer_pytorch",)}
EXTRAS={"Chronos":"foundation","Chronos2":"foundation","TimesFM":"timesfm","Moirai":"moirai","Sundial":"transformers-models","TimeMoE":"transformers-models","TiRex":"tirex","Mamba":"mamba","Reformer":"reformer"}
PACKAGES={"chronos":"chronos-forecasting","tirex":"tirex-ts","mamba_ssm":"mamba-ssm","huggingface_hub":"huggingface-hub","reformer_pytorch":"reformer-pytorch"}
VERSIONS={"timesfm":">=3.0.2,<4","transformers":">=4.57,<5"}


def canonical_model(name):
    normalized="".join(c for c in str(name).casefold() if c.isalnum())
    for candidate in NAMES:
        if "".join(c for c in candidate.casefold() if c.isalnum())==normalized:
            return candidate
    raise ValueError(f"Unknown model {name}; use the models command to list supported implementations")


def model_family(name):
    if name=="UserPrior":return "user_prior"
    name=canonical_model(name)
    if name in BASELINES:return "baseline"
    if name in STATISTICAL_MODEL_NAMES:return "statistical"
    return "foundation" if name in FOUNDATION_MODEL_REPOS else "deep_learning"


def requirements(name):
    name=canonical_model(name)
    return (("torch",)+SPECIAL.get(name,())) if name in NEURAL_MODELS else ()


def missing_requirements(name):
    missing=[]
    for module in requirements(name):
        package=PACKAGES.get(module,module)
        requirement=package+VERSIONS.get(module,"")
        if importlib.util.find_spec(module) is None:
            missing.append(requirement)
        elif module in VERSIONS:
            try:
                if importlib.metadata.version(package) not in SpecifierSet(VERSIONS[module]):missing.append(requirement)
            except importlib.metadata.PackageNotFoundError:missing.append(requirement)
    return missing


def catalog():
    models=[]
    for name in NAMES:
        required=requirements(name)
        missing=missing_requirements(name)
        extra=EXTRAS.get(name,"neural" if name in NEURAL_MODELS else "statistical")
        models.append({"name":name,"family":model_family(name),"dependency_modules":list(required),"missing_dependencies":missing,
                       "dependencies_available":not missing,"install_extra":extra,"foundation_repository":FOUNDATION_MODEL_REPOS.get(name),
                       "weight_download_required":name in FOUNDATION_MODEL_REPOS})
    return {"models":models,"availability_scope":"Python dependency presence only; model weights, hardware and parameter combinations require execution"}


def preflight(names, deterministic=False):
    defaults={"statistical":"ARIMA","deep_learning":"DLinear","foundation":"Chronos","baseline":"Naive"}
    for item in names:
        if item in defaults and not deterministic:
            candidates=[name for name in NAMES if model_family(name)==item]
            if any(not missing_requirements(name) for name in candidates):continue
            extras=sorted({EXTRAS.get(name,"neural") for name in candidates})
            raise ValueError(f"No available model in family {item}; install an appropriate Lite extra: {', '.join(extras)}")
        name=canonical_model(defaults.get(item,item))
        missing=missing_requirements(name)
        if missing:
            extra=EXTRAS.get(name,"neural")
            raise ValueError(f"{name} requires {', '.join(missing)}; install extra '{extra}' in the independent Lite Python environment")
