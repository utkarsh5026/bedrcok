"""Which parsing models can you actually use? Ask Bedrock.

``PARSING_MODELS`` lists the parsers kbforge knows by alias, but what's available depends on
your region and account: new models appear, old ones are retired, and some can only be called
through a cross-region inference profile. ``list_parsing_models()`` asks Bedrock for the vision
models in your region that can read documents, and says for each one exactly what to pass as
``parsing=``. It only reads; nothing is created or billed.

    import kbforge
    kbforge.parsing_models()            # a card in a notebook; `kbforge parsers` in a terminal

If listing isn't allowed (it needs ``bedrock:ListFoundationModels`` and
``bedrock:ListInferenceProfiles``), the known parsers are shown instead, with a note saying so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List

from .aws import Aws, error_message
from .models import DEFAULT_PARSING_MODEL, PARSING_MODELS, is_parser_family, profile_geo


@dataclass
class ParserOption:
    """One model you can parse documents with, and what to pass to use it."""

    #: What to pass as ``parsing=`` (or ``Parsing.foundation_model(...)``): an alias or a model id.
    value: str
    label: str
    provider: str
    #: How Bedrock calls it here, e.g. ``"us. inference profile"`` or ``"on-demand"``.
    route: str
    note: str = ""

    @property
    def recommended(self) -> bool:
        """True for the parser kbforge suggests starting with."""
        return self.value == DEFAULT_PARSING_MODEL


@dataclass
class ParsingModels:
    """The parsing models for one region: from Bedrock when ``live``, else kbforge's built-in list."""

    region: str
    options: List[ParserOption] = field(default_factory=list)
    #: False when Bedrock couldn't be asked and these are the parsers kbforge knows about.
    live: bool = True
    #: Why Bedrock couldn't be asked, when ``live`` is False.
    error: str = ""

    def __iter__(self) -> Iterator[ParserOption]:
        return iter(self.options)

    def __len__(self) -> int:
        return len(self.options)

    @property
    def values(self) -> List[str]:
        """Just the values to pass as ``parsing=``."""
        return [option.value for option in self.options]

    def _repr_html_(self) -> str:
        from .ui.html import render_parsing_models

        return render_parsing_models(self)

    def __str__(self) -> str:
        from .ui.text import render_parsing_models

        return render_parsing_models(self)


def known_parsing_models(region: str, error: str = "") -> ParsingModels:
    """kbforge's built-in parser list, for when Bedrock can't be asked."""
    route = f"{profile_geo(region or 'us-east-1')}. inference profile"
    options = [
        ParserOption(m.alias, m.label, m.model_id.split(".", 1)[0].capitalize(), route, m.note)
        for m in PARSING_MODELS.values()
    ]
    return ParsingModels(region, options, live=False, error=error)


def _profiles_by_model(aws: Aws) -> Dict[str, List[str]]:
    """Every system inference profile in the region, keyed by the model id it calls."""
    bedrock = aws.client("bedrock")
    found: Dict[str, List[str]] = {}
    kwargs: Dict[str, Any] = {"typeEquals": "SYSTEM_DEFINED", "maxResults": 1000}
    while True:
        page = bedrock.list_inference_profiles(**kwargs)
        for profile in page.get("inferenceProfileSummaries", []):
            if profile.get("status", "ACTIVE") != "ACTIVE":
                continue
            for model in profile.get("models", []):
                model_id = model.get("modelArn", "").rsplit("/", 1)[-1]
                found.setdefault(model_id, []).append(profile["inferenceProfileId"])
        token = page.get("nextToken")
        if not token:
            return found
        kwargs["nextToken"] = token


def _option(summary: Dict[str, Any], profiles: List[str], region: str) -> ParserOption:
    """Turn one foundation model into a choice, preferring the profile for the region's geography."""
    model_id = summary["modelId"]
    known = next((m for m in PARSING_MODELS.values() if m.model_id == model_id), None)
    label = summary.get("modelName") or (known.label if known else model_id)
    provider = summary.get("providerName", "")
    note = known.note if known else ""
    geo = profile_geo(region)
    profile = next((p for p in profiles if p.startswith(geo + ".")), None) or (profiles[0] if profiles else None)
    if profile:
        prefix = profile.split(".", 1)[0]
        # The alias picks the profile for the region's geography, so it only fits when that's this one.
        value = known.alias if known and prefix == geo else profile
        return ParserOption(value, label, provider, f"{prefix}. inference profile", note)
    return ParserOption(model_id, label, provider, "on-demand", note)


def list_parsing_models(aws: Aws) -> ParsingModels:
    """Ask Bedrock which models in this region can parse documents for a knowledge base.

    Keeps the active Claude, Nova and Llama 4 models that read images and write text, and only
    those that can be called here (directly or through an inference profile). Known parsers come
    first, in kbforge's order, then the rest by provider and name. Never raises: on an AWS error
    it returns kbforge's built-in list with ``error`` set.
    """
    try:
        summaries = aws.client("bedrock").list_foundation_models(byOutputModality="TEXT")["modelSummaries"]
        profiles = _profiles_by_model(aws)
    except Exception as exc:
        return known_parsing_models(
            aws.region,
            f"Couldn't list the models in your account ({error_message(exc)}). These are the parsers kbforge "
            "knows; to see what's available to you, allow bedrock:ListFoundationModels and "
            "bedrock:ListInferenceProfiles.",
        )
    options: List[ParserOption] = []
    for summary in summaries:
        model_id = summary.get("modelId", "")
        usable_directly = "ON_DEMAND" in summary.get("inferenceTypesSupported", [])
        if (
            not is_parser_family(model_id)
            or "IMAGE" not in summary.get("inputModalities", [])
            or summary.get("modelLifecycle", {}).get("status", "ACTIVE") != "ACTIVE"
            or not (usable_directly or profiles.get(model_id))
        ):
            continue
        options.append(_option(summary, profiles.get(model_id, []), aws.region))
    order = list(PARSING_MODELS)
    options.sort(key=lambda o: (order.index(o.value) if o.value in order else len(order), o.provider, o.label, o.value))
    return ParsingModels(aws.region, options)
