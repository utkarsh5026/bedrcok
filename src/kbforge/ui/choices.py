"""Clickable option cards for the wizard: a nicer replacement for dropdowns.

A dropdown hides the choices and their trade-offs until it's opened. For decisions like
"which vector store?", seeing every option side by side, each with a sentence on when to
pick it, helps people who don't know AWS make a good choice.

ipywidgets has no card widget, so each card is a small box holding an HTML snippet (the
icon, title and explanation) with a transparent button stretched over it. Clicking anywhere
on the card clicks the button; the button also keeps the card reachable with the keyboard.

This module imports ipywidgets at the top, so only import it once you know it is installed
(``Wizard`` does that and shows a friendly message if it isn't).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import ipywidgets as w
import traitlets

from .wizard_style import option


@dataclass(frozen=True)
class Choice:
    """One option card: the value it stands for and what the card says."""

    value: Any
    title: str
    body: str = ""
    icon: str = ""
    #: An optional coloured tag beside the title, e.g. ("Recommended", "c-green").
    tag: Tuple[str, str] = ("", "")
    #: A small extra line under the text, e.g. supported vector sizes.
    meta: str = ""


class OptionCards(w.GridBox):  # type: ignore[misc]
    """A grid of option cards where exactly one is selected, with a ``value`` you can observe.

    It behaves like the other ipywidgets selection controls::

        cards = OptionCards([Choice("a", "Option A"), Choice("b", "Option B")], value="a")
        cards.observe(on_change, names="value")
    """

    value = traitlets.Any()

    def __init__(self, choices: Sequence[Choice], value: Any, columns: int = 2, pills: bool = False):
        self._cards: Dict[Any, w.Box] = {}
        children: List[w.Box] = []
        for choice in choices:
            hit = w.Button(description=choice.title, tooltip=choice.body or choice.title)
            hit.add_class("kbf-hit")
            hit.on_click(lambda _, v=choice.value: setattr(self, "value", v))
            box = w.Box([w.HTML(option(choice.title, choice.body, choice.icon, choice.tag, choice.meta)), hit])
            box.add_class("kbf-opt")
            self._cards[choice.value] = box
            children.append(box)
        super().__init__(children, layout=w.Layout(grid_template_columns=f"repeat({columns}, minmax(0, 1fr))"))
        self.add_class("kbf-opts")
        if pills:
            self.add_class("pills")
        self.value = value

    @traitlets.observe("value")
    def _highlight(self, change: Dict[str, Any]) -> None:
        """Move the selected look to the card that was just picked."""
        for value, box in self._cards.items():
            if value == change["new"]:
                box.add_class("on")
            else:
                box.remove_class("on")
