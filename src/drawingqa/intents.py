"""Conservative bilingual routing for the provided benchmark's 12 count tasks.

Unknown or qualified questions retain their full wording in the general route;
we must not turn 'windows on floor two' into a whole-drawing window count.
"""
from __future__ import annotations

import re
import unicodedata

from .schemas import Intent, Question


COUNT_TASKS = {
    "doors": (
        "How many doors are shown on this floor plan? Count doorways fitted with a door; open doorways without a door leaf do not count.",
        "Kuinka monta ovea pohjapiirustuksessa on? Laske oviaukot, joissa on ovi; pelkkiä oviaukkoja ilman ovilehteä ei lasketa.",
        "Count one doorway fitted with a door as one entity, including a double-leaf door. Exclude open passages, cabinet doors and swing arcs as separate entities. Include sliding doors only when a door leaf is drawn.",
        ["doors", "doorways fitted with doors", "ovea", "ovia"],
    ),
    "windows": (
        "How many windows are shown on this floor plan?",
        "Kuinka monta ikkunaa pohjapiirustuksessa on?",
        "Count distinct architectural window units, not the strokes or glazing panes inside one unit. Separate adjacent framed units only when the drawing supports that distinction; flag ambiguous grouping.",
        ["windows", "ikkunaa", "ikkunoita"],
    ),
    "bedrooms": (
        "How many bedrooms (makuuhuone, usually marked MH) are on this floor plan?",
        "Kuinka monta makuuhuonetta (MH) pohjapiirustuksessa on?",
        "Count rooms explicitly identified as bedrooms, for example MH/makuuhuone or unambiguous bedroom furnishings. Do not invent a bedroom function for an empty unlabeled enclosed space. Mark uncertain room use as uncertain; lack of a label alone is not proof of absence.",
        ["bedrooms", "makuuhuonetta"],
    ),
    "kitchens": (
        "How many kitchens (including kitchenettes, open kitchens and utility kitchens) are on this floor plan?",
        "Kuinka monta keittiötä (myös keittokomerot, avo- ja tupakeittiöt sekä apukeittiöt) pohjapiirustuksessa on?",
        "Count kitchen spaces including kitchenettes, open kitchens and utility kitchens. Use labels and kitchen fittings together; multiple appliances in one kitchen are not multiple kitchens.",
        ["kitchens", "keittiötä"],
    ),
    "bathrooms": (
        "How many bathrooms or washrooms (e.g. KH, KPH, PH, PESUH, SH) are on this floor plan, not counting separate WC rooms?",
        "Kuinka monta kylpy- tai pesuhuonetta (esim. KH, KPH, PH, PESUH, SH) pohjapiirustuksessa on, erillisiä WC-tiloja lukuun ottamatta?",
        "Count bathroom/washroom spaces, supported by labels KH/KPH/PH/PESUH/SH or bathing fittings. Exclude separate WC-only rooms and separate sauna rooms. Count rooms, not fixtures.",
        ["bathrooms", "washrooms", "bathrooms or washrooms", "kylpyhuonetta", "pesuhuonetta"],
    ),
    "wc_rooms": (
        "How many separate WC rooms (toilet rooms marked WC) are on this floor plan?",
        "Kuinka monta erillistä WC-tilaa pohjapiirustuksessa on?",
        "Count separate toilet rooms marked WC or clearly established as WC-only rooms. A toilet bowl inside a bathroom does not make that bathroom a separate WC room.",
        ["separate wc rooms", "separate toilet rooms", "erillistä wc-tilaa"],
    ),
    "saunas": (
        "How many saunas are on this floor plan?",
        "Kuinka monta saunaa pohjapiirustuksessa on?",
        "Count sauna spaces, using S/sauna labels and characteristic benches/heater. Several benches in the same sauna count as one sauna.",
        ["saunas", "saunaa"],
    ),
    "balconies": (
        "How many balconies (parveke, including glazed balconies) are on this floor plan?",
        "Kuinka monta parveketta (myös lasitetut parvekkeet) pohjapiirustuksessa on?",
        "Count balcony spaces including glazed balconies. Do not count a terrace (terassi) or porch (kuisti) as a balcony. Use labels to resolve ambiguous outdoor spaces.",
        ["balconies", "parveketta"],
    ),
    "toilets": (
        "How many toilet bowls (WC seats) are drawn on this floor plan?",
        "Kuinka monta WC-istuinta pohjapiirustukseen on piirretty?",
        "Count individual toilet bowl fixtures wherever drawn. Do not count the WC room label as an additional fixture and do not confuse a basin or bidet with a toilet.",
        ["toilet bowls", "toilets", "wc seats", "wc-istuinta"],
    ),
    "sinks": (
        "How many sinks and wash basins (kitchen and bathroom combined; a double-bowl sink counts as one) are drawn on this floor plan?",
        "Kuinka monta pesu- ja tiskiallasta (keittiö ja kylpytilat yhteensä; kaksoisallas lasketaan yhdeksi) pohjapiirustukseen on piirretty?",
        "Count kitchen sinks and bathroom wash basins together. A double-bowl sink is one fixture: one box around the complete unit. Do not count its two bowls separately.",
        ["sinks", "wash basins", "sinks and wash basins", "pesu- ja tiskiallasta", "tiskiallasta"],
    ),
    "showers_bathtubs": (
        "How many showers and bathtubs are drawn on this floor plan in total?",
        "Kuinka monta suihkua ja kylpyammetta pohjapiirustukseen on piirretty yhteensä?",
        "Count drawn shower fixtures plus bathtub fixtures. Count distinct fixtures, not bathrooms, drain symbols, or individual lines. Flag uncertainty when shower and tub symbols overlap or cannot be distinguished.",
        ["showers and bathtubs", "showers and baths", "suihkua ja kylpyammetta"],
    ),
    "floors": (
        "How many floors (storeys) does this floor plan show?",
        "Kuinka monta kerrosta pohjapiirustuksessa on esitetty?",
        "Count distinct floor/storey levels depicted. Box each floor panel and use floor labels to avoid counting multiple views of one level twice. A staircase does not imply an additional depicted storey. Do not count isolated legends or details as floors.",
        ["floors", "storeys", "stories", "floors (storeys)", "kerrosta"],
    ),
}


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).strip().casefold()).rstrip(" .?!")


def resolve_intent(question: Question) -> Intent:
    text = _normalise(question.question)
    for category, (english, finnish, rule, nouns) in COUNT_TASKS.items():
        exact = {_normalise(english), _normalise(finnish)}
        for noun in nouns:
            for template in (
                "how many {noun}", "how many {noun} are there", "count {noun}", "count the {noun}",
                "how many {noun} are shown on this floor plan", "how many {noun} are on this floor plan",
                "how many {noun} are drawn on this floor plan", "kuinka monta {noun}",
                "kuinka monta {noun} pohjapiirustuksessa on", "kuinka monta {noun} pohjapiirustukseen on piirretty",
            ):
                exact.add(template.format(noun=noun))
        if text in exact:
            return Intent(operation="count", category=category, canonical_question=english, rule=rule)
    return Intent(
        operation="general", category="general", canonical_question=question.question,
        rule="Answer only from visible evidence in this drawing. Preserve the requested scope, units and exclusions. If essential scale, geometry, text or other evidence is missing, return answer=null and complete=false.",
    )
