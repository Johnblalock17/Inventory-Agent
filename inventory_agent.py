"""
inventory_agent.py
==================

Supply Chain Analytics Inventory Decision Agent.

Pipeline
--------
    Customer Question
      -> Identify Model      (keyword scoring, or Claude when --llm is used)
      -> Extract Inputs      (regex parser, or Claude; values + units exactly as stated)
      -> Normalize           (Python unit conversion: periods, percents, price tiers)
      -> Select Python Tool  (inventory_tools.TOOLS)
      -> Calculate           (inventory_tools.*)
      -> Verify              (inventory_tools.verify + input sanity checks)
      -> Explain             (plain-language answer with formulas and assumptions)

The language layer (regex or LLM) only *reads* the question.  Every number in
the answer, including unit conversions, is computed in Python.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import inventory_tools as tools
from inventory_tools import CalculationResult, InventoryInputError

# --------------------------------------------------------------------------- #
# Model catalogue
# --------------------------------------------------------------------------- #
MODEL_NAMES = {
    "eoq": "Economic Order Quantity (EOQ)",
    "quantity_discount": "Quantity Discount (all-units EOQ)",
    "safety_stock": "Safety Stock and Reorder Point",
    "fixed_period": "Fixed-Period Review (order-up-to level)",
    "newsvendor": "Newsvendor (single-period)",
}

TOOL_NAMES = {
    "eoq": "inventory_tools.calculate_eoq",
    "quantity_discount": "inventory_tools.calculate_quantity_discount",
    "safety_stock": "inventory_tools.calculate_rop",
    "fixed_period": "inventory_tools.calculate_periodic_review",
    "newsvendor": "inventory_tools.calculate_newsvendor",
}

# Human-readable descriptions of each input, used when asking for missing data.
INPUT_LABELS = {
    "annual_demand": "annual demand (units per year)",
    "ordering_cost": "ordering / setup cost per order ($)",
    "holding": "holding cost ($ per unit per year) or holding-cost rate (% of unit cost) plus unit cost",
    "holding_qd": "holding-cost rate (% of unit price per year) or fixed holding cost ($ per unit per year)",
    "price_breaks": "price breaks (e.g. '0-999 at $5.00, 1000+ at $4.80')",
    "mean_demand": "average demand per period",
    "std_demand": "standard deviation of demand per period",
    "lead_time": "lead time",
    "service_level": "service level (e.g. 95%) or z-value",
    "review_period": "review period (time between reviews)",
    "on_hand": "current inventory on hand (units) or inventory position",
    "overage": "purchase cost per unit or overage (excess) cost per unit ($)",
    "selling_price": "selling price per unit ($)",
    "unit_cost": "unit cost ($)",
    "demand_distribution": "demand distribution (mean and standard deviation, or uniform low and high)",
}

# Which raw fields to ask for when an input is missing.
FOLLOW_UPS: Dict[str, List[Tuple[str, Optional[str], Any]]] = {
    "annual_demand": [("demand", "Annual demand (units per year)", None), ("demand_period", None, "year")],
    "ordering_cost": [("ordering_cost", "Ordering cost per order ($)", None)],
    "holding": [("holding_cost", "Holding cost per unit per year ($)", None), ("holding_cost_period", None, "year")],
    "holding_qd": [("holding_rate", "Holding-cost rate (% of unit price per year)", None)],
    "price_breaks": [("price_breaks", "Price breaks as 'min:price, min:price' (e.g. 0:5, 1000:4.8)", None)],
    "mean_demand": [("demand", "Average demand per period", None)],
    "std_demand": [("std_demand", "Standard deviation of demand per period", None)],
    "lead_time": [("lead_time", "Lead time (in the same period as demand)", None)],
    "service_level": [("service_level", "Service level (%)", None)],
    "review_period": [("review_period", "Review period (in the same period as demand)", None)],
    "on_hand": [("on_hand", "Current inventory on hand (units)", None)],
    "overage": [("overage_cost", "Overage (excess) cost per unsold unit ($)", None)],
    "selling_price": [("selling_price", "Selling price per unit ($)", None)],
    "unit_cost": [("unit_cost", "Unit cost ($)", None)],
    "demand_distribution": [("demand", "Mean demand", None), ("std_demand", "Standard deviation of demand", None)],
}


# Keyword weights for model identification.  Distinctive cues get high weight.
MODEL_KEYWORDS: Dict[str, List[Tuple[str, int]]] = {
    "newsvendor": [
        (r"news ?vendor|newsboy", 8), (r"single[- ]period|one[- ]time (?:order|purchase|buy)", 6),
        (r"salvage", 6), (r"\bunsold\b", 6), (r"critical (?:ratio|fractile)", 6),
        (r"\boverage\b|\bunderage\b", 6), (r"left ?over", 3), (r"end of (?:the )?(?:season|day|event)", 3),
        (r"perishable|spoil|expire", 3), (r"marked? down|markdown|clearance", 3),
        (r"selling price|sells? (?:\w+ )?(?:for|at) \$|retail price", 2), (r"season|holiday|event|game day", 1),
        (r"excess (?:inventory )?cost|cost of (?:over|under)-?stocking|cost per unit (?:short|left over)", 6),
        (r"shortage cost", 2), (r"goodwill", 3), (r"uniform(?:ly)? distributed", 2),
        (r"how many .{0,30}(?:stock|order|bake|print|buy|make|produce)", 1),
    ],
    "quantity_discount": [
        (r"quantity discount|price[- ]breaks?|all[- ]units", 8), (r"discount", 6),
        (r"price schedule|pricing tiers?|tiered pric", 5), (r"\btiers?\b", 2),
        # price tables such as "0-999 units at $5.00" or "1,000+ units: $4.80"
        (r"\d[\d,]*\s*(?:units?\s*)?(?:-|to)\s*\d[\d,]*\s*(?:units?\s*)?(?:at|:|@|for|cost|costs)?\s*\$", 6),
        (r"\d(?:\s*\+|\s*(?:units?\s*)?or more)\s*(?:units?\s*)?(?:at|:|@|for|cost|costs)?\s*\$", 4),
        (r"\$\s*\d[\d.,]*\s*(?:per unit\s*|each\s*|/unit\s*)?(?:for|on) (?:orders? |purchases? )?(?:of |below |under |above |over |between )?\d", 5),
        (r"regardless of (?:the )?(?:unit )?price", 3),
        (r"or more|and above|\d\s*\+", 1),
    ],
    "fixed_period": [
        (r"fixed[- ](?:period|interval|time)|periodic review", 8), (r"review period|review interval", 7),
        (r"review(?:ed|s)? (?:the )?(?:inventory |stock |it )?(?:every|once|weekly|monthly|daily|biweekly)", 7),
        (r"order[- ]up[- ]to|up[- ]to level", 7), (r"target (?:inventory|stock)? ?level", 4),
        (r"\bp[- ](?:system|model)\b", 5), (r"(?:checked|counted|orders? (?:are )?placed) every", 4),
        (r"every \d+(?:\.\d+)? (?:days?|weeks?|months?)\b[^.]{0,50}(?:review|order|count|check)", 6),
        (r"(?:reviews?|checks?|counts?) (?:the )?(?:inventory|stock)\b", 3),
        (r"inventory position|on[- ]order|backorder", 2), (r"on[- ]hand|in stock", 1),
    ],
    "safety_stock": [
        (r"safety stock|buffer stock", 4), (r"re-?order point|\brop\b", 5), (r"continuous review|\bq[- ]system\b", 5),
        (r"service level|in-stock probability", 2), (r"stock-?outs?|run(?:ning)? out", 2),
        (r"when (?:should|to|do) .{0,20}re-?order", 3), (r"standard deviation|variab", 1), (r"lead[- ]?time", 1),
    ],
    "eoq": [
        (r"\beoq\b|economic order quantity|economic lot size", 8),
        (r"ordering cost|order cost|set-?up (?:cost|expense)|cost per order|cost (?:of|to) plac|placing (?:each|an|a) order", 3),
        (r"holding cost|carrying cost|storage cost", 3), (r"order quantity|order size|lot size", 2),
        (r"orders? per year|number of orders|how often", 2), (r"how (?:much|many) .{0,25}order", 1),
        (r"costs? \$\s?[\d.,]+ (?:to place|to order|per order|for each order|each time|to set ?up)", 3),
    ],
}
# Tie-break order: more specific models win ties.
MODEL_PRIORITY = ["newsvendor", "quantity_discount", "fixed_period", "safety_stock", "eoq"]

# --------------------------------------------------------------------------- #
# Regex building blocks
# --------------------------------------------------------------------------- #
_NUMBODY = r"(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"


def _num(name: str = "num") -> str:
    return rf"(?P<{name}>{_NUMBODY})(?:\s*(?P<{name}_scale>k|thousand|million)\b)?"


GAP = r"(?:(?!\.\s)[^\d$;!?\n%])"          # filler that contains no numbers
MGAP = r"(?:(?!\.\s)[^$;!?\n])"            # filler that may contain numbers (money search)
TIME_UNIT = r"(?:business |working |calendar )?(?P<unit>days?|weeks?|wks?|months?|years?|yrs?)\b"
PERIOD_TRAIL = re.compile(
    r"^\s*(?:[a-z]+\s+){0,2}?(?:per|a|an|each|every|/|in a|in one)\s*(?P<p>day|week|wk|month|mo|year|yr|annum)\b"
    r"|^\s*(?:[a-z]+\s+){0,2}?(?P<adj>daily|weekly|monthly|annually|yearly|annual)\b",
    re.I,
)
PERIOD_ALIASES = {
    "day": "day", "days": "day", "daily": "day",
    "week": "week", "weeks": "week", "wk": "week", "wks": "week", "weekly": "week",
    "month": "month", "months": "month", "mo": "month", "monthly": "month",
    "year": "year", "years": "year", "yr": "year", "yrs": "year", "annum": "year",
    "annual": "year", "annually": "year", "yearly": "year",
}
WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fourteen": 14, "fifteen": 15,
    "twenty": 20, "thirty": 30,
}
_SCALES = {"k": 1e3, "thousand": 1e3, "million": 1e6}


def _value(m: re.Match, name: str = "num") -> float:
    v = float(m.group(name).replace(",", ""))
    scale = m.groupdict().get(f"{name}_scale")
    return v * _SCALES[scale.lower()] if scale else v


def _period(word: Optional[str]) -> Optional[str]:
    return PERIOD_ALIASES.get(word.lower()) if word else None


def _alt(keywords: List[str]) -> str:
    return "(?:" + "|".join(keywords) + ")"


def preprocess(text: str) -> str:
    t = " " + text.replace("–", "-").replace("—", "-").replace("’", "'") + " "
    t = re.sub(r"(\d)\s*(?:percent|per cent|pct)\b", r"\1%", t, flags=re.I)
    t = re.sub(rf"({_NUMBODY})\s*(?:dollars?|usd)\b", r"$\1", t, flags=re.I)
    t = re.sub(rf"\busd\s*({_NUMBODY})", r"$\1", t, flags=re.I)
    t = re.sub(r"\$\s+", "$", t)
    for word, n in WORD_NUMBERS.items():
        t = re.sub(rf"\b{word}\b(?=[\s-]+(?:business |working )?(?:days?|weeks?|months?|years?|units?)\b)", str(n), t, flags=re.I)
    t = re.sub(r"\b(every|each)\s+(day|week|month|year)\b", r"\1 1 \2", t, flags=re.I)
    t = re.sub(r"\bonce (?:a|an|per|each|every) (day|week|month|year)\b", r"every 1 \1", t, flags=re.I)
    t = re.sub(r"\b(?:bi-?weekly|fortnightly)\b", "every 2 weeks", t, flags=re.I)
    return t


class _Scanner:
    """Holds the question text; matched spans are blanked so they are not reused."""

    def __init__(self, text: str):
        self.text = text

    def consume(self, m: re.Match) -> None:
        s, e = m.span()
        self.text = self.text[:s] + " " * (e - s) + self.text[e:]

    def trailing(self, m: re.Match, n: int = 40) -> str:
        return self.text[m.end(): m.end() + n]

    def search(self, pattern: str, flags: int = re.I) -> Optional[re.Match]:
        return re.search(pattern, self.text, flags)

    def after(self, keywords: List[str], money: bool = False, percent: Optional[bool] = None,
              window: int = 40) -> Optional[Tuple[float, re.Match, str]]:
        """Find '<keyword> ... <number>'.  Returns (value, match, trailing_text)."""
        kw = _alt(keywords)
        if money:
            pat = rf"{kw}{MGAP}{{0,{window}}}?\$\s*{_num()}"
        else:
            pat = rf"{kw}{GAP}{{0,{window}}}?\$?\s*{_num()}(?P<pct>\s*%)?"
        for m in re.finditer(pat, self.text, re.I):
            has_pct = bool(m.groupdict().get("pct"))
            if percent is True and not has_pct:
                continue
            if percent is False and has_pct:
                continue
            trail = self.trailing(m)
            self.consume(m)
            return _value(m), m, trail
        return None

    def before(self, keywords: List[str], money: bool = False, percent: Optional[bool] = None,
               window: int = 25) -> Optional[Tuple[float, re.Match, str]]:
        """Find '<number> ... <keyword>'."""
        kw = _alt(keywords)
        pat = rf"(?P<dollar>\$)?\s*{_num()}(?P<pct>\s*%)?{GAP}{{0,{window}}}?{kw}"
        for m in re.finditer(pat, self.text, re.I):
            if money and not m.group("dollar"):
                continue
            has_pct = bool(m.group("pct"))
            if percent is True and not has_pct:
                continue
            if percent is False and has_pct:
                continue
            trail = self.trailing(m)
            self.consume(m)
            return _value(m), m, trail
        return None


def _trail_period(trail: str) -> Optional[str]:
    m = PERIOD_TRAIL.search(trail)
    if not m:
        return None
    return _period(m.group("p") or m.group("adj"))


def _clause(text: str) -> str:
    """Cut text at the end of its clause so a period word from the next fact is not borrowed."""
    return re.split(r"[,;]|\.\s|\.$|\band\b|\bwhile\b|\bbut\b", text, maxsplit=1)[0]


def _find_period(text: str) -> Optional[str]:
    m = re.search(r"\bper (day|week|month|year|annum)\b|\ba (day|week|month|year)\b|/\s*(day|week|wk|month|mo|year|yr)\b"
                  r"|\b(annually|monthly|weekly|daily|yearly)\b", text, re.I)
    return _period(next(g for g in m.groups() if g)) if m else None


def _trail_time_unit(trail: str) -> Optional[str]:
    m = re.match(r"^\s*-?\s*" + TIME_UNIT, trail, re.I)
    return _period(m.group("unit")) if m else None


# --------------------------------------------------------------------------- #
# Rule-based extraction (question -> raw values with their stated units)
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Inventory terminology: synonyms recognised for each concept
# --------------------------------------------------------------------------- #
ORDERING_COST_TERMS = [
    r"order(?:ing)?\s*/\s*set-?up (?:costs?|expenses?)", r"set-?up\s*/\s*order(?:ing)? (?:costs?|expenses?)",
    r"order(?:ing)? (?:costs?|expenses?|fees?|charges?)", r"cost of ordering", r"order processing cost",
    r"set-?up (?:costs?|expenses?|fees?|charges?)(?: per (?:order|production run|run|batch|set-?up))?",
    r"cost (?:per|of each|for each|of an?) (?:order|set-?up|production run|batch|run)",
    r"costs? (?:of|to) plac(?:e|ing)(?: (?:an|each|a|one) (?:new )?order)?", r"placing (?:an|each|a|one) order costs?",
    r"fixed cost per (?:order|shipment|run|batch)", r"costs? per (?:order|set-?up|shipment|run|batch)",
    r"procurement cost(?: per order)?",
]
HOLDING_TERMS = [
    r"holding (?:cost|rate|charge|expense)s?", r"carrying (?:cost|rate|charge|expense)s?", r"inventory carrying(?: costs?)?",
    r"storage (?:cost|charge|expense)s?", r"cost of (?:holding|carrying|storing)", r"\bholding\b", r"\bcarrying\b",
    r"\bstorage\b", r"\bto (?:hold|carry|store)\b",
]
UNIT_COST_TERMS = [
    r"unit (?:purchase )?(?:cost|price)", r"cost per (?:unit|item)", r"per-unit (?:cost|price)", r"purchase (?:price|cost)",
    r"item (?:cost|price)", r"wholesale (?:price|cost)", r"price per unit",
    r"(?:regular|list|base|standard|normal|original|undiscounted) (?:unit )?price",
    r"buys? (?:\w+\s){0,3}?(?:for|at)", r"pays?", r"(?:bought|purchased|acquired) (?:for|at)", r"costs?",
    r"\bprice[ds]?\b", r"\bvalued? at",
]
UNDERAGE_TERMS = [r"\bunderage (?:cost|penalty)", r"\bcost of (?:under-?stocking|a shortage|being short|running short)",
                  r"cost (?:of|for|per) (?:each |every |a |one )?units? (?:short|of shortage|under-?stocked|not available)",
                  r"each unit short costs", r"lost (?:profit|margin) per unit", r"\bcu\s*(?:=|is|of)"]
OVERAGE_TERMS = [r"\boverage cost", r"\bexcess (?:inventory )?cost", r"\bcost of (?:over-?stocking|excess|surplus|leftovers?)",
                 r"cost (?:of|for|per) (?:each |every |a |one )?units? (?:left ?over|unsold|of excess|in excess|remaining|over-?stocked)",
                 r"each unit left ?over costs", r"\bco\s*(?:=|is|of)"]

STD_KW = r"(?:standard deviation|std\.?\s*dev\.?|stdev|\bstd\b|\bs\.?d\.?\b|σ|\bsigma\b)"
DEMAND_KW = (
    r"(?:(?P<adj>daily|weekly|monthly|annual|yearly)\s+)?(?:average\s+|mean\s+|expected\s+|forecast(?:ed)?\s+)?"
    r"(?:(?P<adj2>daily|weekly|monthly|annual|yearly)\s+)?(?:demand|usage|sales|consumption|requirements?)"
    r"|\b(?:sells|uses|consumes|needs|requires|averages?|ships)\b"
)


def _extract_symbols(sc: _Scanner, raw: Dict[str, Any]) -> None:
    """Explicit symbols such as 'D = 5000, S = $49, H = 2'."""
    symbols = {"D": "demand", "S": "ordering_cost", "H": "holding_cost", "C": "unit_cost",
               "L": "lead_time", "z": "z", "Z": "z", "i": "holding_rate", "I": "holding_rate"}
    for sym, key in symbols.items():
        m = re.search(rf"(?<![A-Za-z]){sym}\s*=\s*\$?\s*{_num()}(?P<pct>\s*%)?", sc.text)
        if m and key not in raw:
            val = _value(m)
            raw[key] = val
            sc.consume(m)


def _extract_days_per_year(sc: _Scanner, raw: Dict[str, Any]) -> None:
    m = sc.search(rf"{_num()}\s*(?:working|business|operating|production)\s*days")
    if not m:
        m = sc.search(rf"(?:operates?|open|works?)\s*{_num()}\s*days\s*(?:a|per|each)\s*year")
    if m:
        raw["days_per_year"] = _value(m)
        sc.consume(m)


def _extract_demand(sc: _Scanner, raw: Dict[str, Any]) -> None:
    # Explicit normal notation: N(500, 100)
    m = sc.search(rf"\bN\s*\(\s*{_num('a')}\s*,\s*{_num('b')}\s*\)")
    if m:
        raw["demand"], raw["std_demand"] = _value(m, "a"), _value(m, "b")
        sc.consume(m)
        return
    pat = rf"(?:{DEMAND_KW}){GAP}{{0,40}}?{_num()}(?!\s*%)"
    for m in re.finditer(pat, sc.text, re.I):
        gap_text = m.group(0)
        trail = sc.trailing(m)
        if re.search(r"lead|review|deviation|variab", gap_text, re.I):
            continue
        if re.match(r"^\s*-?\s*" + TIME_UNIT, trail, re.I) and not re.match(r"^\s*units?", trail, re.I):
            continue  # "needs 3 days" is a time, not demand
        raw["demand"] = _value(m)
        adj = m.groupdict().get("adj") or m.groupdict().get("adj2")
        period = _period(adj) or _trail_period(trail)
        if not period:
            pm = re.search(r"\b(daily|weekly|monthly|annual|yearly|per (?:day|week|month|year))\b", gap_text, re.I)
            if pm:
                period = _period(pm.group(1).split()[-1])
        if period:
            raw["demand_period"] = period
        sc.consume(m)
        return
    # Fallback: "12,000 units per year"
    m = sc.search(rf"(?<![$\d.,]){_num()}\s*(?:[a-z]+\s+){{0,2}}?(?:per|a|an|each|every|/)\s*(?P<p>day|week|month|year|yr)\b")
    if m:
        raw["demand"] = _value(m)
        raw["demand_period"] = _period(m.group("p"))
        sc.consume(m)


def _extract_mean_demand(sc: _Scanner, raw: Dict[str, Any]) -> None:
    if "demand" in raw:
        return
    _extract_demand(sc, raw)
    if "demand" in raw:
        return
    hit = sc.after([r"\bmean\b", r"\baverage\b", r"\bexpected\b", r"\bμ\b", r"\bmu\b"], percent=False)
    if hit:
        raw["demand"] = hit[0]
        p = _trail_period(hit[2])
        if p:
            raw["demand_period"] = p


def _extract_std(sc: _Scanner, raw: Dict[str, Any]) -> None:
    # 1) "standard deviation of (the) lead time is 2 days"
    m = sc.search(rf"{STD_KW}\s*(?:of|in|for)\s*(?:the\s*)?(?:supplier'?s?\s*)?lead[- ]?times?{GAP}{{0,25}}?{_num()}")
    if m:
        raw["std_lead_time"] = _value(m)
        unit = _trail_time_unit(sc.trailing(m))
        if unit:
            raw["std_lead_time_unit"] = unit
        sc.consume(m)
    # 2) remaining standard deviations
    pat = rf"{STD_KW}{GAP}{{0,40}}?{_num()}(?!\s*%)"
    while True:
        m = sc.search(pat)
        if not m:
            break
        trail = sc.trailing(m)
        unit = _trail_time_unit(trail)
        if unit and "std_lead_time" not in raw:
            raw["std_lead_time"] = _value(m)
            raw["std_lead_time_unit"] = unit
        elif "std_demand" not in raw:
            raw["std_demand"] = _value(m)
            pm = re.search(r"\b(daily|weekly|monthly|annual|yearly)\b", m.group(0), re.I)
            period = _period(pm.group(1)) if pm else _trail_period(trail)
            if period:
                raw["std_demand_period"] = period
        sc.consume(m)
    if "std_demand" not in raw:  # "demand varies by +/- 20 units"
        hit = sc.after([r"varies by (?:\+/-|±)?", r"(?:\+/-|±)"], percent=False, window=5)
        if hit:
            raw["std_demand"] = hit[0]


def _extract_lead_time(sc: _Scanner, raw: Dict[str, Any]) -> None:
    m = sc.search(rf"{_num()}\s*-?\s*{TIME_UNIT}\s*(?:of\s*)?lead[- ]?time")
    if m:
        raw["lead_time"], raw["lead_time_unit"] = _value(m), _period(m.group("unit"))
        sc.consume(m)
        return
    pat = (rf"(?:lead[- ]?time|delivery time|replenishment time|shipping time|takes|arrives? (?:in|after|within)|"
           rf"delivered (?:in|after|within)|to arrive|to be delivered|supplier needs){GAP}{{0,30}}?{_num()}")
    for m in re.finditer(pat, sc.text, re.I):
        unit = _trail_time_unit(sc.trailing(m))
        needs_unit = not re.match(r"lead", m.group(0), re.I)
        if not unit and (needs_unit or re.search(r"demand|cost|price|stock|order|units", m.group(0), re.I)):
            continue  # "lead time, daily demand is 30" -- the number belongs to demand
        raw["lead_time"] = _value(m)
        if unit:
            raw["lead_time_unit"] = unit
        sc.consume(m)
        return


def _extract_service_level(sc: _Scanner, raw: Dict[str, Any]) -> None:
    m = sc.search(rf"\bz\s*(?:-?\s*(?:score|value))?\s*(?:=|of|is|:)\s*(?P<neg>-)?{_num()}")
    if m:
        raw["z"] = -_value(m) if m.group("neg") else _value(m)
        sc.consume(m)
    sl_kw = [r"(?:cycle )?service level", r"\bcsl\b", r"in-?stock probability",
             r"probability of not (?:stocking|running) out", r"chance of not (?:stocking|running) out"]
    hit = sc.before(sl_kw, percent=True, window=20) or sc.after(sl_kw, percent=None, window=25)
    if hit:
        raw["service_level"] = hit[0]
        return
    so_kw = [r"(?:risk|chance|probability) of (?:a )?(?:stock-?out|running out|stocking out)",
             r"stock-?out (?:risk|probability|chance|rate)"]
    hit = sc.before(so_kw, percent=True, window=20) or sc.after(so_kw, window=25)
    if hit:
        v = hit[0] / 100 if hit[0] > 1 else hit[0]
        raw["service_level"] = 1 - v
        return
    hit = sc.before([r"(?:service|in-?stock|of the time|confiden)"], percent=True, window=15)
    if hit:
        raw["service_level"] = hit[0]


def _extract_review_period(sc: _Scanner, raw: Dict[str, Any]) -> None:
    m = sc.search(rf"{_num()}\s*-?\s*{TIME_UNIT}\s*(?:review|reorder|ordering|order) (?:period|interval|cycle)")
    if m:
        raw["review_period"], raw["review_period_unit"] = _value(m), _period(m.group("unit"))
        sc.consume(m)
        return
    kws = [r"review (?:period|interval|cycle)", r"review(?:ed|s)?\s*(?:the\s*)?(?:inventory\s*|stock\s*|it\s*)?(?:once\s*)?every",
           r"(?:checked|counted|reviewed) (?:once )?every", r"checks?\s*(?:inventory\s*|stock\s*)?every",
           r"orders? (?:are|is) placed (?:once )?every", r"places? (?:an )?orders? (?:once )?every",
           r"orders? (?:once )?every", r"\bevery"]
    for kw in kws:
        m = sc.search(rf"{kw}{GAP}{{0,15}}?{_num()}")
        if not m:
            continue
        unit = _trail_time_unit(sc.trailing(m))
        if not unit and kw == r"\bevery":
            continue
        raw["review_period"] = _value(m)
        if unit:
            raw["review_period_unit"] = unit
        sc.consume(m)
        return
    m = sc.search(r"(?:review(?:ed|s)?\s*(?:inventory\s*|stock\s*)?|order(?:s|ed)?\s*)(?P<adj>daily|weekly|monthly)"
                  r"|(?P<adj2>daily|weekly|monthly)\s*(?:review|basis|order)")
    if m:
        raw["review_period"] = 1.0
        raw["review_period_unit"] = _period(m.group("adj") or m.group("adj2"))
        sc.consume(m)


def _extract_given_safety_stock(sc: _Scanner, raw: Dict[str, Any]) -> None:
    hit = (sc.after([r"safety stock(?: level)?(?: is| of| =|:)?(?: (?:set|fixed|held|kept|maintained))?(?: at| to| of)?"],
                    percent=False, window=12)
           or sc.before([r"units? (?:of|as) safety stock"], percent=False, window=3))
    if hit:
        raw["safety_stock"] = hit[0]


def _extract_positions(sc: _Scanner, raw: Dict[str, Any]) -> None:
    hit = sc.after([r"inventory position"], percent=False, window=20)
    if hit:
        raw["inventory_position"] = hit[0]
    hit = (sc.before([r"(?:currently )?(?:on[- ]hand|in stock|in inventory|on the shel(?:f|ves)|in the warehouse)"], percent=False, window=20)
           or sc.after([r"on[- ]hand(?: inventory)?", r"inventory on hand", r"current (?:inventory|stock)(?: level)?",
                        r"currently (?:has|holds|have)", r"stock on hand", r"inventory level (?:is|of)"],
                       percent=False, window=20))
    if hit:
        raw["on_hand"] = hit[0]
    hit = (sc.before([r"(?:already |currently )?(?:on[- ]order|in transit|outstanding)"], percent=False, window=20)
           or sc.after([r"on[- ]order(?: inventory| stock| quantity)?", r"outstanding orders?", r"already ordered",
                        r"in transit", r"scheduled receipts?", r"open orders?"], percent=False, window=20))
    if hit:
        raw["on_order"] = hit[0]
    hit = sc.before([r"backorder(?:ed|s)?|back-orders?"], percent=False, window=15) or \
        sc.after([r"backorders?|back-orders?"], percent=False, window=15)
    if hit:
        raw["backorders"] = hit[0]


def _extract_holding(sc: _Scanner, raw: Dict[str, Any]) -> None:
    """Percentage holding cost (rate x unit price) or fixed holding cost ($ per unit per period)."""
    hit = sc.after(HOLDING_TERMS, percent=True, window=40)
    if not hit:
        hit = sc.before([r"(?:of (?:the )?(?:unit |item |purchase )?(?:cost|price|value)\s*)?(?:per year |annually |a year )?"
                         r"(?:for |as |to )?(?:holding|carrying|hold|carry|store|storage)"], percent=True, window=40)
    if hit:
        raw["holding_rate"] = hit[0]
        period = _find_period(hit[1].group(0)) or _find_period(_clause(hit[2]))
        if period:
            raw["holding_rate_period"] = period
        return
    hit = sc.after(HOLDING_TERMS, window=40)
    if not hit:
        hit = sc.before([r"(?:per unit\s*)?(?:per (?:year|month|week|day)|annually|a year|/year)?\s*to (?:hold|store|carry)"],
                        money=True, window=30)
    if hit:
        raw["holding_cost"] = hit[0]
        period = _find_period(_clause(hit[2])) or _find_period(hit[1].group(0))
        if period:
            raw["holding_cost_period"] = period


def _extract_ordering_cost(sc: _Scanner, raw: Dict[str, Any]) -> None:
    """Ordering cost = order cost = setup cost/expense = cost per order = cost of placing an order."""
    hit = sc.after(ORDERING_COST_TERMS, percent=False)
    if not hit:
        hit = sc.before([r"(?:per|for each|each|/|to place an?|to place each|for an?|for every|every time (?:it|they|we) "
                         r"(?:places?|orders?)(?: an?)?)\s*(?:new )?(?:order|setup|set-up|shipment|purchase order|production run)"],
                        money=True, window=10)
    if hit:
        raw["ordering_cost"] = hit[0]


def _extract_unit_cost(sc: _Scanner, raw: Dict[str, Any], money: bool = True) -> None:
    hit = sc.after(UNIT_COST_TERMS, money=money)
    if not hit:  # "the $12 unit cost", "$8 item cost", "$12 per unit"
        hit = sc.before([r"(?:per[- ]unit |purchase )?(?:unit cost|unit price|item cost|item price|cost per unit|purchase price|per unit|each)"],
                        money=True, window=5)
    if hit:
        raw["unit_cost"] = hit[0]


def _extract_price_breaks(sc: _Scanner, raw: Dict[str, Any]) -> None:
    breaks: List[Tuple[float, float]] = []
    pct_breaks: List[Tuple[float, float]] = []
    U = r"(?:\s*units?)?"
    patterns = [
        # 0-999 units at $5.00
        (rf"{_num('lo')}{U}\s*(?:-|to|through)\s*{_num('hi')}{U}\s*(?:[:@=,]|at|cost|costs|for|is|are|priced at|sell for)?\s*:?\s*\$\s*{_num('p')}", 0),
        # quantity break at 1000 units at a price of $4.50
        (rf"(?:quantity |price |volume )?break (?:at|of|for|starting at|begins at)\s*{_num('lo')}{U}{MGAP}{{0,30}}?\$\s*{_num('p')}", 0),
        # $7.60 for 2,000-4,999 units
        (rf"\$\s*{_num('p')}\s*(?:per unit\s*|each\s*|/unit\s*)?(?:for|on) (?:orders? |purchases? )?(?:of |between )?{_num('lo')}{U}\s*(?:-|to|and)\s*{_num('hi')}", 0),
        # $8.00 per unit for orders below 2,000 units
        (rf"\$\s*{_num('p')}\s*(?:per unit\s*|each\s*|/unit\s*)?(?:for|on) (?:orders? |purchases? )?(?:of )?(?:less than|fewer than|under|below|up to)\s*{_num('lo')}", "zero"),
        # $7.20 for 5,000 units or more
        (rf"\$\s*{_num('p')}\s*(?:per unit\s*|each\s*|/unit\s*)?(?:for|on) (?:orders? |purchases? )?(?:of )?{_num('lo')}{U}\s*(?:\+|or more|and (?:above|up|over|more)|or above|or greater)", 0),
        (rf"\$\s*{_num('p')}\s*(?:per unit\s*|each\s*|/unit\s*)?(?:for|on) (?:orders? |purchases? )?(?P<kw>at least|more than|over|above)\s*{_num('lo')}", "kw"),
        # less than 1000 units: $5.00
        (rf"(?:less than|fewer than|under|below|up to|upto)\s*{_num('lo')}{U}{MGAP}{{0,25}}?\$\s*{_num('p')}", "zero"),
        # at least / more than 1000 units: $4.80
        (rf"(?P<kw>at least|more than|over|minimum of|of at least|above)\s*{_num('lo')}{U}{MGAP}{{0,25}}?\$\s*{_num('p')}", "kw"),
        # 1000+ / 1000 or more units at $4.80
        (rf"{_num('lo')}{U}\s*(?:\+|or more|and (?:above|up|over|more)|or above|or greater|and greater|plus){U}{MGAP}{{0,25}}?\$\s*{_num('p')}", 0),
    ]
    for pat, mode in patterns:
        while True:
            m = sc.search(pat)
            if not m:
                break
            lo = 0.0 if mode == "zero" else _value(m, "lo")
            if mode == "kw" and m.group("kw").lower() in ("more than", "over", "above"):
                lo += 1
            breaks.append((lo, _value(m, "p")))
            sc.consume(m)
    # Percentage discounts: "2% discount for 1,000 or more", "1000+ units get 3% off"
    pct_patterns = [
        rf"{_num('pct')}\s*%\s*(?:discount|off)\s*(?:\w+\s){{0,6}}?(?:at least|of|for|on|over|above)?\s*{_num('lo')}",
        rf"{_num('lo')}{U}\s*(?:\+|or more|and above|or above)?{MGAP}{{0,25}}?{_num('pct')}\s*%\s*(?:discount|off)",
    ]
    for pat in pct_patterns:
        while True:
            m = sc.search(pat)
            if not m:
                break
            pct_breaks.append((_value(m, "lo"), _value(m, "pct")))
            sc.consume(m)
    if breaks:
        raw["price_breaks"] = [list(b) for b in breaks]
    if pct_breaks:
        raw["discount_breaks"] = [list(b) for b in pct_breaks]


def _extract_newsvendor_prices(sc: _Scanner, raw: Dict[str, Any]) -> None:
    hit = sc.after(UNDERAGE_TERMS, percent=False)
    if hit:
        raw["underage_cost"] = hit[0]
    hit = sc.after(OVERAGE_TERMS, percent=False)
    if hit:
        raw["overage_cost"] = hit[0]
    hit = sc.after([r"\bshortage costs?"], percent=False)   # meaning depends on whether prices are given
    if hit:
        raw["shortage_cost"] = hit[0]
    if sc.search(r"worthless|thrown (?:away|out)|discarded|no (?:salvage )?value|(?:spoil|expire)s? (?:and|at)|cannot be (?:sold|returned)"):
        raw["salvage_value"] = 0.0
    hit = sc.after([r"salvage(?: value| price)?", r"\bunsold\b", r"left ?over", r"returned (?:to the \w+ )?for",
                    r"\b(?:day-old|stale|expired|end-of-(?:season|day)|surplus|remaining|excess) \w+",
                    r"(?:sold|given|passed) (?:off )?to (?:a |an |the )?[\w -]{0,25}? for", r"(?:recycled|donated|liquidated) for",
                    r"buy-?back", r"marked down to", r"clearance(?: price)?", r"at a discount for",
                    r"scrap value", r"residual value", r"after the (?:season|event|day)"], money=True, window=60)
    if hit:
        raw["salvage_value"] = hit[0]
    hit = sc.after([r"lost goodwill", r"goodwill(?: cost| loss| penalty)?", r"shortage penalty", r"stock-?out (?:cost|penalty)",
                    r"penalty"], money=True)
    if hit:
        raw["shortage_penalty"] = hit[0]
    hit = sc.after([r"selling price", r"sale price", r"retail price", r"\bsells?\b", r"\bsold\s*(?:for|at)",
                    r"retails? (?:for|at)", r"revenue (?:of|per)", r"customers? pays?", r"charges?"], money=True, window=45)
    if hit:
        raw["selling_price"] = hit[0]
    _extract_unit_cost(sc, raw)
    if "selling_price" not in raw:
        hit = sc.after([r"\bprice\b"], money=True)
        if hit:
            raw["selling_price"] = hit[0]


def _extract_uniform(sc: _Scanner, raw: Dict[str, Any]) -> None:
    m = sc.search(rf"uniform(?:ly)?\s*(?:distributed\s*)?(?:between|from|over|on)?\s*[\[(]?\s*{_num('a')}{GAP}{{0,12}}?(?:and|to|-|,)\s*{_num('b')}")
    if not m and sc.search(r"uniform"):
        m = sc.search(rf"(?:between|from|range[sd]? from)\s*{_num('a')}\s*(?:units?\s*)?(?:and|to|-)\s*{_num('b')}")
    if m:
        raw["distribution"] = "uniform"
        raw["low"], raw["high"] = _value(m, "a"), _value(m, "b")
        sc.consume(m)


def rule_based_extract(question: str, model_key: str) -> Dict[str, Any]:
    """Return raw values exactly as stated in the question, with their units."""
    sc = _Scanner(preprocess(question))
    raw: Dict[str, Any] = {}
    _extract_symbols(sc, raw)
    _extract_days_per_year(sc, raw)

    if model_key in ("eoq", "quantity_discount"):
        if model_key == "quantity_discount":
            _extract_price_breaks(sc, raw)
        if "holding_rate" not in raw and "holding_cost" not in raw:
            _extract_holding(sc, raw)
        if "ordering_cost" not in raw:
            _extract_ordering_cost(sc, raw)
        if "unit_cost" not in raw:
            _extract_unit_cost(sc, raw)
        if "lead_time" not in raw:
            _extract_lead_time(sc, raw)
        if "demand" not in raw:
            _extract_demand(sc, raw)
    elif model_key in ("safety_stock", "fixed_period"):
        if model_key == "fixed_period":
            _extract_given_safety_stock(sc, raw)
        _extract_service_level(sc, raw)
        if model_key == "fixed_period":
            _extract_review_period(sc, raw)
            _extract_positions(sc, raw)
        _extract_std(sc, raw)
        if "lead_time" not in raw:
            _extract_lead_time(sc, raw)
        _extract_mean_demand(sc, raw)
    elif model_key == "newsvendor":
        _extract_newsvendor_prices(sc, raw)
        _extract_uniform(sc, raw)
        _extract_std(sc, raw)
        _extract_mean_demand(sc, raw)
        if "demand" not in raw and "low" not in raw:
            hit = sc.before([r"(?:units?|customers?|copies|items?)"], percent=False, window=10)
            if hit:
                raw["demand"] = hit[0]

    # Examine the whole question again before declaring anything missing.
    _, _, missing = normalize(model_key, raw)
    if missing:
        _rescue_missing(sc, raw, missing)
    raw["_unused"] = [_value(m) for m in re.finditer(rf"(?<![\w.]){_num()}", sc.text)]
    raw["_evidence"] = _evidence(preprocess(question), raw)
    return raw


# Broad terms used only by the second pass, for inputs still missing after the first pass.
RESCUE_TERMS: Dict[str, Tuple[str, List[str], str]] = {
    # missing input -> (raw field, words that signal it, kind: money | percent | count | time)
    "annual_demand": ("demand", [r"demand", r"usage", r"sales", r"requirements?", r"volume", r"consum\w*", r"sells?",
                                 r"uses?", r"needs?"], "count"),
    "mean_demand": ("demand", [r"demand", r"usage", r"sales", r"consum\w*", r"sells?", r"uses?", r"average", r"mean"], "count"),
    "ordering_cost": ("ordering_cost", [r"order\w*", r"set-?up", r"procure\w*", r"replenish\w*"], "money"),
    "holding": ("holding_cost", [r"hold\w*", r"carry\w*", r"stor\w*", r"warehous\w*"], "money"),
    "holding_qd": ("holding_rate", [r"hold\w*", r"carry\w*", r"stor\w*"], "percent"),
    "std_demand": ("std_demand", [r"deviation", r"variab\w*", r"σ", r"sigma", r"\bsd\b", r"\bstd\b", r"fluctuat\w*"], "count"),
    "lead_time": ("lead_time", [r"lead", r"deliver\w*", r"arriv\w*", r"replenish\w*", r"supplier", r"ship\w*"], "time"),
    "service_level": ("service_level", [r"service", r"fill", r"in-?stock", r"confiden\w*", r"availability"], "percent"),
    "review_period": ("review_period", [r"review\w*", r"every", r"interval", r"cycle", r"count\w*", r"check\w*"], "time"),
    "on_hand": ("on_hand", [r"hand", r"stock", r"inventory", r"shel\w*", r"warehouse", r"have"], "count"),
    "selling_price": ("selling_price", [r"sell\w*", r"sale", r"retail\w*", r"revenue", r"charge\w*"], "money"),
    "unit_cost": ("unit_cost", [r"cost\w*", r"buy\w*", r"purchas\w*", r"pay\w*", r"wholesale", r"make", r"produce"], "money"),
}


def _rescue_missing(sc: _Scanner, raw: Dict[str, Any], missing: List[str]) -> None:
    """Second pass: for each missing input, look in every remaining clause for a signal word and the
    nearest unused number of the right kind.  Anything found this way is flagged for the user to check."""
    for need in missing:
        if need not in RESCUE_TERMS:
            continue
        field_name, words, kind = RESCUE_TERMS[need]
        if field_name in raw:
            continue
        best = None
        for clause in re.finditer(r"[^.;!?\n]+", sc.text):
            ctext = clause.group(0)
            kw = re.search(_alt(words), ctext, re.I)
            if not kw:
                continue
            for m in re.finditer(rf"(?P<dollar>\$)?\s*(?<![\w.]){_num()}(?P<pct>\s*%)?", ctext):
                if kind == "money" and not m.group("dollar"):
                    continue
                if kind == "percent" and not m.group("pct"):
                    continue
                if kind in ("count", "time") and (m.group("dollar") or m.group("pct")):
                    continue
                unit = _trail_time_unit(ctext[m.end():m.end() + 20])
                if kind == "time" and not unit:
                    continue
                dist = abs(m.start() - kw.start())
                if best is None or dist < best[0]:
                    best = (dist, clause.start() + m.start("num"), clause.start() + m.end(), _value(m), unit, ctext.strip())
        if best:
            _, start, end, value, unit, ctext = best
            raw[field_name] = value
            if unit and kind == "time":
                raw[f"{field_name}_unit"] = unit
            if field_name == "demand":
                period = _find_period(_clause(sc.text[end:end + 40]))
                if period:
                    raw["demand_period"] = period
            raw.setdefault("_rescued", []).append(f"{field_name} = {value:g} (read from: \"{ctext}\")")
            sc.text = sc.text[:start] + " " * (end - start) + sc.text[end:]


RAW_LABELS = {
    "demand": "Demand (average)", "std_demand": "Demand standard deviation", "lead_time": "Lead time (average)",
    "std_lead_time": "Lead-time standard deviation", "review_period": "Review period", "service_level": "Service level",
    "z": "z-value", "ordering_cost": "Ordering / setup cost", "holding_cost": "Holding cost (fixed $ per unit)",
    "holding_rate": "Holding cost (% of unit price)", "unit_cost": "Unit purchase cost / price",
    "selling_price": "Selling price", "salvage_value": "Salvage value", "shortage_penalty": "Shortage penalty (goodwill)",
    "shortage_cost": "Shortage cost", "underage_cost": "Underage (shortage) cost", "overage_cost": "Overage (excess) cost",
    "on_hand": "On-hand inventory", "on_order": "On-order inventory", "backorders": "Backorders",
    "inventory_position": "Inventory position", "safety_stock": "Safety stock", "low": "Demand low (uniform)",
    "high": "Demand high (uniform)", "days_per_year": "Working days per year",
}
_UNIT_KEYS = {"demand": "demand_period", "std_demand": "std_demand_period", "lead_time": "lead_time_unit",
              "std_lead_time": "std_lead_time_unit", "review_period": "review_period_unit",
              "holding_cost": "holding_cost_period", "holding_rate": "holding_rate_period"}


def _evidence(text: str, raw: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Link each extracted number back to the clause of the question it came from."""
    tokens = [(m, _value(m)) for m in re.finditer(rf"(?<![\w.]){_num()}", text)]
    used: set = set()
    out: Dict[str, Dict[str, Any]] = {}
    for key, value in raw.items():
        if key not in RAW_LABELS or not isinstance(value, (int, float)):
            continue
        for idx, (m, v) in enumerate(tokens):
            if idx in used or abs(v - value) > 1e-9 * max(1.0, abs(value)):
                continue
            used.add(idx)
            s0 = max(text.rfind(c, 0, m.start()) for c in (". ", ";", ",", "\n", "(")) + 1
            ends = [i for i in (text.find(c, m.end()) for c in (". ", ";", ",", "\n", ")", "?")) if i != -1]
            e0 = min(ends) if ends else len(text)
            snippet = text[s0:e0].strip(" .")
            if len(snippet) > 90:  # keep the words nearest the number
                a = max(0, m.start() - s0 - 55)
                snippet = ("..." if a else "") + snippet[a:a + 90].strip() + "..."
            out[key] = {"label": RAW_LABELS[key], "value": value, "unit": raw.get(_UNIT_KEYS.get(key, ""), None),
                        "text": snippet, "pos": m.start()}
            break
        else:
            out[key] = {"label": RAW_LABELS[key], "value": value, "unit": raw.get(_UNIT_KEYS.get(key, ""), None),
                        "text": "", "pos": len(text)}
    return dict(sorted(out.items(), key=lambda kv: kv[1]["pos"]))


# --------------------------------------------------------------------------- #
# Normalization (raw values + units -> tool inputs).  All arithmetic is here.
# --------------------------------------------------------------------------- #
def _ratio(from_unit: str, to_unit: str, days_per_year: Optional[float]) -> float:
    """How many `to_unit` periods are in one `from_unit` period.

    Uses 52 weeks / 12 months per year, `days_per_year` days per year (365 unless
    working days are stated), and 7 days per week when no working-day count is given.
    """
    if from_unit == to_unit:
        return 1.0
    if {from_unit, to_unit} == {"day", "week"} and not days_per_year:
        return 7.0 if from_unit == "week" else 1 / 7.0
    per_year = {"year": 1.0, "month": 12.0, "week": 52.0, "day": days_per_year or 365.0}
    return per_year[to_unit] / per_year[from_unit]


def _fraction(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    return v / 100 if v >= 1 else v


def normalize(model_key: str, raw: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """Convert raw extracted values to tool inputs. Returns (inputs, assumptions, missing)."""
    raw = {k: v for k, v in raw.items() if v is not None}
    inputs: Dict[str, Any] = {}
    notes: List[str] = []
    missing: List[str] = []
    dpy = raw.get("days_per_year")

    def convert_time(value: float, unit: Optional[str], target: str, label: str) -> float:
        if not unit:
            notes.append(f"{label} assumed to be in {target}s (same unit as demand).")
            return value
        if unit == target:
            return value
        converted = value * _ratio(unit, target, dpy)
        notes.append(f"{label} converted: {value:g} {unit}(s) = {converted:,.4g} {target}(s).")
        return converted

    if model_key in ("eoq", "quantity_discount"):
        if "demand" in raw:
            period = raw.get("demand_period")
            if not period:
                period = "year"
                notes.append("Demand assumed to be annual (no time period stated).")
            factor = _ratio("year", period, dpy)
            inputs["annual_demand"] = raw["demand"] * factor
            if period != "year":
                notes.append(f"Annual demand = {raw['demand']:g} per {period} x {factor:g} = {inputs['annual_demand']:,.2f}.")
        else:
            missing.append("annual_demand")
        if "ordering_cost" in raw:
            inputs["ordering_cost"] = raw["ordering_cost"]
        else:
            missing.append("ordering_cost")
        if "holding_rate" in raw:
            rate = _fraction(raw["holding_rate"])
            rp = raw.get("holding_rate_period")
            if rp and rp != "year":
                factor = _ratio("year", rp, dpy)
                notes.append(f"Holding-cost rate annualized: {rate:.4g} per {rp} x {factor:g} = {rate * factor:.4g} per year.")
                rate *= factor
            inputs["holding_rate"] = rate
        elif "holding_cost" in raw:
            hp = raw.get("holding_cost_period")
            if hp and hp != "year":
                factor = _ratio("year", hp, dpy)
                inputs["holding_cost"] = raw["holding_cost"] * factor
                notes.append(f"Holding cost annualized: ${raw['holding_cost']:g} per {hp} x {factor:g} = ${inputs['holding_cost']:,.4f} per year.")
            else:
                inputs["holding_cost"] = raw["holding_cost"]
                if not hp:
                    notes.append("Holding cost assumed to be per unit per year.")
        if "unit_cost" in raw:
            inputs["unit_cost"] = raw["unit_cost"]

    if model_key == "eoq":
        if "holding_cost" not in inputs and not ("holding_rate" in inputs and "unit_cost" in inputs):
            missing.append("holding")
        if "lead_time" in raw:
            inputs["lead_time_days"] = convert_time(raw["lead_time"], raw.get("lead_time_unit") or "day", "day", "Lead time")
        if dpy:
            inputs["days_per_year"] = dpy

    elif model_key == "quantity_discount":
        breaks = [tuple(b) for b in raw.get("price_breaks", [])]
        base = raw.get("unit_cost")
        if raw.get("discount_breaks"):
            if base:
                for lo, pct in raw["discount_breaks"]:
                    price = base * (1 - _fraction(pct))
                    breaks.append((lo, price))
                    notes.append(f"{pct:g}% discount at {lo:g}+ units -> ${price:,.4f} per unit.")
            else:
                missing.append("unit_cost")
        if breaks and base and min(b[0] for b in breaks) > 1 and all(abs(b[1] - base) > 1e-9 for b in breaks):
            breaks.append((0.0, base))
            notes.append(f"Base price ${base:g} applies below the first price break.")
        if breaks and min(b[0] for b in breaks) > 1 and not base:
            notes.append(f"No base price given: assumed the minimum order is {min(b[0] for b in breaks):g} units.")
        if breaks:
            inputs["price_breaks"] = sorted(breaks)
            inputs.pop("unit_cost", None)
        else:
            missing.append("price_breaks")
        if "holding_rate" not in inputs and "holding_cost" not in inputs:
            missing.append("holding_qd")

    elif model_key in ("safety_stock", "fixed_period"):
        period = raw.get("demand_period") or raw.get("std_demand_period")
        if not period:
            period = raw.get("lead_time_unit") or "day"
            notes.append(f"Demand assumed to be per {period}.")
        inputs["period"] = period
        if "demand" in raw:
            inputs["mean_demand"] = raw["demand"]
        else:
            missing.append("mean_demand")
        if "std_demand" in raw:
            sp = raw.get("std_demand_period") or period
            sd = raw["std_demand"]
            if sp != period:
                ratio = _ratio(period, sp, dpy)
                sd = sd * math.sqrt(ratio)
                notes.append(f"Std. dev. converted from per-{sp} to per-{period}: {raw['std_demand']:g} x sqrt({ratio:.4g}) = {sd:,.4f} "
                             "(assumes independent periods).")
            inputs["std_demand"] = sd
        if "lead_time" in raw:
            inputs["lead_time"] = convert_time(raw["lead_time"], raw.get("lead_time_unit"), period, "Lead time")
        else:
            missing.append("lead_time")
        if "std_lead_time" in raw:
            inputs["std_lead_time"] = convert_time(raw["std_lead_time"], raw.get("std_lead_time_unit") or raw.get("lead_time_unit"),
                                                   period, "Lead-time std. dev.")
        has_sl = "z" in raw or "service_level" in raw
        has_var = bool(raw.get("std_demand")) or bool(raw.get("std_lead_time"))
        if "z" in raw:
            inputs["z"] = raw["z"]
        elif "service_level" in raw:
            inputs["service_level"] = _fraction(raw["service_level"])
        given_ss = model_key == "fixed_period" and "safety_stock" in raw
        if given_ss:
            inputs["safety_stock"] = raw["safety_stock"]
        if has_var and not has_sl and not given_ss:
            missing.append("service_level")
        if has_sl and not given_ss and "std_demand" not in raw and "std_lead_time" not in raw:
            missing.append("std_demand")  # a service level implies uncertainty; never assume it is zero
        if model_key == "fixed_period":
            if "review_period" in raw:
                inputs["review_period"] = convert_time(raw["review_period"], raw.get("review_period_unit"), period, "Review period")
            else:
                missing.append("review_period")
            for key in ("on_hand", "on_order", "backorders", "inventory_position"):
                if key in raw:
                    inputs[key] = raw[key]
            if "inventory_position" in raw:
                for key in ("on_hand", "on_order", "backorders"):
                    inputs.pop(key, None)
            elif "on_hand" not in raw:
                missing.append("on_hand")

    elif model_key == "newsvendor":
        for key in ("selling_price", "unit_cost", "salvage_value", "shortage_penalty", "underage_cost", "overage_cost"):
            if key in raw:
                inputs[key] = raw[key]
        if "shortage_cost" in raw and "underage_cost" not in raw:
            if "selling_price" in raw and "unit_cost" in raw:
                inputs["shortage_penalty"] = raw["shortage_cost"]
                notes.append("Shortage cost treated as a penalty on top of the lost margin, because price and cost are also given.")
            else:
                inputs["underage_cost"] = raw["shortage_cost"]
                notes.append("Shortage cost used as the underage cost (cost of each unit of unmet demand).")
        if "underage_cost" not in inputs and not ("selling_price" in inputs and "unit_cost" in inputs):
            missing += [k for k in ("selling_price", "unit_cost") if k not in inputs]
        if "overage_cost" not in inputs and "unit_cost" not in inputs and "unit_cost" not in missing:
            missing.append("overage")
        dist = raw.get("distribution")
        if dist == "uniform" or ("low" in raw and "high" in raw and "demand" not in raw):
            if "low" in raw and "high" in raw:
                inputs.update(demand_distribution="uniform", low=raw["low"], high=raw["high"])
            else:
                missing.append("demand_distribution")
        else:
            if "demand" in raw and "std_demand" in raw:
                inputs.update(demand_distribution="normal", mean=raw["demand"], std=raw["std_demand"])
                if dist != "normal":
                    notes.append("Demand assumed to be normally distributed.")
            else:
                missing.append("demand_distribution")
    return inputs, notes, missing


# --------------------------------------------------------------------------- #
# Optional Claude-powered understanding (identification + extraction only)
# --------------------------------------------------------------------------- #
_NUM_OR_NULL = {"anyOf": [{"type": "number"}, {"type": "null"}]}
_PERIOD_OR_NULL = {"anyOf": [{"type": "string", "enum": ["day", "week", "month", "year"]}, {"type": "null"}]}
_RAW_FIELDS = {
    "demand": _NUM_OR_NULL, "demand_period": _PERIOD_OR_NULL,
    "std_demand": _NUM_OR_NULL, "std_demand_period": _PERIOD_OR_NULL,
    "lead_time": _NUM_OR_NULL, "lead_time_unit": _PERIOD_OR_NULL,
    "std_lead_time": _NUM_OR_NULL, "std_lead_time_unit": _PERIOD_OR_NULL,
    "review_period": _NUM_OR_NULL, "review_period_unit": _PERIOD_OR_NULL,
    "ordering_cost": _NUM_OR_NULL, "holding_cost": _NUM_OR_NULL, "holding_cost_period": _PERIOD_OR_NULL,
    "holding_rate": _NUM_OR_NULL, "unit_cost": _NUM_OR_NULL, "service_level": _NUM_OR_NULL, "z": _NUM_OR_NULL,
    "on_hand": _NUM_OR_NULL, "on_order": _NUM_OR_NULL, "backorders": _NUM_OR_NULL,
    "inventory_position": _NUM_OR_NULL, "safety_stock": _NUM_OR_NULL, "holding_rate_period": _PERIOD_OR_NULL,
    "shortage_cost": _NUM_OR_NULL,
    "selling_price": _NUM_OR_NULL, "salvage_value": _NUM_OR_NULL, "shortage_penalty": _NUM_OR_NULL,
    "underage_cost": _NUM_OR_NULL, "overage_cost": _NUM_OR_NULL,
    "distribution": {"anyOf": [{"type": "string", "enum": ["normal", "uniform"]}, {"type": "null"}]},
    "low": _NUM_OR_NULL, "high": _NUM_OR_NULL, "days_per_year": _NUM_OR_NULL,
    "price_breaks": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["min_quantity", "unit_price"],
        "properties": {"min_quantity": {"type": "number"}, "unit_price": {"type": "number"}}}},
    "discount_breaks": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["min_quantity", "percent_off"],
        "properties": {"min_quantity": {"type": "number"}, "percent_off": {"type": "number"}}}},
}
LLM_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["model", "reason", "raw"],
    "properties": {
        "model": {"type": "string", "enum": list(MODEL_NAMES) + ["unsupported"]},
        "reason": {"type": "string"},
        "raw": {"type": "object", "additionalProperties": False, "required": list(_RAW_FIELDS), "properties": _RAW_FIELDS},
    },
}
LLM_SYSTEM = """You are the language-understanding front end of an inventory decision agent.
Your only jobs are to (1) pick which inventory model answers the question and (2) copy the numbers out of it.
A separate Python program does ALL arithmetic, so never calculate, convert, or annualize anything.

Models: eoq (economic order quantity), quantity_discount (EOQ with price breaks), safety_stock
(safety stock / reorder point, continuous review), fixed_period (periodic review / order-up-to level),
newsvendor (single-period stocking with selling price, cost, salvage). Use "unsupported" otherwise.

Extraction rules:
- Copy each number exactly as written in the question and record its stated time unit in the matching *_period / *_unit field.
- "demand" is the demand amount (for newsvendor and safety stock, the mean demand). Percentages: copy the number as written (20% -> 20).
- holding_rate is a percent of unit cost; holding_cost is a dollar amount. Record "per month" etc. in the matching *_period field.
- Ordering cost synonyms: order cost, setup cost/expense, cost per order, cost of placing an order.
- underage_cost = cost of each unit short (underage); overage_cost = cost of each unit left over (overage / excess).
  Use shortage_cost only when the question literally says "shortage cost" and its meaning is ambiguous.
- For periodic review, copy safety_stock and inventory_position if the question states them. A stockout risk of 5% means service_level 95 -- that one restatement is allowed.
- price_breaks: one entry per price tier with its minimum quantity (use 0 for the lowest tier). "more than 999" -> 1000.
- Use null for anything not stated. Do not guess values."""


def llm_extract(question: str, client: Any = None, model: str = "claude-opus-5-5") -> Tuple[str, str, Dict[str, Any]]:
    """Ask Claude to identify the model and copy out raw values. Returns (model_key, reason, raw)."""
    import anthropic  # imported lazily so the rule-based agent has no dependency on it

    client = client or anthropic.Anthropic()
    response = client.beta.messages.create(
        model=model,
        max_tokens=4000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=LLM_SYSTEM,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": LLM_SCHEMA}},
        messages=[{"role": "user", "content": question}],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("The model declined to process this question.")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("The model's response was cut off.")
    text = next(b.text for b in response.content if b.type == "text")
    data = json.loads(text)
    raw = {k: v for k, v in data["raw"].items() if v not in (None, [])}
    if "price_breaks" in raw:
        raw["price_breaks"] = [[b["min_quantity"], b["unit_price"]] for b in raw["price_breaks"]]
    if "discount_breaks" in raw:
        raw["discount_breaks"] = [[b["min_quantity"], b["percent_off"]] for b in raw["discount_breaks"]]
    return data["model"], data["reason"], raw


def _evidence_lines(evidence: Dict[str, Any]) -> List[str]:
    lines = []
    for e in evidence.values():
        unit = f" per {e['unit']}" if e.get("unit") and e["label"] not in ("Lead time (average)", "Review period",
                                                                               "Lead-time standard deviation") \
            else (f" {e['unit']}(s)" if e.get("unit") else "")
        src = f'  <- "{e["text"]}"' if e.get("text") else ""
        value = f"{e['value']:g}%" if e["label"] in ("Service level", "Holding cost (% of unit price)") and e["value"] > 1 \
            else f"{e['value']:g}"
        lines.append(f"  {e['label']}: {value}{unit}{src}")
    return lines


# --------------------------------------------------------------------------- #
# Explanation helpers
# --------------------------------------------------------------------------- #
def _n(x: float, d: int = 2) -> str:
    return f"{x:,.{d}f}"


def _units(x: float) -> str:
    return f"{x:,.1f}" if abs(x - round(x)) > 0.05 else f"{x:,.0f}"


def _headline(model_key: str, r: CalculationResult) -> List[str]:
    o, i = r.outputs, r.inputs
    if model_key == "eoq":
        lines = [
            f"Order about {_units(o['optimal_order_quantity'])} units each time "
            f"(round up to {math.ceil(o['optimal_order_quantity']):,} in practice).",
            f"That is {_n(o['orders_per_year'])} orders per year, roughly every "
            + (f"{_n(o['days_between_orders'], 1)} days." if "days_between_orders" in o
               else f"{_n(52 * o['cycle_time_years'], 1)} weeks."),
            f"Annual ordering cost ${_n(o['annual_ordering_cost'])} + holding cost ${_n(o['annual_holding_cost'])} "
            f"= total relevant cost ${_n(o['total_annual_relevant_cost'])}.",
        ]
        if "reorder_point" in o:
            lines.append(f"Reorder when inventory falls to {_units(o['reorder_point'])} units (demand during lead time, no safety stock).")
        return lines
    if model_key == "quantity_discount":
        lines = [f"Order {_units(o['optimal_order_quantity'])} units at ${o['unit_price']:,.2f} per unit "
                 f"for the lowest total annual cost of ${_n(o['total_annual_cost'])}."]
        for t in o["tiers"]:
            rng = f"{t['min_quantity']:,.0f}+" if t["max_quantity"] is None else f"{t['min_quantity']:,.0f}-{t['max_quantity'] - 1:,.0f}"
            if t["feasible"]:
                lines.append(f"  - ${t['unit_price']:,.2f} tier ({rng}): Q = {_units(t['order_quantity'])}, total cost ${_n(t['total_cost'])} ({t['note']})")
            else:
                lines.append(f"  - ${t['unit_price']:,.2f} tier ({rng}): not feasible ({t['note']})")
        return lines
    if model_key == "safety_stock":
        p = i["period"]
        return [
            f"Safety stock: {_units(o['safety_stock'])} units.",
            f"Reorder point: {_units(o['reorder_point'])} units "
            f"(expected lead-time demand {_units(o['expected_lead_time_demand'])} + safety stock).",
            (f"At a {o['service_level']:.1%} cycle service level (z = {o['z']:.3f}), the chance of a stockout in any "
             f"replenishment cycle is {1 - o['service_level']:.1%}; expected shortage per cycle about {_n(o['expected_shortage_per_cycle'])} units."
             if o["z"] is not None else "Demand and lead time are both constant, so no safety stock is needed."),
            f"(Demand and time measured per {p}.)",
        ]
    if model_key == "fixed_period":
        return [
            f"Order {_units(o['order_quantity'])} units now.",
            f"Target (order-up-to) level: {_units(o['target_inventory_level'])} units, covering {i['review_period']:g} + "
            f"{i['lead_time']:g} = {o['protection_interval']:g} {i['period']}s of protection.",
            (f"Safety stock inside the target: {_units(o['safety_stock'])} units at a {o['service_level']:.1%} service level (z = {o['z']:.3f})."
             if o["z"] is not None else "No safety stock needed: demand and lead time are constant."),
            f"Current inventory position: {_units(o['inventory_position'])} units.",
        ]
    if model_key == "newsvendor":
        return [
            f"Stock {_units(o['optimal_order_quantity'])} units (round to {round(o['optimal_order_quantity']):,}).",
            f"Critical ratio = {o['critical_ratio']:.4f}: it costs ${o['underage_cost']:,.2f} for each unit short and "
            f"${o['overage_cost']:,.2f} for each unit left over, so stock where P(demand <= Q) = {o['critical_ratio']:.1%}.",
            f"Expected sales {_units(o['expected_sales'])}, leftover {_units(o['expected_leftover'])}, "
            f"shortage {_units(o['expected_lost_sales'])}; "
            + (f"expected profit ${_n(o['expected_profit'])}." if "expected_profit" in o
               else f"expected cost of leftovers + shortages ${_n(o['expected_mismatch_cost'])}."),
        ]
    return []


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #
@dataclass
class AgentResponse:
    question: str
    status: str                       # ok | needs_input | unsupported | error
    model_key: Optional[str] = None
    model_name: Optional[str] = None
    identification_reason: str = ""
    raw_extraction: Dict[str, Any] = field(default_factory=dict)
    inputs: Dict[str, Any] = field(default_factory=dict)
    assumptions: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    unused_numbers: List[float] = field(default_factory=list)
    tool: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    checks: List[Dict[str, Any]] = field(default_factory=list)
    explanation: str = ""
    trace: List[str] = field(default_factory=list)

    @property
    def verified(self) -> bool:
        return bool(self.checks) and all(c["passed"] for c in self.checks)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["verified"] = self.verified
        return d


class InventoryAgent:
    """Customer Question -> Identify -> Extract -> Select Tool -> Calculate -> Verify -> Explain."""

    def __init__(self, use_llm: bool = False, client: Any = None, llm_model: str = "claude-opus-5-5"):
        self.use_llm = use_llm
        self.client = client
        self.llm_model = llm_model

    # 1. Identify model -------------------------------------------------------
    @staticmethod
    def identify_model(question: str) -> Tuple[Optional[str], str, Dict[str, int]]:
        text = question.lower()
        scores: Dict[str, int] = {}
        hits: Dict[str, List[str]] = {}
        for key, pats in MODEL_KEYWORDS.items():
            scores[key] = 0
            hits[key] = []
            for pat, weight in pats:
                m = re.search(pat, text)
                if m:
                    scores[key] += weight
                    hits[key].append(m.group(0))
        has_costs = re.search(r"order(?:ing)? cost|set-?up (?:cost|expense)|cost per order|to place (?:an|each|a) order|per order", text) \
            and re.search(r"holding|carrying|storage", text)
        has_uncertainty = re.search(r"service level|standard deviation|std\.? ?dev|\bsd\b|σ|safety stock|stock-?out|in-stock|variab", text)
        if has_costs and not has_uncertainty and scores["safety_stock"] >= scores["eoq"]:
            scores["eoq"] = scores["safety_stock"] + 1
            hits["eoq"].append("ordering + holding costs with no demand uncertainty")
        best = max(scores.values())
        if best == 0:
            return None, "No inventory-model keywords found.", scores
        key = next(k for k in MODEL_PRIORITY if scores[k] == best)
        reason = f"Matched cues: {', '.join(repr(h) for h in hits[key])} (score {best})."
        return key, reason, scores

    # 2. Extract inputs -------------------------------------------------------
    @staticmethod
    def extract_inputs(question: str, model_key: str) -> Dict[str, Any]:
        return rule_based_extract(question, model_key)

    # 3. Select tool ----------------------------------------------------------
    @staticmethod
    def select_tool(model_key: str) -> Callable[..., CalculationResult]:
        return tools.TOOLS[model_key]

    # 4-5. Calculate and verify ----------------------------------------------
    @staticmethod
    def calculate(model_key: str, inputs: Dict[str, Any]) -> CalculationResult:
        return InventoryAgent.select_tool(model_key)(**inputs)

    @staticmethod
    def verify(result: CalculationResult) -> List[Dict[str, Any]]:
        return tools.verify(result)

    # 6. Explain --------------------------------------------------------------
    @staticmethod
    def explain(model_key: str, result: CalculationResult, assumptions: List[str], checks: List[Dict[str, Any]],
                evidence: Optional[Dict[str, Any]] = None, unused: Optional[List[float]] = None) -> str:
        """Student-facing worked solution.  Every number comes from the Python tool's own record of its
        calculation (result.to_text); this method only adds the words each input was read from."""
        header = [f"Model: {MODEL_NAMES[model_key]}" + (f"  (case: {result.case.replace('_', ' ')})" if result.case else ""),
                  f"Answer: {result.answer}", ""]
        from_question = _evidence_lines(evidence or {})
        if unused:
            from_question.append("  Not used: " + ", ".join(f"{u:g}" for u in unused) + " (check whether these matter)")
        agent_notes = [a for a in assumptions if a not in result.assumptions]
        return "\n".join(header) + result.to_text(from_question or None, agent_notes)

    # Full pipeline -----------------------------------------------------------
    def run(self, question: str, overrides: Optional[Dict[str, Any]] = None,
            model_key: Optional[str] = None) -> AgentResponse:
        resp = AgentResponse(question=question, status="error")
        raw: Dict[str, Any] = {}

        # Identify + extract
        if self.use_llm and model_key is None:
            try:
                key, reason, raw = llm_extract(question, self.client, self.llm_model)
                resp.trace.append(f"Identify + extract: Claude ({self.llm_model}) chose '{key}'.")
                model_key, resp.identification_reason = (None if key == "unsupported" else key), reason
            except Exception as exc:  # fall back to the deterministic parser
                resp.trace.append(f"Claude unavailable ({type(exc).__name__}: {exc}); using rule-based parser.")
                resp.assumptions.append("Language model unavailable; rule-based parser used.")
                raw = {}
        if model_key is None and not raw:
            model_key, resp.identification_reason, scores = self.identify_model(question)
            resp.trace.append(f"Identify model: keyword scores {scores}.")
        elif resp.identification_reason == "":
            resp.identification_reason = "Model specified by caller."
        if model_key is None:
            resp.status = "unsupported"
            resp.explanation = ("I could not match this question to a supported model. I can answer EOQ, quantity discount, "
                                "safety stock / reorder point, fixed-period review, and newsvendor questions.")
            return resp
        resp.model_key, resp.model_name = model_key, MODEL_NAMES[model_key]

        if not raw:
            raw = self.extract_inputs(question, model_key)
        resp.evidence = raw.pop("_evidence", {})
        resp.unused_numbers = raw.pop("_unused", [])
        for note in raw.pop("_rescued", []):
            resp.assumptions.append(f"Read on a second pass (please check): {note}")
        if overrides:
            overrides = {k: v for k, v in overrides.items() if not str(k).startswith("_")}
            raw.update({k: v for k, v in overrides.items() if v is not None})
        resp.raw_extraction = raw
        resp.trace.append(f"Extract inputs: {raw}.")

        inputs, notes, missing = normalize(model_key, raw)
        resp.inputs, resp.missing = inputs, missing
        resp.assumptions += notes
        resp.trace.append(f"Normalize: {inputs}.")
        if missing:
            resp.status = "needs_input"
            found = _evidence_lines(resp.evidence)
            resp.explanation = (f"Model identified: {MODEL_NAMES[model_key]}. To calculate it I still need: "
                                + "; ".join(INPUT_LABELS[m] for m in missing) + "."
                                + ("\n\nAlready read from your question:\n" + "\n".join(found) if found else ""))
            return resp

        resp.tool = TOOL_NAMES[model_key]
        resp.trace.append(f"Select tool: {resp.tool}.")
        try:
            result = self.calculate(model_key, inputs)
        except InventoryInputError as exc:
            if exc.missing:
                resp.status, resp.missing = "needs_input", exc.missing
                resp.explanation = f"Model identified: {MODEL_NAMES[model_key]}. {exc}"
                return resp
            resp.status = "error"
            resp.explanation = f"The inputs are not valid for {MODEL_NAMES[model_key]}: {exc}"
            return resp
        resp.result = result.to_dict()
        resp.assumptions += [a for a in result.assumptions if a not in resp.assumptions]
        resp.trace.append(f"Calculate: {result.answer}")

        resp.checks = result.checks  # verification run by the tool itself
        resp.trace.append(f"Verify: {sum(c['passed'] for c in resp.checks)}/{len(resp.checks)} checks passed.")
        resp.explanation = self.explain(model_key, result, resp.assumptions, resp.checks, resp.evidence, resp.unused_numbers)
        resp.status = "ok"
        return resp
