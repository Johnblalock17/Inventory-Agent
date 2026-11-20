"""
inventory_tools.py
==================

Deterministic Python tools for five inventory decision models.  Every number
the agent reports -- including every intermediate step shown to students --
is produced here, never by a language model.

Tools
-----
    calculate_eoq                Economic Order Quantity           Q* = sqrt(2DS/H)
    calculate_quantity_discount  All-units quantity discount       compare total annual cost of feasible quantities
    calculate_rop                Safety stock + reorder point      ROP = dL + z * sigma_LT
    calculate_periodic_review    Fixed-period (order-up-to)        order = max(0, target - inventory position)
    calculate_newsvendor         Single-period stocking            Q* = F^-1(Cu / (Cu + Co))

Worked solution (every tool)
----------------------------
Each tool records its working *while it calculates*, so students can check
every number by hand.  ``result.worked_solution()`` returns eight sections:

    1. inputs_extracted         every input used, its symbol, unit, and whether it was given or a stated default
    2. variable_definitions     every symbol that appears in a formula
    3. formula                  each formula in symbols
    4. substituted              the same formula with the numbers put in
    5. intermediate             the value of each step
    6. final_answer             the recommendation and its key numbers
    7. verification_checks      independent checks run on the result
    8. assumptions              model assumptions and any interpretation of optional inputs

The substituted expressions use plain numbers and the operators x, /, +, -, sqrt,
Phi, Phi^-1 and phi, so each one can be re-typed into a calculator (the test suite
re-evaluates every one and checks it reproduces the stated value).

Contract
--------
* Inputs are validated; ``InventoryInputError`` lists *all* missing required
  inputs in ``.missing``.  The tools never invent a missing number.
* Optional inputs (salvage value, shortage penalty, on-order stock, standard
  deviations, days per year) have a defined meaning when omitted, and every such
  default is listed as "not given" in section 1 and explained in section 8.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from statistics import NormalDist
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

STD_NORMAL = NormalDist(0.0, 1.0)
TOL = 1e-6
DEFAULT_DAYS_PER_YEAR = 365.0

SECTION_TITLES = [
    ("inputs_extracted", "1. Inputs Extracted"),
    ("variable_definitions", "2. Variable Definitions"),
    ("formula", "3. Formula"),
    ("substituted", "4. Formula with Substituted Values"),
    ("intermediate", "5. Intermediate Calculations"),
    ("final_answer", "6. Final Answer"),
    ("verification_checks", "7. Verification Checks"),
    ("assumptions", "8. Assumptions"),
]


class InventoryInputError(ValueError):
    """Raised when inputs are missing or invalid.  ``missing`` lists absent required inputs."""

    def __init__(self, message: str, missing: Optional[List[str]] = None):
        super().__init__(message)
        self.missing = missing or []


# --------------------------------------------------------------------------- #
# Number formatting for substituted expressions (plain, calculator-friendly)
# --------------------------------------------------------------------------- #
def _n(x: Optional[float]) -> str:
    """Format a number for a worked step: integers as integers, otherwise 4 decimals
    (6 significant digits for small values), negatives in parentheses."""
    if x is None:
        return "-"
    if abs(x - round(x)) < 1e-9 and abs(x) < 1e15:
        s = str(int(round(x)))
    elif abs(x) >= 1:
        s = f"{x:.4f}".rstrip("0").rstrip(".")
    else:
        s = f"{x:.6g}"
    return f"({s})" if x < 0 else s


def _show(x: Optional[float], unit: str = "") -> str:
    """Human-readable value with thousands separators for sections 1, 5 and 6."""
    if x is None:
        return "-"
    if abs(x - round(x)) < 1e-9:
        s = f"{x:,.0f}"
    elif abs(x) >= 1:
        s = f"{x:,.4f}".rstrip("0").rstrip(".")
    else:
        s = f"{x:.6g}"
    return f"{s} {unit}".strip()


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class CalculationResult:
    model: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    steps: List[str] = field(default_factory=list)            # one-line text of each worked step
    intermediate: Dict[str, Any] = field(default_factory=dict)  # step key -> value
    answer: str = ""
    assumptions: List[str] = field(default_factory=list)
    case: Optional[str] = None
    title: str = ""
    inputs_extracted: List[Dict[str, Any]] = field(default_factory=list)
    variables: List[Dict[str, Any]] = field(default_factory=list)
    work: List[Dict[str, Any]] = field(default_factory=list)
    final_values: List[Dict[str, Any]] = field(default_factory=list)
    checks: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["worked_solution"] = self.worked_solution()
        return d

    def worked_solution(self) -> Dict[str, Any]:
        """The eight student-facing sections, all built from values computed by this tool."""
        calc = [w for w in self.work if w["value"] is not None]
        return {
            "inputs_extracted": [dict(i, display=_show(i["value"], i["unit"])) for i in self.inputs_extracted],
            "variable_definitions": self.variables,
            "formula": [{"key": w["key"], "symbol": w["symbol"], "formula": w["formula"]} for w in self.work],
            "substituted": [{"key": w["key"], "symbol": w["symbol"], "expression": w["substituted"],
                             "line": f"{w['symbol']} = {w['substituted']}" if w["value"] is not None else w["substituted"]}
                            for w in self.work],
            "intermediate": [{"key": w["key"], "symbol": w["symbol"], "meaning": w["meaning"], "value": w["value"],
                              "unit": w["unit"], "display": _show(w["value"], w["unit"])} for w in calc],
            "final_answer": {"text": self.answer, "case": self.case,
                             "values": [dict(v, display=_show(v["value"], v["unit"])) for v in self.final_values]},
            "verification_checks": self.checks,
            "assumptions": self.assumptions,
        }

    def to_text(self, from_question: Optional[List[str]] = None, extra_assumptions: Optional[List[str]] = None) -> str:
        """Plain-text worked solution in the eight numbered sections."""
        ws = self.worked_solution()
        out: List[str] = []
        out.append(SECTION_TITLES[0][1])
        if from_question:
            out.append("  Read from the question:")
            out += [f"  {line}" for line in from_question]
            out.append("  Values used by the calculation:")
        for i in ws["inputs_extracted"]:
            src = "" if i["source"] == "given" else "  [not given - default, see Assumptions]"
            out.append(f"  - {i['label']} ({i['symbol']}): {_show(i['value'], i['unit'])}{src}")
        out += ["", SECTION_TITLES[1][1]]
        out += [f"  - {v['symbol']}: {v['meaning']}" + (f" ({v['unit']})" if v["unit"] else "") for v in ws["variable_definitions"]]
        out += ["", SECTION_TITLES[2][1]] + [f"  {f['formula']}" for f in ws["formula"]]
        out += ["", SECTION_TITLES[3][1]] + [f"  {s['line']}" for s in ws["substituted"]]
        out += ["", SECTION_TITLES[4][1]] + [f"  {c['symbol']} = {c['display']}" for c in ws["intermediate"]]
        out += ["", SECTION_TITLES[5][1], f"  {self.answer}"]
        out += [f"  - {v['label']}: {_show(v['value'], v['unit'])}" for v in self.final_values]
        passed = sum(c["passed"] for c in self.checks)
        out += ["", f"{SECTION_TITLES[6][1]} ({passed}/{len(self.checks)} passed)"]
        out += [f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['name']} - {c['detail']}" for c in self.checks]
        notes = list(extra_assumptions or []) + [a for a in self.assumptions if a not in (extra_assumptions or [])]
        out += ["", SECTION_TITLES[7][1]] + [f"  - {a}" for a in notes]
        return "\n".join(out)


class _Work:
    """Records inputs, variable definitions and each calculation step as the tool computes it."""

    def __init__(self) -> None:
        self.inputs: List[Dict[str, Any]] = []
        self.variables: List[Dict[str, Any]] = []
        self.steps: List[Dict[str, Any]] = []
        self.assumptions: List[str] = []
        self.finals: List[Dict[str, Any]] = []

    def _define(self, symbol: str, meaning: str, unit: str) -> None:
        if all(v["symbol"] != symbol for v in self.variables):
            self.variables.append({"symbol": symbol, "meaning": meaning, "unit": unit})

    def given(self, key: str, symbol: str, label: str, value: Any, unit: str = "") -> Any:
        self.inputs.append({"key": key, "symbol": symbol, "label": label, "value": value, "unit": unit, "source": "given"})
        self._define(symbol, label, unit)
        return value

    def default(self, key: str, symbol: str, label: str, value: float, unit: str, assumption: str) -> float:
        """An optional input that was not supplied: record the default AND say so in the assumptions."""
        self.inputs.append({"key": key, "symbol": symbol, "label": label, "value": value, "unit": unit,
                            "source": "not given (default)"})
        self._define(symbol, label, unit)
        self.assumptions.append(assumption)
        return value

    def calc(self, key: str, symbol: str, meaning: str, formula: str, substituted: str, value: float,
             unit: str = "") -> float:
        self.steps.append({"key": key, "symbol": symbol, "meaning": meaning, "formula": formula,
                           "substituted": substituted, "value": value, "unit": unit})
        self._define(symbol, meaning, unit)
        return value

    def note(self, key: str, symbol: str, meaning: str, formula: str, text: str) -> None:
        """A decision step with no numeric value (e.g. a price tier that is infeasible)."""
        self.steps.append({"key": key, "symbol": symbol, "meaning": meaning, "formula": formula,
                           "substituted": text, "value": None, "unit": ""})
        self._define(symbol, meaning, "")

    def final(self, label: str, value: Any, unit: str = "") -> None:
        self.finals.append({"label": label, "value": value, "unit": unit})

    def result(self, model: str, title: str, inputs: Dict[str, Any], outputs: Dict[str, Any], answer: str,
               case: Optional[str] = None, extra_intermediate: Optional[Dict[str, Any]] = None) -> CalculationResult:
        intermediate = {s["key"]: s["value"] for s in self.steps if s["value"] is not None}
        intermediate.update(extra_intermediate or {})
        lines = [f"{s['formula']} = {s['substituted']} = {_show(s['value'], s['unit'])}" if s["value"] is not None
                 else f"{s['formula']}: {s['substituted']}" for s in self.steps]
        r = CalculationResult(model=model, inputs=inputs, outputs=outputs, steps=lines, intermediate=intermediate,
                              answer=answer, assumptions=self.assumptions, case=case, title=title,
                              inputs_extracted=self.inputs, variables=self.variables, work=self.steps,
                              final_values=self.finals)
        r.checks = verify(r)
        return r


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #
def _require(**values: Any) -> None:
    """Raise one error naming every required input that is missing."""
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise InventoryInputError(f"Missing required input(s): {', '.join(missing)}.", missing)


def _number(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise InventoryInputError(f"'{name}' must be a number (got {value!r}).")
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise InventoryInputError(f"'{name}' must be a number (got {value!r}).")
    if not math.isfinite(value):
        raise InventoryInputError(f"'{name}' must be finite.")
    return value


def _positive(name: str, value: Any) -> float:
    value = _number(name, value)
    if value <= 0:
        raise InventoryInputError(f"'{name}' must be greater than 0 (got {value:g}).")
    return value


def _non_negative(name: str, value: Any) -> float:
    value = _number(name, value)
    if value < 0:
        raise InventoryInputError(f"'{name}' cannot be negative (got {value:g}).")
    return value


def _fraction(name: str, value: Any) -> float:
    value = _number(name, value)
    if not 0 < value <= 1:
        raise InventoryInputError(f"'{name}' must be a fraction between 0 and 1, e.g. 0.20 for 20% (got {value:g}).")
    return value


# --------------------------------------------------------------------------- #
# Statistics helpers
# --------------------------------------------------------------------------- #
def z_from_service_level(service_level: float) -> float:
    """z such that P(Z <= z) = service_level for a standard normal Z."""
    service_level = _number("service_level", service_level)
    if not 0 < service_level < 1:
        raise InventoryInputError(
            f"service_level must be strictly between 0 and 1, e.g. 0.95 (got {service_level:g}).")
    return STD_NORMAL.inv_cdf(service_level)


def service_level_from_z(z: float) -> float:
    return STD_NORMAL.cdf(_number("z", z))


def standard_normal_loss(z: float) -> float:
    """Standard normal loss function L(z) = E[(Z - z)+] = phi(z) - z(1 - Phi(z))."""
    return STD_NORMAL.pdf(z) - z * (1.0 - STD_NORMAL.cdf(z))


def _variability_case(sd: float, sL: float) -> Tuple[str, str]:
    var_d, var_L = sd > 0, sL > 0
    if var_d and var_L:
        return "variable_demand_variable_lead_time", "variable demand and variable lead time"
    if var_d:
        return "variable_demand_constant_lead_time", "variable demand and constant lead time"
    if var_L:
        return "constant_demand_variable_lead_time", "constant demand and variable lead time"
    return "constant_demand_constant_lead_time", "constant demand and constant lead time"


def _optional_std(w: _Work, key: str, symbol: str, label: str, value: Optional[float], unit: str,
                  what: str) -> float:
    if value is None:
        return w.default(key, symbol, label, 0.0, unit,
                         f"No {label.lower()} provided: {what} treated as constant ({symbol} = 0).")
    return w.given(key, symbol, label, _non_negative(key, value), unit)


def _sigma_step(w: _Work, key: str, sym: str, meaning: str, d: float, sd: float, T: float, sL: float,
                Tsym: str, unit: str) -> float:
    """sigma over T periods for the four variability cases (records the case-specific formula)."""
    if sd > 0 and sL > 0:
        return w.calc(key, sym, meaning, f"{sym} = √({Tsym} × σ_d² + d² × σ_L²)",
                      f"√({_n(T)} × {_n(sd)}² + {_n(d)}² × {_n(sL)}²)", math.sqrt(T * sd ** 2 + d ** 2 * sL ** 2), unit)
    if sd > 0:
        return w.calc(key, sym, meaning, f"{sym} = σ_d × √{Tsym}", f"{_n(sd)} × √({_n(T)})", sd * math.sqrt(T), unit)
    if sL > 0:
        return w.calc(key, sym, meaning, f"{sym} = d × σ_L", f"{_n(d)} × {_n(sL)}", d * sL, unit)
    return w.calc(key, sym, meaning, f"{sym} = 0 (demand and lead time are both constant)", "0", 0.0, unit)


def _z_step(w: _Work, service_level: Optional[float], z: Optional[float]) -> Tuple[float, float]:
    """Record z (from the service level) or the service level (from z)."""
    if z is not None:
        zv = w.given("z", "z", "Safety factor (z-value)", _number("z", z))
        sl = w.calc("service_level", "SL", "Cycle service level implied by z", "SL = Φ(z)", f"Φ({_n(zv)})",
                    STD_NORMAL.cdf(zv))
        return zv, sl
    _require(service_level_or_z=service_level)
    sl = w.given("service_level", "SL", "Cycle service level (probability of no stockout per cycle)",
                 _number("service_level", service_level))
    zv = w.calc("z", "z", "Safety factor: standard normal value for the service level", "z = Φ⁻¹(SL)",
                f"Φ⁻¹({_n(sl)})", z_from_service_level(sl))
    return zv, sl


# --------------------------------------------------------------------------- #
# 1. Economic Order Quantity
# --------------------------------------------------------------------------- #
def calculate_eoq(
    annual_demand: Optional[float] = None,
    ordering_cost: Optional[float] = None,
    holding_cost: Optional[float] = None,
    holding_rate: Optional[float] = None,
    unit_cost: Optional[float] = None,
    lead_time_days: Optional[float] = None,
    days_per_year: Optional[float] = None,
) -> CalculationResult:
    """
    Q* = sqrt(2DS / H).

    Holding cost H is either given directly ($/unit/year) or as holding_rate x unit_cost
    (holding_rate as a fraction, 0.20 = 20%).  lead_time_days is optional; when given the
    tool also reports the reorder point d x L (no safety stock).
    """
    if holding_cost is None and holding_rate is None:
        _require(annual_demand=annual_demand, ordering_cost=ordering_cost, holding_cost_or_holding_rate=None)
    if holding_cost is None:
        _require(annual_demand=annual_demand, ordering_cost=ordering_cost, unit_cost=unit_cost)
    else:
        _require(annual_demand=annual_demand, ordering_cost=ordering_cost)
    if holding_cost is not None and holding_rate is not None:
        raise InventoryInputError("Give either holding_cost or holding_rate (with unit_cost), not both.")

    w = _Work()
    D = w.given("annual_demand", "D", "Annual demand", _positive("annual_demand", annual_demand), "units/year")
    S = w.given("ordering_cost", "S", "Ordering (setup) cost per order", _positive("ordering_cost", ordering_cost), "$/order")
    C = None
    if unit_cost is not None:
        C = w.given("unit_cost", "C", "Unit purchase cost", _positive("unit_cost", unit_cost), "$/unit")
    if holding_cost is not None:
        H = w.given("holding_cost", "H", "Holding cost per unit per year", _positive("holding_cost", holding_cost),
                    "$/unit/year")
    else:
        i = w.given("holding_rate", "i", "Annual holding-cost rate (fraction of unit cost)",
                    _fraction("holding_rate", holding_rate), "per year")
        H = w.calc("H", "H", "Holding cost per unit per year", "H = i × C", f"{_n(i)} × {_n(C)}", i * C, "$/unit/year")
    days = None
    if days_per_year is not None:
        days = w.given("days_per_year", "W", "Working days per year", _positive("days_per_year", days_per_year), "days/year")
    L = None
    if lead_time_days is not None:
        L = w.given("lead_time_days", "L", "Lead time", _non_negative("lead_time_days", lead_time_days), "days")
        if days is None:
            days = w.default("days_per_year", "W", "Days per year", DEFAULT_DAYS_PER_YEAR, "days/year",
                             "No days-per-year given: 365 calendar days used to turn annual demand into daily demand.")
    w.assumptions += [
        "Demand is constant and known; each order arrives all at once.",
        "No shortages are allowed and no quantity discounts apply.",
    ]

    two_ds = w.calc("2DS", "2DS", "Twice annual demand times ordering cost", "2DS = 2 × D × S",
                    f"2 × {_n(D)} × {_n(S)}", 2 * D * S, "$·units/year")
    ratio = w.calc("2DS/H", "2DS/H", "Value under the square root", "2DS/H = 2DS ÷ H",
                   f"{_n(two_ds)} / {_n(H)}", two_ds / H, "units²")
    Q = w.calc("Q*", "Q*", "Economic order quantity", "Q* = √(2DS / H)",
               f"√(2 × {_n(D)} × {_n(S)} / {_n(H)})", math.sqrt(ratio), "units")
    N = w.calc("N", "N", "Number of orders per year", "N = D / Q*", f"{_n(D)} / {_n(Q)}", D / Q, "orders/year")
    T = w.calc("T", "T", "Time between orders (fraction of a year)", "T = Q* / D", f"{_n(Q)} / {_n(D)}", Q / D, "years")
    ordering = w.calc("ordering_cost_annual", "OC", "Annual ordering cost", "OC = (D / Q*) × S",
                      f"({_n(D)} / {_n(Q)}) × {_n(S)}", D / Q * S, "$/year")
    holding = w.calc("holding_cost_annual", "HC", "Annual holding cost", "HC = (Q* / 2) × H",
                     f"({_n(Q)} / 2) × {_n(H)}", Q / 2 * H, "$/year")
    total = w.calc("TC", "TC", "Total annual relevant cost (ordering + holding)", "TC = OC + HC",
                   f"{_n(ordering)} + {_n(holding)}", ordering + holding, "$/year")

    outputs: Dict[str, Any] = {
        "optimal_order_quantity": Q, "orders_per_year": N, "cycle_time_years": T,
        "annual_ordering_cost": ordering, "annual_holding_cost": holding, "total_annual_relevant_cost": total,
        "holding_cost_per_unit_per_year": H, "average_inventory": Q / 2,
    }
    w.final("Economic order quantity Q*", Q, "units")
    w.final("Orders per year N", N, "orders/year")
    w.final("Total annual relevant cost TC", total, "$/year")
    if days is not None:
        outputs["days_between_orders"] = w.calc("days_between_orders", "T_days", "Days between orders",
                                                "T_days = W / N", f"{_n(days)} / {_n(N)}", days / N, "days")
    if C is not None:
        outputs["annual_purchase_cost"] = w.calc("purchase_cost", "PC", "Annual purchase cost", "PC = D × C",
                                                 f"{_n(D)} × {_n(C)}", D * C, "$/year")
        outputs["total_annual_cost_including_purchase"] = w.calc(
            "TC_total", "TC_all", "Total annual cost including purchases", "TC_all = TC + PC",
            f"{_n(total)} + {_n(D * C)}", total + D * C, "$/year")
    if L is not None:
        d = w.calc("daily_demand", "d", "Average daily demand", "d = D / W", f"{_n(D)} / {_n(days)}", D / days,
                   "units/day")
        outputs["daily_demand"] = d
        outputs["reorder_point"] = w.calc("ROP", "ROP", "Reorder point (no safety stock)", "ROP = d × L",
                                          f"{_n(d)} × {_n(L)}", d * L, "units")
        w.final("Reorder point ROP", outputs["reorder_point"], "units")
        w.assumptions.append("Reorder point carries no safety stock (demand during lead time is certain).")

    answer = f"Order {Q:,.1f} units each time ({N:,.2f} orders per year), for an annual ordering + holding cost of ${total:,.2f}."
    if "reorder_point" in outputs:
        answer += f" Reorder when inventory falls to {outputs['reorder_point']:,.1f} units."
    inputs = {"annual_demand": D, "ordering_cost": S, "holding_cost": holding_cost, "holding_rate": holding_rate,
              "unit_cost": C, "lead_time_days": L, "days_per_year": days}
    return w.result("eoq", "Economic Order Quantity (EOQ)", inputs, outputs, answer)


# --------------------------------------------------------------------------- #
# 2. Quantity discount (all-units)
# --------------------------------------------------------------------------- #
def _discount_total_cost(D: float, S: float, H: float, price: float, Q: float) -> float:
    return D * price + D / Q * S + Q / 2 * H


def _parse_breaks(price_breaks: Iterable[Any]) -> List[Tuple[float, float]]:
    tiers = []
    for item in price_breaks:
        if isinstance(item, dict):
            q, p = item.get("min_quantity"), item.get("unit_price")
        else:
            try:
                q, p = item
            except (TypeError, ValueError):
                raise InventoryInputError(f"Price break {item!r} must be (min_quantity, unit_price).")
        tiers.append((_non_negative("min_quantity", q), _positive("unit_price", p)))
    return sorted(tiers)


def calculate_quantity_discount(
    annual_demand: Optional[float] = None,
    ordering_cost: Optional[float] = None,
    price_breaks: Optional[Sequence[Any]] = None,
    holding_rate: Optional[float] = None,
    holding_cost: Optional[float] = None,
) -> CalculationResult:
    """
    All-units quantity discount.

    price_breaks: [(min_quantity, unit_price), ...] or [{"min_quantity", "unit_price"}, ...];
    tier k covers q_k <= Q < q_(k+1).  Holding cost is a percentage of the unit price
    (holding_rate, a fraction -- H changes with the tier's price) or a fixed dollar amount
    per unit per year (holding_cost -- the same in every tier).

    For each tier: EOQ at that tier's H; raise it to the tier minimum if below; if it is at or
    above the next break the tier is infeasible (a cheaper tier dominates).  Then compare
    TC = D x P + (D/Q) x S + (Q/2) x H.
    """
    _require(annual_demand=annual_demand, ordering_cost=ordering_cost, price_breaks=price_breaks or None,
             holding_rate_or_holding_cost=holding_rate if holding_rate is not None else holding_cost)
    if holding_rate is not None and holding_cost is not None:
        raise InventoryInputError("Give either holding_rate (percentage) or holding_cost (fixed), not both.")
    tiers = _parse_breaks(price_breaks)
    if len({q for q, _ in tiers}) != len(tiers):
        raise InventoryInputError("Price breaks contain duplicate minimum quantities.")

    w = _Work()
    D = w.given("annual_demand", "D", "Annual demand", _positive("annual_demand", annual_demand), "units/year")
    S = w.given("ordering_cost", "S", "Ordering (setup) cost per order", _positive("ordering_cost", ordering_cost), "$/order")
    if holding_rate is not None:
        rate = w.given("holding_rate", "i", "Annual holding-cost rate (fraction of unit price)",
                       _fraction("holding_rate", holding_rate), "per year")
        fixed_h = None
        h_mode = f"percentage holding cost: H_k = {rate:g} x unit price of tier k"
    else:
        rate = None
        fixed_h = w.given("holding_cost", "H", "Fixed holding cost per unit per year (same in every tier)",
                          _positive("holding_cost", holding_cost), "$/unit/year")
        h_mode = f"fixed holding cost: H = ${fixed_h:g} per unit per year in every tier"
    for k, (lo, price) in enumerate(tiers, 1):
        w.given(f"q{k}", f"q{k}", f"Minimum order quantity for price tier {k}", lo, "units")
        w.given(f"P{k}", f"P{k}", f"Unit price in tier {k}", price, "$/unit")

    w.assumptions += [
        "All-units discount: the tier's price applies to every unit in the order.",
        "Demand is constant and known; no shortages.",
        f"Holding cost is a {h_mode}.",
    ]
    if tiers[0][0] > 0:
        w.assumptions.append(f"No price is given below {tiers[0][0]:g} units, so smaller orders are not considered.")
    if any(tiers[k + 1][1] >= tiers[k][1] for k in range(len(tiers) - 1)):
        w.assumptions.append("Warning: prices do not decrease with quantity; tiers are still evaluated as given.")

    rows: List[Dict[str, Any]] = []
    for k, (lower, price) in enumerate(tiers, 1):
        upper = tiers[k][0] if k < len(tiers) else math.inf
        if rate is not None:
            H = w.calc(f"H{k}", f"H{k}", f"Holding cost per unit per year in tier {k}", f"H{k} = i × P{k}",
                       f"{_n(rate)} × {_n(price)}", rate * price, "$/unit/year")
            hs = f"H{k}"
        else:
            H, hs = fixed_h, "H"
        q_eoq = w.calc(f"EOQ{k}", f"EOQ{k}", f"EOQ at the tier {k} price", f"EOQ{k} = √(2DS / {hs})",
                       f"√(2 × {_n(D)} × {_n(S)} / {_n(H)})", math.sqrt(2 * D * S / H), "units")
        row: Dict[str, Any] = {"min_quantity": lower, "max_quantity": None if upper == math.inf else upper,
                               "unit_price": price, "holding_cost": H, "eoq": q_eoq}
        if q_eoq >= upper:
            w.note(f"Q{k}", f"Q{k}", f"Order quantity for tier {k}", f"EOQ{k} ≥ q{k + 1} → tier {k} not feasible",
                   f"{_n(q_eoq)} ≥ {_n(upper)}, so tier {k} cannot be bought at its EOQ; a lower-priced tier is cheaper")
            row.update(feasible=False, order_quantity=None, total_cost=None,
                       note="EOQ above this tier's range; a cheaper tier dominates")
        else:
            q = w.calc(f"Q{k}", f"Q{k}", f"Feasible order quantity for tier {k} (raised to the tier minimum if needed)",
                       f"Q{k} = max(EOQ{k}, q{k})", f"max({_n(q_eoq)}, {_n(lower)})", max(q_eoq, lower), "units")
            tc = w.calc(f"TC{k}", f"TC{k}", f"Total annual cost in tier {k} (purchase + ordering + holding)",
                        f"TC{k} = D × P{k} + (D / Q{k}) × S + (Q{k} / 2) × {hs}",
                        f"{_n(D)} × {_n(price)} + ({_n(D)} / {_n(q)}) × {_n(S)} + ({_n(q)} / 2) × {_n(H)}",
                        _discount_total_cost(D, S, H, price, q), "$/year")
            row.update(feasible=True, order_quantity=q, total_cost=tc, purchase_cost=D * price,
                       ordering_cost=D / q * S, holding_cost_total=q / 2 * H,
                       note="raised to tier minimum" if q > q_eoq else "EOQ feasible")
        rows.append(row)

    feasible = [(k, r) for k, r in enumerate(rows, 1) if r["feasible"]]
    if not feasible:
        raise InventoryInputError("No feasible price tier; check the price breaks.")
    best_k, best = min(feasible, key=lambda kr: kr[1]["total_cost"])
    if len(feasible) == 1:
        w.calc("TC*", "TC*", "Lowest total annual cost (only one feasible tier)", f"TC* = TC{best_k}",
               _n(best["total_cost"]), best["total_cost"], "$/year")
    else:
        w.calc("TC*", "TC*", "Lowest total annual cost among feasible tiers",
               "TC* = min(" + ", ".join(f"TC{k}" for k, _ in feasible) + ")",
               "min(" + ", ".join(_n(r["total_cost"]) for _, r in feasible) + ")", best["total_cost"], "$/year")
    Q = best["order_quantity"]
    w.final("Order quantity Q*", Q, "units")
    w.final("Unit price paid", best["unit_price"], "$/unit")
    w.final("Total annual cost TC*", best["total_cost"], "$/year")
    outputs = {
        "optimal_order_quantity": Q, "unit_price": best["unit_price"], "total_annual_cost": best["total_cost"],
        "annual_purchase_cost": D * best["unit_price"], "annual_ordering_cost": D / Q * S,
        "annual_holding_cost": Q / 2 * best["holding_cost"], "orders_per_year": D / Q, "best_tier": best_k,
        "tiers": rows,
    }
    candidates = [{"unit_price": r["unit_price"], "eoq": r["eoq"], "order_quantity": r["order_quantity"],
                   "total_cost": r["total_cost"], "feasible": r["feasible"]} for r in rows]
    answer = (f"Order {Q:,.0f} units at ${best['unit_price']:,.2f} per unit (tier {best_k}); "
              f"total annual cost ${best['total_cost']:,.2f} is the lowest of the feasible options.")
    inputs = {"annual_demand": D, "ordering_cost": S, "price_breaks": [list(t) for t in tiers],
              "holding_rate": rate, "holding_cost": fixed_h}
    return w.result("quantity_discount", "Quantity Discount (all-units)", inputs, outputs, answer,
                    extra_intermediate={"candidates": candidates})


# --------------------------------------------------------------------------- #
# 3. Safety stock and reorder point (continuous review)
# --------------------------------------------------------------------------- #
def calculate_rop(
    mean_demand: Optional[float] = None,
    lead_time: Optional[float] = None,
    std_demand: Optional[float] = None,
    std_lead_time: Optional[float] = None,
    service_level: Optional[float] = None,
    z: Optional[float] = None,
    period: str = "day",
) -> CalculationResult:
    """
    Safety stock and reorder point.  Demand per period: mean d, std. dev. sigma_d; lead time
    L periods with std. dev. sigma_L (same period unit).

        constant demand, constant lead time:  sigma_LT = 0
        variable demand, constant lead time:  sigma_LT = sigma_d * sqrt(L)
        constant demand, variable lead time:  sigma_LT = d * sigma_L
        variable demand, variable lead time:  sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2)

        SS = z * sigma_LT,   ROP = d * L + SS

    A service level (fraction) or z-value is required whenever there is any variability.
    """
    _require(mean_demand=mean_demand, lead_time=lead_time)
    w = _Work()
    p = period
    d = w.given("mean_demand", "d", f"Average demand per {p}", _non_negative("mean_demand", mean_demand), f"units/{p}")
    sd = _optional_std(w, "std_demand", "σ_d", f"Standard deviation of demand per {p}", std_demand, f"units/{p}", "demand")
    L = w.given("lead_time", "L", "Average lead time", _positive("lead_time", lead_time), f"{p}s")
    sL = _optional_std(w, "std_lead_time", "σ_L", "Standard deviation of lead time", std_lead_time, f"{p}s", "lead time")
    case, label = _variability_case(sd, sL)

    if sd > 0 or sL > 0:
        zv, sl = _z_step(w, service_level, z)
        w.assumptions.append("Demand during lead time is normally distributed; periods are independent.")
        w.assumptions.append("Service level is a cycle service level (probability of no stockout per order cycle).")
    else:
        zv, sl = None, None
        if service_level is not None or z is not None:
            w.assumptions.append("No variability, so no safety stock is needed; the service level does not change the answer.")

    sigma = _sigma_step(w, "sigma_LT", "σ_LT", f"Standard deviation of demand during lead time ({label})",
                        d, sd, L, sL, "L", "units")
    if zv is not None:
        ss = w.calc("SS", "SS", "Safety stock", "SS = z × σ_LT", f"{_n(zv)} × {_n(sigma)}", zv * sigma, "units")
    else:
        ss = w.calc("SS", "SS", "Safety stock", "SS = 0 (no variability)", "0", 0.0, "units")
    lt_demand = w.calc("dL", "LTD", "Expected demand during lead time", "LTD = d × L", f"{_n(d)} × {_n(L)}",
                       d * L, "units")
    rop = w.calc("ROP", "ROP", "Reorder point", "ROP = LTD + SS", f"{_n(lt_demand)} + {_n(ss)}", lt_demand + ss, "units")
    shortage = 0.0
    if zv is not None:
        loss = w.calc("L(z)", "L(z)", "Standard normal loss function", "L(z) = φ(z) − z × (1 − Φ(z))",
                      f"φ({_n(zv)}) - {_n(zv)} × (1 - Φ({_n(zv)}))", standard_normal_loss(zv))
        shortage = w.calc("E_short", "E[short]", "Expected units short per replenishment cycle", "E[short] = σ_LT × L(z)",
                          f"{_n(sigma)} × {_n(loss)}", sigma * loss, "units")

    w.final("Safety stock SS", ss, "units")
    w.final("Reorder point ROP", rop, "units")
    outputs = {"case": case, "z": zv, "service_level": sl, "expected_lead_time_demand": lt_demand,
               "std_lead_time_demand": sigma, "safety_stock": ss, "reorder_point": rop,
               "expected_shortage_per_cycle": shortage}
    answer = (f"Reorder when inventory position falls to {rop:,.1f} units "
              f"(lead-time demand {lt_demand:,.1f} + safety stock {ss:,.1f}).")
    inputs = {"mean_demand": d, "std_demand": sd, "lead_time": L, "std_lead_time": sL,
              "service_level": sl, "z": zv, "period": period}
    return w.result("safety_stock", "Safety Stock and Reorder Point", inputs, outputs, answer, case,
                    extra_intermediate={"case": case})


# --------------------------------------------------------------------------- #
# 4. Fixed-period review (order-up-to)
# --------------------------------------------------------------------------- #
def calculate_periodic_review(
    mean_demand: Optional[float] = None,
    review_period: Optional[float] = None,
    lead_time: Optional[float] = None,
    on_hand: Optional[float] = None,
    std_demand: Optional[float] = None,
    std_lead_time: Optional[float] = None,
    service_level: Optional[float] = None,
    z: Optional[float] = None,
    on_order: Optional[float] = None,
    backorders: Optional[float] = None,
    inventory_position: Optional[float] = None,
    safety_stock: Optional[float] = None,
    period: str = "day",
) -> CalculationResult:
    """
        Protection period  P = T + L
        sigma_P            = sqrt(P sigma_d^2 + d^2 sigma_L^2)   (same four variability cases as calculate_rop)
        Safety stock       = z * sigma_P      (or the safety stock given directly)
        Target inventory   = d * P + safety stock
        Inventory position = on hand + on order - backorders   (or given directly)
        Order quantity     = max(0, target - inventory position)

    The current stock is required: give on_hand, or inventory_position directly.
    """
    _require(mean_demand=mean_demand, review_period=review_period, lead_time=lead_time,
             on_hand_or_inventory_position=on_hand if on_hand is not None else inventory_position)
    w = _Work()
    p = period
    d = w.given("mean_demand", "d", f"Average demand per {p}", _non_negative("mean_demand", mean_demand), f"units/{p}")
    sd = _optional_std(w, "std_demand", "σ_d", f"Standard deviation of demand per {p}", std_demand, f"units/{p}", "demand")
    T = w.given("review_period", "T", "Review period (time between reviews)", _positive("review_period", review_period), f"{p}s")
    L = w.given("lead_time", "L", "Average lead time", _non_negative("lead_time", lead_time), f"{p}s")
    sL = _optional_std(w, "std_lead_time", "σ_L", "Standard deviation of lead time", std_lead_time, f"{p}s", "lead time")
    case, label = _variability_case(sd, sL)

    if inventory_position is not None:
        position_given = w.given("inventory_position", "IP", "Inventory position", _number("inventory_position", inventory_position), "units")
        I = oo = bo = None
    else:
        position_given = None
        I = w.given("on_hand", "OH", "On-hand inventory", _non_negative("on_hand", on_hand), "units")
        oo = (w.given("on_order", "OO", "On-order inventory", _non_negative("on_order", on_order), "units")
              if on_order is not None else
              w.default("on_order", "OO", "On-order inventory", 0.0, "units", "No on-order quantity given: nothing is on order (OO = 0)."))
        bo = (w.given("backorders", "BO", "Backorders", _non_negative("backorders", backorders), "units")
              if backorders is not None else
              w.default("backorders", "BO", "Backorders", 0.0, "units", "No backorders given: none outstanding (BO = 0)."))

    P = w.calc("protection_period", "P", "Protection period (review period + lead time)", "P = T + L",
               f"{_n(T)} + {_n(L)}", T + L, f"{p}s")
    zv = sl = None
    if safety_stock is not None:
        ss_given = w.given("safety_stock", "SS", "Safety stock (given)", _non_negative("safety_stock", safety_stock), "units")
        w.assumptions.append(f"Safety stock given directly ({ss_given:g} units); it is used as stated, not recomputed.")
        if service_level is not None or z is not None:
            w.assumptions.append("A service level was also given but is not needed because the safety stock is set.")
    elif sd > 0 or sL > 0:
        zv, sl = _z_step(w, service_level, z)
        w.assumptions.append("Demand over the protection period is normally distributed; periods are independent.")
    elif service_level is not None or z is not None:
        w.assumptions.append("No variability, so no safety stock is needed; the service level does not change the answer.")
    w.assumptions.append("Inventory is reviewed at fixed intervals and an order is placed at every review.")

    sigma = _sigma_step(w, "sigma_P", "σ_P", f"Standard deviation of demand over the protection period ({label})",
                        d, sd, P, sL, "P", "units")
    if safety_stock is not None:
        ss = ss_given
    elif zv is not None:
        ss = w.calc("SS", "SS", "Safety stock", "SS = z × σ_P", f"{_n(zv)} × {_n(sigma)}", zv * sigma, "units")
    else:
        ss = w.calc("SS", "SS", "Safety stock", "SS = 0 (no variability)", "0", 0.0, "units")
    dP = w.calc("dP", "D_P", "Expected demand during the protection period", "D_P = d × P",
                f"{_n(d)} × {_n(P)}", d * P, "units")
    target = w.calc("target", "S_target", "Target (order-up-to) inventory level", "S_target = D_P + SS",
                    f"{_n(dP)} + {_n(ss)}", dP + ss, "units")
    if position_given is not None:
        position = position_given
    else:
        position = w.calc("IP", "IP", "Inventory position", "IP = OH + OO − BO", f"{_n(I)} + {_n(oo)} - {_n(bo)}",
                          I + oo - bo, "units")
    order = w.calc("Q", "Q", "Order quantity this review", "Q = max(0, S_target − IP)",
                   f"max(0, {_n(target)} - {_n(position)})", max(0.0, target - position), "units")

    w.final("Order quantity Q", order, "units")
    w.final("Target inventory level S_target", target, "units")
    w.final("Safety stock SS", ss, "units")
    outputs = {"case": case, "z": zv, "service_level": sl, "protection_interval": P,
               "expected_demand_over_protection_interval": dP, "std_demand_over_protection_interval": sigma,
               "safety_stock": ss, "safety_stock_source": "given" if safety_stock is not None else "computed",
               "target_inventory_level": target, "inventory_position": position, "order_quantity": order}
    answer = (f"Order {order:,.1f} units now to bring inventory position up to the target of {target:,.1f} units."
              if order > 0 else
              f"Do not order this period: inventory position {position:,.1f} already meets the target of {target:,.1f}.")
    inputs = {"mean_demand": d, "std_demand": sd, "review_period": T, "lead_time": L, "std_lead_time": sL,
              "service_level": sl, "z": zv, "on_hand": I, "on_order": oo, "backorders": bo,
              "inventory_position": position, "period": period}
    return w.result("fixed_period", "Fixed-Period Review (order-up-to level)", inputs, outputs, answer, case,
                    extra_intermediate={"case": case})


# --------------------------------------------------------------------------- #
# 5. Newsvendor
# --------------------------------------------------------------------------- #
def _expected_lost_sales(Q: float, dist: str, a: float, b: float) -> float:
    """E[(D - Q)+].  normal: a=mean, b=std;  uniform: a=low, b=high."""
    if dist == "normal":
        return b * standard_normal_loss((Q - a) / b)
    if Q <= a:
        return (a + b) / 2 - Q
    if Q >= b:
        return 0.0
    return (b - Q) ** 2 / (2 * (b - a))


def newsvendor_expected_profit(
    Q: float, p: float, c: float, v: float, g: float, dist: str, a: float, b: float
) -> Dict[str, float]:
    mean = a if dist == "normal" else (a + b) / 2
    lost = _expected_lost_sales(Q, dist, a, b)
    sales = mean - lost
    leftover = Q - sales
    profit = p * sales + v * leftover - c * Q - g * lost
    return {"expected_sales": sales, "expected_lost_sales": lost,
            "expected_leftover": leftover, "expected_profit": profit}


def newsvendor_expected_cost(Q: float, cu: float, co: float, dist: str, a: float, b: float) -> Dict[str, float]:
    """Expected mismatch cost Co*E[leftover] + Cu*E[shortage] -- minimized at the newsvendor quantity."""
    mean = a if dist == "normal" else (a + b) / 2
    lost = _expected_lost_sales(Q, dist, a, b)
    leftover = Q - (mean - lost)
    return {"expected_lost_sales": lost, "expected_leftover": leftover,
            "expected_mismatch_cost": co * leftover + cu * lost}


def calculate_newsvendor(
    selling_price: Optional[float] = None,
    unit_cost: Optional[float] = None,
    salvage_value: Optional[float] = None,
    shortage_penalty: Optional[float] = None,
    demand_distribution: str = "normal",
    mean: Optional[float] = None,
    std: Optional[float] = None,
    low: Optional[float] = None,
    high: Optional[float] = None,
    underage_cost: Optional[float] = None,
    overage_cost: Optional[float] = None,
) -> CalculationResult:
    """
        Underage (shortage) cost  Cu = selling price - purchase cost + shortage penalty   (or given directly)
        Overage (excess) cost     Co = purchase cost - salvage value                      (or given directly)
        Critical ratio            CR = Cu / (Cu + Co)
        Order quantity            Q* = F^-1(CR):  normal -> mean + z*std;  uniform -> low + CR*(high - low)

    Give prices (selling_price, unit_cost, optional salvage_value / shortage_penalty) or give
    underage_cost and/or overage_cost directly.  Expected profit is reported when prices are
    known; the expected mismatch cost (Co x leftover + Cu x shortage) always is.
    """
    dist = (demand_distribution or "").lower()
    if dist not in ("normal", "uniform"):
        raise InventoryInputError("demand_distribution must be 'normal' or 'uniform'.")
    missing: List[str] = []
    if underage_cost is None:
        missing += [k for k, v in (("selling_price", selling_price), ("unit_cost", unit_cost)) if v is None]
    if overage_cost is None and unit_cost is None and "unit_cost" not in missing:
        missing.append("unit_cost")
    dist_args = (("mean", mean), ("std", std)) if dist == "normal" else (("low", low), ("high", high))
    missing += [k for k, v in dist_args if v is None]
    if missing:
        raise InventoryInputError(f"Missing required input(s): {', '.join(missing)}.", missing)

    w = _Work()
    p = w.given("selling_price", "p", "Selling price per unit", _positive("selling_price", selling_price), "$/unit") \
        if selling_price is not None else None
    c = w.given("unit_cost", "c", "Purchase cost per unit", _positive("unit_cost", unit_cost), "$/unit") \
        if unit_cost is not None else None
    v: Optional[float] = None
    g = 0.0

    if underage_cost is not None:
        cu = w.given("underage_cost", "Cu", "Underage (shortage) cost per unit short", _positive("underage_cost", underage_cost), "$/unit")
        if shortage_penalty is not None:
            g = _non_negative("shortage_penalty", shortage_penalty)
        if p is not None and c is not None:
            w.assumptions.append("Underage cost given directly; it is used instead of price - cost.")
    else:
        g = (w.given("shortage_penalty", "g", "Shortage penalty per unit short (lost goodwill)",
                     _non_negative("shortage_penalty", shortage_penalty), "$/unit")
             if shortage_penalty is not None else
             w.default("shortage_penalty", "g", "Shortage penalty per unit short", 0.0, "$/unit",
                       "No shortage penalty given: a shortage costs only the lost margin (g = 0)."))
        cu = w.calc("Cu", "Cu", "Underage cost: profit lost on each unit of unmet demand", "Cu = p − c + g",
                    f"{_n(p)} - {_n(c)} + {_n(g)}", p - c + g, "$/unit")
    if overage_cost is not None:
        co = w.given("overage_cost", "Co", "Overage (excess) cost per unit left over", _positive("overage_cost", overage_cost), "$/unit")
        if salvage_value is not None:
            v = _non_negative("salvage_value", salvage_value)
    else:
        v = (w.given("salvage_value", "v", "Salvage value per unsold unit", _non_negative("salvage_value", salvage_value), "$/unit")
             if salvage_value is not None else
             w.default("salvage_value", "v", "Salvage value per unsold unit", 0.0, "$/unit",
                       "No salvage value given: unsold units have no recovery value (v = 0)."))
        co = w.calc("Co", "Co", "Overage cost: loss on each unit left over", "Co = c − v", f"{_n(c)} - {_n(v)}", c - v, "$/unit")
    if cu <= 0:
        raise InventoryInputError("Underage cost (price - cost + penalty) must be positive.")
    if co <= 0:
        raise InventoryInputError("Overage cost (cost - salvage) must be positive; salvage must be below cost.")
    w.assumptions.insert(0, "Single selling period: unsold units cannot be carried over to a later period.")

    if dist == "normal":
        mu = w.given("mean", "μ", "Mean demand", _non_negative("mean", mean), "units")
        sigma = w.given("std", "σ", "Standard deviation of demand", _positive("std", std), "units")
        a, b = mu, sigma
        w.assumptions.append(f"Demand is normally distributed (mean {mu:g}, standard deviation {sigma:g}).")
    else:
        lo = w.given("low", "a", "Lowest possible demand (uniform)", _non_negative("low", low), "units")
        hi = w.given("high", "b", "Highest possible demand (uniform)", _number("high", high), "units")
        if hi <= lo:
            raise InventoryInputError("Uniform demand needs high > low.")
        a, b = lo, hi
        w.assumptions.append(f"Demand is uniformly distributed between {lo:g} and {hi:g}.")

    cr = w.calc("CR", "CR", "Critical ratio: target probability that demand ≤ Q*", "CR = Cu / (Cu + Co)",
                f"{_n(cu)} / ({_n(cu)} + {_n(co)})", cu / (cu + co))
    extra: Dict[str, Any] = {}
    if dist == "normal":
        zv = w.calc("z", "z", "Standard normal value with probability CR below it", "z = Φ⁻¹(CR)", f"Φ⁻¹({_n(cr)})",
                    STD_NORMAL.inv_cdf(cr))
        Q = w.calc("Q*", "Q*", "Optimal order quantity", "Q* = μ + z × σ", f"{_n(mu)} + {_n(zv)} × {_n(sigma)}",
                   mu + zv * sigma, "units")
        if Q < 0:
            raise InventoryInputError("The normal model gives a negative quantity; demand is too variable for a normal fit.")
        extra["z"] = zv
        loss = w.calc("L(z)", "L(z)", "Standard normal loss function", "L(z) = φ(z) − z × (1 − Φ(z))",
                      f"φ({_n(zv)}) - {_n(zv)} × (1 - Φ({_n(zv)}))", standard_normal_loss(zv))
        short = w.calc("E_short", "E[short]", "Expected units short (unmet demand)", "E[short] = σ × L(z)",
                       f"{_n(sigma)} × {_n(loss)}", sigma * loss, "units")
        mean_d = mu
        left = w.calc("E_left", "E[left]", "Expected units left over", "E[left] = Q* − μ + E[short]",
                      f"{_n(Q)} - {_n(mu)} + {_n(short)}", Q - mu + short, "units")
    else:
        Q = w.calc("Q*", "Q*", "Optimal order quantity", "Q* = a + CR × (b − a)", f"{_n(a)} + {_n(cr)} × ({_n(b)} - {_n(a)})",
                   a + cr * (b - a), "units")
        short = w.calc("E_short", "E[short]", "Expected units short (unmet demand)", "E[short] = (b − Q*)² / (2 × (b − a))",
                       f"({_n(b)} - {_n(Q)})² / (2 × ({_n(b)} - {_n(a)}))", _expected_lost_sales(Q, dist, a, b), "units")
        mean_d = w.calc("mean_demand", "μ", "Mean demand", "μ = (a + b) / 2", f"({_n(a)} + {_n(b)}) / 2", (a + b) / 2, "units")
        left = w.calc("E_left", "E[left]", "Expected units left over", "E[left] = Q* − μ + E[short]",
                      f"{_n(Q)} - {_n(mean_d)} + {_n(short)}", Q - mean_d + short, "units")
    sales = w.calc("E_sales", "E[sales]", "Expected units sold", "E[sales] = Q* − E[left]", f"{_n(Q)} - {_n(left)}",
                   Q - left, "units")
    mismatch = w.calc("E_cost", "E[cost]", "Expected cost of leftovers and shortages",
                      "E[cost] = Co × E[left] + Cu × E[short]", f"{_n(co)} × {_n(left)} + {_n(cu)} × {_n(short)}",
                      co * left + cu * short, "$")
    metrics: Dict[str, Any] = {"expected_lost_sales": short, "expected_leftover": left, "expected_sales": sales,
                               "expected_mismatch_cost": mismatch}
    w.final("Optimal order quantity Q*", Q, "units")
    w.final("Critical ratio CR", cr)
    prices_known = p is not None and c is not None and underage_cost is None and overage_cost is None
    if prices_known:
        profit = w.calc("E_profit", "E[profit]", "Expected profit",
                        "E[profit] = p × E[sales] + v × E[left] − c × Q* − g × E[short]",
                        f"{_n(p)} × {_n(sales)} + {_n(v)} × {_n(left)} - {_n(c)} × {_n(Q)} - {_n(g)} × {_n(short)}",
                        p * sales + v * left - c * Q - g * short, "$")
        metrics["expected_profit"] = profit
        w.final("Expected profit", profit, "$")
        answer = f"Stock {Q:,.1f} units (critical ratio {cr:.3f}); expected profit ${profit:,.2f}."
    else:
        w.final("Expected cost of leftovers and shortages", mismatch, "$")
        answer = (f"Stock {Q:,.1f} units (critical ratio {cr:.3f}); expected cost of leftovers and shortages "
                  f"${mismatch:,.2f}.")
    outputs = {"underage_cost": cu, "overage_cost": co, "critical_ratio": cr, "optimal_order_quantity": Q,
               **extra, **metrics}
    inputs = {"selling_price": p, "unit_cost": c, "salvage_value": v, "shortage_penalty": g,
              "underage_cost": cu, "overage_cost": co,
              "demand_distribution": dist, "mean": mean, "std": std, "low": low, "high": high}
    return w.result("newsvendor", "Newsvendor (single-period)", inputs, outputs, answer)


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def _check(name: str, passed: bool, detail: str) -> Dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _close(a: float, b: float, rel: float = 1e-6) -> bool:
    return math.isclose(a, b, rel_tol=rel, abs_tol=1e-6)


def _verify_eoq(r: CalculationResult) -> List[Dict[str, Any]]:
    o, i = r.outputs, r.inputs
    D, S, H, Q = i["annual_demand"], i["ordering_cost"], o["holding_cost_per_unit_per_year"], o["optimal_order_quantity"]
    tc = lambda q: D / q * S + q / 2 * H  # noqa: E731
    checks = [
        _check("positive quantity", Q > 0, f"Q* = {Q:,.2f}"),
        _check("ordering cost = holding cost at optimum",
               _close(o["annual_ordering_cost"], o["annual_holding_cost"]),
               f"{o['annual_ordering_cost']:,.4f} vs {o['annual_holding_cost']:,.4f}"),
        _check("Q* minimizes total cost (+/-10% perturbation)",
               tc(Q) <= tc(0.9 * Q) + TOL and tc(Q) <= tc(1.1 * Q) + TOL,
               f"TC(0.9Q)={tc(0.9 * Q):,.2f}, TC(Q*)={tc(Q):,.2f}, TC(1.1Q)={tc(1.1 * Q):,.2f}"),
        _check("total cost recomputed as sqrt(2DSH)", _close(tc(Q), math.sqrt(2 * D * S * H)),
               f"sqrt(2DSH) = {math.sqrt(2 * D * S * H):,.4f}"),
    ]
    if "reorder_point" in o:
        checks.append(_check("reorder point non-negative", o["reorder_point"] >= 0, f"ROP = {o['reorder_point']:,.2f}"))
    return checks


def _verify_discount(r: CalculationResult) -> List[Dict[str, Any]]:
    o, i = r.outputs, r.inputs
    D, S = i["annual_demand"], i["ordering_cost"]
    feasible = [t for t in o["tiers"] if t["feasible"]]
    best = min(t["total_cost"] for t in feasible)
    chosen = next(t for t in feasible if t["unit_price"] == o["unit_price"])
    Q = o["optimal_order_quantity"]
    upper = chosen["max_quantity"] if chosen["max_quantity"] is not None else math.inf
    recomputed = _discount_total_cost(D, S, chosen["holding_cost"], chosen["unit_price"], Q)
    scan_best = math.inf  # independent grid search over every feasible tier
    for t in feasible:
        hi = t["max_quantity"] if t["max_quantity"] is not None else t["order_quantity"] * 3
        lo = max(t["min_quantity"], 1)
        step = max(1, int((hi - lo) / 2000))
        q = lo
        while q < hi:
            scan_best = min(scan_best, _discount_total_cost(D, S, t["holding_cost"], t["unit_price"], q))
            q += step
    return [
        _check("chosen tier has the lowest total cost", _close(o["total_annual_cost"], best),
               f"{o['total_annual_cost']:,.2f} vs min {best:,.2f}"),
        _check("order quantity lies inside its price tier", chosen["min_quantity"] <= Q + TOL and Q < upper,
               f"{chosen['min_quantity']:g} <= {Q:,.2f} < {upper:g}"),
        _check("total cost recomputed independently", _close(recomputed, o["total_annual_cost"]),
               f"recomputed {recomputed:,.2f}"),
        _check("no cheaper quantity found by grid search", o["total_annual_cost"] <= scan_best + 0.01,
               f"grid-search minimum {scan_best:,.2f}"),
    ]


def _verify_z(o: Dict[str, Any]) -> List[Dict[str, Any]]:
    if o.get("safety_stock_source") == "given":
        return [_check("given safety stock used as stated", o["safety_stock"] >= 0, f"SS = {o['safety_stock']:,.2f}")]
    if o["z"] is None:
        return [_check("no variability -> zero safety stock", _close(o["safety_stock"], 0.0), "SS = 0")]
    return [_check("z reproduces service level", _close(STD_NORMAL.cdf(o["z"]), o["service_level"]),
                   f"Phi({o['z']:.4f}) = {STD_NORMAL.cdf(o['z']):.4f}")]


def _verify_safety_stock(r: CalculationResult) -> List[Dict[str, Any]]:
    o, i = r.outputs, r.inputs
    sigma = math.sqrt(i["lead_time"] * i["std_demand"] ** 2 + i["mean_demand"] ** 2 * i["std_lead_time"] ** 2)
    return _verify_z(o) + [
        _check("sigma of lead-time demand recomputed", _close(sigma, o["std_lead_time_demand"]), f"{sigma:,.4f}"),
        _check("safety stock non-negative", o["safety_stock"] >= -TOL,
               "OK" if o["safety_stock"] >= 0 else "service level below 50% gives negative safety stock"),
        _check("ROP = expected lead-time demand + safety stock",
               _close(o["reorder_point"], o["expected_lead_time_demand"] + o["safety_stock"]),
               f"{o['expected_lead_time_demand']:,.2f} + {o['safety_stock']:,.2f}"),
    ]


def _verify_fixed_period(r: CalculationResult) -> List[Dict[str, Any]]:
    o, i = r.outputs, r.inputs
    P = i["review_period"] + i["lead_time"]
    sigma = math.sqrt(P * i["std_demand"] ** 2 + i["mean_demand"] ** 2 * i["std_lead_time"] ** 2)
    sl = o["service_level"] if o["service_level"] is not None else 1.0
    return _verify_z(o) + [
        _check("protection-period sigma recomputed", _close(sigma, o["std_demand_over_protection_interval"]),
               f"{sigma:,.4f}"),
        _check("target covers expected demand over T+L (if SL >= 50%)",
               sl < 0.5 or o["target_inventory_level"] >= o["expected_demand_over_protection_interval"] - TOL,
               f"{o['target_inventory_level']:,.2f} vs {o['expected_demand_over_protection_interval']:,.2f}"),
        _check("order brings inventory position up to target",
               (o["order_quantity"] == 0 and o["inventory_position"] >= o["target_inventory_level"] - TOL)
               or _close(o["inventory_position"] + o["order_quantity"], o["target_inventory_level"]),
               f"{o['inventory_position']:,.2f} + {o['order_quantity']:,.2f} vs target {o['target_inventory_level']:,.2f}"),
    ]


def _verify_newsvendor(r: CalculationResult) -> List[Dict[str, Any]]:
    o, i = r.outputs, r.inputs
    dist = i["demand_distribution"]
    a, b = (i["mean"], i["std"]) if dist == "normal" else (i["low"], i["high"])
    mean = a if dist == "normal" else (a + b) / 2
    Q, cu, co = o["optimal_order_quantity"], i["underage_cost"], i["overage_cost"]
    cdf = STD_NORMAL.cdf((Q - a) / b) if dist == "normal" else (Q - a) / (b - a)
    delta = max(1.0, 0.02 * Q)
    cost = lambda q: newsvendor_expected_cost(q, cu, co, dist, a, b)["expected_mismatch_cost"]  # noqa: E731
    c0, c_lo, c_hi = cost(Q), cost(max(Q - delta, 0.0)), cost(Q + delta)
    return [
        _check("critical ratio strictly between 0 and 1", 0 < o["critical_ratio"] < 1, f"CR = {o['critical_ratio']:.4f}"),
        _check("F(Q*) equals critical ratio", _close(cdf, o["critical_ratio"], 1e-5), f"F(Q*) = {cdf:.4f}"),
        _check("expected leftover + shortage cost is minimized at Q*", c0 <= c_lo + TOL and c0 <= c_hi + TOL,
               f"cost(Q-{delta:.1f})={c_lo:,.2f}, cost(Q*)={c0:,.2f}, cost(Q+{delta:.1f})={c_hi:,.2f}"),
        _check("E[leftover] - E[shortage] = Q - mean demand",
               _close(o["expected_leftover"] - o["expected_lost_sales"], Q - mean), "inventory balance holds"),
    ]


_VERIFIERS = {
    "eoq": _verify_eoq,
    "quantity_discount": _verify_discount,
    "safety_stock": _verify_safety_stock,
    "fixed_period": _verify_fixed_period,
    "newsvendor": _verify_newsvendor,
}


def verify(result: CalculationResult) -> List[Dict[str, Any]]:
    """Run model-specific sanity and optimality checks on a calculation result."""
    return _VERIFIERS[result.model](result)


TOOLS = {
    "eoq": calculate_eoq,
    "quantity_discount": calculate_quantity_discount,
    "safety_stock": calculate_rop,
    "fixed_period": calculate_periodic_review,
    "newsvendor": calculate_newsvendor,
}
