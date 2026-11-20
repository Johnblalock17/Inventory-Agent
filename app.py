"""
app.py -- command-line front end for the Inventory Decision Agent.

Usage
-----
    python app.py                                   # interactive session
    python app.py -q "Annual demand is 1000 ..."    # answer one question
    python app.py -q "..." --json                   # machine-readable output
    python app.py --examples                        # run built-in sample questions
    python app.py --web 8080                        # interactive web app at http://localhost:8080
    python app.py --llm                             # let Claude read the questions
                                                    # (needs `pip install anthropic` and an API key;
                                                    #  Python still does all the math)
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, Optional

from inventory_agent import FOLLOW_UPS, AgentResponse, InventoryAgent

EXAMPLES = [
    "Annual demand is 1,000 units, ordering cost is $10 per order, and holding cost is $0.50 per unit per year. "
    "What is the EOQ?",
    "Annual demand is 5,000 units, ordering cost $49, holding cost is 20% of unit price. Supplier prices: "
    "0-999 units at $5.00, 1,000-1,999 at $4.80, 2,000 or more at $4.75. What order quantity minimizes total cost?",
    "Daily demand averages 100 units with a standard deviation of 20 units. Lead time is 4 days. "
    "What safety stock and reorder point give a 95% service level?",
    "We review inventory every 2 weeks. Daily demand is 10 units with a standard deviation of 3. Lead time is "
    "7 days, service level 95%, and we have 50 units on hand. How much should we order?",
    "A newsstand buys papers for $4 and sells them for $10. Unsold papers are recycled for $2. Daily demand is "
    "normally distributed with mean 500 and standard deviation 100. How many should it stock?",
]

def _parse_answer(key: str, text: str) -> Any:
    text = text.strip().replace("$", "").replace("%", "").replace(",", "") if key != "price_breaks" else text
    if key == "price_breaks":
        tiers = []
        for part in text.split(","):
            lo, price = part.replace("$", "").split(":")
            tiers.append([float(lo), float(price)])
        return tiers
    return float(text)


def _render(resp: AgentResponse, verbose: bool) -> str:
    out = []
    if verbose and resp.trace:
        out.append("Pipeline trace:")
        out += [f"  {i + 1}. {t}" for i, t in enumerate(resp.trace)]
        out.append("")
    if resp.identification_reason and resp.status == "ok":
        out.append(f"(Why this model: {resp.identification_reason})")
    out.append(resp.explanation)
    return "\n".join(out)


def ask_for_missing(agent: InventoryAgent, resp: AgentResponse) -> AgentResponse:
    """Interactively collect missing inputs, then rerun the pipeline."""
    overrides: Dict[str, Any] = dict(resp.raw_extraction)
    for missing in resp.missing:
        for key, prompt, default in FOLLOW_UPS.get(missing, []):
            if prompt is None:
                overrides.setdefault(key, default)
                continue
            while True:
                answer = input(f"  {prompt}: ").strip()
                if not answer:
                    return resp
                try:
                    overrides[key] = _parse_answer(key, answer)
                    break
                except (ValueError, IndexError):
                    print("    Please enter a number (or leave blank to skip).")
    return agent.run(resp.question, overrides=overrides, model_key=resp.model_key)


def answer(agent: InventoryAgent, question: str, as_json: bool, verbose: bool, interactive: bool) -> int:
    resp = agent.run(question)
    if resp.status == "needs_input" and interactive and not as_json:
        print(resp.explanation)
        resp = ask_for_missing(agent, resp)
    if as_json:
        print(json.dumps(resp.to_dict(), indent=2, default=str))
    else:
        print(_render(resp, verbose))
    return 0 if resp.status == "ok" else 1


def interactive_loop(agent: InventoryAgent, verbose: bool) -> None:
    print("Inventory Decision Agent -- EOQ, quantity discount, safety stock/ROP, fixed-period review, newsvendor.")
    print("Type a question, 'examples' to see samples, or 'quit' to exit.\n")
    while True:
        try:
            question = input("Question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            return
        if question.lower() == "examples":
            for e in EXAMPLES:
                print(f"  - {e}")
            continue
        answer(agent, question, as_json=False, verbose=verbose, interactive=True)
        print()


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Supply Chain Analytics Inventory Decision Agent")
    parser.add_argument("-q", "--question", help="answer a single question and exit")
    parser.add_argument("--json", action="store_true", help="print the full structured response as JSON")
    parser.add_argument("--examples", action="store_true", help="run the built-in example questions")
    parser.add_argument("--llm", action="store_true", help="use Claude to identify the model and read the inputs")
    parser.add_argument("-v", "--verbose", action="store_true", help="show the pipeline trace")
    parser.add_argument("--web", nargs="?", type=int, const=8080, metavar="PORT",
                        help="launch the interactive web app on PORT (default 8080)")
    parser.add_argument("--host", default="127.0.0.1", help="web server host (use 0.0.0.0 to share on your network)")
    parser.add_argument("--open", action="store_true", help="open the web app in your browser")
    args = parser.parse_args(argv)

    if args.web is not None:
        from web_server import serve
        serve(port=args.web, host=args.host, use_llm=args.llm, open_browser=args.open)
        return 0

    agent = InventoryAgent(use_llm=args.llm)
    if args.examples:
        status = 0
        for q in EXAMPLES:
            print("=" * 78)
            print(f"Q: {q}\n")
            status |= answer(agent, q, args.json, args.verbose, interactive=False)
        return status
    if args.question:
        return answer(agent, args.question, args.json, args.verbose, interactive=sys.stdin.isatty())
    interactive_loop(agent, args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
