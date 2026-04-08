import json
import math
import os
import sys
from datetime import datetime, timezone

# Agrega la raíz del repo al path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.api.macro import macro_snapshot


def clean_value(x):
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(xf) or math.isinf(xf):
        return None
    return xf


def main():
    macro = macro_snapshot()

    data = {
        "risk_free_rate_pct": clean_value(macro.get("risk_free_rate_pct")),
        "inflation_yoy": clean_value(macro.get("inflation_yoy")),
        "cop_per_usd": clean_value(macro.get("cop_per_usd")),
        "usdcop_market": clean_value(macro.get("usdcop_market")),
        "source": "github_actions_cache",
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }

    with open(os.path.join(ROOT_DIR, "data", "macro_cache.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()