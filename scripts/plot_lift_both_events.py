"""Combined early-warning lift figure: rows = horizon (h = 5, 22), columns = event (COVID, 2022), ext set.

Same forecasts, windows and cutoff markers as covid_lift / bear2022_lift in notebooks/02_model_comparison.ipynb:

    uv run python scripts/plot_lift_both_events.py   # -> reports/eval/lift_both_events.{pdf,png}
"""
import matplotlib

matplotlib.use("Agg")

from volforecast import adapters, evaluation as E, evaluation_plots as P  # noqa: E402
from volforecast.config import PROJECT_ROOT  # noqa: E402

P.use_paper_style()
features = adapters.load_features()
splits = {e: adapters.load_split(e) for e in ("covid", "bear2022")}
df = E.load_forecasts(basis="longrun", features=features)
fig, _ = P.plot_lift_grid(df, splits)
out = PROJECT_ROOT / "reports" / "eval" / "lift_both_events"
P.finish(fig, out)
print(out.with_suffix(".png"), out.with_suffix(".pdf"))
