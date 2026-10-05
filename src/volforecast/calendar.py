"""Stage 2: build the NYSE trading calendar (open/close per session, incl. early closes)."""
from __future__ import annotations

import logging

import pandas as pd
import pandas_market_calendars as mcal

from volforecast.config import ensure_parent, load_config, resolve_path

logger = logging.getLogger("volforecast.calendar")

NY_TZ = "America/New_York"


def build_calendar(start: str, end: str, exchange: str = "NYSE") -> pd.DataFrame:
    """One row per trading session with open/close in America/New_York and an early-close flag.

    A "normal" NYSE session is 09:30-16:00 ET (6.5 hours = 390 minutes).
    Anything shorter is flagged as an early close.
    """
    cal = mcal.get_calendar(exchange)
    sched = cal.schedule(start_date=start, end_date=end).reset_index(drop=True)

    # Keep these tz-aware (America/New_York). Do NOT pull out `.values` on a
    # tz-aware Series: numpy has no tz-aware dtype, so `.values` silently
    # converts to UTC and strips the tz, mislabeling the clock time.
    market_open = sched["market_open"].dt.tz_convert(NY_TZ)
    market_close = sched["market_close"].dt.tz_convert(NY_TZ)
    session_minutes = (market_close - market_open).dt.total_seconds() / 60.0

    out = pd.DataFrame(
        {
            "date": market_open.dt.tz_localize(None).dt.normalize().dt.date,
            "market_open": market_open,
            "market_close": market_close,
            "session_minutes": session_minutes,
        }
    )
    out["is_early_close"] = out["session_minutes"] < 389.0  # normal session is 390 min
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    start = cfg["sample_period"]["download_start"]
    end = cfg["sample_period"]["download_end"]
    cal_df = build_calendar(start, end, cfg["calendar"]["exchange"])
    out_path = resolve_path(cfg["paths"]["calendar_path"])
    ensure_parent(out_path)
    cal_df.to_parquet(out_path, index=False)
    logger.info(
        "Saved %d trading sessions to %s (%d early closes)",
        len(cal_df),
        out_path,
        int(cal_df["is_early_close"].sum()),
    )


if __name__ == "__main__":
    main()
