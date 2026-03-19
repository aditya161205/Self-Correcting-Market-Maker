import pandas as pd


EPS = 1e-12


def compute_step_pnl(df: pd.DataFrame) -> pd.Series:
    return df["wealth"].diff().fillna(0.0)


def compute_spread_series(df: pd.DataFrame) -> pd.Series:
    return df["ask"] - df["bid"]


def compute_pnl_metrics(df: pd.DataFrame) -> dict:
    step_pnl = compute_step_pnl(df)

    return {
        "final_wealth": float(df["wealth"].iloc[-1]),
        "mean_step_pnl": float(step_pnl.mean()),
        "std_step_pnl": float(step_pnl.std()),
    }


def compute_sharpe_like(df: pd.DataFrame) -> dict:
    step_pnl = compute_step_pnl(df)
    mean_step_pnl = float(step_pnl.mean())
    std_step_pnl = float(step_pnl.std())
    sharpe_like = mean_step_pnl / (std_step_pnl + EPS)

    return {
        "sharpe_like": float(sharpe_like),
    }


def compute_drawdown_metrics(df: pd.DataFrame) -> dict:
    running_max = df["wealth"].cummax()
    drawdown = running_max - df["wealth"]

    return {
        "max_drawdown": float(drawdown.max()),
        "mean_drawdown": float(drawdown.mean()),
    }


def compute_inventory_metrics(df: pd.DataFrame) -> dict:
    inventory = df["inventory"]

    return {
        "final_inventory": int(inventory.iloc[-1]),
        "mean_inventory": float(inventory.mean()),
        "mean_abs_inventory": float(inventory.abs().mean()),
        "inventory_std": float(inventory.std()),
        "max_inventory": int(inventory.max()),
        "min_inventory": int(inventory.min()),
    }


def compute_fill_metrics(df: pd.DataFrame) -> dict:
    num_bid_fills = int(df["bid_fill"].sum())
    num_ask_fills = int(df["ask_fill"].sum())
    total_fills = num_bid_fills + num_ask_fills
    fill_imbalance = abs(num_bid_fills - num_ask_fills)

    return {
        "num_bid_fills": num_bid_fills,
        "num_ask_fills": num_ask_fills,
        "total_fills": total_fills,
        "fill_imbalance": int(fill_imbalance),
    }


def compute_tail_risk_metrics(df: pd.DataFrame) -> dict:
    step_pnl = compute_step_pnl(df)

    return {
        "step_pnl_p05": float(step_pnl.quantile(0.05)),
        "step_pnl_p01": float(step_pnl.quantile(0.01)),
    }


def compute_quote_stability_metrics(df: pd.DataFrame) -> dict:
    bid_changes = df["bid"].diff().fillna(0.0)
    ask_changes = df["ask"].diff().fillna(0.0)
    spread_changes = compute_spread_series(df).diff().fillna(0.0)

    return {
        "bid_change_std": float(bid_changes.std()),
        "ask_change_std": float(ask_changes.std()),
        "spread_change_std": float(spread_changes.std()),
    }


def compute_all_metrics(df: pd.DataFrame) -> dict:
    metrics = {}
    metrics.update(compute_pnl_metrics(df))
    metrics.update(compute_sharpe_like(df))
    metrics.update(compute_drawdown_metrics(df))
    metrics.update(compute_inventory_metrics(df))
    metrics.update(compute_fill_metrics(df))
    metrics.update(compute_tail_risk_metrics(df))
    metrics.update(compute_quote_stability_metrics(df))
    return metrics