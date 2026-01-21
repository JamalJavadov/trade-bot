from .planner import Plan

def format_binance_futures_form(plan: Plan, sizing: dict, manual_cfg: dict) -> str:
    side_btn = "Buy/Long" if plan.side == "LONG" else "Sell/Short"
    order_type = "Limit"  # pending üçün
    trigger = manual_cfg.get("trigger_type", "Mark")
    margin_mode = manual_cfg.get("margin_mode", "Isolated")
    tif = manual_cfg.get("tif", "GTC")
    lev = manual_cfg.get("leverage", 3)
    expiry = manual_cfg.get("expiry_days", 7)

    if not sizing.get("ok"):
        return f"NO SIZE: {sizing.get('reason','')}"

    return (
        "=== BINANCE FUTURES FORM (manual doldurma) ===\n"
        f"Market: {plan.symbol}\n"
        f"Margin: {margin_mode}\n"
        f"Leverage: {lev}x\n\n"
        f"Tab: {order_type}\n"
        f"Price (Entry): {sizing['entry']}\n"
        f"Size (Qty): {sizing['qty']}\n"
        f"TP/SL: ON\n"
        f"  Take Profit: {sizing['tp']}   Trigger: {trigger}\n"
        f"  Stop Loss:   {sizing['sl']}   Trigger: {trigger}\n"
        f"Reduce-Only: OFF (entry açırsan)\n"
        f"TIF: {tif}\n"
        f"Action: {side_btn}\n\n"
        f"Expiry: {expiry} gün (trigger olmazsa cancel)\n"
        f"Risk target: {sizing['risk_target']:.4f} USDT | Risk actual: {sizing['risk_actual']:.4f} USDT\n"
        + ("[QEYD] Notional cap tətbiq olundu.\n" if sizing.get("capped") else "")
    )
