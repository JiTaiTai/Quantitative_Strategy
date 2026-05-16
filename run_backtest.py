"""
事件驱动量化策略 — 回测入口（日线版）

当前限制：akshare 分钟线 API 暂时不可用，先用日线数据验证策略框架。
QMT 接入后，数据源替换为实时 Level-2 / Tick，事件检测逻辑不变。

用法：python run_backtest.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from backtest.data_loader import load_stock_pool
from backtest.engine import BacktestEngine
from backtest.reporter import generate_report, print_report, plot_report


def main():
    CONFIG = {
        # 资金
        "initial_cash": 200_000,
        "position_per_trade": 40_000,

        # 风控
        "stop_loss": -0.08,
        "take_profit": 10.0,
        "daily_loss_limit": -5000,
        "weekly_loss_limit": -12000,
        "consecutive_loss_limit": 5,
        "loss_cooldown_days": 10,
        "max_holdings": 5,
        "max_daily_trades": 5,
        "max_sector_holdings": 3,
        "cooldown_days": 5,
        "max_hold_days": 180,
        "min_hold_days": 3,
        "trailing_drawdown": 0.20,
        "profit_exit_ma": "ma60",
        "min_position_cash": 10_000,
        "regime_exposure": {
            "bull": 0.95,
            "neutral": 0.65,
            "bear": 0.30,
        },

        # 股票宇宙：只允许 A 股主板股票。
        # sh = 沪市主板 60xxxx；sz = 深市主板 00xxxx。
        # 不允许创业板 30xxxx、科创板 68xxxx、北交所 43/83/87/88xxxx。
        "allowed_boards": ["sh", "sz"],
        "exclude_st": True,
        "max_universe_size": 120,
        "max_new_downloads_per_run": 30,
        "download_delay": 0.1,
        "min_history_days": 180,

        # 执行约束（日线信号 -> 次日开盘成交）
        "buy_slippage": 0.001,
        "sell_slippage": 0.001,
        "limit_buffer": 0.002,
        "order_timeout_days": 3,

        # 事件参数
        "enabled_events": ["趋势底仓", "趋势启动", "放量突破"],
        "volume_mult": 2.0,
        "gap_pct": 0.02,
        "ma_short": 5,
        "ma_long": 20,
    }

    # 回测时间范围（日线数据可以拿更长的）
    START_DATE = "2024-01-01"
    END_DATE = "2026-05-14"

    print("=" * 60)
    print("  事件驱动量化策略 — 回测（日线版）")
    print(f"  回测区间: {START_DATE} ~ {END_DATE}")
    print(f"  数据频率: 日线")
    print(f"  初始资金: {CONFIG['initial_cash']:,} 元")
    print(f"  说明: 收盘生成信号，次日开盘按滑点/涨跌停/T+1约束成交")
    print(f"        当前仅启用: {', '.join(CONFIG['enabled_events'])}")
    print("=" * 60)
    print()

    # ==================== 1. 加载数据 ====================
    print("[1/3] 获取行情数据...")
    data = load_stock_pool(START_DATE, END_DATE, CONFIG)

    if not data:
        print("\n[错误] 没有获取到任何数据。")
        return

    total_bars = sum(len(df) for df in data.values())
    print(f"总数据量: {total_bars:,} 条日线\n")

    # ==================== 2. 运行回测 ====================
    print("[2/3] 运行回测...")
    engine = BacktestEngine(data, CONFIG)
    engine.run()

    # ==================== 3. 生成报告 ====================
    print("[3/3] 生成报告...")
    report = generate_report(
        trades=engine.portfolio.trades,
        equity_curve=engine.portfolio.equity_curve,
        initial_cash=CONFIG["initial_cash"],
    )

    if report:
        print_report(report)
        chart_path = os.path.join(os.path.dirname(__file__), "backtest_result.png")
        plot_report(report, engine.portfolio.equity_curve, save_path=chart_path)

        closed = [t for t in engine.portfolio.trades if t.sell_date is not None]
        if closed:
            print("\n最近 15 笔交易明细：")
            print(f"  {'买入日':<12} {'股票':<12} {'事件':<12} {'买入价':>8} {'卖出价':>8} {'盈亏%':>8} {'盈亏额':>8} {'原因'}")
            for t in closed[-15:]:
                print(f"  {str(t.buy_date.date()):<12} {t.name:<12} {t.event:<12} "
                      f"{t.buy_price:>8.2f} {t.sell_price:>8.2f} {t.pnl_pct:>7.1%} "
                      f"{t.pnl_amount:>8.0f} {t.sell_reason}")
    else:
        print("未能生成报告。")


if __name__ == "__main__":
    main()
