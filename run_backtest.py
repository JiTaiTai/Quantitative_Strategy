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
        "position_per_trade": 20_000,

        # 风控
        "stop_loss": -0.05,
        "take_profit": 0.08,
        "daily_loss_limit": -5000,
        "weekly_loss_limit": -12000,
        "consecutive_loss_limit": 5,
        "max_holdings": 5,
        "max_daily_trades": 3,
        "cooldown_days": 5,

        # 事件参数
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
    print(f"  说明: 分钟线 API 暂不可用，日线验证策略框架")
    print(f"        QMT 接入后替换为实时数据源")
    print("=" * 60)
    print()

    # ==================== 1. 加载数据 ====================
    print("[1/3] 获取行情数据...")
    data = load_stock_pool(START_DATE, END_DATE)

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
