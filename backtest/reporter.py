"""
回测报告生成
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt
from typing import List
from .engine import Trade

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def generate_report(trades: List[Trade], equity_curve: List[dict],
                    initial_cash: float, benchmark_name: str = "中证500") -> dict:
    """生成回测报告"""
    closed_trades = [t for t in trades if t.sell_date is not None]
    open_trades = [t for t in trades if t.sell_date is None]

    if not equity_curve:
        print("权益曲线为空，无法生成报告")
        return {}

    eq_df = pd.DataFrame(equity_curve)
    eq_df.set_index("time", inplace=True)

    final_equity = eq_df["equity"].iloc[-1]
    total_return = (final_equity - initial_cash) / initial_cash
    exposure = eq_df["position_value"] / eq_df["equity"].replace(0, np.nan)
    avg_exposure = exposure.fillna(0).mean()
    max_exposure = exposure.fillna(0).max()

    # 年化收益率
    days = (eq_df.index[-1] - eq_df.index[0]).days
    years = max(days / 365, 0.01)
    annual_return = (1 + total_return) ** (1 / years) - 1

    # 最大回撤
    cummax = eq_df["equity"].cummax()
    drawdown = (eq_df["equity"] - cummax) / cummax
    max_drawdown = drawdown.min()

    # 夏普比率（日频近似）
    daily_returns = eq_df["equity"].pct_change().dropna()
    if daily_returns.std() > 0:
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    else:
        sharpe = 0

    # 胜率
    if closed_trades:
        win_rate = sum(1 for t in closed_trades if t.pnl_amount > 0) / len(closed_trades)
        total_pnl = sum(t.pnl_amount for t in closed_trades)
        avg_win = np.mean([t.pnl_amount for t in closed_trades if t.pnl_amount > 0]) if any(t.pnl_amount > 0 for t in closed_trades) else 0
        avg_loss = np.mean([t.pnl_amount for t in closed_trades if t.pnl_amount < 0]) if any(t.pnl_amount < 0 for t in closed_trades) else 0
        profit_factor = abs(sum(t.pnl_amount for t in closed_trades if t.pnl_amount > 0) / sum(t.pnl_amount for t in closed_trades if t.pnl_amount < 0)) if sum(t.pnl_amount for t in closed_trades if t.pnl_amount < 0) != 0 else float("inf")
    else:
        win_rate = 0
        total_pnl = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0

    # 按事件类型统计
    event_stats = {}
    for t in closed_trades:
        event = t.event
        if event not in event_stats:
            event_stats[event] = {"count": 0, "win": 0, "total_pnl": 0, "avg_hold_days": 0}
        event_stats[event]["count"] += 1
        if t.pnl_amount > 0:
            event_stats[event]["win"] += 1
        event_stats[event]["total_pnl"] += t.pnl_amount
        event_stats[event]["avg_hold_days"] += t.holding_days

    for e in event_stats:
        s = event_stats[e]
        s["win_rate"] = s["win"] / s["count"] if s["count"] > 0 else 0
        s["avg_hold_days"] = s["avg_hold_days"] / s["count"] if s["count"] > 0 else 0

    report = {
        "初始资金": initial_cash,
        "最终权益": final_equity,
        "总收益率": total_return,
        "年化收益率": annual_return,
        "最大回撤": max_drawdown,
        "夏普比率": sharpe,
        "交易总数": len(closed_trades),
        "胜率": win_rate,
        "总盈亏": total_pnl,
        "平均盈利": avg_win,
        "平均亏损": avg_loss,
        "盈亏比": profit_factor,
        "平均仓位": avg_exposure,
        "最高仓位": max_exposure,
        "持仓中": len(open_trades),
        "事件统计": event_stats,
    }
    return report


def print_report(report: dict):
    """打印报告"""
    print("=" * 60)
    print("  回 测 报 告")
    print("=" * 60)
    print(f"  初始资金      : {report['初始资金']:>12,.0f} 元")
    print(f"  最终权益      : {report['最终权益']:>12,.0f} 元")
    print(f"  总收益率      : {report['总收益率']:>11.2%}")
    print(f"  年化收益率    : {report['年化收益率']:>11.2%}")
    print(f"  最大回撤      : {report['最大回撤']:>11.2%}")
    print(f"  夏普比率      : {report['夏普比率']:>11.2f}")
    print("-" * 60)
    print(f"  交易总数      : {report['交易总数']:>11} 笔")
    print(f"  胜率          : {report['胜率']:>11.1%}")
    print(f"  总盈亏        : {report['总盈亏']:>11,.0f} 元")
    print(f"  平均盈利      : {report['平均盈利']:>11,.0f} 元")
    print(f"  平均亏损      : {report['平均亏损']:>11,.0f} 元")
    print(f"  盈亏比        : {report['盈亏比']:>11.2f}")
    print(f"  平均仓位      : {report['平均仓位']:>11.1%}")
    print(f"  最高仓位      : {report['最高仓位']:>11.1%}")
    print(f"  持仓中        : {report['持仓中']:>11} 笔")
    print("-" * 60)

    if report["事件统计"]:
        print("  各事件表现：")
        print(f"  {'事件':<16} {'次数':>5} {'胜率':>7} {'累计盈亏':>10} {'平均持天':>8}")
        for event, stats in sorted(report["事件统计"].items()):
            print(f"  {event:<16} {stats['count']:>5} {stats['win_rate']:>6.1%} "
                  f"{stats['total_pnl']:>10,.0f} {stats['avg_hold_days']:>7.0f}天")
    print("=" * 60)


def plot_report(report: dict, equity_curve: List[dict], save_path: str = None):
    """绘制回测图表"""
    eq_df = pd.DataFrame(equity_curve)
    eq_df.set_index("time", inplace=True)

    # 全部转 numpy，避免 pandas 2.x + matplotlib 多维索引兼容问题
    t = eq_df.index.values
    equity = eq_df["equity"].values
    pos_value = eq_df["position_value"].values
    cummax = np.maximum.accumulate(equity)
    drawdown = (equity - cummax) / cummax * 100

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("事件驱动策略回测报告", fontsize=16, fontweight="bold")

    # ---- 左上：权益曲线 ----
    ax1 = axes[0, 0]
    ax1.plot(t, equity, color="#1f77b4", linewidth=1, label="策略权益")
    ax1.axhline(y=report["初始资金"], color="gray", linestyle="--", alpha=0.5, label="初始资金")
    ax1.fill_between(t, report["初始资金"], equity,
                     where=equity >= report["初始资金"],
                     color="#1f77b4", alpha=0.1)
    ax1.fill_between(t, report["初始资金"], equity,
                     where=equity < report["初始资金"],
                     color="#d62728", alpha=0.1)
    ax1.set_title("权益曲线")
    ax1.set_ylabel("权益（元）")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # ---- 右上：回撤曲线 ----
    ax2 = axes[0, 1]
    ax2.fill_between(t, 0, drawdown, color="#d62728", alpha=0.3)
    ax2.plot(t, drawdown, color="#d62728", linewidth=0.5)
    ax2.set_title("回撤曲线")
    ax2.set_ylabel("回撤 (%)")
    ax2.grid(True, alpha=0.3)

    # ---- 左下：持仓市值 ----
    ax3 = axes[1, 0]
    ax3.fill_between(t, 0, pos_value, color="#2ca02c", alpha=0.3)
    ax3.plot(t, pos_value, color="#2ca02c", linewidth=0.5)
    ax3.set_title("持仓市值变化")
    ax3.set_ylabel("持仓市值（元）")
    ax3.grid(True, alpha=0.3)

    # ---- 右下：事件统计 ----
    ax4 = axes[1, 1]
    if report["事件统计"]:
        events = list(report["事件统计"].keys())
        counts = [report["事件统计"][e]["count"] for e in events]
        wins = [report["事件统计"][e]["win"] for e in events]
        losses = [c - w for c, w in zip(counts, wins)]

        y_pos = range(len(events))
        ax4.barh(y_pos, wins, color="#2ca02c", alpha=0.7, label="盈利")
        ax4.barh(y_pos, losses, left=wins, color="#d62728", alpha=0.7, label="亏损")
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(events)
        ax4.set_title("各事件交易次数")
        ax4.legend(loc="lower right")
        ax4.set_xlabel("次数")
    else:
        ax4.text(0.5, 0.5, "无交易记录", ha="center", va="center", transform=ax4.transAxes)
    ax4.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"图表已保存至: {save_path}")
    else:
        plt.show()
    plt.close()
