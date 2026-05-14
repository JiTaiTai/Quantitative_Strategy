"""
回测引擎（日线版）
- 事件检测：6 类事件适配到日频（用日 OHLCV 近似盘中异动）
- 投资组合管理：仓位、现金、T+1
- 风控：止损止盈、熔断

信号基于当日收盘数据触发，执行于次日开盘。

QMT 接入后，将 EventDetector 的数据源从日线 DataFrame
切换为实时 Tick/Level-2，事件逻辑保持不变。
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from .data_loader import STOCK_POOL, get_limit_price


# ============================================================
# 交易记录
# ============================================================

@dataclass
class Trade:
    code: str
    name: str
    event: str
    buy_date: pd.Timestamp
    buy_price: float
    buy_shares: int = 0
    sell_date: Optional[pd.Timestamp] = None
    sell_price: Optional[float] = None
    sell_reason: str = ""
    pnl_pct: float = 0.0
    pnl_amount: float = 0.0
    holding_days: int = 0


# ============================================================
# 事件检测器（日线适配版）
# ============================================================

class EventDetector:
    """
    事件检测器。日线版本用 OHLCV 近似盘中行为：

    局限性（日线数据无法精确模拟的）：
    - 无法判断价格运动的先后顺序（先跌后涨 vs 先涨后跌）
    - 无法判断成交的日内分布
    - 无法获取 Level-2 盘口数据

    这些局限性在 QMT 接入后自动消除——事件逻辑不变，数据精度提升。
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.vol_mult = config.get("volume_mult", 2.0)
        self.gap_pct = config.get("gap_pct", 0.02)
        self.ma_short = config.get("ma_short", 5)
        self.ma_long = config.get("ma_long", 20)

    # -------- 事件 1：跌停板翘板 --------
    def detect_limit_down_reversal(
        self, code: str, bar: pd.Series, hist: pd.DataFrame
    ) -> bool:
        """
        日线版本：日内触及跌停但收盘弹起。
        条件：
        1. 最低价触及跌停价（或非常接近）
        2. 收盘价相对最低点回升 > 3%
        3. 成交量放大
        """
        if len(hist) < 20:
            return False
        prev_close = hist.iloc[-2]["close"] if len(hist) >= 2 else bar["open"]
        limit_down, _ = get_limit_price(code, prev_close)

        # 条件 1：最低触及跌停
        if bar["low"] > limit_down * 1.005:
            return False

        # 条件 2：翘板 — 收盘从最低回升 > 3%
        rebound = (bar["close"] - bar["low"]) / bar["low"] if bar["low"] > 0 else 0
        if rebound < 0.03:
            return False

        # 条件 3：放量
        avg_vol = hist["volume"].tail(20).mean()
        if avg_vol <= 0 or bar["volume"] < avg_vol * self.vol_mult:
            return False

        return True

    # -------- 事件 2：持续资金流入（日线版） --------
    def detect_sustained_inflow(
        self, code: str, bar: pd.Series, hist: pd.DataFrame
    ) -> bool:
        """
        日线版：连续 N 天放量收阳（买方主导）。
        条件：
        1. 连续 3 天收盘 > 开盘（阳线）
        2. 每天成交量 > 20 日均量
        3. 累计涨幅 < 8%（还没爆拉）
        4. 当前日成交量放大
        """
        if len(hist) < 20:
            return False

        recent = hist.tail(3)
        if len(recent) < 3:
            return False

        avg_vol = hist["volume"].tail(20).mean()
        if avg_vol <= 0:
            return False

        # 连续 3 天阳线且放量
        for _, r in recent.iterrows():
            if r["close"] <= r["open"]:
                return False
            if r["volume"] < avg_vol:
                return False

        # 累计涨幅 < 8%
        cum_return = (bar["close"] - recent.iloc[0]["open"]) / recent.iloc[0]["open"]
        if cum_return >= 0.08:
            return False

        # 当前日进一步放量
        if bar["volume"] < avg_vol * self.vol_mult:
            return False

        return True

    # -------- 事件 3：放量突破（大单扫货日线版） --------
    def detect_volume_breakout(
        self, code: str, bar: pd.Series, hist: pd.DataFrame
    ) -> bool:
        """
        日线版：单日巨量上涨，收盘接近最高价。
        条件：
        1. 成交量 > 20 日均量 × 2.5
        2. 收盘在当日上 1/3（买方主导全天）
        3. 涨幅 > +2%
        4. 成交额 > 大单阈值
        """
        if len(hist) < 20:
            return False

        avg_vol = hist["volume"].tail(20).mean()
        if avg_vol <= 0:
            return False

        # 条件 1：极端放量
        if bar["volume"] < avg_vol * 2.5:
            return False

        # 条件 2：收盘在高位
        bar_range = bar["high"] - bar["low"]
        if bar_range <= 0:
            return False
        close_pos = (bar["close"] - bar["low"]) / bar_range
        if close_pos < 0.667:
            return False

        # 条件 3：涨幅
        if bar["pct_change"] < 2.0:
            return False

        # 条件 4：成交额
        threshold = self.config.get("big_order_amount", 300_000)
        if bar["amount"] < threshold:
            return False

        return True

    # -------- 事件 4：涨停封板 --------
    def detect_limit_up_seal(
        self, code: str, bar: pd.Series, hist: pd.DataFrame
    ) -> bool:
        """
        日线版：收盘封涨停。
        条件：
        1. 收盘距涨停 < 0.5%
        2. 当日非一字板（open < limit_up，有机会进场）
        3. 换手率适中（3%-15%，太高可能是出货）
        4. 前一日换手不能太低（排除无量一字）
        """
        if len(hist) < 2:
            return False
        prev_close = hist.iloc[-2]["close"]
        _, limit_up = get_limit_price(code, prev_close)

        # 条件 1：封板
        if bar["close"] < limit_up * 0.995:
            return False

        # 条件 2：非一字板（有机会买入）
        if bar["open"] >= limit_up * 0.998:
            return False

        # 条件 3：换手率适中
        turnover = bar.get("turnover", 5)
        if turnover is None or np.isnan(turnover):
            return True  # 无换手率数据时放行
        if turnover < 3 or turnover > 15:
            return False

        return True

    # -------- 事件 5：开盘跳空确认 --------
    def detect_opening_gap(
        self, code: str, bar: pd.Series, hist: pd.DataFrame
    ) -> bool:
        """
        日线版：跳空高开 + 收盘不补缺口。
        条件：
        1. 开盘相对前日收盘跳空 > 2%
        2. 最低价 > 开盘价 × 0.99（没有回补缺口）
        3. 收盘 > 开盘（阳线确认）
        4. 成交量 > 20 日均量
        """
        if len(hist) < 20:
            return False
        prev_close = hist.iloc[-2]["close"]

        # 条件 1：跳空
        gap = (bar["open"] - prev_close) / prev_close
        if gap < self.gap_pct:
            return False

        # 条件 2：不补缺口
        if bar["low"] < bar["open"] * 0.99:
            return False

        # 条件 3：阳线确认
        if bar["close"] <= bar["open"]:
            return False

        # 条件 4：放量
        avg_vol = hist["volume"].tail(20).mean()
        if avg_vol <= 0 or bar["volume"] < avg_vol:
            return False

        return True

    # -------- 事件 6：趋势启动（均线多头 + 放量） --------
    def detect_trend_launch(
        self, code: str, bar: pd.Series, hist: pd.DataFrame
    ) -> bool:
        """
        日线版：均线刚形成多头排列 + 放量突破。
        条件：
        1. MA5 > MA20（金叉不久）
        2. 前一日 MA5 <= MA20（刚金叉，不是已经走很久了）
        3. 收盘站上 MA5
        4. 成交量放大
        """
        if len(hist) < 60:
            return False

        closes = hist["close"]
        ma5 = closes.rolling(5).mean()
        ma20 = closes.rolling(20).mean()

        if len(ma5) < 21:
            return False

        # 条件 1：当前金叉
        if ma5.iloc[-1] <= ma20.iloc[-1]:
            return False

        # 条件 2：昨日未金叉 → 今天是金叉日
        if ma5.iloc[-2] > ma20.iloc[-2]:
            return False

        # 条件 3：收盘 > MA5
        if bar["close"] <= ma5.iloc[-1]:
            return False

        # 条件 4：放量
        avg_vol = hist["volume"].tail(20).mean()
        if avg_vol <= 0 or bar["volume"] < avg_vol * 1.5:
            return False

        return True

    def detect_all(self, code: str, bar: pd.Series, hist: pd.DataFrame) -> List[str]:
        """运行所有事件检测，返回触发的事件名列表"""
        events = []
        if self.detect_limit_down_reversal(code, bar, hist):
            events.append("跌停翘板")
        if self.detect_sustained_inflow(code, bar, hist):
            events.append("持续吸筹")
        if self.detect_volume_breakout(code, bar, hist):
            events.append("放量突破")
        if self.detect_limit_up_seal(code, bar, hist):
            events.append("涨停封板")
        if self.detect_opening_gap(code, bar, hist):
            events.append("开盘跳空")
        if self.detect_trend_launch(code, bar, hist):
            events.append("趋势启动")
        return events


# ============================================================
# 风控管理器
# ============================================================

class RiskManager:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.stop_loss = config.get("stop_loss", -0.05)
        self.take_profit = config.get("take_profit", 0.08)
        self.daily_loss_limit = config.get("daily_loss_limit", -5000)
        self.weekly_loss_limit = config.get("weekly_loss_limit", -12000)
        self.consecutive_loss_limit = config.get("consecutive_loss_limit", 5)
        self.max_holdings = config.get("max_holdings", 5)
        self.max_daily_trades = config.get("max_daily_trades", 3)
        self.position_per_trade = config.get("position_per_trade", 20000)
        self.cooldown_days = config.get("cooldown_days", 5)

        # 运行时状态
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.consecutive_losses = 0
        self.daily_trade_count = 0
        self.current_date = None
        self.current_week = None
        self.blacklist: Dict[str, pd.Timestamp] = {}
        self.last_sell_date: Dict[str, pd.Timestamp] = {}

    def start_day(self, date: pd.Timestamp):
        if self.current_date != date:
            self.daily_pnl = 0.0
            self.daily_trade_count = 0
            self.current_date = date
        week = date.isocalendar()[1]
        if self.current_week != week:
            self.weekly_pnl = 0.0
            self.current_week = week

    def can_open(self, code: str, current_date: pd.Timestamp) -> Tuple[bool, str]:
        """检查是否允许开仓"""
        if self.daily_pnl < self.daily_loss_limit:
            return False, f"日内亏损熔断 ({self.daily_pnl:.0f})"
        if self.weekly_pnl < self.weekly_loss_limit:
            return False, f"本周亏损熔断 ({self.weekly_pnl:.0f})"
        if self.consecutive_losses >= self.consecutive_loss_limit:
            return False, f"连续亏损 {self.consecutive_losses} 笔"
        if self.daily_trade_count >= self.max_daily_trades:
            return False, "今日交易次数已满"
        if code in self.blacklist and current_date < self.blacklist[code]:
            return False, "黑名单中"
        if code in self.last_sell_date:
            days_since = (current_date - self.last_sell_date[code]).days
            if days_since < self.cooldown_days:
                return False, f"冷却中 ({days_since}/{self.cooldown_days} 天)"
        return True, ""

    def record_sell(self, code: str, date: pd.Timestamp):
        self.last_sell_date[code] = date

    def record_trade_close(self, trade: Trade):
        self.daily_pnl += trade.pnl_amount
        self.weekly_pnl += trade.pnl_amount
        if trade.pnl_amount < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

    def add_blacklist(self, code: str, until: pd.Timestamp):
        self.blacklist[code] = until


# ============================================================
# 投资组合
# ============================================================

class Portfolio:
    def __init__(self, initial_cash: float = 200_000):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, dict] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[dict] = []

    def buy(self, code: str, name: str, event: str, price: float, date: pd.Timestamp,
            risk_mgr: RiskManager = None) -> Optional[Trade]:
        """买入。信号日收盘触发，次日开盘执行。用当日收盘价近似次日开盘价。"""
        amount_per = risk_mgr.position_per_trade if risk_mgr else 20000
        amount = min(amount_per, self.cash * 0.95)  # 留 5% 现金

        shares = int(amount / price / 100) * 100
        if shares == 0:
            return None

        cost = shares * price * (1 + 0.0003)
        if cost > self.cash:
            shares = int(self.cash * 0.95 / (price * 1.0003) / 100) * 100
            if shares == 0:
                return None
            cost = shares * price * (1 + 0.0003)

        self.cash -= cost
        self.positions[code] = {
            "shares": shares,
            "avg_cost": price,
            "buy_date": date,
            "buy_price": price,
            "peak_price": price,
            "event": event,
        }

        trade = Trade(code=code, name=name, event=event,
                      buy_date=date, buy_price=price, buy_shares=shares)
        self.trades.append(trade)
        return trade

    def sell(self, code: str, price: float, date: pd.Timestamp, reason: str):
        if code not in self.positions:
            return
        pos = self.positions.pop(code)
        shares = pos["shares"]
        proceeds = shares * price * (1 - 0.0003 - 0.001)  # 佣金 + 印花税

        for trade in self.trades:
            if trade.code == code and trade.sell_date is None:
                trade.sell_date = date
                trade.sell_price = price
                trade.sell_reason = reason
                trade.pnl_pct = (price - pos["avg_cost"]) / pos["avg_cost"]
                trade.pnl_amount = (price - pos["avg_cost"]) * shares
                buy_fee = pos["avg_cost"] * shares * 0.0003
                sell_fee = price * shares * (0.0003 + 0.001)
                trade.pnl_amount -= (buy_fee + sell_fee)
                trade.holding_days = (date - pos["buy_date"]).days
                break

        self.cash += proceeds

    def update_peak(self, code: str, price: float):
        if code in self.positions and price > self.positions[code]["peak_price"]:
            self.positions[code]["peak_price"] = price

    def total_value(self, prices: Dict[str, float]) -> float:
        pos_value = sum(
            pos["shares"] * prices.get(code, pos["avg_cost"])
            for code, pos in self.positions.items()
        )
        return self.cash + pos_value

    def record_equity(self, date: pd.Timestamp, prices: Dict[str, float]):
        pos_value = sum(
            pos["shares"] * prices.get(code, pos["avg_cost"])
            for code, pos in self.positions.items()
        )
        self.equity_curve.append({
            "time": date,
            "equity": self.cash + pos_value,
            "cash": self.cash,
            "position_value": pos_value,
        })


# ============================================================
# 回测引擎
# ============================================================

class BacktestEngine:
    def __init__(self, data: dict, config: dict = None):
        self.data = data
        self.config = config or {}
        self.detector = EventDetector(config)
        self.risk_mgr = RiskManager(config)
        self.portfolio = Portfolio(config.get("initial_cash", 200_000))
        self._build_timeline()

    def _build_timeline(self):
        all_dates = set()
        for code, df in self.data.items():
            all_dates.update(df.index)
        self.timeline = sorted(all_dates)

    def run(self):
        print(f"回测时间线: {self.timeline[0].date()} ~ {self.timeline[-1].date()}")
        print(f"共 {len(self.timeline)} 个交易日\n")

        for i, date in enumerate(self.timeline):
            if i % 30 == 0:
                eq = self.portfolio.total_value({})
                print(f"\r  进度: {i/len(self.timeline)*100:.0f}% | "
                      f"净值: {eq:,.0f} | 持仓: {len(self.portfolio.positions)} | "
                      f"已平仓: {len([t for t in self.portfolio.trades if t.sell_date])}",
                      end="", flush=True)

            self.risk_mgr.start_day(date)

            # 收集当日价格
            prices = {}
            for code in self.data:
                df = self.data[code]
                if date in df.index:
                    prices[code] = df.loc[date, "close"]

            # ---- 检查持仓的止损止盈 ----
            for code in list(self.portfolio.positions.keys()):
                if code not in prices:
                    continue
                pos = self.portfolio.positions[code]
                price = prices[code]
                self.portfolio.update_peak(code, price)

                pnl = (price - pos["avg_cost"]) / pos["avg_cost"]

                # 止损
                if pnl <= self.risk_mgr.stop_loss:
                    self.portfolio.sell(code, price, date, "止损")
                    self.risk_mgr.record_sell(code, date)
                    continue

                # 止盈
                if pnl >= self.risk_mgr.take_profit:
                    self.portfolio.sell(code, price, date, "止盈")
                    self.risk_mgr.record_sell(code, date)
                    continue

                # 回撤止盈
                peak = pos["peak_price"]
                dd = (price - peak) / peak if peak > 0 else 0
                if dd <= -0.08:
                    self.portfolio.sell(code, price, date, "回撤止盈")
                    self.risk_mgr.record_sell(code, date)
                    continue

                # T+1 后才可卖，但日线每个 bar 都是一天，所以 T+1 自动满足

                # 时限止盈：持有 10 个交易日
                holding = (date - pos["buy_date"]).days
                if holding >= 10:
                    self.portfolio.sell(code, price, date, "时限到期")
                    self.risk_mgr.record_sell(code, date)

            # ---- 事件检测 ----
            for code in self.data:
                if code in self.portfolio.positions:
                    continue

                df = self.data[code]
                if date not in df.index:
                    continue

                bar = df.loc[date]
                hist = df[df.index <= date]

                # 跌停/涨停无法买入
                if len(hist) >= 2:
                    prev_close = hist.iloc[-2]["close"]
                    ld, lu = get_limit_price(code, prev_close)
                    if bar["close"] <= ld * 1.002 or bar["close"] >= lu * 0.998:
                        continue

                # 风控
                can_open, _ = self.risk_mgr.can_open(code, date)
                if not can_open:
                    continue

                # 事件检测
                events = self.detector.detect_all(code, bar, hist)
                if not events:
                    continue

                # 触发买入
                self.portfolio.buy(
                    code, STOCK_POOL[code]["name"], events[0],
                    bar["close"], date, risk_mgr=self.risk_mgr
                )
                self.risk_mgr.daily_trade_count += 1

            # 每日记录权益曲线
            self.portfolio.record_equity(date, prices)

        # 最后强制平仓
        final_prices = {}
        for code in self.data:
            df = self.data[code]
            if not df.empty:
                final_prices[code] = df.iloc[-1]["close"]
        for code in list(self.portfolio.positions.keys()):
            price = final_prices.get(code, self.portfolio.positions[code]["avg_cost"])
            self.portfolio.sell(code, price, self.timeline[-1], "回测结束")
        self.portfolio.record_equity(self.timeline[-1], final_prices)

        print("\n回测完成。\n")
