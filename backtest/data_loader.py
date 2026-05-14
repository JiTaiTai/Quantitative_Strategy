"""
数据获取模块
- 使用 baostock 获取 A 股日线 K 线数据（免费、无需注册）
- 本地缓存，避免重复下载
- QMT 接入后，替换为实时 Level-2 / Tick 数据源即可
"""

import os
import time
import pandas as pd
import numpy as np
import baostock as bs

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data_cache")

# 股票池：用户关注的硬科技赛道
STOCK_POOL = {
    "sz.300308": {"name": "中际旭创", "sector": "CPO", "board": "cyb"},
    "sz.300502": {"name": "新易盛", "sector": "CPO", "board": "cyb"},
    "sz.300394": {"name": "天孚通信", "sector": "CPO", "board": "cyb"},
    "sz.002371": {"name": "北方华创", "sector": "芯片", "board": "sz"},
    "sz.002463": {"name": "沪电股份", "sector": "PCB", "board": "sz"},
    "sz.300124": {"name": "汇川技术", "sector": "机器人", "board": "cyb"},
    "sz.300750": {"name": "宁德时代", "sector": "电池", "board": "cyb"},
    "sz.002460": {"name": "赣锋锂业", "sector": "锂矿", "board": "sz"},
    "sh.603501": {"name": "韦尔股份", "sector": "芯片", "board": "sh"},
    "sh.601899": {"name": "紫金矿业", "sector": "有色", "board": "sh"},
}

# 不同板块涨跌停幅度
LIMIT_PCT = {"cyb": 0.20, "sz": 0.10, "sh": 0.10}


def _ensure_login():
    """确保 baostock 已登录"""
    bs.login()


def load_one_stock(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取单只股票日线数据（后复权）。
    返回 DataFrame，索引为 date，列: open, high, low, close, volume, amount, pct_change, turnover
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"{code.replace('.', '_')}_daily_{start_date}_{end_date}.parquet")

    if os.path.exists(cache_file):
        df = pd.read_parquet(cache_file)
        if not df.empty:
            return df

    _ensure_login()

    try:
        rs = bs.query_history_k_data_plus(
            code,
            "date,open,high,low,close,volume,amount,pctChg,turn",
            start_date=start_date, end_date=end_date,
            frequency="d", adjustflag="2",  # 2 = 后复权
        )
        if rs.error_code != "0":
            print(f"  [警告] {code} 查询失败: {rs.error_msg}")
            return pd.DataFrame()

        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())

        if not data_list:
            return pd.DataFrame()

        df = pd.DataFrame(data_list, columns=rs.fields)

        # 类型转换（baostock 返回全是字符串）
        numeric_cols = ["open", "high", "low", "close", "volume", "amount", "pctChg", "turn"]
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)

        # 统一列名
        df.rename(columns={
            "pctChg": "pct_change",
            "turn": "turnover",
        }, inplace=True)

        # 过滤无效数据
        df = df[df["close"] > 0]

        df.to_parquet(cache_file)
        return df

    except Exception as e:
        print(f"  [警告] {code} 下载失败: {e}")
        return pd.DataFrame()


def load_stock_pool(start_date: str, end_date: str) -> dict:
    """加载整个股票池的日线数据。返回 {code: DataFrame}"""
    _ensure_login()

    data = {}
    codes = list(STOCK_POOL.keys())
    print(f"加载股票池 {len(codes)} 只股票 (日线, baostock)...")
    for i, code in enumerate(codes):
        name = STOCK_POOL[code]["name"]
        print(f"  [{i+1}/{len(codes)}] {code} {name}...", end=" ")
        df = load_one_stock(code, start_date, end_date)
        if not df.empty:
            data[code] = df
            date_range = f"{df.index[0].date()} ~ {df.index[-1].date()}"
            print(f"{len(df)} 条 ({date_range})")
        else:
            print("无数据")
        if i < len(codes) - 1:
            time.sleep(0.3)  # 避免请求过快
    print(f"成功加载 {len(data)} 只股票\n")

    bs.logout()
    return data


def get_limit_price(code: str, prev_close: float) -> tuple:
    """返回 (跌停价, 涨停价)"""
    board = STOCK_POOL[code]["board"]
    pct = LIMIT_PCT[board]
    limit_down = round(prev_close * (1 - pct), 2)
    limit_up = round(prev_close * (1 + pct), 2)
    return limit_down, limit_up
