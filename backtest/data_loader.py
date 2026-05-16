"""
数据获取模块
- 使用 baostock 获取 A 股日线 K 线数据（免费、无需注册）
- 本地缓存，避免重复下载
- QMT 接入后，替换为实时 Level-2 / Tick 数据源即可
"""

import os
import time
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import baostock as bs

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data_cache")
_BAOSTOCK_LOGGED_IN = False
STOCK_INFO: Dict[str, dict] = {}
DEFAULT_ALLOWED_BOARDS = ["sh", "sz"]

# 不同板块涨跌停幅度。上市前 5 日、ST、退市整理等特殊情形仍需更细数据源。
LIMIT_PCT = {"cyb": 0.20, "kcb": 0.20, "bj": 0.30, "sz": 0.10, "sh": 0.10}


def _ensure_login():
    """确保 baostock 已登录"""
    global _BAOSTOCK_LOGGED_IN
    if not _BAOSTOCK_LOGGED_IN:
        bs.login()
        _BAOSTOCK_LOGGED_IN = True


def _logout():
    """仅在本进程实际登录过 baostock 时退出。"""
    global _BAOSTOCK_LOGGED_IN
    if _BAOSTOCK_LOGGED_IN:
        bs.logout()
        _BAOSTOCK_LOGGED_IN = False


def normalize_code(code: str) -> str:
    """把 300750.SZ / sz300750 / sz.300750 等写法统一为 baostock 代码。"""
    code = str(code).strip().lower()
    if "." in code:
        left, right = code.split(".", 1)
        if left in {"sh", "sz", "bj"}:
            return f"{left}.{right}"
        if right in {"sh", "sz", "bj"}:
            return f"{right}.{left}"
    digits = "".join(ch for ch in code if ch.isdigit())
    if len(digits) != 6:
        return code
    if digits.startswith(("60", "68", "90")):
        return f"sh.{digits}"
    if digits.startswith(("00", "30", "20")):
        return f"sz.{digits}"
    if digits.startswith(("43", "83", "87", "88")):
        return f"bj.{digits}"
    return code


def _safe_text(value, default: str = "") -> str:
    if value is None or pd.isna(value):
        return default
    return str(value).strip()


def infer_board(code: str) -> str:
    code = normalize_code(code)
    market, symbol = code.split(".", 1) if "." in code else ("", code)
    if market == "sz" and symbol.startswith("30"):
        return "cyb"
    if market == "sh" and symbol.startswith("68"):
        return "kcb"
    if market == "bj":
        return "bj"
    if market == "sz":
        return "sz"
    return "sh"


def register_stock_info(stock_infos: List[dict]):
    """登记股票元信息，供回测引擎查询名称、行业、板块。"""
    for info in stock_infos:
        code = normalize_code(info["code"])
        STOCK_INFO[code] = {
            "code": code,
            "name": _safe_text(info.get("name"), code) or code,
            "sector": _safe_text(info.get("sector")),
            "board": info.get("board") or infer_board(code),
        }


def get_stock_info(code: str) -> dict:
    code = normalize_code(code)
    if code not in STOCK_INFO:
        STOCK_INFO[code] = {
            "code": code,
            "name": code,
            "sector": "",
            "board": infer_board(code),
        }
    return STOCK_INFO[code]


def _valid_a_share_code(code: str) -> bool:
    code = normalize_code(code)
    if "." not in code:
        return False
    market, symbol = code.split(".", 1)
    if market == "sh":
        return symbol.startswith(("60", "68"))
    if market == "sz":
        return symbol.startswith(("00", "30"))
    if market == "bj":
        return symbol.startswith(("43", "83", "87", "88"))
    return False


def _spread_select(stock_infos: List[dict], max_count: Optional[int]) -> List[dict]:
    if not max_count or len(stock_infos) <= max_count:
        return stock_infos
    indexes = np.linspace(0, len(stock_infos) - 1, max_count, dtype=int)
    return [stock_infos[i] for i in sorted(set(indexes))]


def load_universe_from_csv(path: str) -> List[dict]:
    """从 CSV 读取股票宇宙。至少需要 code 列，可选 name/sector/board。"""
    df = pd.read_csv(path)
    if "code" not in df.columns:
        raise ValueError(f"股票池文件缺少 code 列: {path}")

    stock_infos = []
    for _, row in df.iterrows():
        code = normalize_code(row["code"])
        if not _valid_a_share_code(code):
            continue
        board = infer_board(code)
        stock_infos.append({
            "code": code,
            "name": _safe_text(row.get("name"), code) or code,
            "sector": _safe_text(row.get("sector")),
            "board": board,
        })
    return stock_infos


def query_market_universe(end_date: str, exclude_st: bool = True) -> List[dict]:
    """用 baostock 获取全市场股票列表。"""
    _ensure_login()
    rs = bs.query_all_stock(day=end_date)
    if rs.error_code != "0":
        raise RuntimeError(f"查询全市场股票失败: {rs.error_msg}")

    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    df = pd.DataFrame(rows, columns=rs.fields)
    if df.empty:
        return []

    name_col = "code_name" if "code_name" in df.columns else "name"
    stock_infos = []
    for _, row in df.iterrows():
        code = normalize_code(row["code"])
        name = _safe_text(row.get(name_col), code) or code
        trade_status = str(row.get("tradeStatus", "1"))
        if trade_status not in {"1", "1.0"}:
            continue
        if not _valid_a_share_code(code):
            continue
        if exclude_st and ("ST" in str(name).upper() or "退" in str(name)):
            continue
        stock_infos.append({
            "code": code,
            "name": name,
            "sector": "",
            "board": infer_board(code),
        })
    return sorted(stock_infos, key=lambda x: x["code"])


def build_universe(start_date: str, end_date: str, config: dict = None) -> List[dict]:
    """构建股票宇宙。优先 CSV；否则动态查询全市场。"""
    config = config or {}
    allowed_boards = set(config.get("allowed_boards", DEFAULT_ALLOWED_BOARDS))
    universe_csv = config.get("universe_csv")
    if universe_csv:
        stock_infos = load_universe_from_csv(universe_csv)
    else:
        stock_infos = query_market_universe(end_date, config.get("exclude_st", True))

    stock_infos = [
        info for info in stock_infos
        if info.get("board", infer_board(info["code"])) in allowed_boards
    ]
    stock_infos = _spread_select(stock_infos, config.get("max_universe_size"))
    register_stock_info(stock_infos)
    return stock_infos


def get_cache_file(code: str, start_date: str, end_date: str) -> str:
    return os.path.join(CACHE_DIR, f"{code.replace('.', '_')}_daily_{start_date}_{end_date}.parquet")


def load_one_stock(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取单只股票日线数据（后复权）。
    返回 DataFrame，索引为 date，列: open, high, low, close, volume, amount, pct_change, turnover
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = get_cache_file(code, start_date, end_date)

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


def load_stock_pool(start_date: str, end_date: str, config: dict = None) -> dict:
    """加载股票宇宙的日线数据。返回 {code: DataFrame}"""
    config = config or {}
    stock_infos = build_universe(start_date, end_date, config)
    max_new_downloads = config.get("max_new_downloads_per_run", 30)
    download_delay = config.get("download_delay", 0.1)
    min_history_days = config.get("min_history_days", 120)

    data = {}
    cached_count = 0
    downloaded_count = 0
    skipped_count = 0
    print(f"加载股票宇宙 {len(stock_infos)} 只股票 (日线, baostock)...", flush=True)
    if max_new_downloads is not None:
        print(f"  本轮最多新下载 {max_new_downloads} 只；已有缓存会直接读取。", flush=True)
    for i, info in enumerate(stock_infos):
        code = info["code"]
        name = info["name"]
        cache_file = get_cache_file(code, start_date, end_date)
        has_cache = os.path.exists(cache_file)
        if not has_cache and max_new_downloads is not None and downloaded_count >= max_new_downloads:
            skipped_count += 1
            print(
                f"  [{i+1}/{len(stock_infos)}] {code} {name}... 跳过（达到本轮新下载上限）",
                flush=True,
            )
            continue

        source = "缓存" if has_cache else "下载"
        print(f"  [{i+1}/{len(stock_infos)}] {code} {name} ({source})...", end=" ", flush=True)
        df = load_one_stock(code, start_date, end_date)
        if has_cache:
            cached_count += 1
        elif not df.empty:
            downloaded_count += 1

        if not df.empty and len(df) >= min_history_days:
            data[code] = df
            date_range = f"{df.index[0].date()} ~ {df.index[-1].date()}"
            print(f"{len(df)} 条 ({date_range})", flush=True)
        else:
            print("无数据或历史不足", flush=True)
        if not has_cache and i < len(stock_infos) - 1:
            time.sleep(download_delay)  # 避免请求过快
    print(
        f"成功加载 {len(data)} 只股票；缓存 {cached_count}，新下载 {downloaded_count}，跳过 {skipped_count}\n",
        flush=True,
    )

    _logout()
    return data


def get_limit_price(code: str, prev_close: float) -> tuple:
    """返回 (跌停价, 涨停价)"""
    board = get_stock_info(code).get("board") or infer_board(code)
    pct = LIMIT_PCT.get(board, 0.10)
    limit_down = round(prev_close * (1 - pct), 2)
    limit_up = round(prev_close * (1 + pct), 2)
    return limit_down, limit_up
