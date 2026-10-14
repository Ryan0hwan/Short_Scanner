"""
숏 후보 스캐너 (이평선 이탈 → 되돌림 → 저항 확인)
- S&P 500 + 내 관심종목에서
- [추세 전환형] 20일선·50일선을 이탈했다가 아래에서 다시 올라와 그 선에 막히고 밀린 종목
- [하락 지속형] 역배열 하락 추세에서 반등이 20일선·50일선에 막히고 밀린 종목을 찾아
- 손절가(되돌림 고가)·목표가(이탈 후 최저가)·손익비와 함께 디스코드로 보냅니다.
"""
import datetime as dt
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

# ───────────── 설정 (여기만 고치면 됩니다) ─────────────
GROUPS = os.environ.get("SCAN_GROUPS", "sp500,watchlist").split(",")
MAS = [int(x) for x in os.environ.get("MAS", "20,50").split(",")]   # 저항선으로 볼 SMA

BRK_MIN, BRK_MAX = int(os.environ.get("BRK_MIN", "3")), int(os.environ.get("BRK_MAX", "20"))  # ① 이탈이 오늘 기준 3~20거래일 전
ABOVE_BEFORE = int(os.environ.get("ABOVE_BEFORE", "10"))        # ① 이탈 전 최소 이 기간 연속으로 선 위에 있었어야 함
TOUCH_TOL_PCT = float(os.environ.get("TOUCH_TOL_PCT", "0.5"))   # ② 오늘 고가가 선의 ±0.5% 안
CLOSE_POS_MAX = float(os.environ.get("CLOSE_POS_MAX", "0.5"))   # ③ 종가 위치가 당일 범위 하단 50% 이내

REQUIRE_SLOPE = os.environ.get("REQUIRE_SLOPE", "1") == "1"     # ④ 선이 5거래일 전보다 낮음 (하락 중)
SLOPE_DAYS = int(os.environ.get("SLOPE_DAYS", "5"))
REQUIRE_BRK_VOL = os.environ.get("REQUIRE_BRK_VOL", "1") == "1" # ⑤ 이탈일 거래량 ≥ 50일 평균 × 1.5
BRK_VOL_X = float(os.environ.get("BRK_VOL_X", "1.5"))
REQUIRE_WEAK_BOUNCE = os.environ.get("REQUIRE_WEAK_BOUNCE", "1") == "1"  # ⑥ 되돌림 구간 평균 거래량 < 이탈일 거래량
REQUIRE_WEAK_RS = os.environ.get("REQUIRE_WEAK_RS", "1") == "1" # ⑦ RS < 50 또는 3개월 수익률 < SPY
RS_MAX = float(os.environ.get("RS_MAX", "50"))

MIN_PRICE = float(os.environ.get("MIN_PRICE", "10"))
MIN_DOLLAR_VOL = float(os.environ.get("MIN_DOLLAR_VOL", "20000000"))  # 50일 평균 거래대금 $20M 이상
DOWN_ENABLE = os.environ.get("DOWN_ENABLE", "1") == "1"         # 하락 지속형 목록 켜기
DOWN_TOP = int(os.environ.get("DOWN_TOP", "10"))                # 하락 지속형은 손익비 상위 이 개수만
SWING_DAYS = int(os.environ.get("SWING_DAYS", "20"))            # 하락 지속형: 반등 시작점(최근 저점)을 찾는 기간
LEADER_DAYS = int(os.environ.get("LEADER_DAYS", "60"))          # 최근 이 기간 안에 추세 템플릿(가격 조건)을 만족했으면 LDR 표시

WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL", "")
FORCE = os.environ.get("FORCE", "0") == "1"                     # 주말에도 실행 (최근 거래일 기준)
# ──────────────────────────────────────────────────────

NY = ZoneInfo("America/New_York")
STATE = Path(__file__).parent / "state"
STATE.mkdir(exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (short-scanner)"}
GROUP_LABEL = {"sp500": "S&P 500", "watchlist": "Watchlist"}


# ───────────── 종목 목록 ─────────────
def _fetch_sp500() -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url, headers=UA, timeout=30).text
    t = pd.read_html(io.StringIO(html))[0]
    return pd.DataFrame({"ticker": t["Symbol"], "name": t["Security"], "industry": t["GICS Sub-Industry"]})


def _load_watchlist() -> pd.DataFrame:
    p = Path(__file__).parent / "watchlist.txt"
    rows = []
    if p.exists():
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.split("#")[0].strip()
            if not ln:
                continue
            tk, _, name = ln.partition(",")
            tk = re.sub(r"\.([A-Z])$", r"-\1", tk.strip().upper())  # MOG.A → MOG-A
            rows.append({"ticker": tk, "name": name.strip(), "industry": "Watchlist"})
    return pd.DataFrame(rows, columns=["ticker", "name", "industry"]).drop_duplicates("ticker")


def load_universe(group: str) -> pd.DataFrame:
    if group == "watchlist":
        return _load_watchlist()
    cache = STATE / f"universe_{group}.csv"
    fresh = cache.exists() and (time.time() - cache.stat().st_mtime) < 7 * 86400
    if not fresh:
        try:
            df = {"sp500": _fetch_sp500}[group]()
            if len(df) < 50:
                raise ValueError("종목 수가 비정상적으로 적음")
            df.to_csv(cache, index=False)
        except Exception as e:
            print(f"[경고] {group} 목록 갱신 실패: {e}")
    if not cache.exists():
        return pd.DataFrame(columns=["ticker", "name", "industry"])
    df = pd.read_csv(cache).fillna("")
    df["ticker"] = df["ticker"].astype(str).str.strip().str.replace(".", "-", regex=False)
    df = df[df["ticker"].str.fullmatch(r"[A-Z\-]{1,6}")]
    return df.drop_duplicates("ticker")


# ───────────── 시세 ─────────────
def download(tickers: list[str]):
    import yfinance as yf
    for i in range(0, len(tickers), 200):
        chunk = tickers[i:i + 200]
        try:
            data = yf.download(chunk, period="2y", interval="1d", group_by="ticker",
                               auto_adjust=True, threads=True, progress=False)
        except Exception as e:
            print(f"[경고] 시세 다운로드 실패({chunk[0]}…): {e}")
            continue
        if not isinstance(data.columns, pd.MultiIndex):
            data = pd.concat({chunk[0]: data}, axis=1)
        have = set(data.columns.get_level_values(0))
        for t in chunk:
            if t in have:
                yield t, data[t]
        time.sleep(1)


# ───────────── 계산 ─────────────
def rs_score(c: pd.Series) -> float:
    """IBD 방식 가중 수익률 (3개월 40% + 6·9·12개월 각 20%). 기간이 모자라면 있는 기간만으로"""
    num = den = 0.0
    for n, w in ((63, 2), (126, 1), (189, 1), (252, 1)):
        if len(c) > n:
            num += (c.iloc[-1] / c.iloc[-1 - n] - 1) * w
            den += w
    return num / den if den else np.nan


def was_leader(c: pd.Series, h: pd.Series, l: pd.Series) -> bool:
    """최근 LEADER_DAYS일 안에 추세 템플릿 가격 조건(1~7)을 모두 만족한 날이 있었는지"""
    if len(c) < 260:
        return False
    s50, s150, s200 = c.rolling(50).mean(), c.rolling(150).mean(), c.rolling(200).mean()
    hi, lo = h.rolling(252, min_periods=200).max(), l.rolling(252, min_periods=200).min()
    ok = ((c > s150) & (c > s200) & (s150 > s200) & (s200 > s200.shift(21)) & (s50 > s150) & (s50 > s200)
          & (c > s50) & (c >= lo * 1.3) & (c >= hi * 0.75))
    return bool(ok.iloc[-LEADER_DAYS:].any())


def find_setups(df: pd.DataFrame, today: dt.date | None) -> tuple[list[dict], float, float] | None:
    """오늘 조건을 통과한 (이평선별) 셋업 목록, RS 점수, 3개월 수익률"""
    df = df.dropna(subset=["Open", "Close", "High", "Low", "Volume"])
    if len(df) < 120 or (today is not None and df.index[-1].date() != today):
        return None
    o, c, h, l, v = df["Open"], df["Close"], df["High"], df["Low"], df["Volume"]
    last = float(c.iloc[-1])
    if last < MIN_PRICE or (c * v).tail(50).mean() < MIN_DOLLAR_VOL:
        return None
    vavg = v.rolling(50).mean().shift(1)            # 당일 제외 50일 평균
    rng = h.iloc[-1] - l.iloc[-1]
    close_pos = (last - l.iloc[-1]) / rng if rng > 0 else 1.0
    tol = TOUCH_TOL_PCT / 100
    N = len(df)
    out, down = [], []
    s50, s200 = c.rolling(50).mean(), c.rolling(200).mean()
    # 역배열: 주가 < 50일선 < 200일선, 200일선 하락 중
    bearish = (len(c) >= 222 and last < s50.iloc[-1] < s200.iloc[-1] and s200.iloc[-1] < s200.iloc[-22])
    for n in MAS:
        ma = c.rolling(n).mean()
        m = float(ma.iloc[-1])
        if np.isnan(m):
            continue
        # ③ 저항 확인: 종가가 선 아래, 당일 범위 하단 절반에서 마감
        if not (last < m and close_pos <= CLOSE_POS_MAX):
            continue
        # ② 되돌림: 어제 종가는 선 아래, 오늘 고가가 선의 ±tol 안
        if not (c.iloc[-2] < ma.iloc[-2] and m * (1 - tol) <= h.iloc[-1] <= m * (1 + tol)):
            continue
        # ④ 선이 하락 중 (두 유형 공통)
        if REQUIRE_SLOPE and not m < ma.iloc[-1 - SLOPE_DAYS]:
            continue
        # [하락 지속형] 역배열 + 반등 거래량이 평균보다 적음. 이탈 전 선 위 체류 조건 없음
        if DOWN_ENABLE and bearish:
            sw = c.iloc[-SWING_DAYS:]
            k = len(c) - SWING_DAYS + int(np.argmin(sw.to_numpy()))   # 최근 저점(종가 기준)
            if k < len(c) - 1:
                rally_x = float(v.iloc[k + 1:].mean() / vavg.iloc[-1]) if vavg.iloc[-1] else float("nan")
                if rally_x < 1:
                    stop = float(h.iloc[-1])
                    low_k = float(l.iloc[-SWING_DAYS:-1].min())
                    target = low_k if low_k < last else None
                    rr = (last - target) / (stop - last) if target is not None and stop > last else None
                    down.append({"ma_n": n, "ma": m, "rally_x": rally_x,
                                 "below200": float((last / s200.iloc[-1] - 1) * 100),
                                 "last": last, "stop": stop, "target": target, "rr": rr})
        # ① 이탈: 가장 최근에 종가가 선 위 → 아래로 바뀐 날 b
        below = (c < ma).to_numpy()
        b = None
        for j in range(N - 2, 0, -1):
            if below[j] and not below[j - 1]:
                b = j
                break
        if b is None:
            continue
        ago = N - 1 - b
        if not (BRK_MIN <= ago <= BRK_MAX):
            continue
        if not below[b:N].all():                    # 이탈 후 오늘까지 계속 선 아래
            continue
        if b - ABOVE_BEFORE < 0 or below[b - ABOVE_BEFORE:b].any():   # 이탈 전에는 선 위에 있었음
            continue
        # ⑤ 이탈일 거래량
        brk_x = float(v.iloc[b] / vavg.iloc[b]) if vavg.iloc[b] else float("nan")
        if REQUIRE_BRK_VOL and not brk_x >= BRK_VOL_X:
            continue
        # ⑥ 되돌림 구간 거래량이 이탈일보다 적음
        bounce_x = float(v.iloc[b + 1:].mean() / v.iloc[b]) if v.iloc[b] else float("nan")
        if REQUIRE_WEAK_BOUNCE and not bounce_x < 1:
            continue
        stop = float(h.iloc[-1])
        low_since = float(l.iloc[b:N - 1].min())        # 이탈 후 어제까지 최저가
        target = low_since if low_since < last else None
        rr = (last - target) / (stop - last) if target is not None and stop > last else None
        out.append({"ma_n": n, "ma": m, "ago": ago, "brk_x": brk_x, "bounce_x": bounce_x,
                    "last": last, "stop": stop, "target": target, "rr": rr,
                    "ldr": was_leader(c, h, l)})
    r3m = float(last / c.iloc[-64] - 1) if len(c) > 63 else np.nan
    # 같은 종목·같은 선이 추세 전환형에 잡혔으면 하락 지속형에서는 뺌
    done = {x["ma_n"] for x in out}
    down = [x for x in down if x["ma_n"] not in done]
    return (out, down), rs_score(c), r3m


# ───────────── 디스코드 ─────────────
def company_name(ticker: str) -> str:
    cache_file = STATE / "names.json"
    try:
        cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    except Exception:
        cache = {}
    if ticker in cache:
        return cache[ticker]
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        name = info.get("shortName") or info.get("longName") or ""
    except Exception:
        name = ""
    cache[ticker] = name
    cache_file.write_text(json.dumps(cache))
    return name


def post(content: str):
    if not WEBHOOK:
        print(content)
        return
    for _ in range(5):
        r = requests.post(WEBHOOK, json={"content": content}, timeout=30)
        if r.status_code == 429:
            time.sleep(float(r.json().get("retry_after", 2)) + 0.5)
            continue
        r.raise_for_status()
        time.sleep(0.6)
        return


def send_table(title: str, rows: list[dict]):
    head = (f"{'TICKER':<7}{'NAME':<12}{'RS':>3}{'LDR':>4}{'BRK':>4}{'VOLx':>5}{'BNC':>5}"
            f"{'MA':>8}{'LAST':>8}{'STOP':>8}{'TGT':>8}{'R:R':>5}")
    def line(r):
        tgt = f"{r['target']:>8.2f}" if r["target"] is not None else f"{'-':>8}"
        rr = f"{r['rr']:>5.1f}" if r["rr"] is not None else f"{'-':>5}"
        return (f"{r['t']:<7}{r['name'][:10]:<12}{r['rs']:>3}{('Y' if r['ldr'] else '-'):>4}{r['ago']:>4}"
                f"{r['brk_x']:>5.1f}{r['bounce_x']:>5.2f}{r['ma']:>8.2f}{r['last']:>8.2f}{r['stop']:>8.2f}{tgt}{rr}")
    lines = [line(r) for r in rows]
    first, part = True, 1
    while lines:
        block, size = [], 0
        while lines and size + len(lines[0]) + 1 < 1750:
            size += len(lines[0]) + 1
            block.append(lines.pop(0))
        header = f"**{title}**\n" if first else f"**{title} ({part})**\n"
        post(f"{header}```\n{head}\n" + "\n".join(block) + "\n```")
        first, part = False, part + 1


def send_down_table(title: str, rows: list[dict]):
    head = (f"{'TICKER':<7}{'NAME':<12}{'RS':>3}{'RVOL':>6}{'200D%':>7}"
            f"{'MA':>8}{'LAST':>8}{'STOP':>8}{'TGT':>8}{'R:R':>5}")
    def line(r):
        tgt = f"{r['target']:>8.2f}" if r["target"] is not None else f"{'-':>8}"
        rr = f"{r['rr']:>5.1f}" if r["rr"] is not None else f"{'-':>5}"
        return (f"{r['t']:<7}{r['name'][:10]:<12}{r['rs']:>3}{r['rally_x']:>6.2f}{r['below200']:>+7.1f}"
                f"{r['ma']:>8.2f}{r['last']:>8.2f}{r['stop']:>8.2f}{tgt}{rr}")
    post(f"**{title}**\n```\n{head}\n" + "\n".join(line(r) for r in rows) + "\n```")


# ───────────── 실행 ─────────────
def main():
    now = dt.datetime.now(NY)
    if now.weekday() >= 5 and not FORCE:
        print("주말. 종료.")
        return
    today = now.date()

    universes = {g.strip(): load_universe(g.strip()) for g in GROUPS if g.strip()}
    meta = {}
    for g, u in universes.items():
        for row in u.itertuples():
            meta.setdefault(row.ticker, row)
    print(f"스캔 대상 {len(meta)}종목 " + ", ".join(f"{g}={len(u)}" for g, u in universes.items()))
    if not meta:
        post("⚠️ 숏 스캐너: 종목 목록을 불러오지 못했습니다. Actions 로그를 확인하세요.")
        return

    # ⑧ 시장 환경 + SPY 3개월 수익률
    mkt, spy3m = {}, np.nan
    for t, d in download(["QQQ", "SPY"]):
        c = d["Close"].dropna()
        if len(c) > 63:
            mkt[t] = {"above50": bool(c.iloc[-1] > c.rolling(50).mean().iloc[-1]),
                      "above20": bool(c.iloc[-1] > c.rolling(20).mean().iloc[-1])}
            if t == "SPY":
                spy3m = float(c.iloc[-1] / c.iloc[-64] - 1)

    cands, scores, scanned = [], {}, 0
    for ticker, df in download(list(meta)):
        scanned += 1
        r = find_setups(df, None if FORCE else today)
        if r is None:
            continue
        (setups, downs), score, r3m = r
        scores[ticker] = score
        for s in setups:
            cands.append({"t": ticker, "r3m": r3m, "kind": "turn", **s})
        for s in downs:
            cands.append({"t": ticker, "r3m": r3m, "kind": "down", **s})

    # ⑦ RS 백분위는 전체 종목 기준으로 계산
    rs = (pd.Series(scores).rank(pct=True) * 98 + 1).round()
    rows = []
    for cnd in cands:
        cnd["rs"] = int(rs.get(cnd["t"], 50))
        weak = cnd["rs"] < RS_MAX or (not np.isnan(spy3m) and cnd["r3m"] < spy3m)
        if REQUIRE_WEAK_RS and not weak:
            continue
        m = meta[cnd["t"]]
        cnd["name"] = str(m.name) or company_name(cnd["t"])
        rows.append(cnd)

    def mk(t):
        s = mkt.get(t)
        return "확인 불가" if not s else ("50일선 위" if s["above50"] else "**50일선 아래**")
    turn = [r for r in rows if r["kind"] == "turn"]
    down = [r for r in rows if r["kind"] == "down"]
    post(f"**{now:%b} {now.day} | 숏 후보**\n"
         f"시장: QQQ {mk('QQQ')} · SPY {mk('SPY')}  (지수가 50일선 아래일 때 숏 성공률이 높음)\n"
         f"추세 전환형 {len(turn)}건" + (f" · 하락 지속형 {len(down)}건" if DOWN_ENABLE else "") + f" · 확인 {scanned}종목")
    for n in MAS:
        sel = sorted([r for r in turn if r["ma_n"] == n],
                     key=lambda r: (-(r["rr"] or -1), r["rs"]))
        if sel:
            send_table(f"[추세 전환형] {n}일선 저항", sel)
        else:
            post(f"[추세 전환형] {n}일선 저항: 오늘은 해당 종목이 없습니다.")
    if DOWN_ENABLE:
        sel = sorted(down, key=lambda r: (-(r["rr"] or -1), r["rs"]))
        if sel:
            extra = f" (전체 {len(sel)}건 중 손익비 상위 {DOWN_TOP})" if len(sel) > DOWN_TOP else ""
            for n in MAS:
                part = [r for r in sel[:DOWN_TOP] if r["ma_n"] == n]
                if part:
                    send_down_table(f"[하락 지속형] {n}일선 저항{extra}", part)
        else:
            post("[하락 지속형]: 오늘은 해당 종목이 없습니다.")
    if scanned < len(meta) * 0.5:
        post(f"⚠️ 숏 스캐너: 시세를 {scanned}/{len(meta)}종목만 받았습니다. 일시적 오류일 수 있습니다.")
    print(f"완료: 확인 {scanned}종목, 후보 {len(rows)}건")


if __name__ == "__main__":
    sys.exit(main())
