"""네이버 증권 데이터 스크래핑.

Ported from shwi-project/stocklens-mcp (MIT License).
https://github.com/shwi-project/stocklens-mcp/blob/main/stock_mcp_server/naver.py

본 프로젝트는 stocklens-mcp의 22개 함수 중 다음만 사용:
- 테마/섹터: list_themes, get_theme_stocks, list_sectors, get_sector_stocks
- 랭킹: get_volume_ranking, get_change_ranking, get_market_cap_ranking, get_market_index
- 컨센서스/리포트: get_consensus, get_reports, get_report_detail
- 수급: get_investor_flow
- 공시: get_disclosure_list
"""

import asyncio
import re

from bs4 import BeautifulSoup

from sources.cache import cached
from sources.http_client import fetch

BASE_URL = "https://finance.naver.com"
WISEREPORT_CONSENSUS_URL = "https://navercomp.wisereport.co.kr/v2/company/c1010001.aspx"
REPORT_LIST_URL = f"{BASE_URL}/research/company_list.naver"
REPORT_READ_URL = f"{BASE_URL}/research/company_read.naver"
DISCLOSURE_URL = f"{BASE_URL}/item/news_notice.naver"


def _parse_int(text: str, default: int = 0) -> int:
    if text is None:
        return default
    cleaned = text.strip().replace(",", "").replace("+", "")
    if not cleaned or cleaned == "-":
        return default
    try:
        return int(cleaned)
    except ValueError:
        return default


def _market_to_sosok(market: str) -> str | None:
    m = market.upper()
    if m == "KOSPI":
        return "0"
    if m == "KOSDAQ":
        return "1"
    return None


# ─────────────────────────────────────────────────────────────
# 투자자별 수급 (E)
# ─────────────────────────────────────────────────────────────

@cached(ttl_market=300, ttl_closed=7200)
async def get_investor_flow(code: str, days: int = 20) -> list[dict]:
    """기관/외국인 순매매 (네이버 frgn.naver, 두 번째 type2 테이블)."""
    url = f"{BASE_URL}/item/frgn.naver"
    results: list[dict] = []
    page = 1

    while len(results) < days:
        params = {"code": code, "page": page}
        resp = await fetch(url, params=params)
        soup = BeautifulSoup(resp.text, "lxml")

        tables = soup.select("table.type2")
        if len(tables) < 2:
            break

        rows = tables[1].select("tr")
        found_in_page = 0
        for row in rows:
            cols = row.select("td")
            if len(cols) != 9:
                continue
            date_text = cols[0].text.strip()
            if not date_text or "." not in date_text:
                continue
            try:
                results.append({
                    "date": date_text,
                    "close": _parse_int(cols[1].text),
                    "change": _parse_int(cols[2].text.split()[-1] if cols[2].text.strip() else "0"),
                    "volume": _parse_int(cols[4].text),
                    "institutional": _parse_int(cols[5].text),
                    "foreign": _parse_int(cols[6].text),
                })
                found_in_page += 1
            except (ValueError, IndexError):
                continue

        if found_in_page == 0:
            break
        if len(results) >= days:
            break
        page += 1
        if page > 10:
            break

    return results[:days]


# ─────────────────────────────────────────────────────────────
# 테마/섹터 (A)
# ─────────────────────────────────────────────────────────────

@cached(ttl_market=300, ttl_closed=3600)
async def list_themes(page: int = 1) -> list[dict]:
    """네이버 테마 목록 (페이지당 40개, 총 7페이지)."""
    url = f"{BASE_URL}/sise/theme.naver"
    resp = await fetch(url, params={"page": page})
    soup = BeautifulSoup(resp.text, "lxml")

    table = soup.select_one("table.type_1.theme")
    if not table:
        return []

    results = []
    for row in table.select("tr"):
        cells = row.select("td")
        if len(cells) != 8:
            continue
        name_tag = cells[0].find("a")
        if not name_tag:
            continue
        href = name_tag.get("href", "")
        m = re.search(r"no=(\d+)", href)
        if not m:
            continue

        leaders = []
        for leader_cell in cells[6:8]:
            leader_a = leader_cell.find("a")
            if leader_a:
                code_match = re.search(r"code=([A-Za-z0-9]{6})", leader_a.get("href", ""))
                leaders.append({
                    "name": leader_a.text.strip(),
                    "code": code_match.group(1) if code_match else "",
                })

        results.append({
            "name": name_tag.text.strip(),
            "theme_id": m.group(1),
            "change_rate": cells[1].text.strip(),
            "recent_3d_rate": cells[2].text.strip(),
            "up_count": _parse_int(cells[3].text),
            "flat_count": _parse_int(cells[4].text),
            "down_count": _parse_int(cells[5].text),
            "leaders": leaders,
        })

    return results


@cached(ttl_market=300, ttl_closed=3600)
async def list_themes_all() -> list[dict]:
    """모든 페이지(1~7)를 병렬로 가져와 합친다."""
    pages = await asyncio.gather(*[list_themes(p) for p in range(1, 8)])
    out: list[dict] = []
    for p in pages:
        out.extend(p)
    return out


@cached(ttl_market=300, ttl_closed=3600)
async def get_theme_stocks(theme_id: str, count: int = 30) -> list[dict]:
    """특정 theme_id의 종목 리스트."""
    resp = await fetch(
        f"{BASE_URL}/sise/sise_group_detail.naver",
        params={"type": "theme", "no": theme_id},
    )
    soup = BeautifulSoup(resp.text, "lxml")

    tables = soup.select("table.type_5")
    stocks: list[dict] = []
    if tables:
        for row in tables[0].select("tr"):
            cells = row.select("td")
            if len(cells) < 11:
                continue
            name_a = cells[0].find("a")
            if not name_a:
                continue
            code_match = re.search(r"code=([A-Za-z0-9]{6})", name_a.get("href", ""))
            if not code_match:
                continue

            reason_tag = cells[1].select_one("p.info_txt")
            reason = reason_tag.text.strip() if reason_tag else ""
            if len(reason) > 80:
                reason = reason[:78] + ".."

            stocks.append({
                "code": code_match.group(1),
                "name": name_a.text.strip().rstrip("*").strip(),
                "price": _parse_int(cells[2].text),
                "change_rate": cells[4].text.strip(),
                "volume": _parse_int(cells[7].text),
                "reason": reason,
            })
            if len(stocks) >= count:
                break
    return stocks


@cached(ttl_market=300, ttl_closed=3600)
async def list_sectors() -> list[dict]:
    """네이버 업종 목록 (1페이지에 약 79개)."""
    url = f"{BASE_URL}/sise/sise_group.naver"
    resp = await fetch(url, params={"type": "upjong"})
    soup = BeautifulSoup(resp.text, "lxml")

    table = soup.select_one("table.type_1")
    if not table:
        return []

    results: list[dict] = []
    for row in table.select("tr"):
        cells = row.select("td")
        if len(cells) < 6:
            continue
        name_tag = cells[0].find("a")
        if not name_tag:
            continue
        href = name_tag.get("href", "")
        m = re.search(r"no=(\d+)", href)
        if not m:
            continue
        results.append({
            "name": name_tag.text.strip(),
            "sector_id": m.group(1),
            "change_rate": cells[1].text.strip(),
            "total_count": _parse_int(cells[2].text),
            "up_count": _parse_int(cells[3].text),
            "flat_count": _parse_int(cells[4].text),
            "down_count": _parse_int(cells[5].text),
        })
    return results


@cached(ttl_market=300, ttl_closed=3600)
async def get_sector_stocks(sector_id: str, count: int = 30) -> list[dict]:
    """특정 sector_id의 종목 리스트."""
    resp = await fetch(
        f"{BASE_URL}/sise/sise_group_detail.naver",
        params={"type": "upjong", "no": sector_id},
    )
    soup = BeautifulSoup(resp.text, "lxml")

    tables = soup.select("table.type_5")
    stocks: list[dict] = []
    if tables:
        for row in tables[0].select("tr"):
            cells = row.select("td")
            if len(cells) < 10:
                continue
            name_a = cells[0].find("a")
            if not name_a:
                continue
            code_match = re.search(r"code=([A-Za-z0-9]{6})", name_a.get("href", ""))
            if not code_match:
                continue
            stocks.append({
                "code": code_match.group(1),
                "name": name_a.text.strip().rstrip("*").strip(),
                "price": _parse_int(cells[1].text),
                "change_rate": cells[3].text.strip(),
                "volume": _parse_int(cells[6].text),
            })
            if len(stocks) >= count:
                break
    return stocks


# ─────────────────────────────────────────────────────────────
# 랭킹 (B)
# ─────────────────────────────────────────────────────────────

@cached(ttl_market=60, ttl_closed=3600)
async def _fetch_ranking_page(url: str, sosok: str | None, page: int = 1) -> list[dict]:
    params: dict = {"page": page}
    if sosok is not None:
        params["sosok"] = sosok
    resp = await fetch(url, params=params)
    soup = BeautifulSoup(resp.text, "lxml")

    table = soup.select_one("table.type_2")
    if not table:
        return []

    results: list[dict] = []
    for row in table.select("tr"):
        cells = row.select("td")
        if len(cells) < 12:
            continue
        rank_text = cells[0].text.strip()
        if not rank_text.isdigit():
            continue
        name_a = cells[1].find("a")
        if not name_a:
            continue
        code_match = re.search(r"code=([A-Za-z0-9]{6})", name_a.get("href", ""))
        if not code_match:
            continue
        price = _parse_int(cells[2].text)
        volume = _parse_int(cells[5].text)
        results.append({
            "rank": int(rank_text),
            "code": code_match.group(1),
            "name": name_a.text.strip(),
            "price": price,
            "change_rate": cells[4].text.strip(),
            "volume": volume,
            "trade_value_krw": price * volume,
        })
    return results


async def _fetch_ranking_multi_page(url: str, sosok: str | None, count: int) -> list[dict]:
    pages_needed = max(1, min((count + 49) // 50, 10))
    pages = await asyncio.gather(
        *[_fetch_ranking_page(url, sosok, page=p) for p in range(1, pages_needed + 1)]
    )
    merged: list[dict] = []
    for p in pages:
        merged.extend(p)
    return merged[:count]


async def get_volume_ranking(
    market: str = "ALL",
    count: int = 50,
    sort_by: str = "volume",
) -> list[dict]:
    count = min(count, 500)
    url = f"{BASE_URL}/sise/sise_quant.naver"
    sort_key = "trade_value_krw" if sort_by == "trade_value" else "volume"

    if market.upper() == "ALL":
        kospi, kosdaq = await asyncio.gather(
            _fetch_ranking_multi_page(url, "0", count),
            _fetch_ranking_multi_page(url, "1", count),
        )
        merged = sorted(kospi + kosdaq, key=lambda x: x.get(sort_key, 0), reverse=True)
        for i, item in enumerate(merged[:count], 1):
            item["rank"] = i
        return merged[:count]
    sosok = _market_to_sosok(market)
    results = await _fetch_ranking_multi_page(url, sosok, count)
    if sort_by == "trade_value":
        results = sorted(results, key=lambda x: x.get("trade_value_krw", 0), reverse=True)
        for i, item in enumerate(results[:count], 1):
            item["rank"] = i
    return results[:count]


async def get_change_ranking(
    direction: str = "up",
    market: str = "ALL",
    count: int = 50,
) -> list[dict]:
    count = min(count, 500)
    page_url = "sise_rise.naver" if direction.lower() == "up" else "sise_fall.naver"
    url = f"{BASE_URL}/sise/{page_url}"

    def parse_rate(s: str) -> float:
        try:
            return float(s.replace("%", "").replace("+", ""))
        except ValueError:
            return 0.0

    if market.upper() == "ALL":
        kospi, kosdaq = await asyncio.gather(
            _fetch_ranking_multi_page(url, "0", count),
            _fetch_ranking_multi_page(url, "1", count),
        )
        merged = kospi + kosdaq
        reverse = direction.lower() == "up"
        merged.sort(key=lambda x: parse_rate(x["change_rate"]), reverse=reverse)
        for i, item in enumerate(merged[:count], 1):
            item["rank"] = i
        return merged[:count]
    sosok = _market_to_sosok(market)
    return await _fetch_ranking_multi_page(url, sosok, count)


@cached(ttl_market=300, ttl_closed=3600)
async def _fetch_market_cap_page(sosok: str, page: int = 1) -> list[dict]:
    url = f"{BASE_URL}/sise/sise_market_sum.naver"
    resp = await fetch(url, params={"sosok": sosok, "page": page})
    soup = BeautifulSoup(resp.text, "lxml")

    table = soup.select_one("table.type_2")
    if not table:
        return []

    results: list[dict] = []
    for row in table.select("tr"):
        cells = row.select("td")
        if len(cells) < 13:
            continue
        rank_text = cells[0].text.strip()
        if not rank_text.isdigit():
            continue
        name_a = cells[1].find("a")
        if not name_a:
            continue
        code_match = re.search(r"code=([A-Za-z0-9]{6})", name_a.get("href", ""))
        if not code_match:
            continue
        results.append({
            "rank": int(rank_text),
            "code": code_match.group(1),
            "name": name_a.text.strip(),
            "price": _parse_int(cells[2].text),
            "change_rate": cells[4].text.strip(),
            "market_cap_billion": _parse_int(cells[6].text),
            "volume": _parse_int(cells[9].text),
        })
    return results


async def get_market_cap_ranking(market: str = "KOSPI", count: int = 50) -> list[dict]:
    count = min(count, 500)
    sosok = _market_to_sosok(market) or "0"
    pages_needed = max(1, min((count + 49) // 50, 10))
    pages = await asyncio.gather(
        *[_fetch_market_cap_page(sosok, page=p) for p in range(1, pages_needed + 1)]
    )
    merged: list[dict] = []
    for p in pages:
        merged.extend(p)
    return merged[:count]


# ─────────────────────────────────────────────────────────────
# 컨센서스 & 리포트 (C)
# ─────────────────────────────────────────────────────────────

@cached(ttl_market=600, ttl_closed=86400)
async def get_consensus(code: str) -> dict:
    """wisereport에서 목표주가, 투자의견, 실적 추정치 추출."""
    resp = await fetch(WISEREPORT_CONSENSUS_URL, params={"cmp_cd": code})
    html = resp.text

    import json as _json

    def _extract(var_name: str) -> dict | None:
        pattern = rf"var\s+{var_name}\s*=\s*(\{{.*?\}});"
        m = re.search(pattern, html, re.DOTALL)
        if not m:
            return None
        try:
            return _json.loads(m.group(1))
        except (_json.JSONDecodeError, ValueError):
            return None

    result: dict = {"code": code}

    chart2 = _extract("chartData2")
    if chart2 and "target_price" in chart2:
        targets = chart2["target_price"]
        valid = [t for t in targets if t.get("y") is not None]
        if valid:
            result["target_price"] = valid[-1]["y"]
        result["target_price_history"] = [
            {"date": t["x"], "price": t["y"]} for t in valid[-6:]
        ]

    chart3 = _extract("chartData3")
    if chart3:
        today = chart3.get("today", [])
        result["opinion"] = {
            item["name"]: int(item["y"]) if item.get("y") else 0
            for item in today
        }
        ago = chart3.get("a_month_ago", [])
        result["opinion_1m_ago"] = {
            item["name"]: int(item["y"]) if item.get("y") else 0
            for item in ago
        }

    res_data = _extract("res")
    if res_data and "yymm" in res_data:
        result["estimate_periods"] = res_data["yymm"]
        labels = ["매출액", "영업이익", "영업이익률"]
        estimates: dict[str, dict] = {}
        for i, row in enumerate(res_data.get("data", [])):
            if i < len(labels):
                vals: dict = {}
                for period_idx, period in enumerate(res_data["yymm"]):
                    vals[period] = row.get(str(period_idx + 1))
                estimates[labels[i]] = vals
        result["estimates"] = estimates

    return result


@cached(ttl_market=600, ttl_closed=3600)
async def get_reports(code: str, count: int = 5) -> list[dict]:
    """최근 증권사 리포트 목록."""
    resp = await fetch(REPORT_LIST_URL, params={
        "searchType": "itemCode",
        "itemCode": code,
        "page": "1",
    })
    soup = BeautifulSoup(resp.text, "lxml")

    table = soup.select_one("table.type_1")
    if not table:
        return []

    results: list[dict] = []
    for row in table.select("tr"):
        cells = row.select("td")
        if len(cells) < 5:
            continue
        title_a = cells[1].find("a")
        if not title_a:
            continue
        href = title_a.get("href", "")
        m = re.search(r"nid=(\d+)", href)
        if not m:
            continue
        results.append({
            "nid": m.group(1),
            "stock": cells[0].get_text(strip=True),
            "title": title_a.get_text(strip=True),
            "broker": cells[2].get_text(strip=True),
            "date": cells[4].get_text(strip=True),
            "views": _parse_int(cells[5].get_text(strip=True)) if len(cells) > 5 else 0,
        })
        if len(results) >= count:
            break
    return results


async def get_report_detail(nid: str) -> dict:
    """리포트 상세 (목표가, 투자의견, 본문 요약, PDF 링크)."""
    resp = await fetch(REPORT_READ_URL, params={"nid": nid})
    soup = BeautifulSoup(resp.text, "lxml")

    result: dict = {"nid": nid}

    for td in soup.select("td"):
        text = " ".join(td.get_text(strip=True).split())
        if "목표가" in text:
            price_m = re.search(r"목표가\s*([\d,]+)", text)
            if price_m:
                result["target_price"] = _parse_int(price_m.group(1))
            opinion_m = re.search(r"투자의견\s*(\w+)", text)
            if opinion_m:
                result["opinion"] = opinion_m.group(1)

    content_td = soup.select_one("td.view_cnt")
    if content_td:
        text = content_td.get_text(strip=True)
        if len(text) > 500:
            text = text[:500] + "..."
        result["summary"] = text

    for a in soup.select("a"):
        href = a.get("href", "")
        if ".pdf" in href.lower():
            result["pdf_url"] = href
            break

    return result


# ─────────────────────────────────────────────────────────────
# 공시 (F)
# ─────────────────────────────────────────────────────────────

@cached(ttl_market=300, ttl_closed=3600)
async def get_disclosure_list(code: str, page: int = 1) -> list[dict]:
    """종목의 최근 공시 목록 (DART/거래소 등)."""
    resp = await fetch(DISCLOSURE_URL, params={"code": code, "page": page})
    soup = BeautifulSoup(resp.text, "lxml")

    results: list[dict] = []
    for row in soup.select("table tr"):
        cells = row.select("td")
        if len(cells) < 3:
            continue
        title_a = cells[0].find("a")
        if not title_a:
            continue
        title = title_a.get_text(strip=True)
        if not title:
            continue
        href = title_a.get("href", "")
        if href and href.startswith("/"):
            href = BASE_URL + href
        results.append({
            "title": title,
            "source": cells[1].get_text(strip=True),
            "date": cells[2].get_text(strip=True),
            "link": href,
        })
    return results


# ─────────────────────────────────────────────────────────────
# 종목 OHLCV (D 지표 분석에서 사용. pykrx 보조용)
# ─────────────────────────────────────────────────────────────

FCHART_URL = "https://fchart.stock.naver.com/siseJson.nhn"


@cached(ttl_market=300, ttl_closed=3600)
async def get_ohlcv(code: str, timeframe: str = "day", count: int = 260) -> list[dict]:
    """네이버 fchart에서 OHLCV 가져오기."""
    params = {
        "symbol": code,
        "timeframe": timeframe,
        "count": count,
        "requestType": "0",
    }
    resp = await fetch(FCHART_URL, params=params)
    text = resp.text.strip()

    rows: list[dict] = []
    for line in text.split("\n"):
        line = line.strip().strip(",")
        if not line:
            continue
        if line.startswith("[") and "날짜" in line:
            continue
        if line == "]":
            continue
        line = line.strip("[]")
        parts = [p.strip().strip("'\"") for p in line.split(",")]
        if len(parts) >= 6:
            try:
                rows.append({
                    "date": parts[0].strip(),
                    "open": int(parts[1]),
                    "high": int(parts[2]),
                    "low": int(parts[3]),
                    "close": int(parts[4]),
                    "volume": int(parts[5]),
                })
            except (ValueError, IndexError):
                continue
    return rows
