"""
EU-China Science & Technology Cooperation Agreement — Policy Tracker
Scraper Module: Xinhua, People's Daily, EU Commission, EU delegation to China
"""

import requests
import time
import random
import re
import json
from datetime import datetime, date
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
}

SEARCH_KEYWORDS_EN = [
    "EU China science technology cooperation",
    "EU China joint research",
    "EU China researcher exchange",
    "EU China scientific cooperation agreement",
    "Horizon China cooperation",
    "EU China S&T committee",
    "EU China flagship initiative science",
]

SEARCH_KEYWORDS_ZH = [
    "中欧科技合作",
    "中欧联合研究",
    "中欧科学技术合作协定",
    "中欧研究人员交流",
    "中欧科研合作",
    "中欧地平线合作",
    "中欧旗舰计划",
]

SOURCES = {
    "xinhua_en": "https://english.news.cn",
    "xinhua_zh": "https://www.xinhuanet.com",
    "peoples_daily_en": "https://en.people.cn",
    "ec_press": "https://ec.europa.eu/commission/presscorner",
    "eu_delegation": "https://www.eeas.europa.eu/delegations/china",
}

OUTPUT_DIR = Path("/home/claude/eu_china_tracker/data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Helper utilities ───────────────────────────────────────────────────────────

def safe_get(url: str, timeout: int = 15) -> requests.Response | None:
    """HTTP GET with retry logic and polite delay."""
    for attempt in range(3):
        try:
            time.sleep(random.uniform(1.0, 2.5))
            resp = requests.get(url, headers=HEADERS, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            print(f"  [attempt {attempt+1}/3] {url} → {exc}")
            time.sleep(2 ** attempt)
    return None


def parse_date(raw: str) -> str | None:
    """Try multiple date formats; return ISO string or None."""
    raw = raw.strip()
    formats = [
        "%Y-%m-%d", "%Y/%m/%d", "%d %B %Y", "%B %d, %Y",
        "%d-%m-%Y", "%Y年%m月%d日",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    # Last-ditch: extract 4-digit year
    m = re.search(r"(20\d{2}|19\d{2})", raw)
    return m.group(1) + "-01-01" if m else None


# ── Source-specific scrapers ───────────────────────────────────────────────────

class XinhuaEnScraper:
    """Scrapes English Xinhua search results."""

    BASE = "https://english.news.cn/search/index.htm"

    def search(self, keyword: str, pages: int = 3) -> list[dict]:
        results = []
        for page in range(1, pages + 1):
            url = f"{self.BASE}?keyword={requests.utils.quote(keyword)}&curPage={page}"
            resp = safe_get(url)
            if not resp:
                continue
            soup = BeautifulSoup(resp.text, "lxml")
            for item in soup.select(".searchList li, .news-item, article"):
                title_el = item.select_one("h3, h4, .title, a[href]")
                date_el = item.select_one(".date, time, .pubtime, span")
                link_el = item.select_one("a[href]")
                if not title_el:
                    continue
                title = title_el.get_text(strip=True)
                href = link_el["href"] if link_el else ""
                if href and not href.startswith("http"):
                    href = "https://english.news.cn" + href
                date_str = date_el.get_text(strip=True) if date_el else ""
                results.append({
                    "title": title,
                    "url": href,
                    "date_raw": date_str,
                    "source": "Xinhua (EN)",
                    "language": "en",
                    "keyword": keyword,
                })
        return results


class PeoplesDailyEnScraper:
    """Scrapes People's Daily English edition search."""

    def search(self, keyword: str, pages: int = 2) -> list[dict]:
        results = []
        base = f"http://en.people.cn/search/search.do?keyword={requests.utils.quote(keyword)}"
        resp = safe_get(base)
        if not resp:
            return results
        soup = BeautifulSoup(resp.text, "lxml")
        for item in soup.select(".news-list li, .search-result li"):
            title_el = item.select_one("a")
            date_el = item.select_one(".date, .time, span")
            if not title_el:
                continue
            title = title_el.get_text(strip=True)
            href = title_el.get("href", "")
            date_str = date_el.get_text(strip=True) if date_el else ""
            results.append({
                "title": title,
                "url": href,
                "date_raw": date_str,
                "source": "People's Daily (EN)",
                "language": "en",
                "keyword": keyword,
            })
        return results


class XinhuaZhScraper:
    """Scrapes Chinese Xinhua search results."""

    def search(self, keyword: str, pages: int = 2) -> list[dict]:
        results = []
        url = f"https://so.news.cn/#search?keyword={requests.utils.quote(keyword)}&sortField=0&lang=cn"
        # Xinhua Chinese search is JS-rendered; we simulate via their API endpoint
        api = (
            "https://search.news.cn/search/uc?"
            f"keyword={requests.utils.quote(keyword)}&curPage=1&pageSize=20&lang=cn"
        )
        resp = safe_get(api)
        if not resp:
            return results
        try:
            data = resp.json()
            items = data.get("data", {}).get("content", [])
            for item in items:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "date_raw": item.get("pubTime", ""),
                    "source": "新华网 (ZH)",
                    "language": "zh",
                    "keyword": keyword,
                })
        except Exception:
            pass
        return results


class EUCommissionScraper:
    """Scrapes EU Commission press corner for China S&T items."""

    def search(self, keyword: str = "China science technology", pages: int = 2) -> list[dict]:
        results = []
        url = (
            "https://ec.europa.eu/commission/presscorner/api/documents?"
            f"keywords={requests.utils.quote(keyword)}&pageNumber=1&pageSize=20"
            "&orderby=docDate&ordertype=DESC&language=en"
        )
        resp = safe_get(url)
        if not resp:
            return results
        try:
            data = resp.json()
            for doc in data.get("documents", []):
                results.append({
                    "title": doc.get("title", ""),
                    "url": doc.get("reference", ""),
                    "date_raw": doc.get("pubDate", ""),
                    "source": "EU Commission Press",
                    "language": "en",
                    "keyword": keyword,
                })
        except Exception:
            pass
        return results


# ── Full article text fetcher ──────────────────────────────────────────────────

def fetch_article_text(url: str) -> str:
    """Fetch and extract main text from an article URL."""
    if not url or not url.startswith("http"):
        return ""
    resp = safe_get(url)
    if not resp:
        return ""
    soup = BeautifulSoup(resp.text, "lxml")
    # Remove nav/footer/script
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    # Try common article containers
    for sel in ["article", ".article-body", "#article-content", ".content", "main"]:
        el = soup.select_one(sel)
        if el:
            return el.get_text(separator=" ", strip=True)[:3000]
    return soup.get_text(separator=" ", strip=True)[:3000]


# ── Orchestrator ───────────────────────────────────────────────────────────────

def run_scraper(max_per_source: int = 30) -> pd.DataFrame:
    """Run all scrapers and return combined raw dataframe."""
    all_records: list[dict] = []

    scrapers_en = [
        (XinhuaEnScraper(), SEARCH_KEYWORDS_EN[:3]),
        (PeoplesDailyEnScraper(), SEARCH_KEYWORDS_EN[:2]),
        (EUCommissionScraper(), ["China science technology", "EU China research"]),
    ]
    scrapers_zh = [
        (XinhuaZhScraper(), SEARCH_KEYWORDS_ZH[:3]),
    ]

    for scraper, keywords in scrapers_en + scrapers_zh:
        for kw in keywords:
            print(f"  Scraping: {scraper.__class__.__name__} → '{kw}'")
            try:
                records = scraper.search(kw)
                all_records.extend(records[:max_per_source])
            except Exception as exc:
                print(f"    Error: {exc}")

    df = pd.DataFrame(all_records)
    if df.empty:
        print("  No live records fetched (expected in sandboxed env) — using seed data.")
        return df

    # Parse dates
    df["date"] = df["date_raw"].apply(parse_date)

    # Fetch article text (throttled)
    print(f"  Fetching article texts for {min(len(df), 50)} records…")
    df["text"] = ""
    for i, row in df.head(50).iterrows():
        df.at[i, "text"] = fetch_article_text(row["url"])

    # Deduplicate by title
    df = df.drop_duplicates(subset=["title"]).reset_index(drop=True)

    path = OUTPUT_DIR / "raw_scraped.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  Saved {len(df)} records → {path}")
    return df


if __name__ == "__main__":
    df = run_scraper()
    print(df.head())
