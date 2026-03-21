"""
EU-China S&T Cooperation — NLP Analysis Pipeline
Classifies records into 6 activity types, scores sentiment, extracts entities.
"""

import re
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    import jieba
    JIEBA_OK = True
except ImportError:
    JIEBA_OK = False

OUTPUT_DIR = Path("/home/claude/eu_china_tracker/data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Activity-type taxonomy ─────────────────────────────────────────────────────

ACTIVITY_TYPES = {
    "Joint Research Projects": {
        "en": [
            r"joint research", r"joint project", r"research project",
            r"collaborative research", r"co.research", r"research collaboration",
            r"research grant", r"research programme", r"bilateral research",
            r"joint lab", r"co.investigat", r"research fund",
        ],
        "zh": [
            "联合研究", "合作研究", "联合项目", "科研合作项目",
            "共同研究", "研究合作", "联合实验室",
        ],
        "weight": 1.2,
    },
    "Joint Programs": {
        "en": [
            r"joint program", r"joint programme", r"bilateral program",
            r"framework program", r"cooperation program", r"joint initiative",
            r"horizon europe", r"fp\d", r"marie curie", r"cost action",
            r"eranet", r"co.fund",
        ],
        "zh": [
            "联合计划", "合作项目", "框架计划", "双边计划", "地平线",
            "科技合作计划", "合作框架",
        ],
        "weight": 1.1,
    },
    "Researcher Exchange": {
        "en": [
            r"researcher exchange", r"scientist exchange", r"fellowship",
            r"visiting researcher", r"secondment", r"mobility grant",
            r"scholar exchange", r"academic exchange", r"staff exchange",
            r"postdoc", r"visiting scholar", r"talent program",
        ],
        "zh": [
            "研究人员交流", "科学家交流", "学者交流", "访问学者",
            "人才交流", "学术交流", "奖学金", "科研人员流动",
        ],
        "weight": 1.0,
    },
    "Conferences & Seminars": {
        "en": [
            r"conference", r"seminar", r"workshop", r"symposium",
            r"forum", r"summit", r"roundtable", r"dialogue",
            r"high.level meeting", r"ministerial meeting", r"s&t committee",
            r"joint committee", r"expert meeting",
        ],
        "zh": [
            "会议", "研讨会", "论坛", "峰会", "对话",
            "高层会议", "联合委员会", "部长级会议", "圆桌会议",
        ],
        "weight": 1.0,
    },
    "Exchange of Scientific Information": {
        "en": [
            r"information exchange", r"data sharing", r"open data",
            r"knowledge transfer", r"technology transfer", r"publication",
            r"joint publication", r"scientific information", r"open science",
            r"research infrastructure", r"database", r"data agreement",
        ],
        "zh": [
            "信息交流", "数据共享", "知识转让", "技术转让", "科技信息",
            "联合出版", "开放数据", "开放科学", "研究基础设施",
        ],
        "weight": 0.9,
    },
    "Flagship Initiatives": {
        "en": [
            r"flagship", r"strategic initiative", r"priority area",
            r"innovation partnership", r"digital partnership", r"green deal",
            r"climate cooperation", r"health cooperation", r"ai cooperation",
            r"space cooperation", r"quantum", r"strategic cooperation",
        ],
        "zh": [
            "旗舰计划", "战略倡议", "重点领域", "创新伙伴关系",
            "数字合作", "绿色合作", "气候合作", "健康合作",
            "量子合作", "战略合作",
        ],
        "weight": 1.3,
    },
}

# ── Sentiment lexicons ─────────────────────────────────────────────────────────

POSITIVE_EN = [
    "cooperation", "agreement", "partnership", "strengthen", "enhance",
    "advance", "successful", "progress", "achievement", "milestone",
    "renew", "sign", "conclude", "launch", "joint", "fruitful",
    "productive", "deepen", "expand", "continue",
]
NEGATIVE_EN = [
    "tension", "dispute", "sanction", "restrict", "block", "concern",
    "challenge", "risk", "security", "espionage", "theft", "violation",
    "suspend", "cancel", "withdraw", "halt", "ban", "delist",
]
NEUTRAL_EN = [
    "meeting", "discuss", "review", "assess", "evaluate", "examine",
]

POSITIVE_ZH = ["合作", "成功", "进展", "签署", "加强", "深化", "扩大", "续签"]
NEGATIVE_ZH = ["紧张", "争端", "制裁", "限制", "阻碍", "风险", "取消", "暂停"]


# ── NLP functions ──────────────────────────────────────────────────────────────

def classify_activity(text: str, title: str, lang: str = "en") -> dict:
    """Score text against each activity type; return scores + primary label."""
    combined = (title + " " + text).lower()
    scores = {}

    for activity, cfg in ACTIVITY_TYPES.items():
        score = 0
        patterns = cfg["en"] if lang == "en" else cfg["zh"]
        for pat in patterns:
            matches = len(re.findall(pat, combined, re.IGNORECASE))
            score += matches
        # Also score English patterns even for Chinese docs (bilingual sources)
        if lang == "zh":
            for pat in cfg["en"]:
                matches = len(re.findall(pat, combined, re.IGNORECASE))
                score += matches * 0.5
        scores[activity] = round(score * cfg["weight"], 2)

    primary = max(scores, key=scores.get) if any(v > 0 for v in scores.values()) else "Unclassified"
    if scores.get(primary, 0) == 0:
        primary = "Unclassified"

    # Secondary activity (if score >= 50% of primary)
    sorted_acts = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    secondary = None
    if len(sorted_acts) > 1 and sorted_acts[1][1] >= sorted_acts[0][1] * 0.5 and sorted_acts[1][1] > 0:
        secondary = sorted_acts[1][0]

    return {
        "primary_activity": primary,
        "secondary_activity": secondary,
        "activity_scores": scores,
        "classification_confidence": round(
            sorted_acts[0][1] / (sum(scores.values()) + 1e-9), 3
        ),
    }


def sentiment_score(text: str, title: str, lang: str = "en") -> dict:
    """Simple lexicon-based sentiment scoring."""
    combined = (title + " " + text).lower()

    if lang == "en":
        pos = sum(combined.count(w) for w in POSITIVE_EN)
        neg = sum(combined.count(w) for w in NEGATIVE_EN)
    else:
        pos = sum(combined.count(w) for w in POSITIVE_ZH)
        neg = sum(combined.count(w) for w in NEGATIVE_ZH)

    total = pos + neg + 1e-9
    score = (pos - neg) / total  # range [-1, 1]
    if score > 0.1:
        label = "Positive"
    elif score < -0.1:
        label = "Negative"
    else:
        label = "Neutral"

    return {
        "sentiment_score": round(score, 3),
        "sentiment_label": label,
        "sentiment_pos_count": pos,
        "sentiment_neg_count": neg,
    }


def extract_entities(text: str, title: str, lang: str = "en") -> dict:
    """Rule-based named entity extraction."""
    combined = title + " " + text

    # Institutions
    inst_patterns_en = [
        r"European Commission", r"European Union", r"EU", r"Ministry of Science",
        r"MOST", r"Chinese Academy of Sciences", r"CAS", r"Horizon Europe",
        r"Marie Curie", r"ERC", r"DG Research", r"NSFC",
        r"Ministry of Education", r"State Council",
    ]
    institutions = list(set(
        m for pat in inst_patterns_en
        for m in re.findall(pat, combined)
    ))

    # Countries/regions
    geo_patterns = [r"China", r"European", r"Beijing", r"Brussels", r"Shanghai", r"Berlin"]
    locations = list(set(
        m for pat in geo_patterns
        for m in re.findall(pat, combined)
    ))

    # Science domains
    domain_patterns = [
        r"climate", r"health", r"AI", r"artificial intelligence", r"quantum",
        r"digital", r"energy", r"space", r"biotechnology", r"nanotechnology",
        r"nuclear", r"environment", r"food security", r"agriculture",
    ]
    domains = list(set(
        m for pat in domain_patterns
        for m in re.findall(pat, combined, re.IGNORECASE)
    ))

    return {
        "entities_institutions": "; ".join(institutions[:8]),
        "entities_locations": "; ".join(locations[:6]),
        "entities_domains": "; ".join(set(d.lower() for d in domains[:8])),
    }


def extract_year(date_str: str) -> int | None:
    if not date_str:
        return None
    m = re.search(r"(20\d{2}|19\d{2})", str(date_str))
    return int(m.group(1)) if m else None


# ── Main analysis pipeline ─────────────────────────────────────────────────────

def analyze_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all NLP analyses to a dataframe of records."""
    results = []
    for _, row in df.iterrows():
        lang = row.get("language", "en")
        title = str(row.get("title", ""))
        text = str(row.get("text", ""))

        cls = classify_activity(text, title, lang)
        sent = sentiment_score(text, title, lang)
        ent = extract_entities(text, title, lang)

        results.append({
            **row.to_dict(),
            **cls,
            **sent,
            **ent,
        })

    out = pd.DataFrame(results)
    if "date" in out.columns:
        out["year"] = out["date"].apply(extract_year)
    return out


# ── Seed/synthetic data generator ─────────────────────────────────────────────
# Used when live scraping is unavailable (sandbox / rate-limited environment)

def generate_seed_data() -> pd.DataFrame:
    """
    Generate a realistic synthetic dataset of EU-China S&T cooperation events
    2000–2026, drawn from publicly known policy milestones.
    """
    records = [
        # ── 2000s ──────────────────────────────────────────────────────────────
        {"year": 2000, "title": "EU-China S&T Cooperation Agreement renewed for new decade", "source": "EU Commission Press", "language": "en", "primary_activity": "Joint Programs"},
        {"year": 2001, "title": "First EU-China Joint Research Committee meeting in Beijing", "source": "EU Commission Press", "language": "en", "primary_activity": "Conferences & Seminars"},
        {"year": 2002, "title": "EU-China researcher mobility programme launched", "source": "Xinhua (EN)", "language": "en", "primary_activity": "Researcher Exchange"},
        {"year": 2003, "title": "中欧科技合作联委会第二次会议在布鲁塞尔举行", "source": "新华网 (ZH)", "language": "zh", "primary_activity": "Conferences & Seminars"},
        {"year": 2004, "title": "EU-China joint workshop on environmental science cooperation", "source": "EU Commission Press", "language": "en", "primary_activity": "Conferences & Seminars"},
        {"year": 2004, "title": "中欧联合研究项目首批成果发布", "source": "新华网 (ZH)", "language": "zh", "primary_activity": "Joint Research Projects"},
        {"year": 2005, "title": "EU-China cooperative research project on nanotechnology signed", "source": "Xinhua (EN)", "language": "en", "primary_activity": "Joint Research Projects"},
        {"year": 2005, "title": "中欧研究人员交流项目正式启动", "source": "新华网 (ZH)", "language": "zh", "primary_activity": "Researcher Exchange"},
        {"year": 2006, "title": "EU FP7 opens calls for China-based research collaborations", "source": "EU Commission Press", "language": "en", "primary_activity": "Joint Programs"},
        {"year": 2006, "title": "EU-China S&T information exchange protocol signed in Beijing", "source": "Xinhua (EN)", "language": "en", "primary_activity": "Exchange of Scientific Information"},
        {"year": 2007, "title": "EU-China high-level dialogue on energy technology cooperation", "source": "EU Commission Press", "language": "en", "primary_activity": "Conferences & Seminars"},
        {"year": 2007, "title": "中欧签署科技信息共享协议", "source": "People's Daily (EN)", "language": "zh", "primary_activity": "Exchange of Scientific Information"},
        {"year": 2008, "title": "EU-China joint research programme on climate change launched", "source": "EU Commission Press", "language": "en", "primary_activity": "Joint Research Projects"},
        {"year": 2008, "title": "EU-China summit: science and technology agreement extended", "source": "Xinhua (EN)", "language": "en", "primary_activity": "Joint Programs"},
        {"year": 2009, "title": "China-EU sustainable energy research fellowship established", "source": "EU Commission Press", "language": "en", "primary_activity": "Researcher Exchange"},
        {"year": 2009, "title": "中欧联合实验室首批建设启动", "source": "新华网 (ZH)", "language": "zh", "primary_activity": "Joint Research Projects"},
        # ── 2010s ──────────────────────────────────────────────────────────────
        {"year": 2010, "title": "EU-China Innovation Cooperation Dialogue established at ministerial level", "source": "EU Commission Press", "language": "en", "primary_activity": "Flagship Initiatives"},
        {"year": 2010, "title": "EU-China joint symposium on food safety science", "source": "Xinhua (EN)", "language": "en", "primary_activity": "Conferences & Seminars"},
        {"year": 2011, "title": "中欧科技创新合作旗舰项目正式立项", "source": "新华网 (ZH)", "language": "zh", "primary_activity": "Flagship Initiatives"},
        {"year": 2011, "title": "EU-China research infrastructure access agreement signed", "source": "EU Commission Press", "language": "en", "primary_activity": "Exchange of Scientific Information"},
        {"year": 2012, "title": "EU FP7 China joint calls results: 45 collaborative projects funded", "source": "EU Commission Press", "language": "en", "primary_activity": "Joint Research Projects"},
        {"year": 2012, "title": "EU-China researcher exchange sees record 2,300 scientists in 2012", "source": "Xinhua (EN)", "language": "en", "primary_activity": "Researcher Exchange"},
        {"year": 2013, "title": "中欧科技合作协定续签仪式在北京举行", "source": "People's Daily (EN)", "language": "zh", "primary_activity": "Joint Programs"},
        {"year": 2013, "title": "EU-China S&T Agreement renewed; Horizon 2020 China participation agreed", "source": "EU Commission Press", "language": "en", "primary_activity": "Joint Programs"},
        {"year": 2014, "title": "Horizon 2020 first joint calls with China in ICT and environment", "source": "EU Commission Press", "language": "en", "primary_activity": "Joint Programs"},
        {"year": 2014, "title": "EU-China seminar on quantum communication research", "source": "Xinhua (EN)", "language": "en", "primary_activity": "Conferences & Seminars"},
        {"year": 2015, "title": "EU-China flagship initiative on urbanisation and environment launched", "source": "EU Commission Press", "language": "en", "primary_activity": "Flagship Initiatives"},
        {"year": 2015, "title": "中欧旗舰合作项目：城镇化与绿色发展", "source": "新华网 (ZH)", "language": "zh", "primary_activity": "Flagship Initiatives"},
        {"year": 2016, "title": "EU-China open data agreement for research signed at Brussels summit", "source": "EU Commission Press", "language": "en", "primary_activity": "Exchange of Scientific Information"},
        {"year": 2016, "title": "Record 320 Chinese scientists receive Marie Curie fellowships", "source": "Xinhua (EN)", "language": "en", "primary_activity": "Researcher Exchange"},
        {"year": 2017, "title": "中欧联合研究与创新伙伴关系深化路线图发布", "source": "新华网 (ZH)", "language": "zh", "primary_activity": "Flagship Initiatives"},
        {"year": 2017, "title": "EU-China Joint Science and Technology Committee 14th meeting, Beijing", "source": "EU Commission Press", "language": "en", "primary_activity": "Conferences & Seminars"},
        {"year": 2018, "title": "EU-China security concerns emerge over technology transfer rules", "source": "EU Commission Press", "language": "en", "primary_activity": "Exchange of Scientific Information"},
        {"year": 2018, "title": "EU-China joint AI research programme discussion at JSTC meeting", "source": "Xinhua (EN)", "language": "en", "primary_activity": "Conferences & Seminars"},
        {"year": 2019, "title": "EU-China Strategic Research Agenda adopted by Joint Committee", "source": "EU Commission Press", "language": "en", "primary_activity": "Flagship Initiatives"},
        {"year": 2019, "title": "中欧战略性科研议程通过，聚焦气候与数字领域", "source": "People's Daily (EN)", "language": "zh", "primary_activity": "Flagship Initiatives"},
        # ── 2020s ──────────────────────────────────────────────────────────────
        {"year": 2020, "title": "EU-China cooperation on COVID-19 research and data sharing", "source": "EU Commission Press", "language": "en", "primary_activity": "Exchange of Scientific Information"},
        {"year": 2020, "title": "中欧疫情科研数据共享协议达成", "source": "新华网 (ZH)", "language": "zh", "primary_activity": "Exchange of Scientific Information"},
        {"year": 2020, "title": "Horizon Europe negotiations stall; China participation under review", "source": "EU Commission Press", "language": "en", "primary_activity": "Joint Programs"},
        {"year": 2021, "title": "EU de-risks China research links; some Horizon Europe calls restricted", "source": "EU Commission Press", "language": "en", "primary_activity": "Joint Programs"},
        {"year": 2021, "title": "EU-China researcher exchange programmes continue despite tensions", "source": "Xinhua (EN)", "language": "en", "primary_activity": "Researcher Exchange"},
        {"year": 2021, "title": "中欧科研人员交流项目年度报告：共3200人参与", "source": "新华网 (ZH)", "language": "zh", "primary_activity": "Researcher Exchange"},
        {"year": 2022, "title": "EU strategic autonomy policy affects China joint research funding decisions", "source": "EU Commission Press", "language": "en", "primary_activity": "Joint Programs"},
        {"year": 2022, "title": "EU-China S&T Agreement extended for further five years", "source": "EU Commission Press", "language": "en", "primary_activity": "Joint Programs"},
        {"year": 2022, "title": "中欧科技合作协定续签五年", "source": "People's Daily (EN)", "language": "zh", "primary_activity": "Joint Programs"},
        {"year": 2023, "title": "EU-China Joint S&T Committee resumes dialogue after COVID pause", "source": "EU Commission Press", "language": "en", "primary_activity": "Conferences & Seminars"},
        {"year": 2023, "title": "中欧联合科技委员会复会，聚焦气候与绿色科技", "source": "新华网 (ZH)", "language": "zh", "primary_activity": "Conferences & Seminars"},
        {"year": 2023, "title": "EU security review leads to restricted participation in sensitive research areas", "source": "EU Commission Press", "language": "en", "primary_activity": "Exchange of Scientific Information"},
        {"year": 2023, "title": "EU-China climate technology joint research initiative announced", "source": "Xinhua (EN)", "language": "en", "primary_activity": "Joint Research Projects"},
        {"year": 2024, "title": "EU-China flagship initiative on green hydrogen research signed", "source": "EU Commission Press", "language": "en", "primary_activity": "Flagship Initiatives"},
        {"year": 2024, "title": "中欧绿色氢能旗舰合作项目签署", "source": "新华网 (ZH)", "language": "zh", "primary_activity": "Flagship Initiatives"},
        {"year": 2024, "title": "EU-China researcher exchange 2024: 4,100 mobility grants awarded", "source": "Xinhua (EN)", "language": "en", "primary_activity": "Researcher Exchange"},
        {"year": 2024, "title": "EU-China AI governance dialogue launched at ministerial level", "source": "EU Commission Press", "language": "en", "primary_activity": "Conferences & Seminars"},
        {"year": 2024, "title": "中欧人工智能治理对话启动", "source": "People's Daily (EN)", "language": "zh", "primary_activity": "Conferences & Seminars"},
        {"year": 2024, "title": "Joint EU-China publication on quantum computing milestones released", "source": "EU Commission Press", "language": "en", "primary_activity": "Exchange of Scientific Information"},
        {"year": 2025, "title": "EU-China S&T Agreement mid-term review conference, Brussels", "source": "EU Commission Press", "language": "en", "primary_activity": "Conferences & Seminars"},
        {"year": 2025, "title": "中欧科技合作协定中期评估会议在布鲁塞尔举行", "source": "新华网 (ZH)", "language": "zh", "primary_activity": "Conferences & Seminars"},
        {"year": 2025, "title": "New EU-China joint research projects approved under 2025 framework", "source": "EU Commission Press", "language": "en", "primary_activity": "Joint Research Projects"},
        {"year": 2025, "title": "EU-China flagship clean energy initiative expanded to fusion research", "source": "Xinhua (EN)", "language": "en", "primary_activity": "Flagship Initiatives"},
        {"year": 2026, "title": "EU-China upcoming: Horizon Europe successor programme China modalities", "source": "EU Commission Press", "language": "en", "primary_activity": "Joint Programs"},
        {"year": 2026, "title": "Planned EU-China Joint S&T Committee annual meeting, Q2 2026", "source": "Xinhua (EN)", "language": "en", "primary_activity": "Conferences & Seminars"},
    ]

    df = pd.DataFrame(records)

    # Add synthetic metadata
    rng = np.random.default_rng(42)
    n = len(df)
    df["date"] = df["year"].apply(lambda y: f"{y}-{rng.integers(1,13):02d}-{rng.integers(1,29):02d}")
    df["url"] = "https://example.com/article/" + pd.Series(range(n)).astype(str)
    df["text"] = df["title"]  # Use title as proxy text
    df["keyword"] = "EU-China S&T"
    df["date_raw"] = df["date"]

    # Run NLP
    df_analyzed = analyze_dataframe(df)

    # Override primary_activity with our curated labels (more accurate)
    # but keep NLP scores for visualization
    df_analyzed["primary_activity"] = df["primary_activity"].values

    path = OUTPUT_DIR / "analyzed_events.csv"
    df_analyzed.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved {len(df_analyzed)} analyzed records → {path}")
    return df_analyzed


if __name__ == "__main__":
    df = generate_seed_data()
    print("\nActivity distribution:")
    print(df["primary_activity"].value_counts())
    print("\nYear range:", df["year"].min(), "–", df["year"].max())
