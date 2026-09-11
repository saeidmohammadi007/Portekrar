# ============================================================
# پیدا کردن الگوهای پرتکرار MACD با DTW
# اجرا روی GitHub Actions
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from dtw import dtw

# ============================================================
# تنظیمات
# ============================================================
TICKERS = [
    "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "SOL-USD",
    "ADA-USD", "DOGE-USD", "LINK-USD", "AVAX-USD", "DOT-USD"
]

START_DATE = "2015-01-01"
END_DATE = None

PATTERN_LENGTH = 85

FAST = 12
SLOW = 26
SIGNAL = 9

MAX_DISTANCE = 0.80
SAKOE_CHIBA_RATIO = 0.5
MIN_OCCURRENCES = 5
TOP_PATTERNS = 20
MIN_GAP = PATTERN_LENGTH


# ============================================================
# توابع
# ============================================================
def get_data(ticker):
    print(f"دریافت {ticker} ...")
    df = yf.download(
        ticker,
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=True,
        progress=False
    )
    if df.empty:
        print(f"❌ داده‌ای برای {ticker} پیدا نشد")
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Close"]].dropna()
    return df


def calculate_macd(df):
    close = df["Close"].astype(float)
    ema_fast = close.ewm(span=FAST, adjust=False).mean()
    ema_slow = close.ewm(span=SLOW, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=SIGNAL, adjust=False).mean()
    histogram = macd - signal
    return pd.DataFrame({
        "MACD": macd, "Signal": signal, "Histogram": histogram
    }).dropna()


def normalize(x):
    x = np.asarray(x, dtype=float)
    mean = np.mean(x)
    std = np.std(x)
    if std < 1e-10:
        return x - mean
    return (x - mean) / std


def dtw_distance(a, b):
    a = normalize(a)
    b = normalize(b)
    window_size = max(1, int(SAKOE_CHIBA_RATIO * max(len(a), len(b))))
    result = dtw(
        a, b,
        keep_internals=False,
        window_type="sakoechiba",
        window_size=window_size,
    )
    return result.distance / len(a)


def patterns_overlap(a, b):
    if a["ticker"] != b["ticker"]:
        return False
    if (a["end_idx"] + MIN_GAP <= b["start_idx"]
            or b["end_idx"] + MIN_GAP <= a["start_idx"]):
        return False
    return True


# ============================================================
# ساخت الگوها
# ============================================================
all_patterns = []

for ticker in TICKERS:
    df = get_data(ticker)
    if df is None:
        continue

    macd_df = calculate_macd(df)

    if len(macd_df) < PATTERN_LENGTH:
        print(f"⚠️ {ticker}: داده کافی ندارد.")
        continue

    macd_values = macd_df["MACD"].values
    dates = macd_df.index

    for end_idx in range(PATTERN_LENGTH, len(macd_values) + 1):
        start_idx = end_idx - PATTERN_LENGTH
        pattern = macd_values[start_idx:end_idx]
        all_patterns.append({
            "ticker": ticker,
            "start_idx": start_idx,
            "end_idx": end_idx - 1,
            "start_date": dates[start_idx],
            "end_date": dates[end_idx - 1],
            "pattern": pattern
        })

print()
print("================================================")
print("تعداد کل الگوهای MACD")
print("================================================")
print(len(all_patterns))
print(f"طول هر الگو: {PATTERN_LENGTH} کندل روزانه")
print(f"DTW با Sakoe-Chiba Band = {SAKOE_CHIBA_RATIO}")


# ============================================================
# جستجوی گروه‌ها
# ============================================================
print()
print("جستجوی الگوهای مشابه با DTW ...")

groups = []
used_patterns = set()

for i in range(len(all_patterns)):
    if i in used_patterns:
        continue

    reference = all_patterns[i]
    candidates = []

    for j in range(i + 1, len(all_patterns)):
        if j in used_patterns:
            continue

        candidate = all_patterns[j]

        if patterns_overlap(reference, candidate):
            continue

        distance = dtw_distance(reference["pattern"], candidate["pattern"])

        if distance <= MAX_DISTANCE:
            candidates.append((j, distance))

    candidates.sort(key=lambda x: x[1])

    selected = []
    for j, distance in candidates:
        candidate = all_patterns[j]
        overlap = False
        for selected_j, _ in selected:
            if patterns_overlap(candidate, all_patterns[selected_j]):
                overlap = True
                break
        if not overlap:
            selected.append((j, distance))

    occurrences = 1 + len(selected)

    if occurrences >= MIN_OCCURRENCES:
        groups.append({
            "reference": i,
            "matches": selected,
            "count": occurrences
        })
        used_patterns.add(i)
        for j, _ in selected:
            used_patterns.add(j)

groups.sort(key=lambda x: x["count"], reverse=True)


# ============================================================
# خروجی
# ============================================================
lines = []
lines.append("#" * 56)
lines.append(" پرتکرارترین الگوهای MACD")
lines.append("#" * 56)

if len(groups) == 0:
    lines.append("")
    lines.append("هیچ الگوی پرتکراری با تنظیمات فعلی پیدا نشد.")
    lines.append("برای پیدا کردن نمونه‌های بیشتر، MAX_DISTANCE را افزایش بده.")
else:
    for rank, group in enumerate(groups[:TOP_PATTERNS], 1):
        reference = all_patterns[group["reference"]]
        lines.append("")
        lines.append("=" * 56)
        lines.append(f"رتبه الگو: {rank}")
        lines.append(f"تعداد تکرار مستقل: {group['count']}")
        lines.append(f"نماد مرجع: {reference['ticker']}")
        lines.append(f"شروع: {reference['start_date'].strftime('%Y-%m-%d')}")
        lines.append(f"پایان: {reference['end_date'].strftime('%Y-%m-%d')}")
        lines.append("")
        lines.append("نمونه‌های مشابه:")
        lines.append("-" * 70)
        lines.append(f"{'نماد':<12}{'شروع':<15}{'پایان':<15}{'DTW':<12}")
        lines.append("-" * 70)
        lines.append(
            f"{reference['ticker']:<12}"
            f"{reference['start_date'].strftime('%Y-%m-%d'):<15}"
            f"{reference['end_date'].strftime('%Y-%m-%d'):<15}"
            f"{0:<12.4f}"
        )
        for j, distance in group["matches"]:
            sample = all_patterns[j]
            lines.append(
                f"{sample['ticker']:<12}"
                f"{sample['start_date'].strftime('%Y-%m-%d'):<15}"
                f"{sample['end_date'].strftime('%Y-%m-%d'):<15}"
                f"{distance:<12.4f}"
            )


summary = []
for rank, group in enumerate(groups[:TOP_PATTERNS], 1):
    reference = all_patterns[group["reference"]]
    summary.append({
        "Rank": rank,
        "Symbol": reference["ticker"],
        "Pattern Start": reference["start_date"].strftime("%Y-%m-%d"),
        "Pattern End": reference["end_date"].strftime("%Y-%m-%d"),
        "Occurrences": group["count"]
    })

summary_df = pd.DataFrame(summary)

lines.append("")
lines.append("=" * 56)
lines.append("خلاصه")
lines.append("=" * 56)
lines.append(summary_df.to_string(index=False) if not summary_df.empty else "خالی")

output_text = "\n".join(lines)
print(output_text)

with open("macd_dtw_result.txt", "w", encoding="utf-8") as f:
    f.write(output_text)

if not summary_df.empty:
    summary_df.to_csv("macd_dtw_summary.csv", index=False)

print()
print("✅ خروجی در macd_dtw_result.txt و macd_dtw_summary.csv ذخیره شد.")
