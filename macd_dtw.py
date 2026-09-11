# ============================================================
# پیدا کردن الگوهای پرتکرار MACD با DTW + ارسال به تلگرام
# (نسخه اصلاح‌شده: امتیازدهی ترکیبی + تنوع نماد)
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import os
import time
import html
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from collections import defaultdict
from dtw import dtw

# ============================================================
# تنظیمات
# ============================================================
TICKERS = [
    "BTC-USD", "BNB-USD", "XTZ-USD", "AVAX-USD", "DOGE-USD"
]

PATTERN_LENGTH = 60
FAST = 12
SLOW = 26
SIGNAL = 9

MAX_DISTANCE = 2.0
SAKOE_CHIBA_RATIO = 0.5
MIN_OCCURRENCES = 3
TOP_PATTERNS = 10
MIN_GAP = PATTERN_LENGTH
L2_PREFILTER = 4.0

# --- پارامترهای امتیازدهی ---
W_COUNT       = 0.40   # وزن تعداد تکرار (لگاریتمی)
W_QUALITY     = 0.30   # وزن کیفیت شباهت (DTW پایین = بهتر)
W_CONSISTENCY = 0.10   # وزن انسجام گروه (std پایین DTW)
W_DIVERSITY   = 0.10   # وزن تنوع نماد
W_TIME_SPREAD = 0.10   # وزن پخش زمانی (چند سال)

MAX_PER_TICKER = 3     # حداکثر گروه از هر نماد در لیست نهایی
SHUFFLE_SEED   = 123   # برای درهم‌ریختن ترتیب الگوها


# ============================================================
# دریافت داده (مقاوم با retry + fallback)
# ============================================================
def _clean_df(df):
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if "Close" not in df.columns:
        return None
    df = df[["Close"]].dropna()
    return df


def get_data(ticker, max_retries=3):
    print(f"دریافت {ticker} ...")
    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            df = yf.download(
                ticker,
                period="5y",
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
                timeout=30,
            )
            df = _clean_df(df)
            if df is not None and len(df) >= PATTERN_LENGTH:
                print(f"  ✅ {ticker}: {len(df)} رکورد.")
                return df
        except Exception as e:
            last_err = e

        try:
            t = yf.Ticker(ticker)
            df = t.history(period="5y", interval="1d", auto_adjust=True)
            df = _clean_df(df)
            if df is not None and len(df) >= PATTERN_LENGTH:
                print(f"  ✅ {ticker} (history): {len(df)} رکورد.")
                return df
        except Exception as e:
            last_err = e

        try:
            df = yf.download(
                ticker, period="2y", interval="1d",
                auto_adjust=True, progress=False,
                threads=False, timeout=30,
            )
            df = _clean_df(df)
            if df is not None and len(df) >= PATTERN_LENGTH:
                print(f"  ✅ {ticker} (2y): {len(df)} رکورد.")
                return df
        except Exception as e:
            last_err = e

        print(f"  ⚠️ تلاش {attempt} ناموفق برای {ticker}")
        time.sleep(2 * attempt)

    print(f"❌ {ticker}: داده پیدا نشد. آخرین خطا: {last_err}")
    return None


# ============================================================
# محاسبات
# ============================================================
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


def l2_norm_dist(a, b):
    a = normalize(a)
    b = normalize(b)
    if len(a) != len(b):
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
    return float(np.linalg.norm(a - b) / len(a))


def dtw_distance(a, b):
    a = normalize(a)
    b = normalize(b)
    window_size = max(1, int(SAKOE_CHIBA_RATIO * max(len(a), len(b))))
    try:
        result = dtw(
            a, b,
            keep_internals=False,
            window_type="sakoechiba",
            window_args={"window_size": window_size},
        )
    except TypeError:
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
# امتیازدهی گروه‌ها
# ============================================================
def group_score(g):
    """امتیاز ترکیبی: count لگاریتمی + کیفیت + انسجام + تنوع + پخش زمانی"""
    dists = np.array([d for _, d in g["matches"]])

    # ۱) تعداد تکرار (لگاریتمی)
    count_score = np.log1p(g["count"])

    # ۲) کیفیت شباهت (DTW پایین‌تر = بهتر)
    if len(dists) > 0:
        quality = max(0.0, 1.0 - float(dists.mean()) / MAX_DISTANCE)
    else:
        quality = 0.0

    # ۳) انسجام (std پایین DTWها)
    if len(dists) > 1:
        consistency = 1.0 / (1.0 + float(dists.std()))
    else:
        consistency = 1.0

    # ۴) تنوع نماد
    refs = [all_patterns[g["reference"]]] + [all_patterns[j] for j, _ in g["matches"]]
    tickers_in = {p["ticker"] for p in refs}
    diversity = len(tickers_in) / len(TICKERS)

    # ۵) پخش زمانی
    years_in = {p["start_date"].year for p in refs}
    time_spread = len(years_in) / 5.0

    return (
        W_COUNT       * count_score +
        W_QUALITY     * quality +
        W_CONSISTENCY * consistency +
        W_DIVERSITY   * diversity +
        W_TIME_SPREAD * time_spread
    )


# ============================================================
# تلگرام
# ============================================================
def send_to_telegram(text, token, chat_id):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    MAX_LEN = 4000
    safe_text = html.escape(text)
    chunks = [safe_text[i:i + MAX_LEN] for i in range(0, len(safe_text), MAX_LEN)]
    for i, chunk in enumerate(chunks, 1):
        payload = {
            "chat_id": chat_id,
            "text": chunk,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        try:
            r = requests.post(url, data=payload, timeout=30)
            if r.status_code != 200:
                print(f"⚠️ تلگرام خطا (بخش {i}): {r.status_code} - {r.text}")
            else:
                print(f"📨 بخش {i}/{len(chunks)} ارسال شد.")
        except Exception as e:
            print(f"⚠️ ارسال بخش {i} ناموفق: {e}")


def send_document(filepath, token, chat_id, caption=""):
    if not os.path.exists(filepath):
        print(f"⚠️ فایل {filepath} پیدا نشد.")
        return
    url = f"https://api.telegram.org/bot{token}/sendDocument"
    try:
        with open(filepath, "rb") as f:
            r = requests.post(
                url,
                files={"document": f},
                data={"chat_id": chat_id, "caption": caption},
                timeout=60,
            )
        if r.status_code == 200:
            print(f"📎 فایل {filepath} ارسال شد.")
        else:
            print(f"⚠️ ارسال فایل ناموفق: {r.status_code} - {r.text}")
    except Exception as e:
        print(f"⚠️ ارسال فایل ناموفق: {e}")


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
        print(f"⚠️ {ticker}: داده کافی ندارد ({len(macd_df)} < {PATTERN_LENGTH}).")
        continue

    macd_values = macd_df["MACD"].values
    dates = macd_df.index

    step = max(1, PATTERN_LENGTH // 4)

    for end_idx in range(PATTERN_LENGTH, len(macd_values) + 1, step):
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

if len(all_patterns) < MIN_OCCURRENCES:
    msg = (
        f"❌ فقط {len(all_patterns)} الگو ساخته شد؛ "
        f"حداقل {MIN_OCCURRENCES} لازم است.\n"
        "یعنی دریافت داده از yfinance ناموفق بود."
    )
    print(msg)
    TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
    CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
    if TOKEN and CHAT_ID:
        send_to_telegram(msg, TOKEN, CHAT_ID)
    raise SystemExit


# ============================================================
# درهم‌ریختن ترتیب الگوها (تا BTC همیشه اول نباشه)
# ============================================================
rng_shuf = np.random.default_rng(SHUFFLE_SEED)
order = rng_shuf.permutation(len(all_patterns))
all_patterns = [all_patterns[i] for i in order]
print(f"🔀 الگوها درهم ریخته شدند (seed={SHUFFLE_SEED})")


# ============================================================
# توزیع فاصله‌ها
# ============================================================
print()
print("نمونه‌گیری از توزیع فاصله‌های DTW ...")
rng = np.random.default_rng(42)
sample_dists = []
n = len(all_patterns)
tries = 0
while len(sample_dists) < 1000 and tries < 20000:
    tries += 1
    i, j = rng.integers(0, n, 2)
    if i == j:
        continue
    a, b = all_patterns[i], all_patterns[j]
    if patterns_overlap(a, b):
        continue
    if l2_norm_dist(a["pattern"], b["pattern"]) > L2_PREFILTER:
        continue
    sample_dists.append(dtw_distance(a["pattern"], b["pattern"]))

if sample_dists:
    sd = np.array(sample_dists)
    print(f"  تعداد نمونه: {len(sd)}")
    print(f"  min={sd.min():.3f}  p50={np.percentile(sd,50):.3f}  max={sd.max():.3f}")
    print(f"  ➜ MAX_DISTANCE فعلی: {MAX_DISTANCE}")


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

        if l2_norm_dist(reference["pattern"], candidate["pattern"]) > L2_PREFILTER:
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

    if (i + 1) % 200 == 0:
        print(f"  پیشرفت: {i+1}/{len(all_patterns)} — گروه: {len(groups)}")

# --- مرتب‌سازی بر اساس امتیاز ترکیبی ---
for g in groups:
    g["score"] = group_score(g)

groups.sort(key=lambda x: x["score"], reverse=True)

print()
print("--- کیفیت گروه‌های یافت‌شده (Top 20 بر اساس امتیاز) ---")
for rank, g in enumerate(groups[:20], 1):
    dists = [d for _, d in g["matches"]]
    ref = all_patterns[g["reference"]]
    avg_dtw = np.mean(dists) if dists else 0.0
    print(f"  رتبه {rank}: {ref['ticker']:<10} "
          f"تکرار={g['count']:<4} میانگین DTW={avg_dtw:.3f}  "
          f"امتیاز={g['score']:.3f}")


# ============================================================
# انتخاب TOP_PATTERNS با محدودیت هر نماد
# ============================================================
ticker_count = defaultdict(int)
top_groups = []

for g in groups:
    ref_ticker = all_patterns[g["reference"]]["ticker"]
    if ticker_count[ref_ticker] >= MAX_PER_TICKER:
        continue
    ticker_count[ref_ticker] += 1
    top_groups.append(g)
    if len(top_groups) >= TOP_PATTERNS:
        break

print()
print("--- ترکیب نمادها در لیست نهایی ---")
final_tickers = defaultdict(int)
for g in top_groups:
    final_tickers[all_patterns[g["reference"]]["ticker"]] += 1
for t, c in final_tickers.items():
    print(f"  {t}: {c} گروه")


# ============================================================
# خروجی متنی
# ============================================================
lines = []
lines.append("#" * 56)
lines.append(" پرتکرارترین الگوهای MACD (۱۰ الگوی برتر)")
lines.append(" (امتیازدهی ترکیبی: تکرار + کیفیت + تنوع + پخش زمانی)")
lines.append("#" * 56)

if len(top_groups) == 0:
    lines.append("")
    lines.append("هیچ الگوی پرتکراری با تنظیمات فعلی پیدا نشد.")
    lines.append("برای پیدا کردن نمونه‌های بیشتر، MAX_DISTANCE را افزایش بده.")
else:
    for rank, group in enumerate(top_groups, 1):
        reference = all_patterns[group["reference"]]
        dists = [d for _, d in group["matches"]]
        avg_dtw = float(np.mean(dists)) if dists else 0.0
        max_dtw = float(max(dists)) if dists else 0.0
        std_dtw = float(np.std(dists)) if len(dists) > 1 else 0.0

        refs_all = [reference] + [all_patterns[j] for j, _ in group["matches"]]
        n_tickers = len({p["ticker"] for p in refs_all})
        n_years = len({p["start_date"].year for p in refs_all})

        lines.append("")
        lines.append("=" * 56)
        lines.append(f"رتبه الگو: {rank}   |   امتیاز: {group['score']:.3f}")
        lines.append(f"تعداد تکرار مستقل: {group['count']}")
        lines.append(f"نماد مرجع: {reference['ticker']}")
        lines.append(f"شروع: {reference['start_date'].strftime('%Y-%m-%d')}")
        lines.append(f"پایان: {reference['end_date'].strftime('%Y-%m-%d')}")
        lines.append(f"میانگین DTW: {avg_dtw:.3f}   |   "
                     f"Max DTW: {max_dtw:.3f}   |   Std DTW: {std_dtw:.3f}")
        lines.append(f"تنوع نماد: {n_tickers}/{len(TICKERS)}   |   "
                     f"پخش زمانی: {n_years} سال")
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
for rank, group in enumerate(top_groups, 1):
    reference = all_patterns[group["reference"]]
    dists = [d for _, d in group["matches"]]
    refs_all = [reference] + [all_patterns[j] for j, _ in group["matches"]]
    summary.append({
        "Rank": rank,
        "Score": round(float(group["score"]), 4),
        "Symbol": reference["ticker"],
        "Pattern Start": reference["start_date"].strftime("%Y-%m-%d"),
        "Pattern End": reference["end_date"].strftime("%Y-%m-%d"),
        "Occurrences": group["count"],
        "Avg DTW": round(float(np.mean(dists)), 4) if dists else 0.0,
        "Max DTW": round(float(max(dists)), 4) if dists else 0.0,
        "Std DTW": round(float(np.std(dists)), 4) if len(dists) > 1 else 0.0,
        "Diversity": f"{len({p['ticker'] for p in refs_all})}/{len(TICKERS)}",
        "Years": len({p["start_date"].year for p in refs_all}),
    })

summary_df = pd.DataFrame(summary)

lines.append("")
lines.append("=" * 56)
lines.append("خلاصه (۱۰ الگوی برتر)")
lines.append("=" * 56)
lines.append(summary_df.to_string(index=False) if not summary_df.empty else "خالی")

output_text = "\n".join(lines)
print()
print(output_text)

with open("macd_dtw_result.txt", "w", encoding="utf-8") as f:
    f.write(output_text)

if not summary_df.empty:
    summary_df.to_csv("macd_dtw_summary.csv", index=False)

print()
print("✅ خروجی ذخیره شد.")


# ============================================================
# ارسال به تلگرام
# ============================================================
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

if TOKEN and CHAT_ID:
    print()
    print("📤 ارسال به تلگرام ...")
    send_to_telegram(output_text, TOKEN, CHAT_ID)
    if not summary_df.empty:
        send_document(
            "macd_dtw_summary.csv",
            TOKEN,
            CHAT_ID,
            "📊 خلاصه ۱۰ الگوی پرتکرار MACD (امتیازدهی ترکیبی)"
        )
else:
    print("⚠️ TELEGRAM_BOT_TOKEN یا TELEGRAM_CHAT_ID تنظیم نشده.")
