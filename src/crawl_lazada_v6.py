import re, json, math, random, time
import pandas as pd
from pathlib import Path
from datetime import datetime

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    NoSuchElementException, TimeoutException, StaleElementReferenceException
)
from tqdm import tqdm

try:
    from fake_useragent import UserAgent
    _UA = UserAgent()
    def random_ua() -> str:
        return _UA.chrome
except Exception:
    _AGENTS = [
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    ]
    def random_ua() -> str:
        return random.choice(_AGENTS)


# ══════════════════════════════════════════════════════════════
#  CẤU HÌNH  ← chỉnh tại đây
# ══════════════════════════════════════════════════════════════
TARGET_URL = (
    "https://www.lazada.vn/catalog/"
    "?spm=a2o4n.pdp_revamp.cate_4_1.1.4ec03bbd8PvV4N"
    "&q=D%C6%B0%E1%BB%A1ng+Da+Chuy%C3%AAn+S%C3%A2u"
    "&from=hp_categories&src=all_channel"
)

CATEGORY_L1 = "Làm Đẹp - Sức Khỏe"
CATEGORY_L2 = "Chăm Sóc Da"
CATEGORY_L3 = "Dưỡng Da Chuyên Sâu"
PLATFORM    = "lazada"
QUOTA       = 20       # số sản phẩm cần thu thập
HEADLESS    = False    # True để chạy ẩn trình duyệt

OUTPUT_CSV   = "lazada_products_v6.csv"
OUTPUT_PAGES = "lazada_pages_visited_v6.json"


# ══════════════════════════════════════════════════════════════
#  DRIVER
# ══════════════════════════════════════════════════════════════
def make_driver() -> webdriver.Chrome:
    ua = random_ua()
    opts = Options()
    if HEADLESS:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument(f"--user-agent={ua}")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.add_argument("--disable-extensions")
    opts.add_argument("--disable-infobars")

    driver = webdriver.Chrome(options=opts)

    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": """
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            Object.defineProperty(navigator, 'plugins',   {get: () => [1,2,3,4,5]});
            Object.defineProperty(navigator, 'languages', {get: () => ['vi-VN','vi','en-US','en']});
            window.chrome = { runtime: {} };
        """
    })
    driver.execute_cdp_cmd("Network.setExtraHTTPHeaders", {
        "headers": {
            "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        }
    })
    print(f"  [driver] UA: {ua[:70]}...")
    return driver


# ══════════════════════════════════════════════════════════════
#  UTILS
# ══════════════════════════════════════════════════════════════
def rand_sleep(lo: float = 5.0, hi: float = 10.0):
    t = random.uniform(lo, hi) + abs(random.gauss(0, 0.5))
    time.sleep(t)

def build_page_url(base: str, page: int) -> str:
    if "page=" in base:
        return re.sub(r"page=\d+", f"page={page}", base)
    sep = "&" if "?" in base else "?"
    return f"{base}{sep}page={page}"

def human_scroll(driver: webdriver.Chrome):
    height = driver.execute_script("return document.body.scrollHeight")
    pos = 0
    while pos < height:
        step = random.randint(200, 500)
        pos  = min(pos + step, height)
        driver.execute_script(f"window.scrollTo(0, {pos});")
        time.sleep(random.uniform(0.1, 0.45))
    # Cuộn ngược lên 1 chút
    driver.execute_script(
        f"window.scrollTo(0, {max(0, height - random.randint(80, 200))});"
    )
    time.sleep(random.uniform(0.3, 0.8))

def dismiss_popup(driver: webdriver.Chrome):
    for xp in [
        "//button[@aria-label='Close']",
        "//button[contains(@class,'close')]",
        "//div[contains(@class,'next-dialog-close')]",
        "/html/body/div[7]/div[2]/div",
    ]:
        try:
            el = driver.find_element(By.XPATH, xp)
            if el.is_displayed():
                el.click()
                time.sleep(0.5)
                return
        except Exception:
            continue

def css_text(driver: webdriver.Chrome, *selectors: str) -> str:
    for sel in selectors:
        try:
            return driver.find_element(By.CSS_SELECTOR, sel).text.strip()
        except NoSuchElementException:
            continue
    return ""

def parse_sold(text: str) -> int:
    if not text:
        return 0
    t = text.lower().replace(",", ".")
    has_k = "k" in t
    digits = re.sub(r"[^\d.]", "", t)
    if not digits:
        return 0
    try:
        val = float(digits)
        return int(val * 1000) if has_k else int(val)
    except ValueError:
        return 0

def clean_price(text: str) -> str:
    return re.sub(r"[^\d]", "", text) if text else ""

def _safe_float(val) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0

def _safe_int(val) -> int:
    try:
        return int(str(val).replace(",", "").replace(".", ""))
    except (TypeError, ValueError):
        return 0

def deep_find(obj, key):
    """Tìm đệ quy một key trong dict/list lồng nhau."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = deep_find(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for item in obj:
            r = deep_find(item, key)
            if r is not None:
                return r
    return None


# ══════════════════════════════════════════════════════════════
#  PHÁT HIỆN TỔNG SỐ TRANG
# ══════════════════════════════════════════════════════════════
def detect_total_pages(driver: webdriver.Chrome) -> int:
    driver.get(build_page_url(TARGET_URL, 1))
    try:
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "div[data-tracking='product-card']")
            )
        )
    except TimeoutException:
        print("  [!] Không load được listing.")
        return 1

    # Cách 1: button số trang (ant-pagination)
    try:
        items = driver.find_elements(By.CSS_SELECTOR, "li.ant-pagination-item")
        nums  = [int(el.get_attribute("title") or 0) for el in items
                 if el.get_attribute("title")]
        if nums:
            total = max(nums)
            print(f"  [pagination] {total} trang.")
            return total
    except Exception:
        pass

    # Cách 2: text "X/Y"
    for sel in ["span.ant-pagination-simple-pager", "span[class*='pagination']"]:
        try:
            txt = driver.find_element(By.CSS_SELECTOR, sel).text
            m   = re.search(r"/\s*(\d+)", txt)
            if m:
                print(f"  [pagination] {m.group(1)} trang.")
                return int(m.group(1))
        except Exception:
            pass

    # Cách 3: ước tính từ tổng items
    for sel in ["span[class*='total']", "div[class*='total']"]:
        try:
            digits = re.sub(r"[^\d]", "",
                            driver.find_element(By.CSS_SELECTOR, sel).text)
            if digits:
                total = max(1, math.ceil(int(digits) / 40))
                print(f"  [total-items] ước tính {total} trang.")
                return total
        except Exception:
            pass

    print("  [!] Không xác định được số trang — dùng 5.")
    return 5


def scrape_listing_page(driver: webdriver.Chrome, url: str) -> list[dict]:
    driver.get(url)
    try:
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "div[data-tracking='product-card']")
            )
        )
    except TimeoutException:
        print(f"  [!] Timeout load trang: {url}")
        return []

    human_scroll(driver)
    time.sleep(random.uniform(1.0, 2.0))

    # Lấy toàn bộ card — thử selector wrapper trước, fallback sang inner card
    cards = driver.find_elements(
        By.CSS_SELECTOR, "div[data-qa-locator='product-item']"
    )
    if not cards:
        cards = driver.find_elements(
            By.CSS_SELECTOR, "div[data-tracking='product-card']"
        )

    items = []

    for card in cards:
        try:
            # ── product_id từ attribute data-item-id ──────────────
            product_id = (
                card.get_attribute("data-item-id")
                or card.get_attribute("data-sku-simple")
                or "N/A"
            )

            # ── Title & URL từ div.RfADt > a ──────────────────────
            a = None
            for sel in ["div.RfADt a", "div[class*='RfADt'] a", "a[title]"]:
                try:
                    a = card.find_element(By.CSS_SELECTOR, sel)
                    break
                except NoSuchElementException:
                    continue

            if a is None:
                continue

            title = (a.get_attribute("title") or a.text or "").strip()
            href  = a.get_attribute("href") or ""
            if not title or not href:
                continue

            # ── Giá: div.aBrP0 > span.ooOxS ──────────────────────
            # Thứ tự ưu tiên khớp DOM trong ảnh, rồi fallback dần
            price = ""
            for sel in [
                "div.aBrP0 span.ooOxS",   # ← chính xác nhất từ DOM ảnh
                "span.ooOxS",             # fallback nếu ko có div.aBrP0
                "div.aBrP0 span",         # fallback giá trong container
                "span[class*='price']",   # fallback generic
            ]:
                try:
                    raw = card.find_element(By.CSS_SELECTOR, sel).text
                    c   = clean_price(raw)
                    if len(c) >= 4:       # giá hợp lệ ≥ 1000đ
                        price = c
                        break
                except NoSuchElementException:
                    continue

            # ── Sold count: div._6uN7R span._1cEkb > span:first-child ──
            # DOM: <span class="_1cEkb"><span>235 Đã bán</span><span class="brHcE"></span></span>
            sold_count = 0
            for sel in [
                "div._6uN7R span._1cEkb span:first-child",  # ← chính xác từ ảnh
                "span._1cEkb span",                         # fallback
                "span._1epib",                              # selector cũ
                "span[class*='sold']",
                "div[class*='sold'] span",
            ]:
                try:
                    elems = card.find_elements(By.CSS_SELECTOR, sel)
                    for el in elems:
                        txt = el.text.strip()
                        # Text phải chứa số (và thường có "đã bán" hoặc "sold")
                        if re.search(r"\d", txt):
                            val = parse_sold(txt)
                            if val > 0:
                                sold_count = val
                                break
                    if sold_count > 0:
                        break
                except NoSuchElementException:
                    continue

            # Fallback cuối: tìm bất kỳ text "X Đã bán" trong card
            if sold_count == 0:
                try:
                    all_spans = card.find_elements(By.TAG_NAME, "span")
                    for sp in all_spans:
                        txt = sp.text.strip()
                        if "đã bán" in txt.lower() or "sold" in txt.lower():
                            val = parse_sold(txt)
                            if val > 0:
                                sold_count = val
                                break
                except Exception:
                    pass

            items.append({
                "product_id":  product_id,
                "title":       title,
                "product_url": href,
                "price":       price,
                "sold_count":  sold_count,
            })

        except (StaleElementReferenceException, NoSuchElementException):
            continue

    return items


# ══════════════════════════════════════════════════════════════
#  DETAIL PAGE
#  Lấy: brand, rating_score, review_count
#  + fallback price / sold nếu listing miss
# ══════════════════════════════════════════════════════════════
def scrape_detail_page(driver: webdriver.Chrome,
                       product_url: str,
                       listing: dict) -> dict:
    driver.get(product_url)
    rand_sleep(5, 10)   # ← bắt buộc 5–10s giữa mỗi detail page
    dismiss_popup(driver)

    result = {
        "brand":        "No Brand",
        "rating_score": 0.0,
        "review_count": 0,
        # Kế thừa từ listing, overwrite nếu tìm được giá trị tốt hơn
        "price":        listing.get("price", ""),
        "sold_count":   listing.get("sold_count", 0),
    }

    # ── 1) JSON-LD (nhanh, đáng tin cậy nhất) ───────────────
    try:
        for s in driver.find_elements(
            By.XPATH, "//script[@type='application/ld+json']"
        ):
            raw = s.get_attribute("innerHTML") or ""
            if '"Product"' not in raw:
                continue
            data = json.loads(raw)

            brand = (data.get("brand") or {}).get("name", "") or ""
            result["brand"] = brand or "No Brand"

            # Price fallback từ JSON-LD
            if not result["price"]:
                offers = data.get("offers") or {}
                if isinstance(offers, list):
                    offers = offers[0]
                pv = offers.get("price") or offers.get("lowPrice")
                if pv:
                    result["price"] = str(int(float(pv)))

            agg = data.get("aggregateRating") or {}
            result["rating_score"] = _safe_float(agg.get("ratingValue"))
            result["review_count"] = _safe_int(agg.get("reviewCount"))
            break
    except Exception:
        pass

    # ── 2) window.pageData inline script ────────────────────
    needs = (
        result["brand"] == "No Brand"
        or result["rating_score"] == 0.0
        or not result["price"]
    )
    if needs:
        try:
            for s in driver.find_elements(By.XPATH, "//script[not(@src)]"):
                raw = s.get_attribute("innerHTML") or ""
                if "pageData" not in raw:
                    continue
                pd_m = re.search(
                    r'(?:window\.pageData|window\.__pageData__)\s*=\s*(\{)', raw
                )
                if not pd_m:
                    continue
                start = pd_m.start(1)
                depth = 0
                end   = start
                for i, ch in enumerate(raw[start:], start):
                    if ch == '{':   depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                page_data = json.loads(raw[start:end])

                if result["brand"] == "No Brand":
                    b = (
                        deep_find(page_data, "brandName")
                        or deep_find(page_data, "brandDisplayName")
                    )
                    if b:
                        result["brand"] = str(b)

                if result["rating_score"] == 0.0:
                    result["rating_score"] = _safe_float(
                        deep_find(page_data, "averageScore")
                        or deep_find(page_data, "ratingScore")
                    )
                if result["review_count"] == 0:
                    result["review_count"] = _safe_int(
                        deep_find(page_data, "reviewCount")
                        or deep_find(page_data, "totalReviewCount")
                    )
                if not result["price"]:
                    sku_list = deep_find(page_data, "skuPriceList")
                    if isinstance(sku_list, list) and sku_list:
                        sp = (
                            deep_find(sku_list[0], "value")
                            or deep_find(sku_list[0], "price")
                        )
                        if sp:
                            result["price"] = str(int(float(sp)))
                break
        except Exception:
            pass

    # ── 3) CSS fallback ─────────────────────────────────────
    # Brand
    if result["brand"] == "No Brand":
        b = css_text(driver,
            ".pdp-product-brand__brand-link",
            "a[class*='brand']",
            "span[class*='brand']",
        )
        if b:
            result["brand"] = b

    # Rating score
    if result["rating_score"] == 0.0:
        for sel in [
            "span.pdp-review-summary__stars-score",
            "div[class*='summary'] span[class*='score']",
            "span[class*='review-score']",
        ]:
            txt = css_text(driver, sel)
            if txt:
                m = re.search(r"[\d.]+", txt)
                if m:
                    val = _safe_float(m.group())
                    if 0 < val <= 5:
                        result["rating_score"] = val
                        break

    # Review count
    # DOM thực tế: <span class="container-star-v2-count">(14)</span>  → strip "()"
    if result["review_count"] == 0:
        for sel in [
            "span.container-star-v2-count",           # ← chính xác từ DOM ảnh
            "span.pdp-review-summary__stars-count",
            "a[class*='review-count']",
            "span[class*='review-count']",
        ]:
            txt = css_text(driver, sel)
            if txt:
                digits = re.sub(r"[^\d]", "", txt)   # strip "(", ")", dấu phẩy...
                if digits:
                    result["review_count"] = int(digits)
                    break

    # Rating score fallback thêm selector từ DOM ảnh
    if result["rating_score"] == 0.0:
        txt = css_text(driver, "span.container-star-v2-score")
        if txt:
            m = re.search(r"[\d.]+", txt)
            if m:
                val = _safe_float(m.group())
                if 0 < val <= 5:
                    result["rating_score"] = val

    # Price fallback CSS (detail page)
    if not result["price"]:
        for sel in [
            "span.pdp-price_type_normal",
            "span.pdp-price_color_orange",
            "span[class*='pdp-price']",
            "div.pdp-product-price span",
        ]:
            try:
                for el in driver.find_elements(By.CSS_SELECTOR, sel):
                    c = clean_price(el.text.strip())
                    if len(c) >= 4:
                        result["price"] = c
                        break
                if result["price"]:
                    break
            except Exception:
                continue

    return result


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    driver = make_driver()
    ts     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Bước 0: phát hiện tổng số trang ──
    print("\n" + "═"*60)
    print("  Bước 0 — Phát hiện tổng số trang...")
    print("═"*60)
    total_pages = detect_total_pages(driver)

    pages_needed   = max(1, math.ceil(QUOTA / 40))
    pool           = list(range(1, total_pages + 1))
    k              = min(pages_needed * 3, len(pool))
    pages_selected = sorted(random.sample(pool, k))

    log = {
        "timestamp":       ts,
        "target_url":      TARGET_URL,
        "total_pages":     total_pages,
        "pages_selected":  pages_selected,
        "quota":           QUOTA,
    }
    Path(OUTPUT_PAGES).write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n  Tổng trang: {total_pages}  |  Chọn: {pages_selected}")
    print(f"  Log → {OUTPUT_PAGES}\n")

    # ── Phase 1: Listing ──────────────────────────────────────
    print("═"*60)
    print("  Phase 1 — Listing (product_id, title, url, price, sold_count)")
    print("═"*60)
    all_listings: list[dict] = []

    with tqdm(total=QUOTA, desc="Listing", unit="sp") as pbar:
        for page in pages_selected:
            if len(all_listings) >= QUOTA:
                break
            url   = build_page_url(TARGET_URL, page)
            items = scrape_listing_page(driver, url)
            if not items:
                tqdm.write(f"  [!] Trang {page} rỗng — bỏ qua.")
                continue
            prev = len(all_listings)
            all_listings.extend(items)
            added = len(all_listings) - prev
            pbar.update(added)
            n_price = sum(1 for x in items if x["price"])
            n_sold  = sum(1 for x in items if x["sold_count"] > 0)
            tqdm.write(
                f"  [trang {page:>3}] +{added} sp | "
                f"price {n_price}/{len(items)} | sold {n_sold}/{len(items)}"
            )
            rand_sleep(3, 6)   # nghỉ ngắn hơn giữa các trang listing

    all_listings = all_listings[:QUOTA]
    print(f"\n  → {len(all_listings)} sản phẩm vào Phase 2.\n")

    # ── Phase 2: Detail ───────────────────────────────────────
    print("═"*60)
    print("  Phase 2 — Detail (brand, rating_score, review_count) — nghỉ 5–10s/sp")
    print("═"*60 + "\n")

    records: list[dict] = []

    for item in tqdm(all_listings, desc="Detail", unit="sp"):
        try:
            d = scrape_detail_page(driver, item["product_url"], item)
        except Exception as e:
            tqdm.write(f"  [ERR] {item['product_url'][:60]} → {e}")
            d = {
                "brand":        "No Brand",
                "rating_score": 0.0,
                "review_count": 0,
                "price":        item.get("price", ""),
                "sold_count":   item.get("sold_count", 0),
            }

        rec = {
            # Thứ tự cột theo spec yêu cầu
            "title":        item["title"],
            "product_id":   item["product_id"],
            "platform":     PLATFORM,
            "category_l1":  CATEGORY_L1,
            "category_l2":  CATEGORY_L2,
            "category_l3":  CATEGORY_L3,
            "product_url":  item["product_url"],
            "price":        d["price"],
            "rating_score": d["rating_score"],
            "review_count": d["review_count"],
            "sold_count":   d["sold_count"],
            "brand":        d["brand"],
            "time_stamp":   ts,
        }
        records.append(rec)
        tqdm.write(
            f"  ✓ {rec['brand'][:14]:14s} | "
            f"💰{rec['price'] or '?':>10s} | "
            f"⭐{rec['rating_score']} | "
            f"💬{rec['review_count']} | "
            f"🛒{rec['sold_count']} | "
            f"{rec['title'][:30]}"
        )
        # ← Bắt buộc nghỉ 5–10s sau mỗi detail page
        rand_sleep(5, 10)

    driver.quit()

    # ── Lưu & báo cáo ─────────────────────────────────────────
    if records:
        COLS = [
            "title", "product_id", "platform",
            "category_l1", "category_l2", "category_l3",
            "product_url", "price",
            "rating_score", "review_count", "sold_count",
            "brand", "time_stamp",
        ]
        df = pd.DataFrame(records, columns=COLS)
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        print(f"\n✅  Xong! {len(df)} sản phẩm → {OUTPUT_CSV}")

        filled = {
            "price":        (df["price"] != "").sum(),
            "rating_score": (df["rating_score"] > 0).sum(),
            "review_count": (df["review_count"] > 0).sum(),
            "sold_count":   (df["sold_count"] > 0).sum(),
            "brand":        (df["brand"] != "No Brand").sum(),
        }
        print("\n  Tỉ lệ điền được:")
        for col, v in filled.items():
            bar = "█" * int(v / len(df) * 20)
            print(f"    {col:<15}: {v:>3}/{len(df)}  {bar}")

    else:
        print("\n⚠️  Không có dữ liệu.")


if __name__ == "__main__":
    main()