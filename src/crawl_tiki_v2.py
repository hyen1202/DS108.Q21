import math
import requests
import time
import random
import pandas as pd
from tqdm import tqdm
from datetime import datetime

# ══════════════════════════════════════════════════════════════
#  CẤU HÌNH 
# ══════════════════════════════════════════════════════════════
CATEGORY_ID    = "1601"               # Dùng id của catagory_l3
URL_KEY        = "mat-na-cac-loai"    # urlKey dùng cho API listing
QUOTA          = 20                   # số sản phẩm muốn cào 
ITEMS_PER_PAGE = 40                   # Tiki trả về tối đa 40 sp/trang
PLATFORM       = "tiki"
OUTPUT_CSV     = "tiki_matna.csv"     # Đổi tên file tương ứng với tên category_l3


HEADERS_LISTING = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-VN,en;q=0.9,vi-VN;q=0.8,vi;q=0.7,en-US;q=0.6',
    'Referer': 'https://tiki.vn/cham-soc-toc-da-dau/c1591',
    'Connection': 'keep-alive',
}

HEADERS_DETAIL = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-VN,en;q=0.9,vi-VN;q=0.8,vi;q=0.7,en-US;q=0.6',
    'Referer': 'https://tiki.vn/',
    'Connection': 'keep-alive',
}


# ══════════════════════════════════════════════════════════════
#  BƯỚC 1 — LẤY DANH SÁCH PRODUCT ID TỪ LISTING API
# ══════════════════════════════════════════════════════════════
def detect_total_pages() -> int:
    """
    Goi trang 1 de lay tong so san pham -> tinh so trang.
    API tra ve: paging.total (tong so sp cua danh muc)
    """
    params = {
        'limit': str(ITEMS_PER_PAGE),
        'include': 'advertisement',
        'aggregations': '2',
        'version': 'home-persionalized',
        'trackity_id': '9cb12177-ec8f-e97f-6762-e7b82dedc09a',
        'category': CATEGORY_ID,
        'urlKey': URL_KEY,
        'page': '1',
    }
    try:
        resp = requests.get(
            'https://tiki.vn/api/personalish/v1/blocks/listings',
            headers=HEADERS_LISTING,
            params=params,
            timeout=15,
        )
        if resp.status_code == 200:
            paging    = resp.json().get('paging') or {}
            total_sp  = paging.get('total') or 0
            last_page = paging.get('last_page') or 0
            if last_page:
                print(f"  [paging] last_page={last_page}  |  total_sp={total_sp}")
                return int(last_page)
            if total_sp:
                est = math.ceil(int(total_sp) / ITEMS_PER_PAGE)
                print(f"  [paging] total_sp={total_sp}  -> uoc tinh {est} trang")
                return est
    except Exception as e:
        print(f"  [!] detect_total_pages loi: {e}")

    print("  [!] Khong xac dinh duoc so trang -- dung 10")
    return 10


def fetch_product_ids(quota: int) -> list[int]:
    """
    Phat hien tong so trang -> random chon trang 
    -> crawl cac trang do cho den khi du quota IDs.

    So trang can chon = ceil(quota / ITEMS_PER_PAGE) * 3 (du de phong trang rong)
    """
    print("=" * 55)
    print("  Buoc 1 -- Phat hien tong so trang...")
    print("=" * 55)
    total_pages = detect_total_pages()

    pages_needed   = max(1, math.ceil(quota / ITEMS_PER_PAGE))
    pool           = list(range(1, total_pages + 1))
    k              = min(pages_needed * 3, len(pool))
    pages_selected = sorted(random.sample(pool, k))

    print(f"\n  Tong trang: {total_pages}  |  Quota: {quota} sp")
    print(f"  Trang duoc chon ngau nhien: {pages_selected}\n")

    params = {
        'limit': str(ITEMS_PER_PAGE),
        'include': 'advertisement',
        'aggregations': '2',
        'version': 'home-persionalized',
        'trackity_id': '9cb12177-ec8f-e97f-6762-e7b82dedc09a',
        'category': CATEGORY_ID,
        'urlKey': URL_KEY,
        'page': '1',
    }

    product_ids: list[int] = []
    seen: set = set()

    for page in pages_selected:
        if len(product_ids) >= quota:
            break
        params['page'] = str(page)
        try:
            resp = requests.get(
                'https://tiki.vn/api/personalish/v1/blocks/listings',
                headers=HEADERS_LISTING,
                params=params,
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json().get('data') or []
                new_ids = [
                    record.get('id') for record in data
                    if record.get('id') and record.get('id') not in seen
                ]
                for pid in new_ids:
                    seen.add(pid)
                    product_ids.append(pid)
                print(f"  [trang {page:>3}] +{len(new_ids)} IDs  |  tong: {len(product_ids)}")
            else:
                print(f"  [trang {page:>3}] HTTP {resp.status_code} -- bo qua")
        except Exception as e:
            print(f"  [trang {page:>3}] Loi: {e}")

        time.sleep(random.uniform(3, 10))

    product_ids = product_ids[:quota]
    print(f"\n  -> {len(product_ids)} product ID vao Buoc 2\n")
    return product_ids


# ══════════════════════════════════════════════════════════════
#  BƯỚC 2 — LẤY CHI TIẾT TỪNG SẢN PHẨM
# ══════════════════════════════════════════════════════════════
def parse_product(json_data: dict, ts: str) -> dict:
    """
    Mapping:
      title        ← name
      product_id   ← id
      product_url  ← short_url  (fallback: tự build từ id + name slug)
      price        ← current_seller.price  (fallback: price)
      rating_score ← rating_average
      review_count ← review_count
      sold_count   ← quantity_sold.value
      brand        ← brand.name
      category_l1  ← breadcrumbs[0].name
      category_l2  ← breadcrumbs[1].name
      category_l3  ← breadcrumbs[2].name
    """
    # ── title ─────────────────────────────────────────────────
    title = json_data.get('name', '').strip()

    # ── product_id ────────────────────────────────────────────
    product_id = json_data.get('id', '')

    # ── product_url ───────────────────────────────────────────
    product_url = json_data.get('short_url', '') or \
                  f"https://tiki.vn/-p{product_id}.html"

    # ── price: current_seller.price → fallback field 'price' ──
    price = ''
    current_seller = json_data.get('current_seller') or {}
    seller_price   = current_seller.get('price')
    if seller_price is not None:
        price = str(int(seller_price))
    else:
        raw_price = json_data.get('price')
        if raw_price is not None:
            price = str(int(raw_price))

    # ── rating_score ──────────────────────────────────────────
    rating_score = json_data.get('rating_average') or 0.0
    try:
        rating_score = float(rating_score)
    except (TypeError, ValueError):
        rating_score = 0.0

    # ── review_count ──────────────────────────────────────────
    review_count = json_data.get('review_count') or 0
    try:
        review_count = int(str(review_count).replace(',', ''))
    except (TypeError, ValueError):
        review_count = 0

    # ── sold_count: quantity_sold.value ───────────────────────
    sold_count = 0
    qty_sold = json_data.get('quantity_sold') or {}
    if isinstance(qty_sold, dict):
        val = qty_sold.get('value')
        try:
            sold_count = int(val) if val is not None else 0
        except (TypeError, ValueError):
            sold_count = 0

    # ── brand: brand.name ─────────────────────────────────────
    brand_obj = json_data.get('brand') or {}
    brand     = brand_obj.get('name', 'No Brand') or 'No Brand'

    # ── categories từ breadcrumbs ─────────────────────────────
    # breadcrumbs là list: [L1, L2, L3, ...]  (index 0 = root)
    breadcrumbs  = json_data.get('breadcrumbs') or []
    category_l1  = breadcrumbs[0].get('name', '') if len(breadcrumbs) > 0 else ''
    category_l2  = breadcrumbs[1].get('name', '') if len(breadcrumbs) > 1 else ''
    category_l3  = breadcrumbs[2].get('name', '') if len(breadcrumbs) > 2 else ''

    return {
        'title':        title,
        'product_id':   str(product_id),
        'platform':     PLATFORM,
        'category_l1':  category_l1,
        'category_l2':  category_l2,
        'category_l3':  category_l3,
        'product_url':  product_url,
        'price':        price,
        'rating_score': rating_score,
        'review_count': review_count,
        'sold_count':   sold_count,
        'brand':        brand,
        'time_stamp':   ts,
    }


def fetch_product_details(product_ids: list[int]) -> list[dict]:
    """
    Gọi API detail cho từng product_id.
    API: GET https://tiki.vn/api/v2/products/{id}?platform=web&version=3
    """
    ts      = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    records = []

    print("═" * 55)
    print("  Bước 2 — Lấy chi tiết sản phẩm")
    print("═" * 55 + "\n")

    for pid in tqdm(product_ids, desc="Detail", unit="sp"):
        url = f"https://tiki.vn/api/v2/products/{pid}"
        try:
            resp = requests.get(
                url,
                headers=HEADERS_DETAIL,
                params={'platform': 'web', 'version': '3'},
                timeout=15,
            )
            if resp.status_code == 200:
                rec = parse_product(resp.json(), ts)
                records.append(rec)
                tqdm.write(
                    f"  ✓ [{rec['product_id']:>10}] "
                    f"{rec['brand'][:12]:12s} | "
                    f"💰{rec['price']:>10s} | "
                    f"⭐{rec['rating_score']} | "
                    f"💬{rec['review_count']} | "
                    f"🛒{rec['sold_count']} | "
                    f"{rec['title'][:30]}"
                )
            else:
                tqdm.write(f"  [ERR {resp.status_code}] product_id={pid}")

        except Exception as e:
            tqdm.write(f"  [ERR] product_id={pid} → {e}")

        time.sleep(random.uniform(3, 5))

    return records


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    # Bước 1: lấy danh sách ID
    product_ids = fetch_product_ids(QUOTA)

    if not product_ids:
        print("⚠️  Không lấy được product ID nào. Kiểm tra lại token/cookie.")
        return

    # Bước 2: lấy chi tiết từng sản phẩm
    records = fetch_product_details(product_ids)

    if not records:
        print("⚠️  Không có dữ liệu sản phẩm.")
        return

    # Lưu CSV với đúng thứ tự cột
    COLS = [
        'title', 'product_id', 'platform',
        'category_l1', 'category_l2', 'category_l3',
        'product_url', 'price',
        'rating_score', 'review_count', 'sold_count',
        'brand', 'time_stamp',
    ]
    df = pd.DataFrame(records, columns=COLS)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    print(f"\n✅  Xong! {len(df)} sản phẩm → {OUTPUT_CSV}")

    # Báo cáo tỉ lệ điền được
    filled = {
        'price':        (df['price'] != '').sum(),
        'rating_score': (df['rating_score'] > 0).sum(),
        'review_count': (df['review_count'] > 0).sum(),
        'sold_count':   (df['sold_count'] > 0).sum(),
        'brand':        (df['brand'] != 'No Brand').sum(),
        'category_l1':  (df['category_l1'] != '').sum(),
        'category_l2':  (df['category_l2'] != '').sum(),
        'category_l3':  (df['category_l3'] != '').sum(),
    }
    print("\n  Tỉ lệ điền được:")
    for col, v in filled.items():
        bar = '█' * int(v / len(df) * 20)
        print(f"    {col:<15}: {v:>3}/{len(df)}  {bar}")


if __name__ == '__main__':
    main()