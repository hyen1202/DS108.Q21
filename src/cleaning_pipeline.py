import pandas as pd
import numpy as np
import re
import html
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp

class CleaningPipeline:

    ELECTRONICS_LIST = [
        'thiết bị số - phụ kiện số', 'laptop - máy vi tính - linh kiện', 
        'điện tử - điện lạnh', 'máy ảnh - máy quay phim', 
        'điện thoại - máy tính bảng', 'thiết bị điện tử', 'phụ kiện điện tử'
    ]

    FASHION_LIST = [
        'phụ kiện thời trang', 'thời trang nữ', 'đồng hồ và trang sức',
        'thời trang nam', 'balo và vali', 'giày - dép nam',
        'túi thời trang nam', 'túi thời trang nữ', 'giày - dép nữ',
        'thời trang & phụ kiện trẻ em', 'thời trang & phụ kiện nam',
        'thời trang & phụ kiện nữ'
    ]

    HOMELIVING_LIST = [
        'nhà cửa - đời sống', 'điện gia dụng', 'chăm sóc nhà cửa',
        'hàng gia dụng & đời sống', 'tv & thiết bị điện gia dụng'
    ]

    BEAUTY_LIST = [
        'làm đẹp - sức khỏe', 'sức khỏe & làm đẹp'
    ]


    def __init__(self, filepath, near_dup_threshold=0.90):
        """
        Khởi tạo Pipeline làm sạch dữ liệu.
        :param filepath:            Đường dẫn tới file dữ liệu thô (.csv)
        :param near_dup_threshold:  Ngưỡng quyết định trùng mờ chéo sàn (Mục 8)
        """
        self.filepath        = filepath
        self.dup_threshold   = near_dup_threshold
        self.df              = None
        self.log             = {}

    def run_pipeline(self):

        # 1. Load & snapshot ban đầu
        self.df = pd.read_csv(self.filepath)
        self.log['rows_initial'] = len(self.df)
        self.log['cols_initial'] = list(self.df.columns)

        # 2. Dedup cột gốc 
        rows_before_exact = len(self.df)
        self.df = self.df.drop_duplicates(subset=['title'], keep='first')
        self.log['dedup_exact_removed'] = rows_before_exact - len(self.df)

        # 3. Xử lý missing values
        rows_before_missing = len(self.df)
        self.df = self.df.dropna(subset=['title', 'price'])
        self.df['brand'] = self.df['brand'].fillna('No Brand')
        self.log['missing_handle_removed'] = rows_before_missing - len(self.df)
        self.log['rows_after_missing'] = len(self.df)

        # Áp dụng hàm làm sạch tích hợp cho từng tiêu đề (Bao gồm các mục 4, 5, 6)
        self.df['title_clean'] = self.df['title'].apply(self._clean_text_logic)

        # 7. Dedup sau normalize (Gộp các trường hợp trùng lặp format)
        rows_before_norm_dedup = len(self.df)
        self.df['title_lower_tmp'] = self.df['title_clean'].str.lower().str.strip()
        self.df = self.df.drop_duplicates(subset=['title_lower_tmp'], keep='first')
        self.df = self.df.drop(columns=['title_lower_tmp'])
        self.log['dedup_normalize_removed'] = rows_before_norm_dedup - len(self.df)

        # Reset index phẳng trước khi chạy thuật toán Trùng Mờ
        self.df = self.df.reset_index(drop=True)

        # 8. Near-duplicate detection (cross-platform) — TF-IDF + Cosine
        self.df = self._remove_near_duplicates_logic()

        # GÁN NHÓM NGÀNH
        if 'category_l1' in self.df.columns:
            # Ép hạ thường an toàn cho khâu so khớp isin
            self.df['category_l1_lower_tmp'] = self.df['category_l1'].astype(str).str.lower().str.strip()
            self.df['industry_group'] = self.df['category_l1_lower_tmp'].apply(self._map_industry_group_logic)
            self.df = self.df.drop(columns=['category_l1_lower_tmp'])
        else:
            self.df['industry_group'] = 'other'

        # Lưu thông tin phân phối phân tầng nhóm ngành phục vụ ghi log
        self.log['industry_distribution'] = self.df['industry_group'].value_counts().to_dict()

        # Tạo số thứ tự tăng dần định dạng 5 chữ số (00001, 00002...)
        zero_padded_seq = np.arange(1, len(self.df) + 1)
        zero_padded_seq = [f"{x:05d}" for x in zero_padded_seq]
        # Kết hợp ra ID cứng (Ví dụ: lazada_00001, tiki_03300)
        self.df['id'] = self.df['platform'].astype(str) + "_" + zero_padded_seq

        # 9. Chuẩn hóa brand
        self.df['brand_clean'] = self.df['brand'].apply(self._clean_brand_logic)

        # Tính word_count sạch hỗ trợ vẽ Boxplot
        self.df['word_count'] = self.df['title_clean'].str.split().str.len()

        # 10. Cleaning log & báo cáo cuối
        self._generate_final_report()

        return self.df
        

    def _clean_text_logic(self, text):
        if not isinstance(text, str):
            return ""

        # 4. Unicode NFC normalization
        text = unicodedata.normalize('NFC', text)
        text = html.unescape(text)

        # 5. Làm sạch noise ký tự (HTML tag, SĐT, emoji, ký tự rác)
        text = re.sub(r'<[^>]+>', ' ', text)                                # Xóa thẻ HTML
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)                 # Xóa link website
        text = re.sub(r'(?<!\d)(0[3-9]\d{8})(?!\d)', ' ', text)            # SĐT chuẩn 10 số
        text = re.sub(r'(?<!\d)(0[3-9][\d\.]{9,12})(?!\d)', ' ', text)     # SĐT dấu chấm
        text = re.sub(r'@[0-9\._\s\-\+]+', ' ', text)                      # SĐT dạng @ 
        text = re.sub(r'_', ' ', text)                                      # Thay _ bằng khoảng trắng

        # Loại bỏ Emoji và ký tự đặc biệt
        text = re.sub(r'[^\w\s,.\/\"\'\–%\‑\>\°×\+&:*\″\"\℃\-]', ' ', text)

        # 6. Chuẩn hóa khoảng trắng
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _map_industry_group_logic(self, cat_l1_lower):
        # Hàm ánh xạ động từ danh mục cấp 1 về nhóm ngành lớn
        if cat_l1_lower in self.ELECTRONICS_LIST:
            return 'electronics & tech'
        elif cat_l1_lower in self.FASHION_LIST:
            return 'fashion & accessories'
        elif cat_l1_lower in self.HOMELIVING_LIST:
            return 'home & living'
        elif cat_l1_lower in self.BEAUTY_LIST:
            return 'beauty & health'
        else:
            return 'other'

    def _remove_near_duplicates_logic(self):
        """
        Near-duplicate detection chéo sàn bằng TF-IDF + Cosine Similarity.
        """
        titles    = self.df['title_clean'].tolist()
        platforms = self.df['platform'].tolist()

        # Vector hóa toàn bộ title
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4))
        tfidf_matrix = vectorizer.fit_transform(titles)

        # Tính cosine similarity theo batch để tiết kiệm RAM
        BATCH = 1000
        indices_to_drop = set()
        n = len(titles)

        for start in range(0, n, BATCH):
            end = min(start + BATCH, n)
            # Cosine giữa batch[start:end] vs toàn bộ phía sau (upper-triangle)
            batch_matrix = tfidf_matrix[start:end]
            sim_block     = cosine_similarity(batch_matrix, tfidf_matrix[start:])
            # sim_block shape: (end-start) x (n-start)

            rows_block, cols_block = sp.coo_matrix(
                sim_block > self.dup_threshold
            ).nonzero()

            for r, c in zip(rows_block, cols_block):
                i = start + r
                j = start + c          # c là offset từ `start`
                if i >= j:             # chỉ xét upper-triangle
                    continue
                if i in indices_to_drop or j in indices_to_drop:
                    continue
                indices_to_drop.add(j)

        self.log['near_duplicates_removed'] = len(indices_to_drop)
        return self.df.drop(index=list(indices_to_drop)).reset_index(drop=True)

    # ------------------------------------------------------------------
    def _clean_brand_logic(self, brand_name):
        if not isinstance(brand_name, str):
            return "no brand"
        b_clean = re.sub(r'[\t\n\r]+', ' ', brand_name)
        b_clean = re.sub(r'\s+', ' ', b_clean).strip()
        b_lower = b_clean.lower()

        if b_lower in ['nobrand', 'no brand']:
            return "no brand"
        if b_lower in ['locknlock', 'lock&lock', 'lock & lock']:
            return 'locknlock'

        return b_lower

    # ------------------------------------------------------------------
    def _generate_final_report(self):
        final_rows   = len(self.df)
        total_deleted = self.log['rows_initial'] - final_rows

        print("\n" + "="*55)
        print("BÁO CÁO PIPELINE LÀM SẠCH VÀ GHI LOG CUỐI")
        print("="*55)
        print(f"   Tổng số mẫu snapshot ban đầu:      {self.log['rows_initial']} dòng")
        print(f"   Đã xóa trùng lặp tuyệt đối gốc:    {self.log['dedup_exact_removed']} dòng")
        print(f"   Đã xử lý missing value (MCAR):      {self.log['missing_handle_removed']} dòng")
        print(f"   Chuẩn hóa NFC, xóa emoji, ký tự đặc biệt:   [DONE]")
        print(f"   Đã xóa trùng sau chuẩn hóa text:   {self.log['dedup_normalize_removed']} dòng")
        print(f"   Đã xóa trùng mờ (cùng & chéo sàn):     {self.log['near_duplicates_removed']} dòng")
        print(f"   Chuẩn hóa cột 'brand':         [DONE]")
        print(f"   Tự động gán nhãn 'id' độc nhất:    [DONE]")
        print("-"*55)
        print(" THỐNG KÊ PHÂN PHỐI NHÓM NGÀNH (INDUSTRY GROUP):")
        for group, count in self.log.get('industry_distribution', {}).items():
            print(f"    - {group}: {count} dòng ({round((count/final_rows)*100, 2)}%)")
        print("-"*55)
        print(f" TỔNG SỐ DÒNG DATASET SẠCH CUỐI CÙNG:          {final_rows} dòng")
        print(f" Tỷ lệ dữ liệu nhiễu bị tinh lọc:              -{round((total_deleted/self.log['rows_initial'])*100, 2)}%")
        print("="*55 + "\n")


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Run the cleaning pipeline on merged e-commerce product titles.")
    parser.add_argument("--input", "-i", default="data/merged/all_platforms_merged_v2.csv", help="Path to input merged CSV file")
    parser.add_argument("--output", "-o", default="data/processed/data_cleaned.csv", help="Path to save cleaned CSV file")
    parser.add_argument("--threshold", "-t", type=float, default=0.90, help="Near-duplicate cosine similarity threshold (default: 0.90)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        exit(1)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Running Cleaning Pipeline on: {args.input}")
    pipeline = CleaningPipeline(filepath=args.input, near_dup_threshold=args.threshold)
    df_clean = pipeline.run_pipeline()
    df_clean.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"✅ Successfully processed and saved cleaned data to: {args.output}")