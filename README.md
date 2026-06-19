# ViEcomNER: Vietnamese E-commerce Named Entity Recognition Dataset

[**Tiếng Việt**](#tiếng-việt) | [**English**](#english)

---

<a name="english"></a>
# English Version

## 1. Project Overview & Product Release
**ViEcomNER** is the first standard, open-source Named Entity Recognition (NER) dataset specifically designed for Vietnamese E-commerce (e-commerce) product titles. Built using a **Data-Centric AI** approach, ViEcomNER addresses the high linguistic noise, telegraphic writing style, and complex English-Vietnamese code-mixing typical of e-commerce platforms, offering a gold-standard benchmark for downstream NLP tasks.

## 2. Problem Statement & Research Motivation
*   **Specialized Data Scarcity:** Existing Vietnamese NER datasets (such as VLSP, PhoNER_COVID19) focus primarily on standard formal texts (news articles, medical reports). They fail when applied to the informal, highly fragmented domain of e-commerce.
*   **The Noisy Nature of E-Commerce Titles:** E-commerce product titles are fundamentally different from grammatical sentences. They exhibit:
    *   **Telegraphic style** (omitting functional grammar, subject-verb relationships).
    *   **Keyword stuffing** (repeated search keywords to boost SEO).
    *   **Code-mixing** (frequent English/Vietnamese combinations like *"Áo thun polo nam basic form rộng premium"*).
    *   **Non-standard spelling, symbols, and emojis** used to attract consumer attention.

## 3. Methodology & Core Contributions (Data-Centric AI)
1.  **Flat Schema & Tokenizer-Agnostic Design:**
    *   Designed a schema of **24 business entity tags** (15 general tags, 9 industry-specific tags).
    *   Annotations are stored as **character-level offsets** (character coordinates), completely preventing alignment drift or boundary corruption caused by automatic Vietnamese word-segmentation tools.
2.  **9-Step Automated Preprocessing Pipeline:**
    *   **Step 1:** Raw data loading and initial state snapshot logging.
    *   **Step 2:** Exact duplicate removal (dropping duplicate titles on raw text level).
    *   **Step 3:** Missing value handling (imputing missing brands, removing empty records).
    *   **Step 4:** Unicode NFC normalization and HTML unescaping.
    *   **Step 5:** Noise character scrubbing (removing HTML tags, web URLs, emojis, and symbols).
    *   **Step 6:** PII Protection (automatically masking phone numbers in titles using regex).
    *   **Step 7:** Normalized space compaction.
    *   **Step 8:** Cross-platform Near-Deduplication using Char TF-IDF (3-4 ngrams) and Cosine Similarity (threshold $\ge 0.90$) to eliminate data leakage.
    *   **Step 9:** Brand normalization and mapping to 4 major industry groups.
3.  **Human-in-the-Loop Hybrid Labeling Pipeline:**
    *   Used Large Language Models (LLM Claude Sonnet 4.6) for **pre-annotation**, increasing productivity by **60%**.
    *   **Anchor Bias Control:** Countered LLM anchor bias by introducing a manual fallback mechanism and double-blind review processes.
    *   Implemented **Blind Double Annotation** where annotators cross-evaluated 15% of the data blindly, and conflict resolution was managed by an administrator.

## 4. Dataset Scale & Statistics
*   **Data Sources:** Tiki (31.9%) and Lazada (68.1%).
*   **Industry Domains:** 4 primary categories:
    *   *Home & Living* (Nhà cửa & Đời sống)
    *   *Fashion & Accessories* (Thời trang & Phụ kiện)
    *   *Electronics & Technology* (Điện tử & Công nghệ)
    *   *Beauty & Health* (Làm đẹp & Sức khỏe)
*   **Scale:** Filtered from 19,361 raw crawled records down to **2,993 Gold Standard samples** (divided into 2,396 Train, 298 Validation, 299 Test).
*   **Entity Density:** Extremely dense, averaging **6.29 entities/title** (totaling 18,837 entities).
*   **Class Imbalance:** Naturally imbalanced distribution reflecting real e-commerce distributions, reaching a **133:1 ratio** between the most frequent label (`PRODUCT_TYPE`) and the rarest label (`SKIN_TYPE`).

## 5. Quality Evaluation & Experimental Results
*   **Annotation Reliability:** Reached "Almost Perfect Agreement" on the Gold Test Set.
*   **Cohen's Kappa (excluding 'O' tag):** **0.8207**, indicating high consensus.
*   **Exact Span F1-Score:** **81.67%**.
*   **Partial Match F1-Score:** **94.70%**, proving that annotators shared a highly consistent semantic understanding. Minor differences were almost entirely due to ambiguous Vietnamese word boundaries.

---

## 6. Directory Structure
```
.
├── README.md               # Project documentation (EN/VI)
├── requirements.txt        # Python dependency list
├── data/
│   ├── raw/                # ORIGINAL raw crawled data (UNTOUCHED)
│   │   ├── beauty/
│   │   ├── electronics/
│   │   ├── fashion/
│   │   └── home_living/
│   ├── merged/             # Merged crawled files
│   ├── processed/          # Preprocessed and deduped csv files
│   └── annotated/          # Gold standard splits (Train/Val/Test jsonl)
├── src/
│   ├── crawl_tiki_v2.py                  # Tiki crawler script
│   ├── crawl_lazada_v6.py                # Lazada crawler script
│   ├── cleaning_pipeline.py              # Automated data cleaning pipeline class
│   ├── prepare_labelstudio_import.py     # Conversion to Label Studio import tasks
│   ├── label_studio_config.xml           # Label Studio interface XML config
│   ├── split_blind_double_annotation.py # Partitioning for blind double annotation
│   ├── ls_export_to_iaa_jsonl.py         # Converts Label Studio export to calculate_iaa format
│   ├── calculate_iaa.py                  # Computes Inter-Annotator Agreement (Kappa, F1)
│   └── stratify_dev_split.py             # Stratified sampling for train/dev/test splits
├── notebooks/
│   ├── preprocessed_EDA.ipynb            # Exploratory Data Analysis on preprocessed data
│   ├── eda-gold-dataset.ipynb            # Analysis on Gold annotated dataset
│   └── stratified-sampling.ipynb         # Stratified resampling prototype notebook
└── demo/
    └── app.py                            # Streamlit Interactive Dashboard
```

---

## 7. Prerequisites & Installation

### Prerequisites
*   Python 3.8 or higher.
*   Google Chrome (for Selenium-based crawling, if running the crawler).

### Installation Guide
1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd DS108.Q21
    ```

2.  **Create and activate a virtual environment:**
    *   **Using standard `venv`:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate  # On macOS/Linux
        # Or: venv\Scripts\activate  # On Windows
        ```
    *   **Using Conda:**
        ```bash
        conda create -n viecomner python=3.9 -y
        conda activate viecomner
        ```

3.  **Install dependencies:**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

4.  **Download NLTK Corpora:**
    The stratified splitting script uses the NLTK English words corpus. Run this python command to pre-download it:
    ```bash
    python3 -c "import nltk; nltk.download('words')"
    ```

---

## 8. End-to-End Execution & Reproduction Guide

> [!IMPORTANT]
> **Data Integrity Policy:** The `data/raw/` directory contains original raw crawled CSVs. In strict compliance with the DS108 curriculum regulations, **no manual intervention or manual manipulation has been applied** to the files in this directory. All cleaning, normalization, and PII masking are handled programmatically through reproducibility scripts.

To reproduce the dataset creation process from scratch, run the following workflow:

### Step 1: Data Crawling (Optional)
The raw data is already pre-packaged in `data/raw/`. However, if you wish to run the crawlers:
```bash
# Crawl Tiki
python3 src/crawl_tiki_v2.py
# Crawl Lazada (Requires Chrome and Selenium)
python3 src/crawl_lazada_v6.py
```

### Step 2: Merge Raw Data
Ensure your crawled CSV files are placed in their respective category folders in `data/raw/`. Merge them into a single file:
*This step combines all files in `data/raw/` into `data/merged/all_platforms_merged_v2.csv`.*

### Step 3: Run the Cleaning & Near-Deduplication Pipeline
Run the preprocessing pipeline to clean noise, mask PII, and execute TF-IDF character-level Cosine Similarity near-deduplication:
```bash
python3 src/cleaning_pipeline.py \
  --input data/merged/all_platforms_merged_v2.csv \
  --output data/processed/data_cleaned.csv \
  --threshold 0.90
```

### Step 4: Split for Blind Double Annotation
Prepare tasks and split them for the 4 annotators (A, B, C, D) with a 15% blind overlap rotation:
```bash
# Convert cleaned CSV to Label Studio Tasks
python3 src/prepare_labelstudio_import.py \
  --input data/processed/data_cleaned.csv \
  --output data/processed/ls_tasks.json

# Split tasks using double-blind distribution
python3 src/split_blind_double_annotation.py \
  --input data/processed/ls_tasks.json \
  --outdir data/processed/blind_splits/ \
  --seed 42 \
  --qc-ratio 0.15
```
*This outputs `A.json`, `B.json`, `C.json`, `D.json` for annotators, and `blind_mapping.csv` for the project administrator.*

### Step 5: Inter-Annotator Agreement (IAA) Calculation
After importing the splits into Label Studio, completing the annotations, and exporting the JSON results:
1.  **Convert the exports into standard IAA JSONL formats:**
    ```bash
    python3 src/ls_export_to_iaa_jsonl.py --input data/annotated/annotator_A_export.json --output data/annotated/A.jsonl --name Annotator_A
    python3 src/ls_export_to_iaa_jsonl.py --input data/annotated/annotator_B_export.json --output data/annotated/B.jsonl --name Annotator_B
    ```
2.  **Calculate Cohen's Kappa & exact/partial F1 agreement:**
    ```bash
    python3 src/calculate_iaa.py \
      --a data/annotated/A.jsonl \
      --b data/annotated/B.jsonl \
      --outdir data/annotated/iaa_reports/
    ```

### Step 6: Final Stratified Dataset Resplitting
To recreate the final train, validation, and test splits with strict stratum control (industry group, language mixing profile, uppercase/lowercase casing, and title length):
```bash
python3 src/stratify_dev_split.py \
  --train-raw data/annotated/train.jsonl \
  --dev-raw data/annotated/validation.jsonl \
  --test-gold data/annotated/test.jsonl \
  --outdir data/annotated/
```
*This command runs the strict stratified sampling algorithm, validating representative thresholds and outputting the final `train.jsonl` (2,396 records), `validation.jsonl` (298/300 records), and `test.jsonl` (299 records) inside `data/annotated/`.*

### Step 7: Launch the Interactive Dashboard
To view EDA insights, IAA agreement calculations, and browse annotated titles dynamically:
```bash
streamlit run demo/app.py
```

---

## 9. Named Entity Annotation Tagset
ViEcomNER contains **24 entity tags** structured into **Common Labels** (15 tags applicable to all domains) and **Domain Labels** (9 tags specific to electronics, home & living, or beauty & health).

| No. | Entity Label | Domain Scope | Description / Example |
| :--- | :--- | :--- | :--- |
| 1 | `PRODUCT_TYPE` | Common | Category or type of product (*Áo thun, Nồi cơm điện*) |
| 2 | `BRAND` | Common | Product brand name (*Nike, Samsung, LocknLock*) |
| 3 | `MODEL` | Common | Model name or technical code (*iPhone 15 Pro Max, MX-200*) |
| 4 | `QUANTITY` | Common | Number of items in package (*1 cặp, combo 3 cái, hộp 50 chiếc*) |
| 5 | `ORIGIN` | Common | Country/place of origin (*Việt Nam, nội địa Trung*) |
| 6 | `COMPONENT` | Common | Accessory/gift items included (*cáp sạc đi kèm, tặng móc khóa*) |
| 7 | `MATERIAL` | Common | Physical material (*cotton, inox 304, lụa*) |
| 8 | `SIZE` | Common | Dimensions, sizing options (*M, L, XL, 45cm*) |
| 9 | `COMPAT` | Common | Compatibility info (*cho iPhone 14, tương thích iPad*) |
| 10 | `ATTRIBUTE` | Common | Descriptive product attributes (*chống nước, không dây*) |
| 11 | `OCCASION` | Common | Occasion/purpose/season (*đi tiệc, mùa hè, quà sinh nhật*) |
| 12 | `EFFECT` | Common | Health/Beauty benefit (*dưỡng ẩm, giảm mụn, trắng răng*) |
| 13 | `COLOR` | Common | Product color (*xanh mint, đỏ đô, đen*) |
| 14 | `STYLE` | Common | Fashion style, pattern (*họa tiết sọc, form rộng, cổ V*) |
| 15 | `TARGET_GROUP` | Common | Intended user demographics (*cho nam, unisex, em bé*) |
| 16 | `SPEC` | Electronics | Technical specifications (*50000mAh, 120Hz, 8GB/256GB*) |
| 17 | `CONNECTIVITY` | Electronics | Network interface / protocols (*Bluetooth 5.3, Type-C, Wifi 6*) |
| 18 | `POWER` | Home & Living | Electricity consumption rating (*2000W, 220V*) |
| 19 | `CAPACITY` | Home & Living | Internal volume or physical load limits (*1.8L, 7.5kg*) |
| 20 | `INGREDIENT` | Beauty & Health | Chemical or organic active ingredients (*Niacinamide, Retinol*) |
| 21 | `SKIN_TYPE` | Beauty & Health | Target skin categories (*da dầu mụn, nhạy cảm*) |
| 22 | `BODY_PART` | Beauty & Health | Body parts applicable (*cho tóc, da mặt, toàn thân*) |
| 23 | `VOLUME_WEIGHT`| Beauty & Health | Cosmetics weight/volume capacity (*100ml, 50g*) |
| 24 | `CONCENTRATION`| Beauty & Health | Active concentration metrics (*10%, SPF50+*) |

---

## 10. Project Contributors & Course Details
*   **Course:** Data Preprocessing (DS108.Q21) - University of Information Technology (UIT), VNU-HCM.
*   **Project Contributors:**
    *   Đỗ Hoàng Yến (Student ID: 24522068) - Department of Information Science and Engineering
    *   Trần Nguyễn Hoàng Yến (Student ID: 24522070) - Department of Information Science and Engineering
    *   Nguyễn Thị Hải Yến (Student ID: 24522069) - Department of Computer Engineering
*   **Dataset Repository:** [Hugging Face Datasets](https://huggingface.co/datasets/tnhyen/vietnamese-ecommerce-ner)

---

<a name="tiếng-việt"></a>
# Tiếng Việt

## 1. Tổng quan dự án & Công bố Sản phẩm
**ViEcomNER** là bộ dữ liệu gán nhãn Nhận dạng Thực thể có Tên (NER) chuẩn mực đầu tiên được công bố dưới dạng mã nguồn mở (open-source) dành riêng cho tiêu đề sản phẩm Thương mại Điện tử (TMĐT) tiếng Việt. Được thiết kế theo phương pháp tiếp cận **Data-Centric AI**, dự án giải quyết bài toán xử lý dữ liệu đặc thù mang tính "điện tín", ngữ pháp phi chuẩn, lạm dụng từ khóa (keyword stuffing) và hiện tượng trộn mã Anh-Việt (code-mixing) cực kỳ phức tạp trên các sàn TMĐT.

## 2. Đặt vấn đề & Động lực nghiên cứu
*   **Sự thiếu hụt dữ liệu chuyên biệt:** Các bộ dữ liệu NER tiếng Việt hiện tại (như VLSP, PhoNER_COVID19) hầu hết được xây dựng trên văn bản chuẩn mực (tin tức báo chí, y tế). Khi áp dụng vào ngôn ngữ tiêu đề sản phẩm TMĐT vốn mang tính khẩu ngữ và chắp vá, các mô hình học máy gặp hiện tượng suy giảm hiệu năng nghiêm trọng.
*   **Đặc thù nhiễu của tiêu đề sản phẩm TMĐT:** Tiêu đề TMĐT không tuân theo cấu trúc ngữ pháp chuẩn mà mang các đặc tính:
    *   **Ngôn ngữ điện tín** (lược bỏ từ nối, giới từ, chủ ngữ - vị ngữ).
    *   **Lạm dụng từ khóa (Keyword Stuffing)** để tối ưu hóa tìm kiếm (SEO).
    *   **Trộn mã ngôn ngữ (Code-mixing)** phức tạp (Ví dụ: *"Áo thun polo nam basic form rộng premium"*).
    *   **Sử dụng ký tự đặc biệt, biểu tượng cảm xúc (emoji)** tràn lan nhằm thu hút người mua.

## 3. Phương pháp luận & Đóng góp chính (Data-Centric AI)
1.  **Thiết kế Lược đồ phẳng (Flat Schema) & Độc lập phân rã (Tokenizer-Agnosticism):**
    *   Xây dựng hệ thống **24 nhãn nghiệp vụ** (15 nhãn chung, 9 nhãn chuyên biệt theo từng nhóm ngành).
    *   Dữ liệu được lưu trữ dưới dạng **tọa độ ký tự (character-level offset)** thay vì tokenized, giúp ngăn chặn hoàn toàn lỗi xô lệch ranh giới từ vựng do các công cụ tách từ tiếng Việt tự động gây ra.
2.  **Đường ống tiền xử lý tự động (9 bước):**
    *   **Bước 1:** Đọc dữ liệu và lưu log snapshot ban đầu.
    *   **Bước 2:** Khử trùng lặp tuyệt đối (drop exact duplicate trên tiêu đề gốc).
    *   **Bước 3:** Xử lý giá trị thiếu (impute brand trống thành 'No Brand', loại bỏ các dòng thiếu title/price).
    *   **Bước 4:** Chuẩn hóa Unicode NFC và giải mã thực thể HTML.
    *   **Bước 5:** Làm sạch nhiễu ký tự (xóa thẻ HTML, liên kết URL, emoji và ký tự rác).
    *   **Bước 6:** Bảo vệ thông tin cá nhân (PII) bằng cách che số điện thoại thông qua Regular Expression.
    *   **Bước 7:** Chuẩn hóa khoảng trắng dư thừa.
    *   **Bước 8:** Khử trùng lặp gần (Near-deduplication) chéo sàn sử dụng Cosine TF-IDF (ngưỡng tương đồng $\ge 0.90$) để triệt tiêu hiện tượng rò rỉ dữ liệu (data leakage).
    *   **Bước 9:** Chuẩn hóa nhãn thương hiệu và ánh xạ ngành hàng về 4 nhóm chính.
3.  **Quy trình gán nhãn lai kết hợp Người & Máy (Human-in-the-Loop):**
    *   Sử dụng mô hình ngôn ngữ lớn (LLM Claude Sonnet 4.6) để mồi nhãn sơ bộ (pre-annotation), giúp **tăng 60%** năng suất gán nhãn.
    *   **Kiểm soát thiên kiến mỏ neo (Anchor Bias):** Áp dụng cơ chế Fallback gán thủ công và hiệu đính mù đôi (Double-Blind Review).
    *   Triển khai quy trình **Blind Double Annotation** với 15% dữ liệu được phân phối chéo, cho phép đánh giá độ đồng thuận một cách khách quan nhất dưới sự giám sát của Administrator.

## 4. Quy mô & Thống kê dữ liệu
*   **Nguồn dữ liệu:** Tiki (31.9%) và Lazada (68.1%).
*   **Phân bổ ngành hàng:** 4 nhóm ngành chính:
    *   *Nhà cửa & Đời sống* (Home & Living)
    *   *Thời trang & Phụ kiện* (Fashion & Accessories)
    *   *Điện tử & Công nghệ* (Electronics & Technology)
    *   *Làm đẹp & Sức khỏe* (Beauty & Health)
*   **Quy mô:** Từ 19.361 bản ghi thô ban đầu, qua đường ống lọc sạch thu được **2.993 mẫu chuẩn Vàng (Gold Standard)** (phân bổ thành 2.396 Train, 298 Validation, 299 Test).
*   **Mật độ thực thể:** Rất cao, trung bình **6.29 thực thể/tiêu đề** (tổng số 18.837 thực thể).
*   **Mất cân bằng lớp:** Phản ánh đúng phân phối thực tế của thị trường TMĐT với tỷ lệ chênh lệch lên đến **133:1** (giữa nhãn phổ biến nhất là `PRODUCT_TYPE` và nhãn hiếm nhất là `SKIN_TYPE`).

## 5. Đánh giá chất lượng & Kết quả thực nghiệm
*   **Độ tin cậy chú giải:** Đạt ngưỡng "Đồng thuận gần như hoàn hảo" (Almost Perfect Agreement) trên tập Gold Test Set.
*   **Chỉ số Cohen's Kappa (không tính nhãn O):** Đạt **0.8207**, thể hiện sự nhất quán cao độ giữa các người gán nhãn.
*   **Exact Span F1-Score:** Đạt **81.67%**.
*   **Partial Match F1-Score:** Đạt **94.70%**, minh chứng cho việc các annotator đạt được sự đồng thuận cao về mặt ngữ nghĩa, các sai sót chủ yếu nằm ở ranh giới biên của từ ghép tiếng Việt.

---

## 6. Cấu trúc thư mục dự án
```
.
├── README.md               # Tài liệu dự án (Anh/Việt)
├── requirements.txt        # Danh sách thư viện phụ thuộc Python
├── data/
│   ├── raw/                # Dữ liệu thu thập gốc hoàn toàn nguyên bản (KHÔNG SỬA TAY)
│   │   ├── beauty/
│   │   ├── electronics/
│   │   ├── fashion/
│   │   └── home_living/
│   ├── merged/             # File dữ liệu gộp sau thu thập
│   ├── processed/          # File dữ liệu sạch sau tiền xử lý
│   └── annotated/          # Tập dữ liệu chuẩn Vàng (Train/Val/Test jsonl)
├── src/
│   ├── crawl_tiki_v2.py                  # Script thu thập dữ liệu Tiki
│   ├── crawl_lazada_v6.py                # Script thu thập dữ liệu Lazada
│   ├── cleaning_pipeline.py              # Thư viện pipeline làm sạch dữ liệu tự động
│   ├── prepare_labelstudio_import.py     # Chuyển đổi dữ liệu sang task Label Studio
│   ├── label_studio_config.xml           # Cấu hình giao diện Label Studio XML
│   ├── split_blind_double_annotation.py # Phân chia dữ liệu gán nhãn mù đôi chéo
│   ├── ls_export_to_iaa_jsonl.py         # Chuyển đổi Label Studio export sang định dạng tính IAA
│   ├── calculate_iaa.py                  # Tính độ đồng thuận IAA (Kappa, F1)
│   └── stratify_dev_split.py             # Phân chia tập Train/Val/Test phân tầng
├── notebooks/
│   ├── preprocessed_EDA.ipynb            # Phân tích dữ liệu sau khi tiền xử lý
│   ├── eda-gold-dataset.ipynb            # Phân tích tập dữ liệu Gold sau gán nhãn
│   └── stratified-sampling.ipynb         # Notebook thử nghiệm phân tầng lấy mẫu
└── demo/
    └── app.py                            # Ứng dụng Dashboard Streamlit tương tác
```

---

## 7. Yêu cầu hệ thống & Hướng dẫn cài đặt

### Yêu cầu hệ thống
*   Python từ phiên bản 3.8 trở lên.
*   Trình duyệt Google Chrome (nếu cần chạy script crawl Selenium).

### Hướng dẫn cài đặt
1.  **Clone dự án về máy:**
    ```bash
    git clone <repository_url>
    cd DS108.Q21
    ```

2.  **Khởi tạo và kích hoạt môi trường ảo:**
    *   **Sử dụng `venv` tiêu chuẩn:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate  # Trên macOS/Linux
        # Hoặc: venv\Scripts\activate  # Trên Windows
        ```
    *   **Sử dụng Conda:**
        ```bash
        conda create -n viecomner python=3.9 -y
        conda activate viecomner
        ```

3.  **Cài đặt các thư viện cần thiết:**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

4.  **Tải tài nguyên NLTK:**
    Script phân tầng sử dụng tập từ vựng tiếng Anh của NLTK để nhận diện ngôn ngữ. Tải trước bằng lệnh:
    ```bash
    python3 -c "import nltk; nltk.download('words')"
    ```

---

## 8. Quy trình chạy code từ đầu đến cuối (Tái lập kết quả)

> [!IMPORTANT]
> **Cam kết tính toàn vẹn dữ liệu:** Thư mục `data/raw/` chứa các tệp CSV gốc trực tiếp từ quá trình crawl sàn Tiki/Lazada. Để đảm bảo tính trung thực khoa học và tuân thủ tuyệt đối quy định môn học DS108, **không có bất kỳ sự can thiệp thủ công (Manual Manipulation) nào vào các tệp này**. Toàn bộ quy trình làm sạch, xóa trùng mờ, che PII đều được xử lý tự động thông qua code.

Để tái lập toàn bộ quy trình xây dựng bộ dữ liệu từ dữ liệu thô, thực hiện theo các bước sau:

### Bước 1: Thu thập dữ liệu (Không bắt buộc)
Dữ liệu crawl thô đã có sẵn trong `data/raw/`. Nếu bạn muốn tự chạy lại quá trình crawl:
```bash
# Crawl sàn Tiki
python3 src/crawl_tiki_v2.py
# Crawl sàn Lazada (Yêu cầu Chrome và Selenium)
python3 src/crawl_lazada_v6.py
```

### Bước 2: Gộp dữ liệu thô
Đảm bảo tất cả các file CSV thô được đặt đúng thư mục ngành hàng trong `data/raw/`. Tiến hành gộp các file này lại thành một file duy nhất đặt tại `data/merged/all_platforms_merged_v2.csv`.

### Bước 3: Chạy Pipeline Làm sạch & Khử trùng lặp chéo sàn
Chạy script tiền xử lý tự động nhằm loại bỏ nhiễu, che thông tin nhạy cảm và thực hiện thuật toán khử trùng mờ (TF-IDF character n-gram + Cosine Similarity):
```bash
python3 src/cleaning_pipeline.py \
  --input data/merged/all_platforms_merged_v2.csv \
  --output data/processed/data_cleaned.csv \
  --threshold 0.90
```

### Bước 4: Chuẩn bị Task và Chia mẫu gán nhãn mù đôi (Blind Double Annotation)
Chuyển đổi dữ liệu sạch sang định dạng Label Studio và chia nhỏ thành 4 tệp cho 4 người gán nhãn kèm 15% mẫu trùng lặp chéo:
```bash
# Chuyển CSV sạch sang Label Studio JSON Tasks
python3 src/prepare_labelstudio_import.py \
  --input data/processed/data_cleaned.csv \
  --output data/processed/ls_tasks.json

# Chia file gán nhãn chéo
python3 src/split_blind_double_annotation.py \
  --input data/processed/ls_tasks.json \
  --outdir data/processed/blind_splits/ \
  --seed 42 \
  --qc-ratio 0.15
```
*Kết quả sẽ tạo ra `A.json`, `B.json`, `C.json`, `D.json` trong thư mục `blind_splits/` để phân phối cho người gán nhãn, cùng file `blind_mapping.csv` cho Admin đối chiếu.*

### Bước 5: Tính độ đồng thuận gán nhãn (Inter-Annotator Agreement - IAA)
Sau khi hoàn tất gán nhãn trên giao diện Label Studio và xuất dữ liệu ra tệp JSON:
1.  **Chuyển đổi tệp xuất từ Label Studio sang định dạng JSONL chuẩn:**
    ```bash
    python3 src/ls_export_to_iaa_jsonl.py --input data/annotated/annotator_A_export.json --output data/annotated/A.jsonl --name Annotator_A
    python3 src/ls_export_to_iaa_jsonl.py --input data/annotated/annotator_B_export.json --output data/annotated/B.jsonl --name Annotator_B
    ```
2.  **Tính toán Kappa và các độ đo F1 (Exact/Partial):**
    ```bash
    python3 src/calculate_iaa.py \
      --a data/annotated/A.jsonl \
      --b data/annotated/B.jsonl \
      --outdir data/annotated/iaa_reports/
    ```

### Bước 6: Phân chia tập dữ liệu phân tầng (Stratified Resplitting)
Để tái lập tập Train, Validation, Test phân tầng nghiêm ngặt theo nhóm ngành, mức độ trộn ngôn ngữ Anh-Việt, định dạng viết hoa/thường, và độ dài tiêu đề:
```bash
python3 src/stratify_dev_split.py \
  --train-raw data/annotated/train.jsonl \
  --dev-raw data/annotated/validation.jsonl \
  --test-gold data/annotated/test.jsonl \
  --outdir data/annotated/
```
*Script sẽ kiểm tra các phân lớp hiếm, bù quota nếu thiếu, và ghi đè tập dữ liệu cuối cùng vào `train.jsonl` (2.396 mẫu), `validation.jsonl` (300 mẫu) và `test.jsonl` (299 mẫu) trong thư mục `data/annotated/`.*

### Bước 7: Khởi chạy Dashboard tương tác Streamlit
Để xem trực quan biểu đồ EDA, độ đồng thuận IAA và duyệt dữ liệu gán nhãn NER trực quan:
```bash
streamlit run demo/app.py
```

---

## 9. Danh mục 24 Nhãn Thực thể có Tên (Tagset)
Hệ thống nhãn của ViEcomNER gồm **24 thực thể**, chia thành **Common Labels** (15 nhãn áp dụng chung cho mọi ngành) và **Domain Labels** (9 nhãn chuyên biệt cho từng nhóm ngành).

| STT | Nhãn Thực thể | Phạm vi Ngành | Mô tả chi tiết / Ví dụ minh họa |
| :--- | :--- | :--- | :--- |
| 1 | `PRODUCT_TYPE` | Chung | Loại hoặc tên gọi của sản phẩm (*Áo thun, Nồi cơm điện*) |
| 2 | `BRAND` | Chung | Tên thương hiệu của sản phẩm (*Nike, Samsung, LocknLock*) |
| 3 | `MODEL` | Chung | Dòng máy, mã hiệu kỹ thuật (*iPhone 15 Pro Max, MX-200*) |
| 4 | `QUANTITY` | Chung | Số lượng mặt hàng đóng gói (*1 cặp, combo 3 cái, hộp 50 chiếc*) |
| 5 | `ORIGIN` | Chung | Nguồn gốc xuất xứ của sản phẩm (*Việt Nam, nội địa Trung*) |
| 6 | `COMPONENT` | Chung | Phụ kiện, linh kiện đi kèm hoặc hàng tặng (*cáp sạc đi kèm, tặng móc khóa*) |
| 7 | `MATERIAL` | Chung | Chất liệu vật lý sản xuất sản phẩm (*cotton, inox 304, lụa*) |
| 8 | `SIZE` | Chung | Kích cỡ, thông số kích thước vật lý (*M, L, XL, 45cm*) |
| 9 | `COMPAT` | Chung | Thiết bị tương thích phù hợp (*cho iPhone 14, tương thích iPad*) |
| 10 | `ATTRIBUTE` | Chung | Thuộc tính mô tả tính chất sản phẩm (*chống nước, không dây*) |
| 11 | `OCCASION` | Chung | Dịp, hoàn cảnh hoặc mùa vụ sử dụng (*đi tiệc, mùa hè, quà sinh nhật*) |
| 12 | `EFFECT` | Chung | Tác dụng, lợi ích mang lại cho cơ thể (*dưỡng ẩm, giảm mụn, trắng răng*) |
| 13 | `COLOR` | Chung | Màu sắc của sản phẩm (*xanh mint, đỏ đô, đen*) |
| 14 | `STYLE` | Chung | Phong cách, hoa văn, form dáng (*họa tiết sọc, form rộng, cổ V*) |
| 15 | `TARGET_GROUP` | Chung | Đối tượng khách hàng mục tiêu hướng tới (*cho nam, unisex, em bé*) |
| 16 | `SPEC` | Điện tử | Thông số kỹ thuật chuyên biệt điện tử (*50000mAh, 120Hz, 8GB/256GB*) |
| 17 | `CONNECTIVITY` | Điện tử | Chuẩn kết nối, truyền tải dữ liệu (*Bluetooth 5.3, Type-C, Wifi 6*) |
| 18 | `POWER` | Gia dụng | Công suất tiêu thụ điện hoặc điện áp (*2000W, 220V*) |
| 19 | `CAPACITY` | Gia dụng | Dung tích lòng nồi, tải trọng chứa đựng (*1.8L, 7.5kg*) |
| 20 | `INGREDIENT` | Mỹ phẩm & Sức khỏe | Thành phần hóa học, hoạt chất chính (*Niacinamide, Retinol*) |
| 21 | `SKIN_TYPE` | Mỹ phẩm & Sức khỏe | Loại da phù hợp của sản phẩm chăm sóc (*da dầu mụn, nhạy cảm*) |
| 22 | `BODY_PART` | Mỹ phẩm & Sức khỏe | Bộ phận cơ thể tác động (*cho tóc, da mặt, toàn thân*) |
| 23 | `VOLUME_WEIGHT`| Mỹ phẩm & Sức khỏe | Trọng lượng/thể tích đóng gói mỹ phẩm (*100ml, 50g*) |
| 24 | `CONCENTRATION`| Mỹ phẩm & Sức khỏe | Nồng độ hoạt chất hoặc chỉ số bảo vệ (*10%, SPF50+*) |

---

## 10. Thành viên tham gia & Thông tin học phần
*   **Môn học:** Tiền xử lý dữ liệu (DS108.Q21) - Trường Đại học Công nghệ Thông tin, Đại học Quốc gia Thành phố Hồ Chí Minh (UIT).
*   **Thành viên thực hiện:**
    *   Đỗ Hoàng Yến (MSSV: 24522068) - Khoa Khoa học và Kỹ thuật Thông tin
    *   Trần Nguyễn Hoàng Yến (MSSV: 24522070) - Khoa Khoa học và Kỹ thuật Thông tin
    *   Nguyễn Thị Hải Yến (MSSV: 24522069) - Khoa Kỹ thuật Máy tính
*   **Dự án:** Xây dựng và chuẩn hóa bộ dữ liệu Nhận dạng Thực thể có Tên dành riêng cho Tiêu đề sản phẩm Thương mại điện tử tiếng Việt (ViEcomNER).
*   **Công bố bộ dữ liệu tại:** [Hugging Face Datasets](https://huggingface.co/datasets/tnhyen/vietnamese-ecommerce-ner)
