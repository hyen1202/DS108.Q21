"""
stratify_dev_split.py
======================
Apply the strict stratified sampling logic from the user's notebook `l-y-m-u-ph-n-t-ng.ipynb`
to extract a highly representative Dev set of exactly 300 samples from the Silver pool (2,695 samples),
leaving 2,395 samples for the Train set.

This maintains:
  - Strict stratified quota for industry groups (electronics & tech, fashion & accessories, home & living, beauty & health)
  - Strict language profiles (using NLTK words for analyze_language_profile_v2)
  - Standardized JSON formatting
"""

import json
import re
import nltk
import pandas as pd
import numpy as np
from pathlib import Path

# Ensure NLTK words are downloaded
try:
    nltk.data.find('corpora/words')
except LookupError:
    print("Downloading NLTK words corpus...")
    nltk.download('words', quiet=True)

from nltk.corpus import words
english_vocab = set(w.lower() for w in words.words())

vietnamese_signatures = (
    r'[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ'
    r'ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĩỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ]'
)

# ── Clean text helper matching the notebook exactly ─────────────────────────
def clean_text_logic(text):
    if not isinstance(text, str):
        return ""
    import unicodedata
    import html
    text = unicodedata.normalize('NFC', text)
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'@[0-9\._\s\-\+]+', ' ', text)
    text = re.sub(r'(?<!\d)(0|\+84)\d{9,10}(?!\d)', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'_', ' ', text)
    text = re.sub(r'[^\w\s,.\/\"\'\–%\‑\°×\+&:*\″\”\℃\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ── Feature Extractors matching the notebook exactly ─────────────────────────
def analyze_language_profile_v2(text):
    if not isinstance(text, str) or text.strip() == "":
        return "Unknown"
    tokens = [t.lower() for t in re.findall(r'\b\w+\b', text)]
    vi_count, en_count = 0, 0
    for token in tokens:
        if re.match(r'^\d+\w*$', token) or re.match(r'^\w*\d+$', token):
            continue
        if re.search(vietnamese_signatures, token):
            vi_count += 1
        elif token in english_vocab:
            en_count += 1
        elif token in {'cho', 'cua', 'va', 'may', 'tai', 'op', 'lung', 'ao', 'quan', 'giay', 'dep', 'bo', 'treo', 'dan', 'tu', 'gia'}:
            vi_count += 1
    total_valid_words = vi_count + en_count
    if total_valid_words == 0:
        return "Chỉ chứa số/Ký tự nhiễu"
    r_en_pure = en_count / total_valid_words
    if r_en_pure >= 0.90:
        return "Thuần Anh"
    elif r_en_pure <= 0.10:
        return "Thuần Việt"
    elif r_en_pure > 0.50:
        return "Trộn lẫn (Thiên Anh)"
    else:
        return "Trộn lẫn (Thiên Việt)"

def get_case_bucket(text):
    t = str(text).strip()
    if t.isupper():
        return 'All Caps'
    elif t.islower():
        return 'Lower Case'
    else:
        return 'Mixed Case'

def get_length_bucket(text):
    n = len(str(text).split())
    if n <= 12:
        return 'Ngắn'
    elif n <= 20:
        return 'Trung bình'
    elif n <= 30:
        return 'Dài'
    else:
        return 'Rất dài'

# ── Stratified Sampling matching the notebook exactly ──────────────────────
def perform_strict_stratified_sampling(df_source, target_samples, strict_targets=None, random_seed=42):
    df_working = df_source.copy().reset_index(drop=True)
    np.random.seed(random_seed)
    
    if strict_targets is None:
        strict_targets = ['Thuần Anh', 'All Caps', 'Lower Case']

    df_working['title_clean'] = df_working['text'].apply(clean_text_logic)
    df_working['feat_len'] = df_working['title_clean'].apply(get_length_bucket)
    df_working['feat_lang'] = df_working['title_clean'].apply(analyze_language_profile_v2)
    df_working['feat_case'] = df_working['title_clean'].apply(get_case_bucket)

    # 1. Base Quota calculation based on overall ratio (300 target vs 3000 baseline)
    base_ratio = (target_samples * 0.90) / 3000
    industry_quotas = {
        'electronics & tech': int(837 * base_ratio),
        'home & living': int(907 * base_ratio),
        'fashion & accessories': int(789 * base_ratio),
        'beauty & health': int(467 * base_ratio)
    }
    
    base_pool_df = pd.DataFrame()
    buffer_pool_df = pd.DataFrame()

    for industry, quota in industry_quotas.items():
        ind_df = df_working[df_working['industry_group'] == industry]
        shuffled_ind = ind_df.sample(frac=1, random_state=random_seed)
        
        base_pool_df = pd.concat([base_pool_df, shuffled_ind.head(quota)])
        buffer_pool_df = pd.concat([buffer_pool_df, shuffled_ind.iloc[quota:]])
        
    base_pool_df = base_pool_df.reset_index(drop=True)
    buffer_pool_df = buffer_pool_df.reset_index(drop=True)

    # 2. Strict targets check and fill (minimum of 20 samples per strict target, or maximum available)
    global_targets = [
        ('feat_lang', 'Thuần Anh'), ('feat_lang', 'Thuần Việt'),
        ('feat_lang', 'Trộn lẫn (Thiên Anh)'), ('feat_lang', 'Trộn lẫn (Thiên Việt)'),
        ('feat_case', 'All Caps'), ('feat_case', 'Lower Case'), ('feat_case', 'Mixed Case'),
        ('feat_len', 'Ngắn'), ('feat_len', 'Trung bình'), ('feat_len', 'Dài'), ('feat_len', 'Rất dài')
    ]

    print("="*105)
    print(f"📋 TABLE 1: REVIEW AND FILLING FOR STRATIFIED DEV SET ({target_samples} SAMPLES)")
    print("="*105)
    print(f"{'Rare Stratum Target'.ljust(22)} | {'Initial Count'.center(20)} | {'Action Taken'.center(22)} | {'Temporary Count'.center(16)} | {'Status'}")
    print("-" * 105)

    for feat_col, cat_value in global_targets:
        match_condition = base_pool_df[feat_col] == cat_value
        current_global_cnt = len(base_pool_df[match_condition])
        
        action = "Keep Natural"
        final_global_cnt = current_global_cnt
        
        if cat_value in strict_targets:
            # For 300 target samples, minimum floor of 6 samples is representative 
            # (scaling down from 20 samples in 1000 baseline). We will enforce at least 6 samples.
            target_floor = 6
            if current_global_cnt < target_floor:
                needed = target_floor - current_global_cnt
                match_buffer = buffer_pool_df[buffer_pool_df[feat_col] == cat_value]
                
                if len(match_buffer) >= needed:
                    chosen_buffer = match_buffer.head(needed)
                    base_pool_df = pd.concat([base_pool_df, chosen_buffer]).reset_index(drop=True)
                    buffer_pool_df = buffer_pool_df.drop(chosen_buffer.index).reset_index(drop=True)
                    final_global_cnt = current_global_cnt + needed
                    action = f"Fill Floor (+{needed})"
                else:
                    available_in_buffer = len(match_buffer)
                    if available_in_buffer > 0:
                        base_pool_df = pd.concat([base_pool_df, match_buffer]).reset_index(drop=True)
                        buffer_pool_df = buffer_pool_df.drop(match_buffer.index).reset_index(drop=True)
                        needed -= available_in_buffer
                    
                    global_remain_raw = df_working[~df_working.index.isin(base_pool_df.index) & (df_working[feat_col] == cat_value)]
                    if len(global_remain_raw) > 0:
                        force_take = min(needed, len(global_remain_raw))
                        chosen_force = global_remain_raw.head(force_take)
                        base_pool_df = pd.concat([base_pool_df, chosen_force]).reset_index(drop=True)
                        buffer_pool_df = buffer_pool_df[~buffer_pool_df['id'].isin(chosen_force['id'])].reset_index(drop=True)
                        final_global_cnt = current_global_cnt + available_in_buffer + force_take
                        action = f"Force Extract (+{available_in_buffer + force_take})"
                    else:
                        final_global_cnt = current_global_cnt + available_in_buffer
                        action = f"Exhausted (+{available_in_buffer})"
            elif current_global_cnt >= target_floor:
                action = "Keep (Sufficient)"
                final_global_cnt = current_global_cnt
        
        status = "✅ SAFE" if final_global_cnt >= 6 or cat_value not in strict_targets else "⚠️ UNDER"
        print(f"{cat_value.ljust(22)} | {str(current_global_cnt).center(20)} | {action.center(22)} | {str(final_global_cnt).center(16)} | {status}")
    print("-" * 105)

    # 3. Balancing remaining quota back to target samples (60% Electrics : 40% Beauty)
    current_total = len(base_pool_df)
    leftover_quota = target_samples - current_total

    if leftover_quota > 0:
        elec_share = int(leftover_quota * 0.60)
        beauty_share = leftover_quota - elec_share
        
        print(f"⚖️ REBALANCING: Adding +{elec_share} Electrics (60%) and +{beauty_share} Beauty (40%) to reach {target_samples}...")

        for target_ind, share_count in [('electronics & tech', elec_share), ('beauty & health', beauty_share)]:
            match_remain_buffer = buffer_pool_df[buffer_pool_df['industry_group'] == target_ind]
            take_share_cnt = min(share_count, len(match_remain_buffer))
            chosen_share = match_remain_buffer.head(take_share_cnt)
                
            base_pool_df = pd.concat([base_pool_df, chosen_share]).reset_index(drop=True)
            buffer_pool_df = buffer_pool_df.drop(chosen_share.index).reset_index(drop=True)
    
    elif leftover_quota < 0:
        excess = abs(leftover_quota)
        print(f"⚠️ Excess detected! Removing {excess} samples from base pool...")
        base_pool_df = base_pool_df.drop(base_pool_df.index[:excess]).reset_index(drop=True)

    # 4. Save metrics & distribution
    print("\n" + "="*105)
    print("🎯 FINAL STRATIFIED DISTRIBUTION FOR DEV SET:")
    print("="*105)
    ind_counts = base_pool_df['industry_group'].value_counts()
    for k, v in ind_counts.items():
        print(f"  - {k.ljust(22)}: {v} samples ({v/len(base_pool_df)*100:.2f}%)")
    print(f"  --> TOTAL DEV SAMPLES EXTRACTED: {len(base_pool_df)}")
    print("="*105 + "\n")

    drop_cols = ['feat_len', 'feat_lang', 'feat_case', 'title_clean']
    final_cols = [c for c in base_pool_df.columns if c not in drop_cols]
    
    df_result = base_pool_df[final_cols]
    df_remaining = df_working[~df_working['id'].isin(df_result['id'])][final_cols].reset_index(drop=True)
    
    return df_result, df_remaining

# ── Main Runner ──────────────────────────────────────────────────────────────
def main():
    GOLD_DIR = Path(__file__).parent
    
    # 1. Load the original raw Silver dataset (2,695 records) to ensure clean re-stratification
    # If not present, we will load train_raw + dev_raw combined.
    silver_records = []
    
    train_raw_path = GOLD_DIR / "train_raw.jsonl"
    dev_raw_path = GOLD_DIR / "dev_raw.jsonl"
    
    with open(train_raw_path, encoding="utf-8") as f:
        silver_records.extend([json.loads(line) for line in f])
    with open(dev_raw_path, encoding="utf-8") as f:
        silver_records.extend([json.loads(line) for line in f])
        
    df_silver = pd.DataFrame(silver_records)
    df_silver = df_silver.drop_duplicates(subset=['id']).reset_index(drop=True)
    print(f"Total Silver pool loaded: {len(df_silver)} records.")

    # 2. Extract stratified 300 samples for dev, leaving 2,395 for train
    df_dev_strat, df_train_strat = perform_strict_stratified_sampling(
        df_source=df_silver, 
        target_samples=300, 
        random_seed=42
    )

    # 3. Convert back to JSON records and save
    dev_records = df_dev_strat.to_dict(orient='records')
    train_records = df_train_strat.to_dict(orient='records')

    # Update metadata split tags
    for r in dev_records:
        r['split'] = 'dev'
        r['is_iaa_sample'] = False
    for r in train_records:
        r['split'] = 'train'
        r['is_iaa_sample'] = False

    # Load Gold 300 test set to keep the full dataset complete
    test_records = []
    with open(GOLD_DIR / "test_gold_300.jsonl", encoding="utf-8") as f:
        for line in f:
            test_records.append(json.loads(line))

    # Save individual files
    def save_jsonl(records, path):
        with open(path, 'w', encoding='utf-8') as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f"  Saved {len(records)} records → {path.name}")

    save_jsonl(train_records, GOLD_DIR / "train_raw.jsonl")
    save_jsonl(dev_records, GOLD_DIR / "dev_raw.jsonl")
    save_jsonl(test_records, GOLD_DIR / "test_gold_300.jsonl")

    # Combine into full_dataset
    full_dataset = train_records + dev_records + test_records
    save_jsonl(full_dataset, GOLD_DIR / "full_dataset.jsonl")

    # Save splits for the NER training workflow
    # train_full = train_records
    # dev_full = dev_records (or dev_records + test_records depending on user workflow, we keep 10% dev format)
    save_jsonl(train_records, GOLD_DIR / "train_full.jsonl")
    save_jsonl(dev_records, GOLD_DIR / "dev_full.jsonl")

    print("\n✅ Stratified Train/Dev re-splitting successfully completed!")

if __name__ == "__main__":
    main()
