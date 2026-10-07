# ============================================================
# SEMANTIC FEATURES: Transformer Embeddings for Ranking
# ============================================================
# Dùng SentenceTransformer (all-MiniLM-L6-v2) để encode:
#   - User profile text → user_embedding (384-dim)
#   - Item description text → item_embedding (384-dim)
# Output: ../model/semantic_embeddings.pkl

import os
os.environ['USE_TF'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

print("[OK] All libraries imported successfully!")

# ============================================================
# 1. LOAD DATA (self-contained, no dependency on ranking output)
# ============================================================
print("\n" + "="*60)
print("LOADING DATA")
print("="*60)

# Load reviews
review_file = '../data/Luxury_Beauty_5.json'
print(f"Reading reviews from: {review_file}")
reviews = []
with open(review_file, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            r = json.loads(line.strip())
            reviews.append({
                'user_id': r.get('reviewerID'),
                'item_id': r.get('asin'),
                'timestamp': r.get('unixReviewTime'),
            })
        except:
            pass

df = pd.DataFrame(reviews)
print(f"[OK] Loaded {len(df):,} raw reviews")

# Deduplicate (same review crawled multiple times)
before_dedup = len(df)
df = df.drop_duplicates(subset=['user_id', 'item_id', 'timestamp'], keep='first')
print(f"[OK] After dedup: {len(df):,} reviews (removed {before_dedup - len(df):,} duplicates)")

df = df.sort_values('timestamp').reset_index(drop=True)
df = df.dropna(subset=['user_id', 'item_id', 'timestamp'])

# Apply same filtering as retrieval (min_user=5, min_item=3)
min_user_interactions = 5
min_item_interactions = 3

user_counts = df['user_id'].value_counts()
df = df[df['user_id'].isin(user_counts[user_counts >= min_user_interactions].index)]
item_counts = df['item_id'].value_counts()
df = df[df['item_id'].isin(item_counts[item_counts >= min_item_interactions].index)]

# Train/test split matching retrieval (leave-1-out / leave-2-out)
df = df.sort_values(['user_id', 'timestamp'])
user_train_items = {}
for user_id, user_df in df.groupby('user_id'):
    n = len(user_df)    # Count item user buy
    n_test = 1 if n <= 7 else 2
    train_items = user_df.iloc[:-n_test]['item_id'].tolist()
    user_train_items[user_id] = train_items

print(f"[OK] Loaded {len(df):,} interactions ({df['user_id'].nunique():,} users, {df['item_id'].nunique():,} items)")
print(f"[OK] Built user train items: {len(user_train_items)} users")

# Load metadata
meta_file = '../data/meta_Luxury_Beauty.json'
meta_dict = {}
with open(meta_file, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            meta = json.loads(line.strip())
            item_id = meta.get('asin')
            price_str = meta.get('price', '')
            price = None
            if price_str and price_str.startswith('$'):
                try:
                    price = float(price_str.replace('$', '').replace(',', ''))
                except:
                    pass
            meta_dict[item_id] = {
                'title': meta.get('title', ''),
                'brand': meta.get('brand', ''),
                'price': price,
                'description': meta.get('description', []),
            }
        except:
            pass

print(f"[OK] Loaded {len(meta_dict):,} product metadata")

# ============================================================
# 2. BUILD TEXT REPRESENTATIONS
# ============================================================
print("\n" + "="*60)
print("BUILDING TEXT REPRESENTATIONS")
print("="*60)

def build_item_text(item_id):
    """Build item description text from metadata."""
    if item_id not in meta_dict:
        return f"Product {item_id}"
    m = meta_dict[item_id]
    parts = []
    if m['title']:
        parts.append(m['title'])
    if m['brand']:
        parts.append(f"Brand: {m['brand']}")
    if m['price'] and m['price'] > 0:
        parts.append(f"Price: ${m['price']:.2f}")
    if m['description']:
        desc = " ".join(m['description'])[:200]
        if desc:
            parts.append(desc)
    return ". ".join(parts) if parts else f"Product {item_id}"


def build_user_profile(user_id, max_items=10):
    """Build user profile text from purchase history."""
    items = user_train_items.get(user_id, [])
    recent_items = items[-max_items:]
    titles = []
    brands = set()
    for iid in recent_items:
        if iid in meta_dict:
            m = meta_dict[iid]
            if m['title']:
                titles.append(m['title'])
            if m['brand']:
                brands.add(m['brand'])
    profile = "Customer who purchased: " + "; ".join(titles)
    if brands:
        profile += ". Preferred brands: " + ", ".join(list(brands)[:5])
    return profile


# Collect all unique item IDs from user histories
all_item_ids = list(set(
    iid for items in user_train_items.values() for iid in items
))
all_user_ids = list(user_train_items.keys())

print(f"Users to encode: {len(all_user_ids)}")
print(f"Items to encode: {len(all_item_ids)}")

# Build texts
print("\nBuilding user profile texts...")
user_texts = {uid: build_user_profile(uid) for uid in tqdm(all_user_ids, desc="User profiles")}

print("Building item texts...")
item_texts = {iid: build_item_text(iid) for iid in tqdm(all_item_ids, desc="Item texts")}

# ============================================================
# 3. ENCODE WITH SENTENCE TRANSFORMER
# ============================================================
print("\n" + "="*60)
print("ENCODING WITH SENTENCE TRANSFORMER")
print("="*60)

model_name = 'all-MiniLM-L6-v2'
print(f"Loading model: {model_name}")
model = SentenceTransformer(model_name)
embedding_dim = model.get_sentence_embedding_dimension()
print(f"[OK] Model loaded. Embedding dim: {embedding_dim}")

# Encode users
print(f"\nEncoding {len(all_user_ids)} user profiles...")
user_text_list = [user_texts[uid] for uid in all_user_ids]
user_emb_matrix = model.encode(user_text_list, show_progress_bar=True, batch_size=64)
user_embeddings = {uid: user_emb_matrix[i] for i, uid in enumerate(all_user_ids)}
print(f"[OK] User embeddings: {len(user_embeddings)} users x {embedding_dim} dim")

# Encode items
print(f"\nEncoding {len(all_item_ids)} items...")
item_text_list = [item_texts[iid] for iid in all_item_ids]
item_emb_matrix = model.encode(item_text_list, show_progress_bar=True, batch_size=64)
item_embeddings = {iid: item_emb_matrix[i] for i, iid in enumerate(all_item_ids)}
print(f"[OK] Item embeddings: {len(item_embeddings)} items x {embedding_dim} dim")

# ============================================================
# 4. SAVE
# ============================================================
print("\n" + "="*60)
print("SAVING EMBEDDINGS")
print("="*60)

output = {
    'user_embeddings': user_embeddings,
    'item_embeddings': item_embeddings,
    'model_name': model_name,
    'embedding_dim': embedding_dim,
}

output_path = '../model/semantic_embeddings.pkl'
with open(output_path, 'wb') as f:
    pickle.dump(output, f)

file_size = os.path.getsize(output_path) / (1024 * 1024)
print(f"[OK] Saved to: {output_path} ({file_size:.1f} MB)")
print(f"     Users: {len(user_embeddings)}")
print(f"     Items: {len(item_embeddings)}")
print(f"     Model: {model_name}")
print(f"     Dim: {embedding_dim}")

# Quick sanity check: show cosine similarity for a sample user
sample_uid = all_user_ids[0]
sample_items = user_train_items[sample_uid][-3:]
u_emb = user_embeddings[sample_uid]
print(f"\nSanity check - User: {sample_uid}")
for iid in sample_items:
    if iid in item_embeddings:
        i_emb = item_embeddings[iid]
        cos_sim = np.dot(u_emb, i_emb) / (np.linalg.norm(u_emb) * np.linalg.norm(i_emb))
        title = (meta_dict.get(iid, {}).get('title', '') or '')[:40]
        print(f"  {iid} ({title}): cosine = {cos_sim:.4f}")

print(f"\n[DONE] Semantic features ready!")
