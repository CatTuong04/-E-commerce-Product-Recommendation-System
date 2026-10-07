# ============================================================
# MongoDB Initialization: Import data from JSON + PKL files
# ============================================================
# Usage: cd web && python init_db.py
#
# Collections created:
#   - products: {asin, title, brand, price, image, description, category}
#   - reviews:  {user_id, asin, rating, text, summary, timestamp}
#   - recommendations: {user_id, items: [{asin, score, rank}]}

import json
import html
import re
import pickle
from pymongo import MongoClient, ASCENDING, DESCENDING
from tqdm import tqdm


# ============================================================
# Known beauty brands for auto-extraction from title
# ============================================================
KNOWN_BRANDS = {
    'crabtree & evelyn', 'ahava', 'oribe', 'la mer', 'tom ford', 'sk-ii',
    'chanel', 'estee lauder', 'dior', 'ysl', 'yves saint laurent',
    'calvin klein', 'opi', 'revision skincare', 'revision', 'hot tools',
    'mario badescu', 'essie', 'butter london', 'toppik', 'dermablend',
    'cnd', 'zwilling', 'zwilling j.a. henckels', 'proraso', 'unite',
    'unite hair', 'bliss', 'supersmile', 'archipelago', 'phyto',
    'h2o+', 'h2o+ beauty', 'nars', 'clinique', 'lancome', 'bobbi brown',
    'mac', 'bare minerals', 'bareminerals', 'urban decay', 'too faced',
    'benefit', 'murad', 'origins', 'philosophy', 'peter thomas roth',
    'tatcha', 'fresh', 'jack black', "kiehl's", 'bumble and bumble',
    'aveda', 'moroccanoil', 'redken', 'biolage', 'paul mitchell', 'chi',
    'bed head', 'tigi', 'kenra', 'joico', 'sexy hair', 'it cosmetics',
    'nyx', 'revlon', "l'oreal", 'neutrogena', 'cetaphil', 'cerave',
    'la roche-posay', 'vichy', 'bioderma', 'drunk elephant',
    'sunday riley', 'the ordinary', "paula's choice", 'skinceuticals',
    'obagi', 'strivectin', 'perricone md', 'dermalogica', 'clarins',
    'shiseido', 'elizabeth arden', 'helen of troy', 'conair',
    'remington', 'babyliss', 'ghd', 'zoya', 'sally hansen', 'orly',
    'china glaze', 'gelish', 'deborah lippmann', 'julep', 'smith & cult',
    'molton brown', 'acqua di parma', 'jo malone', 'byredo', 'diptyque',
    'nest fragrances', 'tocca', 'lafco', 'voluspa', 'archipelago botanicals',
    'thymes', 'pre de provence', 'crabtree', 'roger & gallet',
}


def extract_brand(title):
    """Extract brand name from product title."""
    t = title.strip()
    tl = t.lower()

    # Check known brands (longest match first)
    for brand in sorted(KNOWN_BRANDS, key=len, reverse=True):
        if tl.startswith(brand):
            return t[:len(brand)]

    # Pattern: 'Brand - Product...'
    m = re.match(r"^([A-Z][\w&.'+]+(?:\s+(?:&|and)\s+[A-Z][\w&.'+]+)?(?:\s+[A-Z][\w.]+)?)\s*[-\u2013]\s", t)
    if m:
        return m.group(1).strip()

    # Pattern: 'ALLCAPS words' at start (like AHAVA, TOPPIK, OPI)
    m = re.match(r'^([A-Z][A-Z.+]+(?:\s+[A-Z][A-Z.+]+)?)\s', t)
    if m and len(m.group(1)) >= 2:
        return m.group(1).strip()

    # Pattern: 'Brand Name Product...' - capitalized words before lowercase
    m = re.match(r"^([A-Z][\w.'+]+(?:\s+[A-Z][\w.'+]+)?)\s+(?:[A-Z][a-z]|[a-z])", t)
    if m:
        candidate = m.group(1).strip()
        skip = {'The', 'New', 'Set', 'Pro', 'Kit', 'Pack', 'Box', 'Day',
                'Night', 'Gel', 'Oil', 'Soy', 'Sea', 'Gold', 'Pure', 'Skin',
                'Hair', 'Nail', 'Face', 'Body', 'Eye', 'Lip', 'Sun'}
        if candidate not in skip and len(candidate) >= 2:
            return candidate

    return ''


def clean_html(text):
    """Remove HTML tags and decode entities."""
    if not text:
        return ''
    text = re.sub(r'<[^>]+>', ' ', text)       # strip HTML tags
    text = html.unescape(text)                   # decode &amp; etc.
    text = re.sub(r'&\w+;', ' ', text)           # leftover entities
    text = re.sub(r'\s+', ' ', text).strip()     # collapse whitespace
    return text


def clean_details(details):
    """Clean product details dict: trim keys, remove junk entries."""
    if not details:
        return {}
    cleaned = {}
    skip_keys = {'shipping weight', 'domestic shipping', 'international shipping',
                 'asin', 'date first available', 'discontinued by manufacturer',
                 'is discontinued by manufacturer'}
    for k, v in details.items():
        ck = k.strip().rstrip(':').strip()
        cv = v.strip() if isinstance(v, str) else str(v)
        # Skip useless entries
        if ck.lower() in skip_keys:
            continue
        if len(ck) > 30 or len(cv) > 80 or not ck:
            continue
        # Clean trailing junk like '(' from "5.6 ounces ("
        cv = re.sub(r'\s*\(\s*$', '', cv)
        cleaned[ck] = cv
    return cleaned

DB_NAME = 'luxury_beauty'

def get_db():
    client = MongoClient('mongodb://localhost:27017/')
    return client[DB_NAME]


def import_products(db):
    """Import product metadata from JSON."""
    print("\n--- Importing products ---")
    db.products.drop()

    seen_asins = set()
    products = []
    with open('../data/meta_Luxury_Beauty.json', 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading metadata"):
            try:
                meta = json.loads(line.strip())
                asin = meta.get('asin')
                if not asin or asin in seen_asins:
                    continue
                seen_asins.add(asin)

                images = meta.get('imageURLHighRes', []) or meta.get('imUrl', [])
                image = ''
                if isinstance(images, list) and images:
                    image = images[0]
                elif isinstance(images, str):
                    image = images

                raw_price = meta.get('price', '')
                price = ''
                if isinstance(raw_price, str) and raw_price.startswith('$') and len(raw_price) < 30:
                    price = raw_price

                desc = meta.get('description', [])
                if isinstance(desc, list):
                    desc = ' '.join(desc)
                desc = clean_html(desc)

                title = clean_html(meta.get('title', ''))

                # Extract brand: use metadata field first, fallback to title
                brand = html.unescape(meta.get('brand', '')).strip()
                if not brand:
                    brand = extract_brand(title)

                products.append({
                    'asin': asin,
                    'title': title,
                    'brand': brand,
                    'price': price,
                    'image': image,
                    'description': desc[:500] if desc else '',
                    'category': meta.get('category', []),
                    'also_buy': meta.get('also_buy', []),
                    'also_view': meta.get('also_view', []),
                    'details': clean_details(meta.get('details', {})),
                })
            except:
                pass

    if products:
        db.products.insert_many(products)
    db.products.create_index('asin', unique=True)
    print(f"[OK] Imported {len(products):,} products")


def import_reviews(db):
    """Import reviews from JSON."""
    print("\n--- Importing reviews ---")
    db.reviews.drop()

    seen_reviews = set()
    reviews = []
    with open('../data/Luxury_Beauty_5.json', 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading reviews"):
            try:
                r = json.loads(line.strip())
                key = (r.get('reviewerID'), r.get('asin'), r.get('unixReviewTime'))
                if key in seen_reviews:
                    continue
                seen_reviews.add(key)
                reviews.append({
                    'user_id': r.get('reviewerID'),
                    'asin': r.get('asin'),
                    'rating': r.get('overall'),
                    'text': html.unescape(r.get('reviewText', '')),
                    'summary': html.unescape(r.get('summary', '')),
                    'timestamp': r.get('unixReviewTime'),
                })
            except:
                pass

    if reviews:
        db.reviews.insert_many(reviews)
    db.reviews.create_index([('user_id', ASCENDING), ('timestamp', DESCENDING)])
    db.reviews.create_index('asin')
    print(f"[OK] Imported {len(reviews):,} reviews")

    # Compute aggregate ratings per product
    print("Computing product ratings...")
    pipeline = [
        {'$group': {
            '_id': '$asin',
            'avg_rating': {'$avg': '$rating'},
            'review_count': {'$sum': 1}
        }}
    ]
    for doc in db.reviews.aggregate(pipeline):
        db.products.update_one(
            {'asin': doc['_id']},
            {'$set': {
                'avg_rating': round(doc['avg_rating'], 1),
                'review_count': doc['review_count']
            }}
        )
    print("[OK] Product ratings updated")


def import_recommendations(db):
    """Import precomputed recommendations from ranking_output.pkl."""
    print("\n--- Importing recommendations ---")
    db.recommendations.drop()

    with open('../model/ranking_output.pkl', 'rb') as f:
        ranking_output = pickle.load(f)

    user_top20 = ranking_output['user_top20']
    test_ground_truth = ranking_output['test_ground_truth']
    user_train_items = ranking_output['user_train_items']

    docs = []
    for user_id, items in tqdm(user_top20.items(), desc="Building recommendations"):
        rec_items = []
        for rank, (asin, score) in enumerate(items[:20], 1):
            rec_items.append({
                'asin': asin,
                'score': round(float(score), 4),
                'rank': rank
            })

        docs.append({
            'user_id': user_id,
            'items': rec_items,
            'ground_truth': test_ground_truth.get(user_id, []),
            'train_items': user_train_items.get(user_id, []),
        })

    if docs:
        db.recommendations.insert_many(docs)
    db.recommendations.create_index('user_id', unique=True)
    print(f"[OK] Imported recommendations for {len(docs):,} users")


def import_users(db):
    """Import all reviewers from dataset as user accounts."""
    print("\n--- Importing users ---")
    db.users.drop()
    db.users.create_index('username', unique=True)
    db.users.create_index('reviewer_id')

    from werkzeug.security import generate_password_hash

    # Collect reviewer info from reviews JSON
    reviewer_names = {}  # {reviewer_id: reviewer_name}
    with open('../data/Luxury_Beauty_5.json', 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading reviewer info"):
            try:
                r = json.loads(line.strip())
                rid = r.get('reviewerID')
                name = r.get('reviewerName', '')
                if rid and rid not in reviewer_names:
                    reviewer_names[rid] = name
            except:
                pass

    # Create user accounts (username = reviewer_id, password = 1234)
    default_hash = generate_password_hash('1234')
    users = []
    for rid, name in tqdm(reviewer_names.items(), desc="Creating users"):
        users.append({
            'username': rid,
            'password_hash': default_hash,
            'reviewer_id': rid,
            'display_name': name or rid,
        })

    if users:
        db.users.insert_many(users)

    # Create admin account
    db.users.insert_one({
        'username': 'admin',
        'password_hash': generate_password_hash('admin123'),
        'reviewer_id': None,
        'display_name': 'Admin',
        'is_admin': True,
    })

    print(f"[OK] Imported {len(users):,} users (password: 1234 for all)")
    print(f"[OK] Admin account created (admin / admin123)")


if __name__ == '__main__':
    print("=" * 60)
    print("INITIALIZING MongoDB DATABASE")
    print("=" * 60)

    db = get_db()

    import_products(db)
    import_reviews(db)
    import_recommendations(db)
    import_users(db)

    # Print summary
    print("\n" + "=" * 60)
    print("DATABASE READY")
    print("=" * 60)
    print(f"  Products:        {db.products.count_documents({}):,}")
    print(f"  Reviews:         {db.reviews.count_documents({}):,}")
    print(f"  Recommendations: {db.recommendations.count_documents({}):,}")
    print(f"  Users:           {db.users.count_documents({}):,}")
