# ============================================================
# WEB DEMO: Luxury Beauty Recommendation System
# ============================================================
# Full-featured Flask app with MongoDB backend
# Usage: cd web && python app.py

import os
os.environ['USE_TF'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_NO_TF'] = '1'

import json
import math
import pickle
import random
import time
from datetime import datetime
import numpy as np
from functools import wraps
from flask import (Flask, render_template, jsonify, request,
                   redirect, url_for, session, flash, abort)
from bson import ObjectId
from pymongo import MongoClient, DESCENDING
from werkzeug.security import generate_password_hash, check_password_hash
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '..', 'model')

app = Flask(__name__)
app.secret_key = 'luxury_beauty_kltn_2025'

PER_PAGE = 10  # items per page for pagination

# ============================================================
# DATABASE
# ============================================================

client = MongoClient('mongodb://localhost:27017/')
db = client['luxury_beauty']

# ============================================================
# LOAD ML MODELS (once at startup)
# ============================================================

print("=" * 60)
print("LOADING ML MODELS...")
print("=" * 60)

with open(os.path.join(MODEL_DIR, 'semantic_embeddings.pkl'), 'rb') as f:
    emb_data = pickle.load(f)

item_embeddings = emb_data['item_embeddings']
model_name = emb_data['model_name']

search_item_ids = list(item_embeddings.keys())
item_matrix = np.array([item_embeddings[iid] for iid in search_item_ids])
item_norms = np.linalg.norm(item_matrix, axis=1, keepdims=True)
item_matrix_normed = item_matrix / item_norms
item_id_to_idx = {iid: idx for idx, iid in enumerate(search_item_ids)}

print(f"[OK] Search index: {len(search_item_ids):,} items x {emb_data['embedding_dim']}-dim")

print(f"Loading SentenceTransformer ({model_name})...")
search_model = SentenceTransformer(model_name)
print("[OK] Search model loaded!")

with open(os.path.join(MODEL_DIR, 'model_info_luxury_beauty.json'), 'r') as f:
    model_info = json.load(f)

# Load per-system top-20 outputs for the demo comparison page.
# Each pkl is produced by train/build_demo_outputs.py.
DEMO_SYSTEMS = [
    ('baseline', 'Baseline (retrieval + ranking)', 'recs_baseline.pkl'),
    ('huim', 'Baseline + HUIM (retrieval)', 'recs_huim.pkl'),
    ('huim_sem', 'Baseline + HUIM (retrieval) + Semantic (ranking)', 'recs_huim_sem.pkl'),
]
demo_outputs = {}
for key, label, fname in DEMO_SYSTEMS:
    path = os.path.join(MODEL_DIR, fname)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            demo_outputs[key] = pickle.load(f)
        n = len(demo_outputs[key].get('user_top20', {}))
        print(f"[OK] Demo system '{key}' loaded ({n:,} users) <- {fname}")
    else:
        print(f"[WARN] Demo system '{key}' missing: {fname} (run train/build_demo_outputs.py)")

print(f"\n{'=' * 60}")
print("SERVER READY!")
print(f"{'=' * 60}\n")


# ============================================================
# HELPERS
# ============================================================

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this page.', 'warning')
            return redirect(url_for('login'))
        user = get_current_user()
        if not user or not user.get('is_admin'):
            flash('Admin access required.', 'error')
            return redirect(url_for('home'))
        return f(*args, **kwargs)
    return decorated


def get_current_user():
    if 'user_id' not in session:
        return None
    return db.users.find_one({'_id': ObjectId(session['user_id'])})


def get_product(asin):
    return db.products.find_one({'asin': asin})


def get_similar_items(asin, top_k=10):
    if asin not in item_id_to_idx:
        return []
    idx = item_id_to_idx[asin]
    query_vec = item_matrix_normed[idx]
    scores = item_matrix_normed @ query_vec
    top_indices = np.argsort(scores)[::-1][1:top_k + 1]
    results = []
    for i in top_indices:
        iid = search_item_ids[int(i)]
        product = get_product(iid)
        if product:
            product['similarity'] = round(float(scores[i]), 4)
            results.append(product)
    return results


def semantic_search(query, top_k=10):
    query_emb = search_model.encode(query)
    query_norm = query_emb / np.linalg.norm(query_emb)
    scores = item_matrix_normed @ query_norm
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for rank, idx in enumerate(top_indices, 1):
        iid = search_item_ids[int(idx)]
        product = get_product(iid)
        if product:
            product['score'] = round(float(scores[idx]), 4)
            product['rank'] = rank
            results.append(product)
    return results


def format_timestamp(ts):
    """Convert unix timestamp to readable date string."""
    if not ts:
        return ''
    try:
        return datetime.fromtimestamp(int(ts)).strftime('%d/%m/%Y')
    except:
        return ''


# Register template filter
app.jinja_env.filters['format_date'] = format_timestamp


# ============================================================
# ROUTES: AUTH
# ============================================================

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        reviewer_id = request.form.get('reviewer_id', '').strip()

        if not username or not password:
            flash('Username and password are required.', 'error')
            return render_template('register.html')

        if len(username) < 3:
            flash('Username must be at least 3 characters.', 'error')
            return render_template('register.html')

        if len(password) < 4:
            flash('Password must be at least 4 characters.', 'error')
            return render_template('register.html')

        if db.users.find_one({'username': username}):
            flash('Username already exists.', 'error')
            return render_template('register.html')

        if reviewer_id:
            rec = db.recommendations.find_one({'user_id': reviewer_id})
            if not rec:
                flash(f'Reviewer ID "{reviewer_id}" not found in the dataset.', 'error')
                return render_template('register.html')

        user_doc = {
            'username': username,
            'password_hash': generate_password_hash(password),
            'reviewer_id': reviewer_id or None,
            'display_name': username,
        }
        result = db.users.insert_one(user_doc)
        session['user_id'] = str(result.inserted_id)
        session['username'] = username
        flash('Registration successful!', 'success')
        return redirect(url_for('home'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        user = db.users.find_one({'username': username})
        if not user or not check_password_hash(user['password_hash'], password):
            flash('Invalid username or password.', 'error')
            return render_template('login.html')

        session['user_id'] = str(user['_id'])
        session['username'] = user['username']
        display = user.get('display_name') or username
        flash(f'Welcome back, {display}!', 'success')
        return redirect(url_for('home'))

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out.', 'success')
    return redirect(url_for('home'))


# ============================================================
# ROUTES: PAGES
# ============================================================

@app.route('/')
def home():
    # Search with pagination + sorting
    query = request.args.get('q', '').strip()
    sort = request.args.get('sort', 'relevance')
    page = max(1, request.args.get('page', 1, type=int))
    search_results = []
    search_total = 0
    search_pages = 0

    if query:
        all_results = semantic_search(query, top_k=50)

        # Sort results
        if sort == 'price_low':
            all_results.sort(key=lambda p: float(p.get('price', '$0').replace('$', '').replace(',', '') or '0'))
        elif sort == 'price_high':
            all_results.sort(key=lambda p: float(p.get('price', '$0').replace('$', '').replace(',', '') or '0'), reverse=True)
        elif sort == 'rating':
            all_results.sort(key=lambda p: p.get('avg_rating', 0) or 0, reverse=True)

        search_total = len(all_results)
        search_pages = math.ceil(search_total / PER_PAGE)
        start = (page - 1) * PER_PAGE
        search_results = all_results[start:start + PER_PAGE]

    # Personalized recommendations (if logged in)
    # Source: "Baseline + HUIM (retrieval) + Semantic (ranking)" system
    # (demo_outputs['huim_sem'] <- recs_huim_sem.pkl), the same system shown
    # as the best column on the /recommend compare page.
    rec_items = []
    user = get_current_user()
    if user and user.get('reviewer_id'):
        huim_sem = demo_outputs.get('huim_sem')
        if huim_sem:
            top = huim_sem.get('user_top20', {}).get(user['reviewer_id'], [])[:10]
            for rank, (asin, score) in enumerate(top, 1):
                product = get_product(asin)
                if product:
                    product['rank'] = rank
                    product['rec_score'] = round(float(score), 4)
                    rec_items.append(product)

    # Top Rated: avg_rating >= 4.5, min 10 reviews
    top_rated = list(db.products.find(
        {'avg_rating': {'$gte': 4.5}, 'review_count': {'$gte': 10}},
        {'_id': 0}
    ).sort([('avg_rating', DESCENDING), ('review_count', DESCENDING)]).limit(10))

    # Best Sellers: most reviewed, exclude top_rated
    top_rated_asins = {p['asin'] for p in top_rated}
    best_sellers = list(db.products.find(
        {'review_count': {'$exists': True}, 'asin': {'$nin': list(top_rated_asins)}},
        {'_id': 0}
    ).sort('review_count', DESCENDING).limit(10))

    return render_template('home.html',
                           query=query, search_results=search_results,
                           search_total=search_total, search_pages=search_pages,
                           page=page, sort=sort, rec_items=rec_items,
                           top_rated=top_rated, best_sellers=best_sellers)


@app.route('/product/<asin>')
def product_detail(asin):
    product = get_product(asin)
    if not product:
        abort(404)

    # Get reviews with display names (batch lookup)
    reviews = list(db.reviews.find(
        {'asin': asin}, {'_id': 0}
    ).sort('timestamp', DESCENDING).limit(10))

    reviewer_ids = [r.get('user_id') for r in reviews]
    users_map = {}
    for u in db.users.find({'reviewer_id': {'$in': reviewer_ids}}):
        users_map[u['reviewer_id']] = u.get('display_name', '')

    for r in reviews:
        r['display_name'] = users_map.get(r.get('user_id'), '')

    similar = get_similar_items(asin, top_k=10)

    # Breadcrumb
    categories = product.get('category', [])

    # Also bought / also viewed (lookup product info, limit 15)
    also_buy = []
    for aid in product.get('also_buy', [])[:15]:
        p = get_product(aid)
        if p:
            also_buy.append(p)

    also_view = []
    for aid in product.get('also_view', [])[:15]:
        p = get_product(aid)
        if p:
            also_view.append(p)

    return render_template('product.html',
                           product=product, reviews=reviews,
                           similar=similar, categories=categories,
                           also_buy=also_buy, also_view=also_view)


@app.route('/profile')
@login_required
def profile():
    user = get_current_user()
    page = max(1, request.args.get('page', 1, type=int))

    history_items = []
    total_history = 0
    history_pages = 0

    if user and user.get('reviewer_id'):
        total_history = db.reviews.count_documents({'user_id': user['reviewer_id']})
        history_pages = math.ceil(total_history / PER_PAGE)
        skip = (page - 1) * PER_PAGE

        reviews = list(db.reviews.find(
            {'user_id': user['reviewer_id']}, {'_id': 0}
        ).sort('timestamp', DESCENDING).skip(skip).limit(PER_PAGE))

        for r in reviews:
            product = get_product(r['asin'])
            if product:
                r['product'] = product
                history_items.append(r)

    return render_template('profile.html', user=user,
                           history_items=history_items,
                           total_history=total_history,
                           history_pages=history_pages, page=page)


@app.route('/profile/link', methods=['POST'])
@login_required
def link_reviewer():
    reviewer_id = request.form.get('reviewer_id', '').strip()
    if not reviewer_id:
        flash('Please enter a Reviewer ID.', 'error')
        return redirect(url_for('profile'))

    rec = db.recommendations.find_one({'user_id': reviewer_id})
    if not rec:
        flash(f'Reviewer ID "{reviewer_id}" not found.', 'error')
        return redirect(url_for('profile'))

    db.users.update_one(
        {'_id': ObjectId(session['user_id'])},
        {'$set': {'reviewer_id': reviewer_id}}
    )
    flash(f'Linked to reviewer {reviewer_id}!', 'success')
    return redirect(url_for('profile'))


@app.route('/profile/password', methods=['POST'])
@login_required
def change_password():
    current_pw = request.form.get('current_password', '').strip()
    new_pw = request.form.get('new_password', '').strip()
    confirm_pw = request.form.get('confirm_password', '').strip()

    user = get_current_user()

    if not check_password_hash(user['password_hash'], current_pw):
        flash('Current password is incorrect.', 'error')
        return redirect(url_for('profile'))

    if len(new_pw) < 4:
        flash('New password must be at least 4 characters.', 'error')
        return redirect(url_for('profile'))

    if new_pw != confirm_pw:
        flash('New passwords do not match.', 'error')
        return redirect(url_for('profile'))

    db.users.update_one(
        {'_id': ObjectId(session['user_id'])},
        {'$set': {'password_hash': generate_password_hash(new_pw)}}
    )
    flash('Password changed successfully!', 'success')
    return redirect(url_for('profile'))


@app.route('/recommend')
@admin_required
def recommend_page():
    return render_template('recommend.html')


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


# ============================================================
# API ROUTES
# ============================================================

@app.route('/api/recommend')
def api_recommend():
    user_id = request.args.get('user_id', '').strip()
    if not user_id:
        return jsonify({'error': 'user_id is required'}), 400

    rec_data = db.recommendations.find_one({'user_id': user_id})
    if not rec_data:
        return jsonify({'error': f'User "{user_id}" not found'}), 404

    history = []
    train_items = rec_data.get('train_items', [])
    for asin in reversed(train_items[-5:]):
        product = get_product(asin)
        info = {
            'item_id': asin,
            'title': product['title'] if product else f'Product {asin}',
            'brand': product.get('brand', '') if product else '',
            'price': product.get('price', '') if product else '',
            'image': product.get('image', '') if product else '',
        }
        review = db.reviews.find_one({'user_id': user_id, 'asin': asin})
        info['rating'] = review['rating'] if review else None
        history.append(info)

    recommendations = []
    rec_items = rec_data.get('items', [])
    rec_asins = {item['asin'] for item in rec_items[:10]}
    for item in rec_items[:10]:
        product = get_product(item['asin'])
        recommendations.append({
            'item_id': item['asin'],
            'title': product['title'] if product else f'Product {item["asin"]}',
            'brand': product.get('brand', '') if product else '',
            'price': product.get('price', '') if product else '',
            'image': product.get('image', '') if product else '',
            'rank': item['rank'],
            'score': item['score'],
        })

    gt_ids = rec_data.get('ground_truth', [])
    ground_truth = []
    for asin in gt_ids:
        product = get_product(asin)
        ground_truth.append({
            'item_id': asin,
            'title': product['title'] if product else f'Product {asin}',
            'brand': product.get('brand', '') if product else '',
            'price': product.get('price', '') if product else '',
            'image': product.get('image', '') if product else '',
            'is_hit': asin in rec_asins,
        })

    return jsonify({
        'user_id': user_id,
        'purchase_history': history,
        'recommendations': recommendations,
        'ground_truth': ground_truth,
    })


def _product_summary(asin):
    p = get_product(asin)
    return {
        'item_id': asin,
        'title': p['title'] if p else f'Product {asin}',
        'brand': p.get('brand', '') if p else '',
        'price': p.get('price', '') if p else '',
        'image': p.get('image', '') if p else '',
    }


@app.route('/api/recommend_compare')
def api_recommend_compare():
    """Return purchase history + top-10 from each demo system + ground truth.

    Used by recommend.html to compare:
      - Baseline (retrieval + ranking)
      - Baseline + HUIM (retrieval)
      - Baseline + HUIM (retrieval) + Semantic (ranking)
    """
    user_id = request.args.get('user_id', '').strip()
    if not user_id:
        return jsonify({'error': 'user_id is required'}), 400

    if not demo_outputs:
        return jsonify({'error': 'Demo outputs not loaded. Run train/build_demo_outputs.py first.'}), 503

    # Resolve a "primary" output (any system) just to get history + GT
    primary = next(iter(demo_outputs.values()))
    if user_id not in primary.get('user_top20', {}) and \
       user_id not in primary.get('user_train_items', {}) and \
       user_id not in primary.get('test_ground_truth', {}):
        return jsonify({'error': f'User "{user_id}" not found in demo outputs'}), 404

    # Purchase history (last 5 from train items)
    train_items = primary.get('user_train_items', {}).get(user_id, [])
    history = []
    for asin in reversed(train_items[-5:]):
        info = _product_summary(asin)
        review = db.reviews.find_one({'user_id': user_id, 'asin': asin})
        info['rating'] = review['rating'] if review else None
        history.append(info)

    # Ground truth (next actual purchases)
    gt_ids = primary.get('test_ground_truth', {}).get(user_id, [])

    # Build per-system top-10 + GT hit info
    systems = []
    for key, label, _ in DEMO_SYSTEMS:
        out = demo_outputs.get(key)
        if not out:
            continue
        top = out.get('user_top20', {}).get(user_id, [])[:10]
        rec_asins = {a for a, _ in top}
        items = []
        for rank, (asin, score) in enumerate(top, 1):
            info = _product_summary(asin)
            info['rank'] = rank
            info['score'] = round(float(score), 4)
            info['is_hit'] = asin in set(gt_ids)
            items.append(info)
        # Hit count for this system
        hit_count = sum(1 for asin in gt_ids if asin in rec_asins)
        systems.append({
            'key': key,
            'label': label,
            'items': items,
            'hit_count': hit_count,
            'gt_count': len(gt_ids),
        })

    # Ground-truth list with per-system hit flags
    ground_truth = []
    for asin in gt_ids:
        info = _product_summary(asin)
        info['hits'] = {
            s['key']: any(it['item_id'] == asin for it in s['items'])
            for s in systems
        }
        info['is_hit'] = any(info['hits'].values())
        ground_truth.append(info)

    return jsonify({
        'user_id': user_id,
        'purchase_history': history,
        'systems': systems,
        'ground_truth': ground_truth,
    })


@app.route('/api/product/<asin>')
def api_product(asin):
    product = get_product(asin)
    if not product:
        return jsonify({'error': 'not found'}), 404
    return jsonify({
        'asin': asin,
        'title': product.get('title', ''),
        'brand': product.get('brand', ''),
        'price': product.get('price', ''),
        'image': product.get('image', ''),
    })


@app.route('/api/search')
def api_search():
    query = request.args.get('q', '').strip()
    top_k = min(int(request.args.get('top_k', 10)), 50)
    if not query:
        return jsonify({'error': 'query (q) is required'}), 400
    results = semantic_search(query, top_k)
    items = [{
        'item_id': p['asin'],
        'title': p.get('title', ''),
        'brand': p.get('brand', ''),
        'price': p.get('price', ''),
        'image': p.get('image', ''),
        'rank': p.get('rank', 0),
        'score': p.get('score', 0),
    } for p in results]
    return jsonify({'query': query, 'results': items})


@app.route('/api/users/random')
def api_random_user():
    count = db.recommendations.count_documents({})
    if count == 0:
        return jsonify({'error': 'No users'}), 404
    skip = random.randint(0, count - 1)
    rec = db.recommendations.find_one(skip=skip)
    return jsonify({'user_id': rec['user_id']})


@app.route('/api/users/random_compare')
def api_random_compare_user():
    """Random user that exists in the demo outputs (for /recommend compare view)."""
    if not demo_outputs:
        return jsonify({'error': 'Demo outputs not loaded'}), 503
    primary = next(iter(demo_outputs.values()))
    users = [uid for uid, recs in primary.get('user_top20', {}).items() if recs]
    if not users:
        return jsonify({'error': 'No demo users'}), 404
    return jsonify({'user_id': random.choice(users)})


@app.route('/api/stats')
def api_stats():
    metrics = model_info.get('ranking', {}).get('metrics', {})
    return jsonify({
        'total_users': db.recommendations.count_documents({}),
        'total_items': db.products.count_documents({}),
        'model': 'LightGBM LambdaRank',
        'features': len(model_info.get('ranking', {}).get('feature_cols', [])),
        'recall_10': round(metrics.get('Recall@10', 0), 4),
        'ndcg_10': round(metrics.get('NDCG@10', 0), 4),
        'mrr_10': round(metrics.get('MRR@10', 0), 4),
    })


@app.route('/api/review', methods=['POST'])
@login_required
def api_submit_review():
    user = get_current_user()
    data = request.get_json()
    asin = data.get('asin', '').strip()
    rating = data.get('rating')
    text = data.get('text', '').strip()

    if not asin or not rating:
        return jsonify({'error': 'asin and rating are required'}), 400

    rating = float(rating)
    if rating < 1 or rating > 5:
        return jsonify({'error': 'Rating must be 1-5'}), 400

    db.reviews.insert_one({
        'user_id': user.get('reviewer_id') or str(user['_id']),
        'asin': asin,
        'rating': rating,
        'text': text,
        'summary': '',
        'timestamp': int(time.time()),
    })

    pipeline = [
        {'$match': {'asin': asin}},
        {'$group': {'_id': None, 'avg': {'$avg': '$rating'}, 'cnt': {'$sum': 1}}}
    ]
    agg = list(db.reviews.aggregate(pipeline))
    if agg:
        db.products.update_one(
            {'asin': asin},
            {'$set': {'avg_rating': round(agg[0]['avg'], 1), 'review_count': agg[0]['cnt']}}
        )

    return jsonify({'status': 'ok', 'message': 'Review submitted!'})


# ============================================================
# TEMPLATE CONTEXT
# ============================================================

@app.context_processor
def inject_user():
    return {'current_user': get_current_user()}


# ============================================================
# RUN
# ============================================================

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
