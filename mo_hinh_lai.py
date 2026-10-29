"""
mo_hinh_lai.py - Mô hình lai (Hybrid) + xử lý người dùng mới (cold-start)

Điểm lai = w_cb * điểm nội dung (TF-IDF) + w_cf * điểm SVD + w_pop * độ phổ biến
  - Mỗi thành phần được chuẩn hóa về thang [0,1] bằng thứ hạng phần trăm, nên các trọng số
    có ý nghĩa như nhau (điểm cosine ~0-0.3 và điểm SVD ~1-5 không còn lệch thang).
  - Trọng số chọn trên tập VALIDATION (15% mới nhất của mỗi người dùng trong TRAIN),
    TEST chỉ dùng một lần ở cuối để báo cáo => không rò rỉ dữ liệu.
  - Người dùng mới (chưa có đánh giá): dùng vài cuốn sách họ chọn lúc khảo sát + độ phổ biến.

Đầu vào : train_data.csv, test_data.csv, books_clean.csv, tfidf_matrix.pkl, tfidf_vectorizer.pkl
Đầu ra  : ket_qua_mo_hinh_lai.csv, svd_model.pkl, hybrid_config.pkl (cho API)
"""
import time

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from surprise import Dataset, Reader, SVD

TRAIN_CSV = 'train_data.csv'
TEST_CSV = 'test_data.csv'
BOOKS_CSV = 'books_clean.csv'
TFIDF_PKL = 'tfidf_matrix.pkl'
VEC_PKL = 'tfidf_vectorizer.pkl'
OUT_CSV = 'ket_qua_mo_hinh_lai.csv'
SVD_PKL = 'svd_model.pkl'
CONFIG_PKL = 'hybrid_config.pkl'

MIN_RATINGS = 50        # chỉ giữ sách có từ 50 lượt đánh giá trở lên trong TRAIN
K = 10                  # độ dài danh sách gợi ý
LIKE_THRESHOLD = 4      # "phù hợp" = chấm từ 4 sao
N_USERS_EVAL = 1000     # số người dùng lấy mẫu ở TEST
N_USERS_VAL = 500       # số người dùng lấy mẫu ở VALIDATION
VAL_FRACTION = 0.15     # phần đánh giá mới nhất của mỗi người dùng trong TRAIN dành cho validation
N_SEED_BOOKS = 3        # người dùng mới chọn 3 cuốn sách ở bước khảo sát
SEED = 42
P, R = f'Precision@{K}', f'Recall@{K}'

# ======================================================================
print("1. Đang nạp dữ liệu...")


def nap(path):
    d = pd.read_csv(path)
    d = d[d['rating'] > 0].copy()                      # bỏ 0 sao
    # Dùng errors='coerce' để tương thích pandas < 2.0 (tránh dùng format='mixed')
    d['date_added'] = pd.to_datetime(d['date_added'], utc=True, errors='coerce')
    d = d.dropna(subset=['date_added'])                # bỏ các dòng ngày tháng lỗi
    return d.sort_values(['user_id', 'date_added'])    # theo thời gian trong từng người dùng


df_train, df_test = nap(TRAIN_CSV), nap(TEST_CSV)
popular = df_train['book_id'].value_counts()
popular = popular[popular >= MIN_RATINGS].index
df_train = df_train[df_train['book_id'].isin(popular)]
df_test = df_test[df_test['book_id'].isin(popular)]

item_ids = np.array(sorted(popular))                   # danh mục ứng viên cố định
n_items = len(item_ids)
gidx = {b: i for i, b in enumerate(item_ids)}
print(f"-> Train: {len(df_train):,} | Test: {len(df_test):,} | Số sách ứng viên: {n_items:,}")

info = pd.read_csv(BOOKS_CSV, usecols=['book_id', 'title', 'authors', 'genres', 'ratings_count'])
info['ratings_count'] = pd.to_numeric(info['ratings_count'], errors='coerce').fillna(0)
book_row = {b: i for i, b in enumerate(info['book_id'])}
tfidf_matrix = joblib.load(TFIDF_PKL).tocsr()
vec = joblib.load(VEC_PKL)
assert tfidf_matrix.shape[0] == len(info), "tfidf_matrix.pkl không khớp books_clean.csv, hãy chạy lại mo_hinh_dg_cso.py"
cand_tfidf = tfidf_matrix[np.array([book_row[b] for b in item_ids])]   # hàng i = sách ứng viên thứ i
item_info = info.set_index('book_id').loc[item_ids]
reader = Reader(rating_scale=(1, 5))


# ======================================================================
# Các hàm thành phần
def rank_norm(x):
    """Chuẩn hóa về [0,1] theo thứ hạng phần trăm (chịu được giá trị cực đoan)."""
    return rankdata(x) / len(x)


def do_pho_bien(df):
    """Weighted rating (IMDb) cho từng sách ứng viên."""
    st = df.groupby('book_id')['rating'].agg(['count', 'mean'])
    m = st['count'].quantile(0.75)
    C = df['rating'].mean()
    wr = st['count'] / (st['count'] + m) * st['mean'] + m / (st['count'] + m) * C
    return wr.reindex(item_ids).fillna(C).values


def lich_su(df):
    """Người dùng -> (chỉ số sách đã đánh giá theo thứ tự thời gian, điểm)."""
    return {u: (g['book_id'].map(gidx).values.astype(int), g['rating'].values.astype(float))
            for u, g in df.groupby('user_id', sort=False)}


def huan_luyen_svd(df):
    ts = Dataset.load_from_df(df[['user_id', 'book_id', 'rating']], reader).build_full_trainset()
    svd = SVD(n_factors=100, random_state=SEED)
    svd.fit(ts)
    inner = np.full(n_items, -1)
    for b, g in gidx.items():
        try:
            inner[g] = ts.to_inner_iid(b)
        except ValueError:
            pass
    ok = inner >= 0
    bi = np.zeros(n_items)
    qi = np.zeros((n_items, svd.qi.shape[1]))
    bi[ok], qi[ok] = svd.bi[inner[ok]], svd.qi[inner[ok]]
    return {'svd': svd, 'ts': ts, 'mu': ts.global_mean, 'bi': bi, 'qi': qi}


def diem_svd(model, raw_u):
    try:
        u = model['ts'].to_inner_uid(raw_u)
    except ValueError:
        return None                                    # người dùng mới, SVD chưa biết
    return model['mu'] + model['svd'].bu[u] + model['bi'] + model['qi'] @ model['svd'].pu[u]


def diem_cb(liked_idx):
    """Điểm nội dung: độ giống với hồ sơ = trung bình vector TF-IDF các sách đã thích."""
    profile = np.asarray(cand_tfidf[liked_idx].mean(axis=0)).ravel()
    return np.asarray(cand_tfidf.dot(profile)).ravel()


def ild(top):
    sub = cand_tfidf[top]
    s = (sub @ sub.T).toarray()
    n = s.shape[0]
    return 1.0 - (s.sum() - np.trace(s)) / (n * (n - 1))


def ket_qua(scores, seen, rel_sets, day_du=True):
    s = np.where(seen, -np.inf, scores)                # không gợi ý lại sách đã biết
    top = np.argpartition(-s, K, axis=1)[:, :K]
    hits = np.array([len(r & set(t)) for r, t in zip(rel_sets, top.tolist())])
    out = {P: hits.mean() / K, R: float(np.mean(hits / np.array([len(r) for r in rel_sets])))}
    if day_du:
        out[f'Coverage@{K}'] = len(set(top.ravel().tolist())) / n_items
        out[f'Diversity@{K}'] = float(np.mean([ild(t) for t in top]))
    return out


def tap_phu_hop(df, hist):
    out = {}
    for u, g in df[df['rating'] >= LIKE_THRESHOLD].groupby('user_id')['book_id']:
        if u in hist:
            s = {gidx[b] for b in g if b in gidx}
            if s:
                out[u] = s
    return out


def chon_mau(users, n):
    users = sorted(users)
    if len(users) <= n:
        return users
    sel = np.random.default_rng(SEED).choice(len(users), n, replace=False)
    return [users[i] for i in sorted(sel)]


def thanh_phan(users, hist, model):
    """Ma trận điểm (đã chuẩn hóa) của nội dung và SVD cho các người dùng đã có lịch sử."""
    cb = np.zeros((len(users), n_items), dtype=np.float32)
    cf = np.zeros_like(cb)
    seen = np.zeros((len(users), n_items), dtype=bool)
    for r, u in enumerate(users):
        idx, rt = hist[u]
        liked = idx[rt >= LIKE_THRESHOLD]
        if len(liked) == 0:
            liked = idx
        cb[r] = rank_norm(diem_cb(liked))
        s = diem_svd(model, u)
        cf[r] = rank_norm(s) if s is not None else 0.5
        seen[r, idx] = True
    return cb, cf, seen


def ca_nguoi_dung_moi(users, hist, rel):
    """Giả lập người dùng mới: chỉ biết N_SEED_BOOKS cuốn họ thích sớm nhất."""
    cb, seen, rs = [], [], []
    for u in users:
        idx, rt = hist[u]
        liked = idx[rt >= LIKE_THRESHOLD]
        if len(liked) < N_SEED_BOOKS:
            continue
        seeds = liked[:N_SEED_BOOKS]
        s = np.zeros(n_items, dtype=bool)
        s[seeds] = True
        cb.append(rank_norm(diem_cb(seeds)))
        seen.append(s)
        rs.append(rel[u])
    return np.array(cb), np.array(seen), rs


def tim_trong_so(cb, cf, seen, rel_sets, pr):
    kq = []
    for wc in range(11):
        for wf in range(11 - wc):
            wp = 10 - wc - wf
            r = ket_qua((wc * cb + wf * cf + wp * pr) / 10, seen, rel_sets, day_du=False)
            kq.append((wc / 10, wf / 10, wp / 10, r[P], r[R]))
    kq.sort(key=lambda x: (x[3], x[4]), reverse=True)
    return kq


# ======================================================================
print("\n2. VALIDATION: chọn trọng số cho mô hình lai (chỉ dùng dữ liệu TRAIN)...")
pos = df_train.groupby('user_id').cumcount()
n_u = df_train.groupby('user_id')['book_id'].transform('size')
is_val = (n_u >= 10) & (pos >= (n_u * (1 - VAL_FRACTION)).astype(int))
df_inner, df_val = df_train[~is_val], df_train[is_val]
print(f"-> Train trong: {len(df_inner):,} | Validation: {len(df_val):,}")

t0 = time.time()
model_in = huan_luyen_svd(df_inner)
hist_in = lich_su(df_inner)
pop_in = rank_norm(do_pho_bien(df_inner))
rel_val = tap_phu_hop(df_val, hist_in)
users_val = chon_mau(rel_val.keys(), N_USERS_VAL)
cb_v, cf_v, seen_v = thanh_phan(users_val, hist_in, model_in)
rs_v = [rel_val[u] for u in users_val]
grid = tim_trong_so(cb_v, cf_v, seen_v, rs_v, pop_in)
print(f"-> Đã thử 66 tổ hợp trọng số trên {len(users_val)} người dùng ({time.time() - t0:.0f}s). Top 8:")
rows = []
for wc, wf, wp, _, _ in grid[:8]:
    r = ket_qua((wc * cb_v + wf * cf_v + wp * pop_in) / 1.0, seen_v, rs_v)
    rows.append({'w_nội dung': wc, 'w_SVD': wf, 'w_phổ biến': wp, **{k: round(v, 4) for k, v in r.items()}})
print(pd.DataFrame(rows).to_string(index=False))
W_CB, W_CF, W_POP = grid[0][:3]
print(f"=> CHỌN: w_nội dung={W_CB}, w_SVD={W_CF}, w_phổ biến={W_POP}")

# Người dùng mới: chỉ trộn nội dung (từ khảo sát) với độ phổ biến
cb_c, seen_c, rs_c = ca_nguoi_dung_moi(users_val, hist_in, rel_val)
cold_grid = []
for w in range(11):
    r = ket_qua((w / 10) * cb_c + (1 - w / 10) * pop_in, seen_c, rs_c, day_du=False)
    cold_grid.append((w / 10, r[P], r[R]))
W_COLD = max(cold_grid, key=lambda x: (x[1], x[2]))[0]
print(f"=> Người dùng mới (chỉ biết {N_SEED_BOOKS} cuốn sách): w_nội dung={W_COLD}, w_phổ biến={1 - W_COLD:.1f}")

# ======================================================================
print("\n3. TEST: huấn luyện lại trên toàn bộ TRAIN và đo kết quả cuối cùng...")
model = huan_luyen_svd(df_train)
hist = lich_su(df_train)
pop_wr = do_pho_bien(df_train)
pop_r = rank_norm(pop_wr)
rel_te = tap_phu_hop(df_test, hist)
users_te = chon_mau(rel_te.keys(), N_USERS_EVAL)
rs_t = [rel_te[u] for u in users_te]
cb_t, cf_t, seen_t = thanh_phan(users_te, hist, model)
print(f"-> Đo trên {len(users_te):,} người dùng.")

pop_mat = np.tile(pop_r, (len(users_te), 1))
ket = [
    ('Baseline (phổ biến)', pop_mat),
    ('TF-IDF (nội dung)', cb_t),
    ('SVD', cf_t),
    (f'LAI ({W_CB}/{W_CF}/{W_POP})', W_CB * cb_t + W_CF * cf_t + W_POP * pop_r),
]
bang = [{'Kịch bản': 'Người dùng cũ', 'Mô hình': ten, **ket_qua(sc, seen_t, rs_t)} for ten, sc in ket]

cb_c, seen_c, rs_c = ca_nguoi_dung_moi(users_te, hist, rel_te)
pop_c = np.tile(pop_r, (len(rs_c), 1))
bang += [
    {'Kịch bản': f'Người dùng mới ({N_SEED_BOOKS} sách)', 'Mô hình': 'Baseline (phổ biến)',
     **ket_qua(pop_c, seen_c, rs_c)},
    {'Kịch bản': f'Người dùng mới ({N_SEED_BOOKS} sách)', 'Mô hình': f'LAI mới ({W_COLD}/{1 - W_COLD:.1f})',
     **ket_qua(W_COLD * cb_c + (1 - W_COLD) * pop_c, seen_c, rs_c)},
]
res = pd.DataFrame(bang)
res.to_csv(OUT_CSV, index=False)

fmt = res.copy()
for c in [P, R]:
    fmt[c] = fmt[c].map(lambda x: f'{x:.4f}')
for c in [f'Coverage@{K}', f'Diversity@{K}']:
    fmt[c] = fmt[c].map(lambda x: f'{x:.3f}')
print("\n==================== KẾT QUẢ MÔ HÌNH LAI (TẬP TEST) ====================")
print(fmt.to_string(index=False))
print("========================================================================")
print(f"(Người dùng mới: đo trên {len(rs_c):,} người dùng đủ {N_SEED_BOOKS} sách đã thích)")
print(f"-> Đã lưu {OUT_CSV}")

# ======================================================================
print("\n4. Đang lưu mô hình cho API...")
joblib.dump(model['svd'], SVD_PKL)
joblib.dump({'item_ids': item_ids, 'pop_wr': pop_wr, 'trong_so': (W_CB, W_CF, W_POP),
             'w_nguoi_dung_moi': W_COLD, 'K': K, 'like_threshold': LIKE_THRESHOLD,
             'min_ratings': MIN_RATINGS}, CONFIG_PKL)
print(f"-> Đã lưu {SVD_PKL} và {CONFIG_PKL}")

# ======================================================================
title_lower = item_info['title'].str.lower()


def _hien_thi(top, scores):
    out = item_info.iloc[top][['title', 'authors', 'genres']].copy()
    out['diem'] = np.round(scores[top], 3)
    out['title'], out['authors'], out['genres'] = out['title'].str[:48], out['authors'].str[:22], out['genres'].str[:30]
    print(out.to_string(index=False))


def goi_y_nguoi_dung_moi(sach_thich, mo_ta='', top_n=5):
    """Gợi ý cho người dùng MỚI từ khảo sát: vài cuốn đã thích + mô tả thể loại/tác giả (tiếng Anh)."""
    seeds = []
    for ten in sach_thich:
        t = ten.lower()
        mask = title_lower.str.startswith(t).values
        if not mask.any():
            mask = title_lower.str.contains(t, regex=False).values
        if mask.any():
            seeds.append(int(np.argmax(np.where(mask, item_info['ratings_count'].values, -1))))
    prof = np.zeros(cand_tfidf.shape[1])
    if seeds:
        prof += np.asarray(cand_tfidf[seeds].mean(axis=0)).ravel()
    if mo_ta:
        prof += vec.transform([mo_ta]).toarray().ravel()
    if prof.any():
        score = W_COLD * rank_norm(np.asarray(cand_tfidf.dot(prof)).ravel()) + (1 - W_COLD) * pop_r
    else:
        score = pop_r.copy()
    score[seeds] = -np.inf
    top = np.argsort(-score)[:top_n]
    print(f"\nNgười dùng mới thích {sach_thich}, mô tả: '{mo_ta}'")
    _hien_thi(top, score)


def goi_y_nguoi_dung_cu(raw_u, top_n=5):
    idx, rt = hist[raw_u]
    cb, cf, seen = thanh_phan([raw_u], hist, model)
    score = (W_CB * cb + W_CF * cf + W_POP * pop_r)[0]
    score[seen[0]] = -np.inf
    top = np.argsort(-score)[:top_n]
    print(f"\nNgười dùng cũ (đã có {len(idx)} đánh giá):")
    _hien_thi(top, score)


print("\n--- CHẠY THỬ ---")
goi_y_nguoi_dung_cu(users_te[0])
goi_y_nguoi_dung_moi(["The Hunger Games", "Divergent", "Harry Potter and the Sorcerer's Stone"], "young adult dystopian")
goi_y_nguoi_dung_moi([], "romance historical fiction")
print("\nHoàn tất mô hình lai.")