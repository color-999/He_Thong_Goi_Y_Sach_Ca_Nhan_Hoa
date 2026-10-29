"""
danh_gia.py - Đánh giá và so sánh 4 mô hình gợi ý:
  Baseline (độ phổ biến) | TF-IDF (nội dung) | SVD | item-kNN
Chỉ số: RMSE, Precision@K, Recall@K, Coverage@K, Intra-list Diversity@K

Đầu vào : train_data.csv, test_data.csv, books_clean.csv, tfidf_matrix.pkl
Đầu ra  : ket_qua_danh_gia.csv (bảng so sánh dùng cho báo cáo)
Chia tập theo thời gian (đã làm ở chia_tach_dlieu.py) nên không rò rỉ dữ liệu.
"""
import time

import joblib
import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD, KNNBasic, accuracy
from scipy.stats import rankdata

TRAIN_CSV = 'train_data.csv'
TEST_CSV = 'test_data.csv'
BOOKS_CSV = 'books_clean.csv'
TFIDF_PKL = 'tfidf_matrix.pkl'
OUT_CSV = 'ket_qua_danh_gia.csv'

MIN_RATINGS = 50       # chỉ giữ sách có từ 50 lượt đánh giá trở lên trong TRAIN (tránh tràn RAM cho kNN)
K = 10                 # độ dài danh sách gợi ý
LIKE_THRESHOLD = 4     # sách "phù hợp" = người dùng chấm từ 4 sao trở lên trong TEST
N_USERS_EVAL = 1000    # số người dùng lấy mẫu để đo xếp hạng (đặt None để đo toàn bộ, sẽ lâu hơn)
K_NN = 40              # số láng giềng của item-kNN (mặc định của Surprise)
SEED = 42

# ----------------------------------------------------------------------
print("1. Đang nạp dữ liệu Train/Test...")
df_train = pd.read_csv(TRAIN_CSV)
df_test = pd.read_csv(TEST_CSV)
df_train = df_train[df_train['rating'] > 0]   # bỏ 0 sao (chưa chấm điểm)
df_test = df_test[df_test['rating'] > 0]

book_counts = df_train['book_id'].value_counts()
popular = book_counts[book_counts >= MIN_RATINGS].index
df_train = df_train[df_train['book_id'].isin(popular)]
df_test = df_test[df_test['book_id'].isin(popular)]
print(f"-> Train: {len(df_train):,} | Test: {len(df_test):,} | Số sách: {len(popular):,}")

reader = Reader(rating_scale=(1, 5))
trainset = Dataset.load_from_df(df_train[['user_id', 'book_id', 'rating']], reader).build_full_trainset()
testset = list(zip(df_test['user_id'], df_test['book_id'], df_test['rating']))
mu = trainset.global_mean
n_items = trainset.n_items

# ----------------------------------------------------------------------
print("2. Đang nạp ma trận TF-IDF (dùng cho mô hình nội dung và đo độ đa dạng)...")
books = pd.read_csv(BOOKS_CSV, usecols=['book_id'])
tfidf_matrix = joblib.load(TFIDF_PKL).tocsr()
assert tfidf_matrix.shape[0] == len(books), "tfidf_matrix.pkl không khớp books_clean.csv, hãy chạy lại mo_hinh_dg_cso.py"
book_row = {b: i for i, b in enumerate(books['book_id'])}
iid_raw = np.array([trainset.to_raw_iid(i) for i in range(n_items)])
iid_row = np.array([book_row.get(b, -1) for b in iid_raw])
assert (iid_row >= 0).all(), "Có sách trong dữ liệu đánh giá không có trong books_clean.csv"
cand_tfidf = tfidf_matrix[iid_row]   # hàng i = sách có mã nội bộ i của trainset

# ----------------------------------------------------------------------
print("3. Đang huấn luyện SVD...")
svd = SVD(n_factors=100, random_state=SEED)
t0 = time.time()
svd.fit(trainset)
svd_time = time.time() - t0
svd_rmse = accuracy.rmse(svd.test(testset), verbose=False)
print(f"   SVD RMSE: {svd_rmse:.4f} ({svd_time:.1f}s)")

print("4. Đang huấn luyện item-kNN (pearson_baseline, có thể mất vài phút)...")
knn = KNNBasic(k=K_NN, sim_options={'name': 'pearson_baseline', 'user_based': False}, random_state=SEED)
t0 = time.time()
knn.fit(trainset)
knn_time = time.time() - t0
knn_rmse = accuracy.rmse(knn.test(testset), verbose=False)
print(f"   item-kNN RMSE: {knn_rmse:.4f} ({knn_time:.1f}s)")

# Baseline: điểm weighted rating tính trên TRAIN; dự đoán rating = điểm trung bình của sách
st = df_train.groupby('book_id')['rating'].agg(['count', 'mean'])
m = st['count'].quantile(0.75)
C = df_train['rating'].mean()
st['wr'] = st['count'] / (st['count'] + m) * st['mean'] + m / (st['count'] + m) * C
pop_scores = st['wr'].reindex(iid_raw).values
base_pred = df_test['book_id'].map(st['mean']).fillna(mu)
base_rmse = float(np.sqrt(np.mean((base_pred.values - df_test['rating'].values) ** 2)))
print(f"   Baseline RMSE (điểm TB của sách): {base_rmse:.4f}")

# ----------------------------------------------------------------------
# Hàm chấm điểm: nhận mã người dùng nội bộ u, trả về mảng điểm cho toàn bộ n_items sách
def diem_baseline(u):
    return pop_scores


def diem_tfidf(u):
    ur = trainset.ur[u]
    liked = [j for j, r in ur if r >= LIKE_THRESHOLD] or [j for j, _ in ur]
    if not liked:
        return np.zeros(n_items)
    profile = np.asarray(cand_tfidf[liked].mean(axis=0)).ravel()   # hồ sơ sở thích = TB vector sách đã thích
    return np.asarray(cand_tfidf.dot(profile)).ravel()


def diem_svd(u):
    return mu + svd.bu[u] + svd.bi + svd.qi @ svd.pu[u]


def diem_knn(u):
    ur = trainset.ur[u]
    if not ur:
        return np.full(n_items, mu)
    items = np.array([j for j, _ in ur])
    r = np.array([x for _, x in ur], dtype=float)
    S = knn.sim[:, items]                       # độ tương đồng giữa mọi sách và sách người dùng đã đọc
    if len(items) > K_NN:
        idx = np.argpartition(-S, K_NN - 1, axis=1)[:, :K_NN]
        sims = np.take_along_axis(S, idx, axis=1)
        rs = r[idx]
    else:
        sims = S
        rs = np.broadcast_to(r, S.shape)
    sims = np.where(sims > 0, sims, 0.0)        # giống KNNBasic: chỉ tính láng giềng có sim > 0
    tong = sims.sum(axis=1)
    return np.where(tong > 0, (sims * rs).sum(axis=1) / np.maximum(tong, 1e-12), mu)


def ild(top):
    """Intra-list diversity = 1 - độ tương đồng cosine TB giữa các cặp sách trong danh sách (theo TF-IDF)."""
    sub = tfidf_matrix[iid_row[top]]
    s = (sub @ sub.T).toarray()
    n = s.shape[0]
    return 1.0 - (s.sum() - np.trace(s)) / (n * (n - 1))

# ----------------------------------------------------------------------
# MÔ HÌNH LAI (Hybrid): trộn CB (TF-IDF) + CF (SVD + item-kNN) + Độ phổ biến
# - Đọc trọng số (W_CB, W_CF, W_POP) từ hybrid_config.pkl do mo_hinh_lai.py đã grid-search
#   → đảm bảo 2 file cho CÙNG kết quả Hybrid, và trọng số có cơ sở validation (không tự bịa).
# - Nếu chưa có hybrid_config.pkl (chưa chạy mo_hinh_lai.py), fallback về 0.4/0.5/0.1.
# - rank_norm chuẩn hóa từng thành phần về [0,1] để 3 thành phần cùng thang đo.
# ----------------------------------------------------------------------
try:
    _hybrid_cfg = joblib.load('hybrid_config.pkl')
    W_CB, W_CF, W_POP = _hybrid_cfg['trong_so']
    print(f"-> Đã nạp trọng số hybrid từ hybrid_config.pkl: {W_CB}/{W_CF}/{W_POP}")
except Exception as _e:
    W_CB, W_CF, W_POP = 0.4, 0.5, 0.1
    print(f"-> Không đọc được hybrid_config.pkl ({_e}); dùng mặc định 0.4/0.5/0.1")


def rank_norm(x):
    """Chuẩn hóa về [0,1] theo thứ hạng phần trăm (chịu được giá trị cực đoan)."""
    return rankdata(x) / len(x)


POP_RANK = rank_norm(pop_scores)   # Độ phổ biến đã chuẩn hóa, dùng chung cho mọi user


def diem_hybrid(u):
    """Điểm lai: W_CB·CB + W_CF·(0.5·SVD + 0.5·kNN) + W_POP·Pop
    CF dùng trung bình SVD và item-kNN (cả hai đều đã chuẩn hóa thứ hạng)."""
    cb = rank_norm(diem_tfidf(u))
    cf = 0.5 * rank_norm(diem_svd(u)) + 0.5 * rank_norm(diem_knn(u))
    return W_CB * cb + W_CF * cf + W_POP * POP_RANK
# ----------------------------------------------------------------------
print("5. Đang chuẩn bị tập người dùng để đo xếp hạng...")
rel = df_test[df_test['rating'] >= LIKE_THRESHOLD]
rel_by_user = {}
for raw_u, grp in rel.groupby('user_id')['book_id']:
    try:
        uid = trainset.to_inner_uid(raw_u)
    except ValueError:
        continue
    items = set()
    for b in grp:
        try:
            items.add(trainset.to_inner_iid(b))
        except ValueError:
            pass
    if items:
        rel_by_user[uid] = items

users = sorted(rel_by_user)
if N_USERS_EVAL and len(users) > N_USERS_EVAL:
    users = list(np.random.default_rng(SEED).choice(users, N_USERS_EVAL, replace=False))
print(f"-> Đo trên {len(users):,} người dùng (có ít nhất 1 sách >= {LIKE_THRESHOLD} sao trong tập Test).")


def danh_gia_xep_hang(ten, ham_diem):
    t0 = time.time()
    prec, recall, div = [], [], []
    goi_y = set()
    for n, u in enumerate(users, 1):
        scores = np.nan_to_num(np.array(ham_diem(u), dtype=float), nan=-1e9)
        scores[[j for j, _ in trainset.ur[u]]] = -np.inf    # không gợi ý lại sách đã đọc
        top = np.argpartition(-scores, K)[:K]
        top = top[np.argsort(-scores[top])]
        hits = len(rel_by_user[u] & set(top.tolist()))
        prec.append(hits / K)
        recall.append(hits / len(rel_by_user[u]))
        goi_y.update(top.tolist())
        div.append(ild(top))
        if n % 250 == 0:
            print(f"   [{ten}] {n}/{len(users)} người dùng...")
    print(f"   {ten}: xong ({time.time() - t0:.1f}s)")
    return {
        f'Precision@{K}': np.mean(prec),
        f'Recall@{K}': np.mean(recall),
        f'Coverage@{K}': len(goi_y) / n_items,
        f'Diversity@{K}': np.mean(div),
    }


print("6. Đang đo Precision/Recall/Coverage/Diversity...")
models = [
    ('Baseline (phổ biến)', diem_baseline, base_rmse, 0.0),
    ('TF-IDF (nội dung)', diem_tfidf, np.nan, 0.0),
    ('SVD', diem_svd, svd_rmse, svd_time),
    ('item-kNN', diem_knn, knn_rmse, knn_time),
    ('Hybrid (CB+CF+Pop)', diem_hybrid, np.nan, svd_time + knn_time),
]
rows = []
for ten, fn, rmse, t_train in models:
    res = danh_gia_xep_hang(ten, fn)
    rows.append({'Mô hình': ten, 'RMSE': rmse, **res, 'Thời gian huấn luyện (s)': t_train})

result = pd.DataFrame(rows)
result.to_csv(OUT_CSV, index=False)

print("\n==================== BẢNG SO SÁNH HIỆU NĂNG ====================")
fmt = result.copy()
fmt['RMSE'] = fmt['RMSE'].map(lambda x: '—' if pd.isna(x) else f'{x:.4f}')
for c in [f'Precision@{K}', f'Recall@{K}']:
    fmt[c] = fmt[c].map(lambda x: f'{x:.4f}')
for c in [f'Coverage@{K}', f'Diversity@{K}']:
    fmt[c] = fmt[c].map(lambda x: f'{x:.3f}')
fmt['Thời gian huấn luyện (s)'] = fmt['Thời gian huấn luyện (s)'].map(lambda x: f'{x:.1f}')
print(fmt.to_string(index=False))
print("================================================================")
print("Ghi chú: RMSE thấp hơn là tốt hơn; Precision/Recall/Coverage/Diversity cao hơn là tốt hơn.")
print(f"-> Đã lưu {OUT_CSV}")


# ---------- BỔ SUNG: mốc ngẫu nhiên + kiểm tra tính đúng đắn ----------
print("\n7. Kiểm tra bổ sung...")
rng_r = np.random.default_rng(SEED)
res_rand = danh_gia_xep_hang('Ngẫu nhiên', lambda u: rng_r.random(n_items))
print("Mốc NGẪU NHIÊN:", {k: round(float(v), 4) for k, v in res_rand.items()})

# Điểm tự tính phải trùng với dự đoán của chính thư viện Surprise
u0 = users[0]
raw_u0 = trainset.to_raw_uid(u0)
seen_u0 = {j for j, _ in trainset.ur[u0]}
chua_doc = [j for j in range(n_items) if j not in seen_u0][:200]
mau = np.random.default_rng(0).choice(chua_doc, 5, replace=False)
s_svd, s_knn = diem_svd(u0), diem_knn(u0)
print("Đối chiếu với Surprise (hai cột phải giống nhau):")
for j in mau:
    raw_i = trainset.to_raw_iid(int(j))
    print(f"   sách #{j}: SVD tự tính={s_svd[j]:.4f} | Surprise={svd.predict(raw_u0, raw_i, clip=False).est:.4f}"
          f"   ||  kNN tự tính={s_knn[j]:.4f} | Surprise={knn.predict(raw_u0, raw_i, clip=False).est:.4f}")