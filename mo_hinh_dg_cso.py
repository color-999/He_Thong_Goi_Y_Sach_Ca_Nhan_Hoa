"""
mo_hinh_dg_cso.py - Baseline (độ phổ biến) + TF-IDF, có gộp các ấn bản của cùng một tác phẩm
Đầu vào : books_clean.csv, train_data.csv (nếu chưa có thì dùng cleaned_goodreads_reviews.csv)
Đầu ra  : mo_hinh_duong_co_so.csv, tfidf_matrix.pkl, tfidf_vectorizer.pkl
Baseline tính trên tập TRAIN để sau này so sánh công bằng với SVD/kNN.
"""
import os
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

BOOKS_CSV = 'books_clean.csv'
TRAIN_CSV = 'train_data.csv'
REVIEWS_CSV = 'cleaned_goodreads_reviews.csv'
OUT_BASELINE = 'mo_hinh_duong_co_so.csv'
OUT_TFIDF_MATRIX = 'tfidf_matrix.pkl'
OUT_TFIDF_VEC = 'tfidf_vectorizer.pkl'
M_PERCENTILE = 0.75  # ngưỡng tin cậy m = phân vị này của số lượt đánh giá

print("Đang nạp danh mục sách...")
books = pd.read_csv(BOOKS_CSV, dtype={'work_id': str})
for col in ['title', 'authors', 'genres', 'shelves', 'description', 'content']:
    books[col] = books[col].fillna('')
books['ratings_count'] = pd.to_numeric(books['ratings_count'], errors='coerce').fillna(0)


def _chuan_hoa(s):
    s = re.sub(r'\s*\(.*?\)\s*', ' ', str(s).lower())   # bỏ "(Harry Potter, #1)"
    return re.sub(r'[^a-z0-9]+', ' ', s).strip()


# Khóa "tác phẩm": các ấn bản khác nhau của cùng một cuốn sách có cùng work_key
if 'work_id' in books.columns and books['work_id'].fillna('').str.strip().ne('').mean() > 0.5:
    wid = books['work_id'].fillna('').str.strip()
    books['work_key'] = np.where(wid != '', 'w' + wid, 'b' + books['book_id'].astype(str))
    print("-> Gộp ấn bản theo work_id.")
else:
    first_author = books['authors'].str.split(',').str[0]
    books['work_key'] = books['title'].map(_chuan_hoa) + '|' + first_author.map(_chuan_hoa)
    print("-> Không có work_id: gộp ấn bản gần đúng theo tên sách + tác giả đầu tiên.")
work_keys = books['work_key'].values
print(f"-> {len(books):,} ấn bản, thuộc {books['work_key'].nunique():,} tác phẩm khác nhau.")

# ===== PHẦN 1: BASELINE (weighted rating kiểu IMDb, tính theo tác phẩm) =====
# WR = v/(v+m)*R + m/(v+m)*C
print("\n=== PHẦN 1: BASELINE ===")
src = TRAIN_CSV if os.path.exists(TRAIN_CSV) else REVIEWS_CSV
print(f"Tính trên: {src}")
df = pd.read_csv(src, usecols=['book_id', 'rating'])
df = df[df['rating'] > 0]
df = df.merge(books[['book_id', 'work_key']], on='book_id', how='left')
df['work_key'] = df['work_key'].fillna('b' + df['book_id'].astype(str))

# Ấn bản đại diện của mỗi tác phẩm = ấn bản có nhiều lượt đánh giá nhất
per_book = df.groupby(['work_key', 'book_id']).size().reset_index(name='n')
rep = per_book.sort_values('n', ascending=False).drop_duplicates('work_key')[['work_key', 'book_id']]

stats = df.groupby('work_key').agg(
    num_ratings=('rating', 'count'),
    avg_rating=('rating', 'mean'),
).reset_index().merge(rep, on='work_key')

C = df['rating'].mean()
m = stats['num_ratings'].quantile(M_PERCENTILE)
print(f"C (điểm TB toàn hệ thống) = {C:.3f} | m (ngưỡng tin cậy) = {m:.0f} lượt")

v, R = stats['num_ratings'], stats['avg_rating']
stats['weighted_score'] = (v / (v + m)) * R + (m / (v + m)) * C

info = books[['book_id', 'title', 'authors', 'genres']]
baseline = stats.merge(info, on='book_id', how='left')
baseline = baseline.sort_values('weighted_score', ascending=False).reset_index(drop=True)
baseline[['avg_rating', 'weighted_score']] = baseline[['avg_rating', 'weighted_score']].round(4)
baseline.to_csv(OUT_BASELINE, index=False)

show = ['title', 'num_ratings', 'avg_rating', 'weighted_score']
print("\nTop 10 Baseline (weighted rating):")
print(baseline[show].head(10).assign(title=lambda d: d['title'].str[:55]).to_string(index=False))

naive = stats.merge(info, on='book_id', how='left').sort_values(
    ['avg_rating', 'num_ratings'], ascending=[False, False])
print("\n[So sánh] Top 5 nếu xếp theo điểm trung bình thuần:")
print(naive[['title', 'num_ratings', 'avg_rating']].head(5)
      .assign(title=lambda d: d['title'].str[:55]).to_string(index=False))
print(f"-> Đã lưu {OUT_BASELINE} ({len(baseline):,} tác phẩm)")

# ===== PHẦN 2: TF-IDF trên cột content =====
print("\n=== PHẦN 2: TF-IDF ===")
tfidf = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.5,
    max_features=50000,
    sublinear_tf=True,
)
tfidf_matrix = tfidf.fit_transform(books['content'])
print(f"Ma trận TF-IDF: {tfidf_matrix.shape[0]:,} sách x {tfidf_matrix.shape[1]:,} từ/cụm từ")

joblib.dump(tfidf_matrix, OUT_TFIDF_MATRIX)
joblib.dump(tfidf, OUT_TFIDF_VEC)
print(f"-> Đã lưu {OUT_TFIDF_MATRIX} và {OUT_TFIDF_VEC}")
print("(Thứ tự hàng của ma trận = thứ tự dòng trong books_clean.csv)")


def _chon_top(sim, top_n, loai_tru=None):
    """Lấy top_n kết quả, mỗi tác phẩm chỉ xuất hiện một lần, bỏ tác phẩm tham chiếu."""
    seen = {loai_tru} if loai_tru else set()
    chon = []
    for i in sim.argsort()[::-1]:
        k = work_keys[i]
        if k in seen:
            continue
        seen.add(k)
        chon.append(i)
        if len(chon) == top_n:
            break
    return chon


def _hien_thi(idx_list, scores):
    out = books.iloc[idx_list][['title', 'authors', 'genres']].copy()
    out['do_tuong_dong'] = np.round(scores, 3)
    out['title'] = out['title'].str[:50]
    out['authors'] = out['authors'].str[:25]
    out['genres'] = out['genres'].str[:35]
    print(out.to_string(index=False))


def goi_y_theo_sach(tu_khoa_ten, top_n=5):
    """Sách tương tự một cuốn sách (Content-based)."""
    t = books['title'].str.lower()
    kw = tu_khoa_ten.lower()
    mask = t.str.startswith(kw).values
    if not mask.any():
        mask = t.str.contains(kw, regex=False).values
    if not mask.any():
        print(f"(Không tìm thấy sách chứa '{tu_khoa_ten}' trong tập dữ liệu)")
        return
    idx = int(np.argmax(np.where(mask, books['ratings_count'].values, -1)))  # ấn bản phổ biến nhất
    print(f"\nSách tham chiếu: {books.iloc[idx]['title']}")
    sim = linear_kernel(tfidf_matrix[idx], tfidf_matrix).ravel()
    top = _chon_top(sim, top_n, loai_tru=work_keys[idx])
    _hien_thi(top, sim[top])


def goi_y_theo_van_ban(van_ban, top_n=5):
    """Gợi ý từ mô tả tự do (khảo sát) -> dùng cho người dùng mới (cold-start)."""
    sim = linear_kernel(tfidf.transform([van_ban]), tfidf_matrix).ravel()
    top = _chon_top(sim, top_n)
    print(f"\nTruy vấn: '{van_ban}'")
    _hien_thi(top, sim[top])


print("\n--- CHẠY THỬ TF-IDF ---")
goi_y_theo_sach("Harry Potter and the Sorcerer's Stone")
goi_y_theo_sach("Pride and Prejudice")
goi_y_theo_sach("The Hunger Games")
goi_y_theo_van_ban("romance historical fiction")
goi_y_theo_van_ban("mystery thriller detective murder")
print("\nHoàn tất bước Baseline + TF-IDF.")