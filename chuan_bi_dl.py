"""
BƯỚC 1 (theo đề cương): Chuẩn hóa dữ liệu sách.

Vấn đề cần sửa:
  - Đánh giá lấy từ Goodreads UCSD (book_id của Goodreads), còn tên sách lại lấy
    từ archive/book1-100k.csv (bộ Kaggle khác, cột Id có hệ đánh số riêng).
    Hai hệ ID này không đảm bảo trùng nhau -> ghép sai/ra rỗng.

Cách sửa:
  - Dùng metadata chính thức của UCSD Book Graph (cùng hệ book_id với đánh giá).
  - Chỉ giữ các sách có trong cleaned_goodreads_reviews.csv.
  - Bổ sung tóm tắt (description) và thể loại, đúng như đề cương mô tả cho TF-IDF.

Các file cần tải về (cùng thư mục với script), từ
https://mengtingwan.github.io/data/goodreads.html :
  - goodreads_books.json.gz
  - goodreads_book_authors.json.gz
  - goodreads_book_genres_initial.json.gz

Kết quả: books_clean.csv
  book_id, title, authors, publisher, description, genres, shelves,
  average_rating, ratings_count, image_url, content
  (genres và shelves ngăn cách nhau bằng dấu '|')
"""
import gzip
import json
import os
import re
import sys

import pandas as pd

REVIEWS_CSV = 'cleaned_goodreads_reviews.csv'
BOOKS_GZ = 'goodreads_books.json.gz'
AUTHORS_GZ = 'goodreads_book_authors.json.gz'
GENRES_GZ = 'goodreads_book_genres_initial.json.gz'
OLD_BOOKS_CSV = 'archive/book1-100k.csv'   # chỉ dùng để kiểm tra mức lệch ID
OUT_CSV = 'books_clean.csv'

# Các "kệ sách" chung chung, không mô tả nội dung -> bỏ qua
GENERIC_SHELVES = {
    'to-read', 'currently-reading', 'favorites', 'favourites', 'owned', 'default',
    'books-i-own', 'library', 'kindle', 'ebook', 'e-book', 'ebooks', 'audiobook',
    'audiobooks', 'my-books', 'wish-list', 'wishlist', 'to-buy', 'read', 're-read',
    'own', 'owned-books', 'favorite', 'favorite-books', 'books', 'novels', 'fiction',
    'did-not-finish', 'dnf', 'unfinished', 'abandoned', 'read-in-2015', 'read-in-2016',
    'read-in-2017', 'read-in-2018', 'read-in-2014', 'read-in-2013', 'owned-and-read',
    'to-read-fiction', 'tbr', 'audible', 'paperback', 'hardcover', 'nook', 'e-books',
}


def read_jsonl(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def clean_id(series):
    return series.astype(str).str.replace(r'\.0$', '', regex=True).str.strip()


def strip_series(title):
    """Bỏ phần '(Harry Potter, #1)' ở cuối tên để so sánh tên sách."""
    return re.sub(r'\s*\(.*?\)\s*$', '', str(title)).strip().lower()


def main():
    # ---------- 0. Kiểm tra file đầu vào ----------
    if not os.path.exists(REVIEWS_CSV):
        sys.exit(f"Không thấy {REVIEWS_CSV}. Hãy chạy count_data.py trước.")
    if not os.path.exists(BOOKS_GZ):
        sys.exit(f"Không thấy {BOOKS_GZ}. Tải từ trang UCSD Book Graph rồi đặt cạnh script này.")

    # ---------- 1. Tập book_id có trong dữ liệu đánh giá ----------
    print("1. Đang đọc danh sách book_id từ dữ liệu đánh giá...")
    review_ids = set(clean_id(pd.read_csv(REVIEWS_CSV, usecols=['book_id'])['book_id']).unique())
    print(f"   -> {len(review_ids):,} cuốn sách có đánh giá.")

    # ---------- 2. Tên tác giả ----------
    author_names = {}
    if os.path.exists(AUTHORS_GZ):
        print("2. Đang đọc bảng tác giả...")
        for rec in read_jsonl(AUTHORS_GZ):
            author_names[str(rec.get('author_id'))] = rec.get('name', '')
        print(f"   -> {len(author_names):,} tác giả.")
    else:
        print(f"2. [CẢNH BÁO] Thiếu {AUTHORS_GZ}: cột authors sẽ để trống.")

    # ---------- 3. Thể loại ----------
    genres_map = {}
    if os.path.exists(GENRES_GZ):
        print("3. Đang đọc thể loại...")
        for rec in read_jsonl(GENRES_GZ):
            bid = str(rec.get('book_id'))
            if bid in review_ids:
                g = rec.get('genres') or {}
                top = sorted(g.items(), key=lambda kv: kv[1], reverse=True)[:3]
                genres_map[bid] = '|'.join(name for name, _ in top)
        print(f"   -> {len(genres_map):,} sách có thể loại.")
    else:
        print(f"3. [CẢNH BÁO] Thiếu {GENRES_GZ}: cột genres sẽ để trống.")

    # ---------- 4. Metadata sách ----------
    print("4. Đang đọc metadata sách (file lớn, có thể mất vài phút)...")
    rows = []
    seen = 0
    for rec in read_jsonl(BOOKS_GZ):
        seen += 1
        if seen % 500_000 == 0:
            print(f"   ...đã quét {seen:,} dòng, tìm thấy {len(rows):,}/{len(review_ids):,}")
        bid = str(rec.get('book_id'))
        if bid not in review_ids:
            continue

        names = [author_names.get(str(a.get('author_id')), '') for a in (rec.get('authors') or [])]
        authors = ', '.join(n for n in names[:3] if n)

        shelves = []
        for s in sorted(rec.get('popular_shelves') or [],
                        key=lambda x: int(x.get('count', 0) or 0), reverse=True):
            name = str(s.get('name', '')).strip().lower()
            if name and name not in GENERIC_SHELVES and not name.startswith('read-in'):
                shelves.append(name)
            if len(shelves) >= 8:
                break

        title = rec.get('title') or rec.get('title_without_series') or ''
        description = re.sub(r'\s+', ' ', str(rec.get('description') or '')).strip()
        genres = genres_map.get(bid, '')

        content = ' '.join([
            title, authors,
            genres.replace('|', ' '),
            ' '.join(sh.replace('-', ' ') for sh in shelves),
            description[:1000],
        ]).strip()

        rows.append({
            'book_id': bid,
            'work_id': rec.get('work_id') or '',
            'title': title,
            'authors': authors,
            'publisher': rec.get('publisher') or '',
            'description': description,
            'genres': genres,
            'shelves': '|'.join(shelves),
            'average_rating': rec.get('average_rating') or '',
            'ratings_count': rec.get('ratings_count') or '',
            'image_url': rec.get('image_url') or '',
            'content': content,
        })
        if len(rows) == len(review_ids):
            break

    df_books = pd.DataFrame(rows)
    print(f"   -> Tìm thấy metadata cho {len(df_books):,}/{len(review_ids):,} cuốn "
          f"({len(df_books) / max(len(review_ids), 1):.1%}).")
    print(f"   -> Có tóm tắt: {(df_books['description'] != '').sum():,} | "
          f"Có thể loại: {(df_books['genres'] != '').sum():,} | "
          f"Có tác giả: {(df_books['authors'] != '').sum():,}")

    df_books.to_csv(OUT_CSV, index=False)
    print(f"5. Đã lưu {OUT_CSV}")

    # ---------- 5. Chẩn đoán: archive/book1-100k.csv có cùng hệ ID không? ----------
    if os.path.exists(OLD_BOOKS_CSV):
        print("\n--- KIỂM TRA book1-100k.csv ---")
        old = pd.read_csv(OLD_BOOKS_CSV, usecols=['Id', 'Name'])
        old['Id'] = clean_id(old['Id'])
        merged = df_books[['book_id', 'title']].merge(old, left_on='book_id', right_on='Id', how='inner')
        print(f"Số ID trùng giữa hai nguồn: {len(merged):,}/{len(df_books):,}")
        if len(merged):
            same = (merged['title'].map(strip_series) == merged['Name'].map(strip_series))
            rate = same.mean()
            print(f"Tỷ lệ tên sách khớp trên các ID trùng: {rate:.1%}")
            print(merged.assign(khop=same).head(5)[['book_id', 'title', 'Name', 'khop']].to_string(index=False))
            if rate < 0.5:
                print("=> KẾT LUẬN: book1-100k.csv KHÔNG cùng hệ ID với dữ liệu đánh giá. "
                      "Hãy dùng books_clean.csv thay cho nó.")
            else:
                print("=> Hai nguồn khá khớp, nhưng vẫn nên dùng books_clean.csv vì có tóm tắt và thể loại.")
        else:
            print("=> Không có ID nào trùng. Hãy dùng books_clean.csv.")


if __name__ == '__main__':
    main()