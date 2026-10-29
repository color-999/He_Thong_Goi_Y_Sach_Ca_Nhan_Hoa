"""
setup_sqlserver.py - Tạo cơ sở dữ liệu SQL Server cho hệ thống gợi ý sách

Các bảng:
  Books                   danh mục sách (từ books_clean.csv: tên, tác giả, tóm tắt, thể loại, ảnh...)
  Reviews                 đánh giá của bộ dữ liệu Goodreads (cleaned_goodreads_reviews.csv)
  Users                   tài khoản của website (user / admin)
  UserPreferenceProfiles  hồ sơ sở thích từ bài khảo sát
  UserRatings             sách do người dùng của website chấm điểm
  WantToRead              danh sách "Muốn đọc"
  ModelEvaluations        kết quả đánh giá mô hình (để admin xem)

Cách chạy : python setup_sqlserver.py            (an toàn khi chạy lại: không xóa dữ liệu người dùng)
            python setup_sqlserver.py --nap-lai  (xóa và nạp lại Books/Reviews/ModelEvaluations)
"""
import getpass
import hashlib
import os
import sys
import time
import uuid

import pandas as pd

SERVER = r'.\SQLEXPRESS01'
DATABASE = 'HeThongGoiYSach'
CONN_URL = (f'mssql+pyodbc://@{SERVER}/{DATABASE}'
            '?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes')

BOOKS_CSV = 'books_clean.csv'
REVIEWS_CSV = 'cleaned_goodreads_reviews.csv'
EVAL_CSVS = [('ket_qua_danh_gia.csv', 'Người dùng cũ'),
             ('ket_qua_mo_hinh_lai.csv', None)]       # file thứ hai đã có cột "Kịch bản"
PBKDF2_ROUNDS = 200_000


# ======================================================================
# Hàm thuần (không cần kết nối SQL)
def bam_mat_khau(mat_khau):
    """Băm mật khẩu: pbkdf2_sha256$số_vòng$salt$hash (api.py dùng cùng định dạng để kiểm tra)."""
    salt = os.urandom(16).hex()
    h = hashlib.pbkdf2_hmac('sha256', mat_khau.encode('utf-8'), bytes.fromhex(salt), PBKDF2_ROUNDS).hex()
    return f'pbkdf2_sha256${PBKDF2_ROUNDS}${salt}${h}'


def chuan_bi_books(path=BOOKS_CSV):
    b = pd.read_csv(path, dtype={'work_id': str})
    if 'work_id' not in b.columns:
        b['work_id'] = ''
    gioi_han = [('work_id', 20), ('title', 1000), ('authors', 1000), ('publisher', 300),
                ('genres', 300), ('shelves', 500), ('image_url', 500)]
    for cot, n in gioi_han:
        b[cot] = b[cot].fillna('').astype(str).str.slice(0, n)
    b['description'] = b['description'].fillna('').astype(str)
    b['average_rating'] = pd.to_numeric(b['average_rating'], errors='coerce')
    b['ratings_count'] = pd.to_numeric(b['ratings_count'], errors='coerce').astype('Int64')
    b = b[b['title'].str.strip() != ''].drop_duplicates('book_id')
    return b[['book_id', 'work_id', 'title', 'authors', 'publisher', 'description',
              'genres', 'shelves', 'average_rating', 'ratings_count', 'image_url']]


def chuan_bi_reviews(path=REVIEWS_CSV):
    r = pd.read_csv(path, usecols=['user_id', 'book_id', 'rating', 'date_added']).dropna()
    r = r[r['rating'].between(0, 5)].copy()
    r['user_id'] = r['user_id'].astype(str).str.slice(0, 40)
    r['book_id'] = r['book_id'].astype('int64')
    r['rating'] = r['rating'].astype('int64')
    r['date_added'] = pd.to_datetime(r['date_added'], utc=True, format='mixed').dt.tz_localize(None)
    return r


def chuan_bi_ket_qua():
    """Gộp các file kết quả đánh giá mô hình thành một bảng cho ModelEvaluations."""
    dong = []
    for path, kich_ban_mac_dinh in EVAL_CSVS:
        if not os.path.exists(path):
            print(f"   (bỏ qua {path}: chưa có file)")
            continue
        df = pd.read_csv(path)

        def cot(tien_to):
            return next((c for c in df.columns if str(c).startswith(tien_to)), None)

        c_p, c_r, c_c, c_d = cot('Precision@'), cot('Recall@'), cot('Coverage@'), cot('Diversity@')
        for _, row in df.iterrows():
            def v(c):
                return None if (c is None or c not in df.columns or pd.isna(row[c])) else float(row[c])
            dong.append({
                'scenario': str(row['Kịch bản']) if 'Kịch bản' in df.columns else kich_ban_mac_dinh,
                'model_name': str(row['Mô hình']),
                'rmse': v('RMSE'), 'precision_k': v(c_p), 'recall_k': v(c_r),
                'coverage_k': v(c_c), 'diversity_k': v(c_d),
                'train_seconds': v('Thời gian huấn luyện (s)'),
                'source': path,
            })
    return pd.DataFrame(dong)


# ======================================================================
DDL = {
    'Books': """CREATE TABLE dbo.Books (
        book_id INT NOT NULL PRIMARY KEY,
        work_id NVARCHAR(20) NULL,
        title NVARCHAR(1000) NOT NULL,
        authors NVARCHAR(1000) NULL,
        publisher NVARCHAR(300) NULL,
        description NVARCHAR(MAX) NULL,
        genres NVARCHAR(300) NULL,
        shelves NVARCHAR(500) NULL,
        average_rating FLOAT NULL,
        ratings_count INT NULL,
        image_url NVARCHAR(500) NULL)""",
    'Reviews': """CREATE TABLE dbo.Reviews (
        user_id NVARCHAR(40) NOT NULL,
        book_id INT NOT NULL,
        rating TINYINT NOT NULL,
        date_added DATETIME2 NULL)""",
    'Users': """CREATE TABLE dbo.Users (
        user_id NVARCHAR(100) NOT NULL PRIMARY KEY,
        username NVARCHAR(100) NOT NULL UNIQUE,
        password_hash NVARCHAR(300) NOT NULL,
        role NVARCHAR(20) NOT NULL DEFAULT 'user',
        created_at DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME())""",
    'UserPreferenceProfiles': """CREATE TABLE dbo.UserPreferenceProfiles (
        user_id NVARCHAR(100) NOT NULL PRIMARY KEY,
        genres NVARCHAR(MAX) NOT NULL DEFAULT '[]',
        authors NVARCHAR(MAX) NOT NULL DEFAULT '[]',
        books NVARCHAR(MAX) NOT NULL DEFAULT '[]',
        profile_source NVARCHAR(30) NOT NULL DEFAULT 'survey',
        updated_at DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME())""",
    'UserRatings': """CREATE TABLE dbo.UserRatings (
        user_id NVARCHAR(100) NOT NULL,
        book_id INT NOT NULL,
        rating TINYINT NOT NULL CHECK (rating BETWEEN 1 AND 5),
        rated_at DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME(),
        CONSTRAINT PK_UserRatings PRIMARY KEY (user_id, book_id),
        CONSTRAINT FK_UserRatings_Books FOREIGN KEY (book_id) REFERENCES dbo.Books(book_id))""",
    'WantToRead': """CREATE TABLE dbo.WantToRead (
        user_id NVARCHAR(100) NOT NULL,
        book_id INT NOT NULL,
        added_at DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME(),
        CONSTRAINT PK_WantToRead PRIMARY KEY (user_id, book_id),
        CONSTRAINT FK_WantToRead_Books FOREIGN KEY (book_id) REFERENCES dbo.Books(book_id))""",
    'ModelEvaluations': """CREATE TABLE dbo.ModelEvaluations (
        eval_id INT IDENTITY(1,1) PRIMARY KEY,
        run_at DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME(),
        scenario NVARCHAR(100) NULL,
        model_name NVARCHAR(100) NOT NULL,
        rmse FLOAT NULL,
        precision_k FLOAT NULL,
        recall_k FLOAT NULL,
        coverage_k FLOAT NULL,
        diversity_k FLOAT NULL,
        train_seconds FLOAT NULL,
        source NVARCHAR(100) NULL)""",
}

INDEXES = [
    ('IX_Reviews_user', 'dbo.Reviews', 'user_id'),
    ('IX_Reviews_book', 'dbo.Reviews', 'book_id'),
    ('IX_Books_work', 'dbo.Books', 'work_id'),
]


def main():
    from sqlalchemy import create_engine, text   # import tại đây để các hàm thuần ở trên chạy được độc lập

    nap_lai = '--nap-lai' in sys.argv
    print("Đang kết nối SQL Server...")
    engine = create_engine(CONN_URL)                              # dùng cho Books (có cột văn bản rất dài)
    engine_fast = create_engine(CONN_URL, fast_executemany=True)  # dùng cho Reviews (1,3 triệu dòng)

    def chay(sql, **tham_so):
        with engine.begin() as conn:
            return conn.execute(text(sql), tham_so)

    def lay_mot(sql, **tham_so):
        with engine.connect() as conn:
            return conn.execute(text(sql), tham_so).fetchone()

    def ton_tai(ten):
        return lay_mot("SELECT OBJECT_ID(:t, 'U')", t=f'dbo.{ten}')[0] is not None

    def dem(ten):
        return lay_mot(f"SELECT COUNT(*) FROM dbo.{ten}")[0]

    def cac_cot(ten):
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT COLUMN_NAME, CHARACTER_MAXIMUM_LENGTH FROM INFORMATION_SCHEMA.COLUMNS "
                                     "WHERE TABLE_SCHEMA='dbo' AND TABLE_NAME=:t"), {'t': ten}).fetchall()
        return {r[0]: r[1] for r in rows}

    print(f"-> Đã kết nối cơ sở dữ liệu: {lay_mot('SELECT DB_NAME()')[0]}")

    # ---------- 1. Bảng cũ (tạo bởi phiên bản setup_sqlserver.py trước) -> tạo lại theo cấu trúc mới ----------
    if ton_tai('Books') and 'description' not in cac_cot('Books'):
        print("1. Phát hiện bảng Books/Reviews cũ (thiếu cột mới), sẽ xóa và tạo lại từ file CSV.")
        if ton_tai('Reviews'):
            chay("DROP TABLE dbo.Reviews")
        chay("DROP TABLE dbo.Books")

    # ---------- 2. Tạo bảng nếu chưa có ----------
    print("2. Đang tạo các bảng (nếu chưa có)...")
    for ten, sql in DDL.items():
        if ton_tai(ten):
            print(f"   {ten}: đã có")
        else:
            chay(sql)
            print(f"   {ten}: đã tạo")

    # ---------- 3. Nạp Books ----------
    print("3. Bảng Books...")
    if nap_lai and dem('Books') > 0:
        try:
            chay("DELETE FROM dbo.Books")
        except Exception as e:
            print(f"   Không xóa được Books (đang có người dùng đánh giá/Muốn đọc tham chiếu tới): {e}")
    if dem('Books') == 0:
        b = chuan_bi_books()
        t0 = time.time()
        b.to_sql('Books', engine, schema='dbo', if_exists='append', index=False, chunksize=500)
        print(f"   -> Đã nạp {len(b):,} cuốn sách ({time.time() - t0:.0f}s).")
    else:
        print(f"   -> Đã có {dem('Books'):,} cuốn, bỏ qua (dùng --nap-lai để nạp lại).")

    # ---------- 4. Nạp Reviews ----------
    print("4. Bảng Reviews (1,3 triệu dòng, mất vài phút)...")
    if nap_lai and dem('Reviews') > 0:
        chay("TRUNCATE TABLE dbo.Reviews")
    if dem('Reviews') == 0:
        r = chuan_bi_reviews()
        t0 = time.time()
        buoc = 100_000
        for i in range(0, len(r), buoc):
            r.iloc[i:i + buoc].to_sql('Reviews', engine_fast, schema='dbo', if_exists='append',
                                      index=False, chunksize=20_000)
            print(f"   ...{min(i + buoc, len(r)):,}/{len(r):,}")
        print(f"   -> Đã nạp {len(r):,} đánh giá ({time.time() - t0:.0f}s).")
    else:
        print(f"   -> Đã có {dem('Reviews'):,} đánh giá, bỏ qua.")

    for ten_idx, bang, cot in INDEXES:
        chay(f"IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name='{ten_idx}' AND object_id=OBJECT_ID('{bang}')) "
             f"CREATE INDEX {ten_idx} ON {bang}({cot})")
    print("   -> Đã tạo chỉ mục tìm kiếm.")

    # ---------- 5. Kết quả đánh giá mô hình ----------
    print("5. Bảng ModelEvaluations...")
    if nap_lai and dem('ModelEvaluations') > 0:
        chay("DELETE FROM dbo.ModelEvaluations")
    if dem('ModelEvaluations') == 0:
        kq = chuan_bi_ket_qua()
        if len(kq):
            kq.to_sql('ModelEvaluations', engine, schema='dbo', if_exists='append', index=False)
            print(f"   -> Đã nạp {len(kq)} dòng kết quả.")
        else:
            print("   -> Chưa có file kết quả nào để nạp.")
    else:
        print(f"   -> Đã có {dem('ModelEvaluations')} dòng, bỏ qua.")

    # ---------- 6. Tài khoản: băm mật khẩu cũ, tạo admin ----------
    print("6. Bảng Users...")
    cot_users = cac_cot('Users')
    dai = cot_users.get('password_hash')
    if dai is not None and dai != -1 and dai < 200:
        try:
            chay("ALTER TABLE dbo.Users ALTER COLUMN password_hash NVARCHAR(300) NOT NULL")
            print("   -> Đã mở rộng cột password_hash (để chứa mật khẩu đã băm).")
        except Exception as e:
            print(f"   [CẢNH BÁO] Không mở rộng được password_hash: {e}")
    with engine.begin() as conn:
        cu = conn.execute(text("SELECT user_id, password_hash FROM dbo.Users "
                               "WHERE password_hash IS NOT NULL AND password_hash NOT LIKE 'pbkdf2_sha256$%'")).fetchall()
        for uid, mat_khau in cu:
            conn.execute(text("UPDATE dbo.Users SET password_hash=:h WHERE user_id=:u"),
                         {'h': bam_mat_khau(mat_khau), 'u': uid})
    if cu:
        print(f"   -> Đã băm lại mật khẩu của {len(cu)} tài khoản cũ (mật khẩu đăng nhập không đổi).")

    if lay_mot("SELECT COUNT(*) FROM dbo.Users WHERE role='admin'")[0] == 0:
        if sys.stdin.isatty():
            print("   Chưa có tài khoản admin. Tạo ngay (bỏ trống tên đăng nhập để bỏ qua):")
            ten = input("   Tên đăng nhập admin [admin]: ").strip() or 'admin'
            mk = getpass.getpass("   Mật khẩu admin (không hiện khi gõ, bỏ trống để bỏ qua): ")
            if mk:
                if lay_mot("SELECT COUNT(*) FROM dbo.Users WHERE username=:u", u=ten)[0]:
                    chay("UPDATE dbo.Users SET role='admin', password_hash=:h WHERE username=:u",
                         h=bam_mat_khau(mk), u=ten)
                else:
                    chay("INSERT INTO dbo.Users (user_id, username, password_hash, role) VALUES (:i, :u, :h, 'admin')",
                         i=str(uuid.uuid4()), u=ten, h=bam_mat_khau(mk))
                print(f"   -> Đã tạo/cập nhật admin '{ten}'.")
            else:
                print("   -> Bỏ qua tạo admin.")
        else:
            print("   Chưa có admin; hãy chạy lại trong terminal để tạo.")

    # ---------- Tổng kết ----------
    print("\n===== TỔNG KẾT CƠ SỞ DỮ LIỆU =====")
    for ten in DDL:
        print(f"{ten:24s}: {dem(ten):>10,} dòng")
    print("===================================")
    print("Hoàn tất. Cơ sở dữ liệu đã sẵn sàng cho API.")


if __name__ == '__main__':
    main()