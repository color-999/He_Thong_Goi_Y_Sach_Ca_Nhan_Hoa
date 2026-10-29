# ============================================================
# API HE THONG GOI Y SACH
#
# Chay API:
# C:\Users\Lenovo\AppData\Local\Programs\Python\Python313\python.exe -m uvicorn api:app --reload
#
# Xem API:
# http://localhost:8000/
# http://localhost:8000/api/books
# http://localhost:8000/docs
#
# Thu muc anh:
# frontend/bia_sach/
# ============================================================

from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional
from contextlib import contextmanager
import pyodbc
import pandas as pd
import joblib
import uuid
import json
import re
import os
import hmac
import base64
import hashlib
import secrets

from sklearn.metrics.pairwise import linear_kernel


# ============================================================
# 1. KHOI TAO API
# ============================================================

app = FastAPI(title="API He Thong Goi Y Sach")

CORS_ORIGINS = [
    x.strip()
    for x in os.getenv("CORS_ORIGINS", "null").split(",")
    if x.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# 2. CAU HINH THU MUC ANH BIA
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "frontend" / "bia_sach"

if IMAGE_DIR.is_dir():
    app.mount(
        "/bia_sach",
        StaticFiles(directory=str(IMAGE_DIR)),
        name="bia_sach"
    )
else:
    print(f"CANH BAO: Khong tim thay thu muc anh: {IMAGE_DIR}")

IMAGE_URL_MAP = {}
IMAGE_URL_DA_NAP = False


def chuan_hoa_book_id(book_id):
    text = str(book_id).strip()
    try:
        return str(int(float(text)))
    except Exception:
        return text


def nap_image_url_tu_sql():
    global IMAGE_URL_MAP, IMAGE_URL_DA_NAP
    IMAGE_URL_DA_NAP = True
    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT book_id, image_url
                FROM Books
                WHERE image_url IS NOT NULL
            """)
            IMAGE_URL_MAP = {
                chuan_hoa_book_id(row[0]): str(row[1]).strip()
                for row in cur.fetchall()
                if row[1] and str(row[1]).strip()
            }
        print(f"-> Nap {len(IMAGE_URL_MAP)} image_url tu SQL Server.")
    except Exception as e:
        print(f"-> Chua nap duoc image_url tu SQL: {e}")


def tim_anh_bia(book_id, title="", image_url=""):
    if not image_url or not str(image_url).strip().startswith(("https://", "http://")):
        if not IMAGE_URL_DA_NAP:
            nap_image_url_tu_sql()
        image_url = IMAGE_URL_MAP.get(chuan_hoa_book_id(book_id), "")

    if (
        image_url
        and str(image_url).strip().startswith(("https://", "http://"))
        and "nophoto" not in str(image_url)
    ):
        return str(image_url).strip()

    for duoi in [".jpg", ".jpeg", ".png", ".jfif", ".webp"]:
        duong_dan = IMAGE_DIR / f"{book_id}{duoi}"
        if duong_dan.is_file():
            return f"bia_sach/{duong_dan.name}"

    title_lower = str(title).lower()
    if "harry" in title_lower:
        match = re.search(r"#\s*(\d+)", title_lower)
        if match:
            tap_so = match.group(1)
            for ext in [".jpg", ".jpeg", ".png", ".jfif", ".webp"]:
                for prefix in ["harry_potter_", "Harry_Potter_"]:
                    ten_file = f"{prefix}{tap_so}{ext}"
                    if (IMAGE_DIR / ten_file).is_file():
                        return f"bia_sach/{ten_file}"

    return "bia_sach/bia_sach_chx_co.png"


# ============================================================
# 3. CAC MODEL DU LIEU
# ============================================================

class UserAuth(BaseModel):
    username: str
    password: str


class UserProfile(BaseModel):
    user_id: str
    genres: List[str] = Field(default_factory=list)
    authors: List[str] = Field(default_factory=list)
    books: List[str] = Field(default_factory=list)
    profile_source: str = "survey"


class FavoriteBook(BaseModel):
    id: str
    title: str = ""
    author: str = ""
    img: str = ""


class FavoritesPayload(BaseModel):
    user_id: str
    favorites: List[FavoriteBook] = Field(default_factory=list)


class UserPref(BaseModel):
    user_id: str
    favorite_genres: str = ""

class ReviewPayload(BaseModel):
    user_id: str
    book_id: str
    rating: float = Field(ge=1, le=5)
    comment: str = ""


class ReviewDelete(BaseModel):
    user_id: str
    review_id: int
# ============================================================
# 4. NAP MO HINH GOI Y AI
# ============================================================

svd_model = None
tfidf_matrix = None
df_books = pd.DataFrame()

try:
    svd_model = joblib.load(BASE_DIR / "svd_model.pkl")
    tfidf_matrix = joblib.load(BASE_DIR / "tfidf_matrix.pkl")
    df_books = joblib.load(BASE_DIR / "df_books.pkl")

    # SUA: fallback Name/Authors de tranh KeyError neu .pkl chi co title/authors
    if isinstance(df_books, pd.DataFrame) and not df_books.empty:
        if "Name" not in df_books.columns and "title" in df_books.columns:
            df_books["Name"] = df_books["title"]
        if "Authors" not in df_books.columns and "authors" in df_books.columns:
            df_books["Authors"] = df_books["authors"]

    print("-> Nap mo hinh AI thanh cong.")

except Exception as e:
    print(f"-> Chua nap du mo hinh .pkl: {e}")
    print("-> API van chay cac chuc nang khong phu thuoc mo hinh.")


THE_LOAI_MAP = None


def lay_the_loai(book_id):
    global THE_LOAI_MAP

    if THE_LOAI_MAP is None:
        THE_LOAI_MAP = {}
        try:
            cot = next(
                (c for c in ["genres", "Genres", "genre", "Genre",
                             "categories", "Categories", "the_loai"]
                 if c in df_books.columns),
                None
            )

            if cot and "book_id" in df_books.columns:
                for bid, val in zip(df_books["book_id"], df_books[cot]):
                    if val is None:
                        items = []
                    elif hasattr(val, "__iter__") and not isinstance(val, str):
                        items = list(val)
                    elif str(val).lower() == "nan":
                        items = []
                    else:
                        items = re.split(r"[,;|]", str(val))

                    THE_LOAI_MAP[chuan_hoa_book_id(bid)] = [
                        str(x).strip() for x in items if str(x).strip()
                    ][:3]
        except Exception as e:
            print(f"-> Khong doc duoc the loai tu df_books: {e}")

    return THE_LOAI_MAP.get(chuan_hoa_book_id(book_id), [])


# ============================================================
# 5. KET NOI SQL SERVER
# ============================================================

conn_str = (
    r"DRIVER={ODBC Driver 17 for SQL Server};"
    r"SERVER=.\SQLEXPRESS01;"
    r"DATABASE=HeThongGoiYSach;"
    r"Trusted_Connection=yes;"
)


@contextmanager
def get_conn():
    conn = pyodbc.connect(conn_str)
    try:
        yield conn
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        conn.close()


PASSWORD_PREFIX = "p1$"
PASSWORD_ITERATIONS = 200_000


def _b64e(raw):
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64d(text):
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def hash_password(password):
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", str(password).encode("utf-8"), salt, PASSWORD_ITERATIONS
    )
    return PASSWORD_PREFIX + _b64e(salt) + "$" + _b64e(digest)


def verify_password(password, stored):
    stored = str(stored or "")

    if stored.startswith(PASSWORD_PREFIX):
        try:
            _, salt_text, hash_text = stored.split("$")
            expected = _b64d(hash_text)
            actual = hashlib.pbkdf2_hmac(
                "sha256",
                str(password).encode("utf-8"),
                _b64d(salt_text),
                PASSWORD_ITERATIONS
            )
            return hmac.compare_digest(actual, expected)
        except Exception:
            return False

    return hmac.compare_digest(
        stored.rstrip().encode("utf-8"),
        str(password).rstrip().encode("utf-8")
    )


def require_admin(
    x_user_id: Optional[str] = Header(default=None),
    admin_id: Optional[str] = None
):
    uid = (x_user_id or admin_id or "").strip()

    if not uid:
        raise HTTPException(
            status_code=401,
            detail="Can dang nhap bang tai khoan quan tri."
        )

    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT role FROM Users WHERE user_id = ?", uid)
            row = cur.fetchone()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not row or str(row[0] or "").strip().lower() != "admin":
        raise HTTPException(
            status_code=403,
            detail="Ban khong co quyen quan tri."
        )

    return uid


# ============================================================
# 6. TAO CAC BANG CA NHAN HOA NEU CHUA CO
# ============================================================

def ensure_personalization_tables(conn):
    cur = conn.cursor()

    cur.execute("""
        IF OBJECT_ID('dbo.UserPreferenceProfiles', 'U') IS NULL
        CREATE TABLE dbo.UserPreferenceProfiles (
            user_id NVARCHAR(100) NOT NULL PRIMARY KEY,
            genres NVARCHAR(MAX) NOT NULL DEFAULT '[]',
            authors NVARCHAR(MAX) NOT NULL DEFAULT '[]',
            books NVARCHAR(MAX) NOT NULL DEFAULT '[]',
            profile_source NVARCHAR(30) NOT NULL DEFAULT 'survey',
            updated_at DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME()
        )
    """)

    cur.execute("""
        IF OBJECT_ID('dbo.UserFavorites', 'U') IS NULL
        CREATE TABLE dbo.UserFavorites (
            user_id NVARCHAR(100) NOT NULL PRIMARY KEY,
            favorites_json NVARCHAR(MAX) NOT NULL DEFAULT '[]',
            updated_at DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME()
        )
    """)

    conn.commit()

def ensure_reviews_table(conn):
    """Tạo bảng Reviews nếu chưa có (đánh giá sách của người dùng)."""
    cur = conn.cursor()
    cur.execute("""
        IF OBJECT_ID('dbo.Reviews', 'U') IS NULL
        CREATE TABLE dbo.Reviews (
            review_id INT IDENTITY(1,1) PRIMARY KEY,
            user_id NVARCHAR(100) NOT NULL,
            book_id NVARCHAR(100) NOT NULL,
            rating FLOAT NOT NULL,
            comment NVARCHAR(MAX) NULL,
            created_at DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME()
        )
    """)
    conn.commit()


# ============================================================
# 7. DOC HO SO SO THICH
# ============================================================

def read_profile(cur, user_id):
    cur.execute("""
        SELECT genres, authors, books, profile_source, updated_at
        FROM dbo.UserPreferenceProfiles
        WHERE user_id = ?
    """, user_id)

    row = cur.fetchone()

    if not row:
        try:
            cur.execute("""
                SELECT favorite_genres
                FROM dbo.UserPreferences
                WHERE user_id = ?
            """, user_id)

            legacy = cur.fetchone()

        except Exception:
            legacy = None

        if legacy and legacy[0]:
            return {
                "user_id": user_id,
                "genres": [
                    x.strip()
                    for x in str(legacy[0]).split(",")
                    if x.strip()
                ],
                "authors": [],
                "books": [],
                "profile_source": "survey",
                "updated_at": None
            }

        return {
            "user_id": user_id,
            "genres": [],
            "authors": [],
            "books": [],
            "profile_source": None,
            "updated_at": None
        }

    def parse_list(value):
        try:
            parsed = json.loads(value or "[]")
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []

    return {
        "user_id": user_id,
        "genres": parse_list(row[0]),
        "authors": parse_list(row[1]),
        "books": parse_list(row[2]),
        "profile_source": row[3],
        "updated_at": row[4].isoformat() if row[4] else None
    }


# ============================================================
# 8. DOC DANH SACH SACH YEU THICH
# ============================================================

def read_favorites(cur, user_id):
    cur.execute("""
        SELECT favorites_json
        FROM dbo.UserFavorites
        WHERE user_id = ?
    """, user_id)

    row = cur.fetchone()

    if not row:
        return []

    try:
        data = json.loads(row[0] or "[]")
        return data if isinstance(data, list) else []
    except Exception:
        return []


# ============================================================
# 9. API TRANG CHU
# ============================================================

@app.get("/")
def read_root():
    return {
        "message": "API dang hoat dong",
        "personalization": True
    }


# ============================================================
# 10. API DANG KY
# ============================================================

@app.post("/api/register")
def register_user(user: UserAuth):
    try:
        with get_conn() as conn:
            cur = conn.cursor()

            cur.execute(
                "SELECT username FROM Users WHERE username = ?",
                user.username
            )

            if cur.fetchone():
                return {
                    "error": "Ten dang nhap da ton tai! Vui long chon ten khac."
                }

            new_id = str(uuid.uuid4())

            cur.execute("""
                INSERT INTO Users
                    (user_id, username, password_hash, role)
                VALUES (?, ?, ?, ?)
            """, (new_id, user.username, hash_password(user.password), "user"))

            conn.commit()

        return {
            "message": "Dang ky thanh cong!",
            "user_id": new_id,
            "username": user.username,
            "role": "user"
        }

    except Exception as e:
        return {"error": f"Loi Server: {str(e)}"}


# ============================================================
# 11. API DANG NHAP
# ============================================================

@app.post("/api/login")
def login_user(user: UserAuth):
    try:
        with get_conn() as conn:
            cur = conn.cursor()

            cur.execute("""
                SELECT user_id, username, role, password_hash
                FROM Users
                WHERE username = ?
            """, user.username)

            row = cur.fetchone()

            if row and verify_password(user.password, row[3]):
                if not str(row[3] or "").startswith(PASSWORD_PREFIX):
                    try:
                        cur.execute("""
                            UPDATE Users
                            SET password_hash = ?
                            WHERE user_id = ?
                        """, (hash_password(user.password), row[0]))
                        conn.commit()
                    except Exception:
                        pass
            else:
                row = None

        if row:
            return {
                "message": "Dang nhap thanh cong!",
                "user_id": row[0],
                "username": row[1],
                "role": row[2]
            }

        return {"error": "Sai ten dang nhap hoac mat khau!"}

    except Exception as e:
        return {"error": f"Loi Server: {str(e)}"}


# ============================================================
# 12. API LAY HO SO SO THICH
# ============================================================

@app.get("/api/user/profile/{user_id}")
def get_user_profile(user_id: str):
    try:
        with get_conn() as conn:
            ensure_personalization_tables(conn)
            profile = read_profile(conn.cursor(), user_id)

        return profile

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 13. API LUU HO SO SO THICH
# ============================================================

@app.post("/api/user/profile")
def save_user_profile(profile: UserProfile):
    if not profile.user_id or profile.user_id == "guest":
        raise HTTPException(
            status_code=400,
            detail="Can dang nhap de luu ho so ca nhan."
        )

    try:
        genres = list(dict.fromkeys(
            x.strip() for x in profile.genres if x and x.strip()
        ))

        authors = list(dict.fromkeys(
            x.strip() for x in profile.authors if x and x.strip()
        ))

        books = list(dict.fromkeys(
            x.strip() for x in profile.books if x and x.strip()
        ))

        with get_conn() as conn:
            ensure_personalization_tables(conn)
            cur = conn.cursor()

            cur.execute("""
                UPDATE dbo.UserPreferenceProfiles
                SET genres = ?,
                    authors = ?,
                    books = ?,
                    profile_source = ?,
                    updated_at = SYSUTCDATETIME()
                WHERE user_id = ?
            """, (
                json.dumps(genres, ensure_ascii=False),
                json.dumps(authors, ensure_ascii=False),
                json.dumps(books, ensure_ascii=False),
                profile.profile_source,
                profile.user_id
            ))

            if cur.rowcount == 0:
                cur.execute("""
                    INSERT INTO dbo.UserPreferenceProfiles
                        (user_id, genres, authors, books, profile_source)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    profile.user_id,
                    json.dumps(genres, ensure_ascii=False),
                    json.dumps(authors, ensure_ascii=False),
                    json.dumps(books, ensure_ascii=False),
                    profile.profile_source
                ))

            try:
                cur.execute("""
                    UPDATE dbo.UserPreferences
                    SET favorite_genres = ?
                    WHERE user_id = ?
                """, (", ".join(genres), profile.user_id))

                if cur.rowcount == 0:
                    cur.execute("""
                        INSERT INTO dbo.UserPreferences
                            (user_id, favorite_genres)
                        VALUES (?, ?)
                    """, (profile.user_id, ", ".join(genres)))

            except Exception:
                pass

            conn.commit()

        return {
            "message": "Da luu ho so so thich",
            "profile": {
                "user_id": profile.user_id,
                "genres": genres,
                "authors": authors,
                "books": books,
                "profile_source": profile.profile_source
            }
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 14. API LUU SO THICH THEO DINH DANG CU
# ============================================================

@app.post("/api/user/preferences")
def save_legacy_preferences(pref: UserPref):
    return save_user_profile(
        UserProfile(
            user_id=pref.user_id,
            genres=[
                x.strip()
                for x in pref.favorite_genres.split(",")
                if x.strip()
            ],
            profile_source="survey"
        )
    )


# ============================================================
# 15. API KIEM TRA DA CO SO THICH HAY CHUA
# ============================================================

@app.get("/api/user/preferences/{user_id}")
def check_user_preferences(user_id: str):
    profile = get_user_profile(user_id)

    return {
        "has_preferences": bool(
            profile.get("genres")
            or profile.get("authors")
            or profile.get("books")
        ),
        "genres": profile.get("genres", []),
        "authors": profile.get("authors", []),
        "books": profile.get("books", []),
        "profile_source": profile.get("profile_source")
    }


# ============================================================
# 16. API DONG BO SACH YEU THICH
# ============================================================

@app.post("/api/user/favorites")
def save_favorites(payload: FavoritesPayload):
    if not payload.user_id or payload.user_id == "guest":
        raise HTTPException(
            status_code=400,
            detail="Can dang nhap de dong bo yeu thich."
        )

    try:
        data = [
            item.model_dump()
            if hasattr(item, "model_dump")
            else item.dict()
            for item in payload.favorites
        ]

        with get_conn() as conn:
            ensure_personalization_tables(conn)
            cur = conn.cursor()

            encoded = json.dumps(data, ensure_ascii=False)

            cur.execute("""
                UPDATE dbo.UserFavorites
                SET favorites_json = ?,
                    updated_at = SYSUTCDATETIME()
                WHERE user_id = ?
            """, (encoded, payload.user_id))

            if cur.rowcount == 0:
                cur.execute("""
                    INSERT INTO dbo.UserFavorites
                        (user_id, favorites_json)
                    VALUES (?, ?)
                """, (payload.user_id, encoded))

            conn.commit()

        return {
            "message": "Da dong bo yeu thich",
            "count": len(data)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 17. API LAY SACH NOI BAT
# ============================================================

@app.get("/api/books/top")
def get_top_books(limit: int = 15):
    try:
        safe_limit = max(1, min(int(limit), 100))

        with get_conn() as conn:
            df = pd.read_sql(
                f"""
                SELECT TOP {safe_limit}
                    book_id, title, authors, publisher
                FROM Books
                """,
                conn
            )

        return df.to_dict(orient="records")

    except Exception as e:
        return {"error": str(e)}


# ============================================================
# 18. API GOI Y SACH HYBRID
# ============================================================

@app.get("/api/recommend")
def get_hybrid_recommendations(
    user_id: str,
    book_name: str,
    weight_cb: float = 0.5,
    weight_cf: float = 0.5,
    top_n: int = 5
):
    try:
        if df_books.empty or tfidf_matrix is None or svd_model is None:
            raise HTTPException(
                status_code=503,
                detail="Mo hinh goi y chua duoc nap."
            )

        idx_list = df_books.index[
            df_books["Name"].astype(str).str.lower().str.contains(
                re.escape(book_name.lower()),
                na=False
            )
        ].tolist()

        if not idx_list:
            return {
                "error": (
                    f"Khong tim thay tua sach nao chua tu khoa: {book_name}"
                )
            }

        target_idx = idx_list[0]

        cosine_sim = linear_kernel(
            tfidf_matrix[target_idx],
            tfidf_matrix
        ).flatten()

        sim_indices = cosine_sim.argsort()[-100:][::-1]

        scores = []

        for idx in sim_indices:
            book = df_books.iloc[idx]
            b_id = str(book['book_id'])
            b_title = str(book['Name'])
            b_author = str(book.get('Authors', 'Chưa rõ'))

            try: 
                cf = float(svd_model.predict(user_id, book['book_id']).est) / 5.0
            except Exception: 
                cf = 0.0

            scores.append({
                "id": b_id,
                "Ten sach": b_title,
                "Tac gia": b_author,
                "cover_image": tim_anh_bia(
                    b_id,
                    b_title,
                    book.get("image_url", "")
                ),
                "rating": get_consistent_rating(
                    b_id,
                    book.get("average_rating")
                ),
                "Diem Noi dung": round(float(cosine_sim[idx]), 4),
                "Diem Hybrid": round(
                    weight_cb * float(cosine_sim[idx]) + weight_cf * cf,
                    4
                )
            })

        target = str(
            df_books.iloc[target_idx]["Name"]
        ).lower()

        result = [
            x for x in sorted(
                scores,
                key=lambda x: x["Diem Hybrid"],
                reverse=True
            )
            if x["Ten sach"].lower() != target
        ]

        return result[:max(1, min(top_n, 50))]

    except HTTPException:
        raise

    except Exception as e:
        return {"error": str(e)}


# ============================================================
# 19. API GOI Y SACH TU DONG THEO SO THICH
# ============================================================

def _translate_genre(g):
    # Chuyển đổi thể loại tiếng Việt sang tiếng Anh cơ bản nếu cần khớp với dữ liệu dataset
    return str(g or "").strip().lower()

def _strip_paren(author_name):
    # Loại bỏ năm sinh/năm mất trong ngoặc của tên tác giả nếu có (Ví dụ: "J.K. Rowling (1965-)" -> "J.K. Rowling")
    return re.sub(r'\s*\(.*?\)', '', str(author_name)).strip()

@app.get("/api/recommend/auto")
def get_auto_recommendations(user_id: str, top_n: int = 12):
    try:
        with get_conn() as conn:
            ensure_personalization_tables(conn)
            cur = conn.cursor()

            profile = read_profile(cur, user_id)
            favorites = read_favorites(cur, user_id)

        if df_books.empty:
            raise HTTPException(
                status_code=503,
                detail="Chua nap duoc danh muc sach that (df_books.pkl)."
            )

        genres = profile["genres"]
        authors = profile["authors"]
        liked_titles = list(profile["books"])

        liked_titles += [
            str(x.get("title", ""))
            for x in favorites
            if x.get("title")
        ]

        if not (genres or authors or liked_titles):
            return {
                "error": "Chua co ho so so thich hoac sach yeu thich.",
                "needs_profile": True
            }

        def norm(value):
            return str(value or "").casefold()

        def _bad_meta(v):
            # SUA: loai NaN va chuoi rong khoi metadata_text de tranh false-positive
            if v is None:
                return True
            try:
                if pd.isna(v):
                    return True
            except (TypeError, ValueError):
                pass
            return str(v).strip().lower() in ("nan", "none", "<na>", "")

        result = []

        for _, book in df_books.iterrows():
            title = str(book.get("Name", book.get("title", "")))
            author = str(book.get("Authors", book.get("authors", "")))
            book_id = str(book.get("book_id", book.get("id", "")))

            # SUA: bo cot book_id/id/index va bo NaN
            metadata_text = " ".join(
                norm(v)
                for col, v in book.to_dict().items()
                if not _bad_meta(v)
                and str(col).lower() not in ("book_id", "id", "index")
            )

            haystack = (
                title + " " + author + " " + metadata_text
            ).casefold()

            matched_genres = []
            for g in genres:
                g_en = _translate_genre(g)
                g_vi = norm(g)
                if g_en in haystack or g_vi in haystack:
                    matched_genres.append(g)

            matched_authors = []
            for a in authors:
                a_clean = _strip_paren(a)
                if norm(a_clean) and (
                    norm(a_clean) in norm(author) or norm(author) in norm(a_clean)
                ):
                    matched_authors.append(a_clean)

            matched_titles = [
                t for t in liked_titles
                if norm(t) and (
                    norm(t) in norm(title)
                    or norm(title) in norm(t)
                )
            ]

            genre_hits = len(matched_genres)
            author_hits = len(matched_authors)
            title_hits = len(matched_titles)

            content_score = (
                genre_hits * 1.0
                + author_hits * 2.0
                + title_hits * 2.5
            )

            try:
                cf_score = (
                    float(
                        svd_model.predict(
                            user_id,
                            book.get("book_id")
                        ).est
                    )
                    if svd_model is not None
                    else 0.0
                )
            except Exception:
                cf_score = 0.0

            score = content_score + max(0.0, cf_score) * 0.1

            if score <= 0:
                if svd_model is not None:
                    score = max(0.01, cf_score * 0.1)
                else:
                    continue

            reasons = []

            if matched_authors:
                reasons.append(
                    f"do ban thich tac gia {matched_authors[0]}"
                )
            elif matched_genres:
                reasons.append(
                    f"phu hop voi the loai {matched_genres[0]}"
                )
            elif matched_titles:
                reasons.append("tuong tu voi sach ban tung thich")
            else:
                reasons.append("dua tren xu huong cong dong (CF)")

            explanation = (
                "Goi y nay duoc de xuat "
                + " va ".join(reasons)
                + "."
            )

            result.append({
                "id": str(book_id),
                "Ten sach": title,
                "Tac gia": author,
                "Diem Phu hop (AI)": round(score, 3),
                "Diem noi dung": round(content_score, 3),
                "The loai phu hop": matched_genres,
                "explanation": explanation,
                "_image_url": book.get("image_url", ""),
                "_avg_rating": book.get("average_rating")
            })

        favorite_ids = {
            str(x.get("id", ""))
            for x in favorites
        }

        result = [
            x for x in result
            if x["id"] not in favorite_ids
        ]

        result.sort(
            key=lambda x: x["Diem Phu hop (AI)"],
            reverse=True
        )

        top_result = result[:max(1, min(top_n, 50))]

        for item in top_result:
            item["cover_image"] = tim_anh_bia(
                item["id"],
                item["Ten sach"],
                item.pop("_image_url", "")
            )
            item["rating"] = get_consistent_rating(
                item["id"],
                item.pop("_avg_rating", None)
            )

        return top_result

    except HTTPException:
        raise

    except Exception as e:
        return {"error": str(e)}


# ============================================================
# 20. API THONG KE DASHBOARD
# ============================================================

@app.get("/api/admin/dashboard", dependencies=[Depends(require_admin)])
def get_dashboard_stats():
    try:
        with get_conn() as conn:
            cur = conn.cursor()

            cur.execute("SELECT COUNT(*) FROM Users")
            users = cur.fetchone()[0]

            try:
                cur.execute("SELECT COUNT(*) FROM Reviews")
                reviews = cur.fetchone()[0]
            except Exception:
                reviews = "Khong xac dinh"

        return {
            "total_users": users,
            "total_books": len(df_books),
            "total_reviews": reviews
        }

    except Exception as e:
        return {"error": str(e)}


# ============================================================
# 21. API THONG KE QUAN TRI
# ============================================================

@app.get("/api/admin/stats", dependencies=[Depends(require_admin)])
def get_admin_stats():
    try:
        with get_conn() as conn:
            cur = conn.cursor()

            cur.execute("SELECT COUNT(*) FROM Users")
            total_users = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM Books")
            total_books = cur.fetchone()[0]

            try:
                cur.execute("SELECT COUNT(*) FROM Reviews")
                total_reviews = cur.fetchone()[0]
            except Exception:
                total_reviews = 0

        return {
            "total_users": total_users,
            "total_books": total_books,
            "total_reviews": total_reviews
        }

    except Exception as e:
        return {"error": str(e)}


# ============================================================
# 22. API LAY DANH SACH NGUOI DUNG QUAN TRI
# ============================================================

@app.get("/api/admin/users", dependencies=[Depends(require_admin)])
def get_admin_users():
    try:
        with get_conn() as conn:
            cur = conn.cursor()

            cur.execute("""
                SELECT user_id, username, role
                FROM Users
            """)

            users = []

            for row in cur.fetchall():
                u_id = row[0]

                try:
                    cur.execute("""
                        SELECT COUNT(*)
                        FROM UserPreferences
                        WHERE user_id = ?
                    """, u_id)

                    has_pref = cur.fetchone()[0] > 0

                except Exception:
                    has_pref = False

                users.append({
                    "user_id": u_id,
                    "username": row[1],
                    "role": row[2],
                    "survey_completed": has_pref
                })

        return {"users": users}

    except Exception as e:
        return {"error": str(e)}

# ============================================================
#  API ĐÁNH GIÁ SÁCH (Reviews)
# ============================================================

@app.post("/api/books/review")
def add_or_update_review(payload: ReviewPayload):
    """Thêm/sửa đánh giá sách. Mỗi user chỉ có 1 review / 1 sách (upsert)."""
    if not payload.user_id or payload.user_id == "guest":
        raise HTTPException(
            status_code=400,
            detail="Can dang nhap de danh gia sach."
        )

    try:
        with get_conn() as conn:
            ensure_reviews_table(conn)
            cur = conn.cursor()

            cur.execute("""
                SELECT review_id FROM dbo.Reviews
                WHERE user_id = ? AND book_id = ?
            """, (payload.user_id, str(payload.book_id)))
            existing = cur.fetchone()

            if existing:
                cur.execute("""
                    UPDATE dbo.Reviews
                    SET rating = ?, comment = ?, created_at = SYSUTCDATETIME()
                    WHERE review_id = ?
                """, (float(payload.rating), payload.comment, existing[0]))
                action = "updated"
            else:
                cur.execute("""
                    INSERT INTO dbo.Reviews (user_id, book_id, rating, comment)
                    VALUES (?, ?, ?, ?)
                """, (
                    payload.user_id,
                    str(payload.book_id),
                    float(payload.rating),
                    payload.comment
                ))
                action = "created"

            conn.commit()

        return {
            "message": "Da luu danh gia" if action == "created" else "Da cap nhat danh gia",
            "action": action,
            "book_id": str(payload.book_id),
            "rating": float(payload.rating)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/books/{book_id}/reviews")
def get_book_reviews(book_id: str):
    """Lấy danh sách đánh giá của một cuốn sách + điểm trung bình cộng đồng."""
    try:
        with get_conn() as conn:
            ensure_reviews_table(conn)
            cur = conn.cursor()
            cur.execute("""
                SELECT r.review_id, r.user_id, r.rating, r.comment,
                       r.created_at, u.username
                FROM dbo.Reviews r
                LEFT JOIN dbo.Users u ON u.user_id = r.user_id
                WHERE r.book_id = ?
                ORDER BY r.created_at DESC
            """, (str(book_id),))

            reviews = []
            total = 0.0
            for row in cur.fetchall():
                total += float(row[2] or 0)
                reviews.append({
                    "review_id": row[0],
                    "user_id": row[1],
                    "username": row[5] or "Ẩn danh",
                    "rating": float(row[2] or 0),
                    "comment": row[3] or "",
                    "created_at": row[4].isoformat() if row[4] else None,
                })

        avg = round(total / len(reviews), 2) if reviews else None
        return {
            "book_id": str(book_id),
            "avg_user_rating": avg,
            "count": len(reviews),
            "reviews": reviews
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/books/review/{review_id}")
def delete_review(review_id: int, user_id: str):
    """Xóa đánh giá (chỉ chủ sở hữu hoặc admin)."""
    try:
        with get_conn() as conn:
            ensure_reviews_table(conn)
            cur = conn.cursor()
            cur.execute("""
                SELECT user_id FROM dbo.Reviews WHERE review_id = ?
            """, (review_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Khong tim thay danh gia.")
            if str(row[0]) != str(user_id):
                cur.execute("SELECT role FROM Users WHERE user_id = ?", user_id)
                adm = cur.fetchone()
                if not adm or str(adm[0]).lower() != "admin":
                    raise HTTPException(status_code=403, detail="Khong co quyen xoa.")
            cur.execute("DELETE FROM dbo.Reviews WHERE review_id = ?", (review_id,))
            conn.commit()
        return {"message": "Da xoa danh gia", "review_id": review_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# ============================================================
# 23. API LAY CHI TIET MOT CUON SACH
# ============================================================

@app.get("/api/books/{book_id}")
def get_book_detail(book_id: str):
    """
    Lay thong tin chi tiet sach tu SQL Server.
    Anh bia duoc tim boi ham tim_anh_bia().
    """

    try:
        with get_conn() as conn:
            cur = conn.cursor()

            cur.execute("""
            SELECT
                book_id,
                title,
                authors,
                publisher,
                description,
                average_rating,
                ratings_count,
                image_url
            FROM Books
            WHERE CAST(book_id AS NVARCHAR(100)) = ?
            """, str(book_id))

            row = cur.fetchone()

            if not row:
                raise HTTPException(
                    status_code=404,
                    detail="Khong tim thay sach trong co so du lieu."
                )

            book_id_value = str(row[0])
            title = str(row[1] or f"Sach #{book_id_value}")
            author = str(row[2] or "Dang cap nhat")
            publisher = row[3] or "Dang cap nhat"
            description = row[4] or ""
            
            rating_value = get_consistent_rating(book_id_value, row[5])
            cover = tim_anh_bia(book_id_value, title, row[7])

            return {
                "book_id": book_id_value,
                "title": title,
                "authors": author,
                "publisher": publisher,
                "publication_year": None,
                "num_pages": None,
                "cover_image": cover,
                "rating": rating_value,
                "description": description
            }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_consistent_rating(book_id, raw_rating=None):
    if raw_rating is not None:
        try:
            val = float(raw_rating)
            if val > 0:
                return round(val, 1)
        except Exception:
            pass
    return None

# ============================================================
# 24. API LAY DANH SACH SACH CHO TRANG CHU
# ============================================================

@app.get("/api/books")
def get_all_books():
    try:
        with get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT TOP 50
                book_id, title, authors, publisher, description,
                average_rating, image_url
            FROM dbo.Books
            """)
            rows = cursor.fetchall()

        books_list = []

        for row in rows:
            book_id = str(row[0])
            title = str(row[1] or f"Sach #{book_id}")
            authors = str(row[2] or "Dang cap nhat")
            publisher = row[3] or "NXB Tre"
            description = row[4] or ""
            
            cover_path = tim_anh_bia(book_id, title, row[6])
            rating_val = get_consistent_rating(book_id, row[5])

            books_list.append({
                "book_id": row[0],
                "title": title,
                "authors": authors,
                "publisher": publisher,
                "genres": lay_the_loai(book_id),
                "cover_image": cover_path,
                "rating": rating_val,
                "description": description
            })

        return {"books": books_list}
    except Exception as e:
        return {"error": str(e)}