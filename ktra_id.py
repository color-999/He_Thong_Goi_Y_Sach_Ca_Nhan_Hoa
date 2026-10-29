# import pandas as pd

# rev = pd.read_csv('cleaned_goodreads_reviews.csv', usecols=['book_id'])
# ids = set(rev['book_id'].astype(str).unique())

# old = pd.read_csv('archive/book1-100k.csv', usecols=['Id', 'Name'])
# old['Id'] = old['Id'].astype(str)

# print("Số sách trong reviews:", len(ids))
# print("Số sách trong book1-100k:", len(old))
# print("Id nhỏ nhất / lớn nhất:", old['Id'].astype(int).min(), "/", old['Id'].astype(int).max())
# print("Số book_id của reviews CÓ trong book1-100k:", len(ids & set(old['Id'])))
# print(old[old['Id'].isin(['1', '2', '3'])].to_string(index=False))




import glob, os
import pandas as pd

ids = set(pd.read_csv('cleaned_goodreads_reviews.csv', usecols=['book_id'])['book_id'].astype(str))
print("Sách cần tìm:", len(ids))
found = set()
first = True
for f in sorted(glob.glob('archive/**/*.csv', recursive=True)):
    try:
        cols = pd.read_csv(f, nrows=0).columns.tolist()
        if 'Id' not in cols or 'Name' not in cols:
            print(f"{os.path.basename(f)}: bỏ qua (cột: {cols[:4]})")
            continue
        if first:
            print("Các cột của file sách:", cols)
            first = False
        s = set(pd.read_csv(f, usecols=['Id'])['Id'].astype(str))
        hit = ids & s
        found |= hit
        print(f"{os.path.basename(f)}: {len(s):,} sách, trùng {len(hit):,}")
    except Exception as e:
        print(f, "lỗi:", e)
print(f"\nTỔNG: {len(found):,}/{len(ids):,} ({len(found)/len(ids):.1%})")