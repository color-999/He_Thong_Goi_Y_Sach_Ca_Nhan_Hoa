import pandas as pd

print("Đang nạp dữ liệu sạch...")
df = pd.read_csv('cleaned_goodreads_reviews.csv')

# Thêm utc=True để đồng bộ với file count_data.py và tránh lỗi múi giờ
df['date_added'] = pd.to_datetime(df['date_added'], utc=True, errors='coerce')
df = df.dropna(subset=['date_added']) # Bỏ các dòng lỗi ngày tháng (nếu có)

print("Đang sắp xếp dữ liệu theo thời gian thực hiện...")
df = df.sort_values(by=['user_id', 'date_added'])

# XỬ LÝ TRÙNG LẶP: Giữ lại đánh giá mới nhất của user cho mỗi cuốn sách
# (Tránh lỗi crash thư viện Surprise khi 1 user đánh giá 1 sách 2 lần)
before_drop = len(df)
df = df.drop_duplicates(subset=['user_id', 'book_id'], keep='last')
print(f"-> Đã xóa {before_drop - len(df):,} dòng trùng lặp (cùng user, cùng sách).")

print("Đang tiến hành chia tập Train/Test (Tỷ lệ 80/20 cho mỗi người dùng)...")
def time_based_split(group):
    split_point = int(len(group) * 0.8)
    if len(group) >= 5:
        train = group.iloc[:split_point]
        test = group.iloc[split_point:]
    else:
        train = group
        test = pd.DataFrame(columns=group.columns)
    return train, test

train_list = []
test_list = []

grouped = df.groupby('user_id')
for _, group in grouped:
    train, test = time_based_split(group)
    train_list.append(train)
    test_list.append(test)

df_train = pd.concat(train_list, ignore_index=True)
df_test = pd.concat(test_list, ignore_index=True)

print(f"Hoàn tất! Số lượng Train: {len(df_train):,} | Số lượng Test: {len(df_test):,}")

# Lưu ra file
df_train.to_csv('train_data.csv', index=False)
df_test.to_csv('test_data.csv', index=False)
print("Đã xuất file 'train_data.csv' và 'test_data.csv'. Sẵn sàng huấn luyện SVD/kNN!")