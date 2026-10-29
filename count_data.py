# Lọc, tìm kiếm & DLS
import pandas as pd
import matplotlib.pyplot as plt

print("Đang đọc dữ liệu, vui lòng chờ một lát vì file khá nặng...")

# Đọc file dữ liệu
df_reviews = pd.read_json('goodreads_reviews_spoiler_raw.json.gz', lines=True, compression='gzip')

# Lấy các con số chính xác
print("Số lượt đánh giá:", len(df_reviews))
print("Số người dùng (User):", df_reviews['user_id'].nunique())
print("Số cuốn sách (Book):", df_reviews['book_id'].nunique())

# Bước làm sạch dữ liệu
print("\nĐang tiến hành làm sạch dữ liệu...")

# Cập nhật danh sách cột cần giữ, bắt buộc phải có 'date_added' để chia tập train/test theo thời gian
columns_to_keep = ['user_id', 'book_id', 'rating', 'date_added'] 
df_clean = df_reviews[columns_to_keep].dropna()

# Chuẩn hóa cột thời gian (chuyển chuỗi văn bản thành định dạng datetime)
print("Đang chuẩn hóa định dạng thời gian...")

# Sử dụng errors='coerce' để tương thích với mọi phiên bản pandas (tránh lỗi format='mixed' trên pandas < 2.0)
df_clean['date_added'] = pd.to_datetime(df_clean['date_added'], utc=True, errors='coerce')
# Xóa các dòng có ngày tháng không hợp lệ (nếu có)
df_clean = df_clean.dropna(subset=['date_added'])

print("\nThông tin dữ liệu sau khi làm sạch:")
print(df_clean.info())

# Lưu dữ liệu đã làm sạch ra file CSV 
print("\nĐang xuất file dữ liệu sạch...")
df_clean.to_csv('cleaned_goodreads_reviews.csv', index=False)
print("-> Đã lưu thành công file 'cleaned_goodreads_reviews.csv'!")

# 3. Phân tích khám phá (EDA): Vẽ biểu đồ phân bố điểm đánh giá (Rating)
print("\nĐang tạo biểu đồ phân bố điểm đánh giá...")
rating_counts = df_clean['rating'].value_counts().sort_index()

plt.figure(figsize=(8, 5))
rating_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Phân bố mức điểm đánh giá của người dùng')
plt.xlabel('Mức điểm (Rating)')
plt.ylabel('Số lượng đánh giá')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Tự động lưu biểu đồ thành file ảnh
plt.savefig('Bieu_do_phan_phoi_diem_dgia_EDA.png')
print("-> Đã lưu biểu đồ thành file 'Bieu_do_phan_phoi_diem_dgia_EDA.png'.")

print("Đang hiển thị biểu đồ...")
plt.show()