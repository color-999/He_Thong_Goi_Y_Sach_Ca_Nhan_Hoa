import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import warnings
warnings.filterwarnings('ignore')

print("Đang nạp file thông tin sách (Metadata)...")
# Lấy mẫu file đầu tiên chứa khoảng 58.000 cuốn sách
df_books = pd.read_csv('archive/book1-100k.csv')

print("Đang làm sạch và gộp các trường văn bản...")
# Xử lý các giá trị rỗng (NaN)
df_books['Name'] = df_books['Name'].fillna('')
df_books['Authors'] = df_books['Authors'].fillna('')
df_books['Publisher'] = df_books['Publisher'].fillna('')

# Gộp Tên sách, Tác giả và Nhà xuất bản thành 1 chuỗi văn bản duy nhất (corpus)
df_books['content'] = df_books['Name'] + " " + df_books['Authors'] + " " + df_books['Publisher']

print("Đang chạy thuật toán TF-IDF để trích xuất đặc trưng...")
# Khởi tạo TF-IDF, loại bỏ các từ vô nghĩa (stop_words) trong tiếng Anh
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_books['content'])

print(f"-> Đã tạo xong ma trận TF-IDF với kích thước: {tfidf_matrix.shape}")

# Hàm gợi ý sách tối ưu RAM (chỉ tính độ tương đồng của 1 cuốn sách mục tiêu với phần còn lại)
def get_content_recommendations(book_name, df, tfidf_mat, top_n=5):
    # Tìm index của cuốn sách trong DataFrame
    idx_list = df.index[df['Name'].str.lower() == book_name.lower()].tolist()
    if not idx_list:
        return f"Không tìm thấy tựa sách '{book_name}' trong dữ liệu."
    
    target_idx = idx_list[0]
    
    # Tính Cosine Similarity chỉ riêng cho cuốn sách này (Cực kỳ tiết kiệm RAM)
    cosine_sim_target = linear_kernel(tfidf_mat[target_idx], tfidf_mat).flatten()
    
    # Sắp xếp và lấy top các sách giống nhất
    sim_scores = list(enumerate(cosine_sim_target))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in sim_scores[1:top_n+1]]
    
    return df[['Name', 'Authors']].iloc[top_indices]

# Test thực tế mô hình
book_to_test = "Harry Potter and the Sorcerer's Stone (Harry Potter, #1)"
print(f"\n=============================================")
print(f" GỢI Ý CÁC CUỐN SÁCH GIỐNG VỚI:")
print(f" '{book_to_test}'")
print(f"=============================================")

# Gọi hàm gợi ý
recommendations = get_content_recommendations(book_to_test, df_books, tfidf_matrix)
print(recommendations.to_string(index=False))
print("=============================================")