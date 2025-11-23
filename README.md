# Trợ lý Phân loại Cảm xúc Tiếng Việt

Đây là đồ án xây dựng một ứng dụng web đơn giản, sử dụng Streamlit và mô hình Transformer (PhoBERT) để phân loại cảm xúc (Tích cực, Tiêu cực, Trung tính) từ một câu văn bản tiếng Việt do người dùng nhập vào. Ứng dụng cũng lưu trữ và hiển thị lịch sử các lần phân loại.

## Tính năng chính

  * **Phân loại Cảm xúc:** Nhận diện 3 nhãn: `POSITIVE`, `NEGATIVE`, `NEUTRAL`.
  * **Tiền xử lý Tiếng Việt:** Tự động chuyển chữ thường và chuẩn hóa các từ viết tắt, tiếng lóng phổ biến (ví dụ: "rat" -> "rất", "iu" -> "yêu").
  * **Xử lý logic nghiệp vụ:** Tự động chuyển các kết quả có độ tin cậy thấp (\< 0.5) về `NEUTRAL` (theo yêu cầu đồ án).
  * **Xử lý lỗi:** Hiển thị cảnh báo cho các câu nhập quá ngắn (\< 5 ký tự).
  * **Lưu trữ Lịch sử:** Tự động lưu mọi kết quả phân loại vào cơ sở dữ liệu SQLite.
  * **Hiển thị Lịch sử:** Hiển thị 50 kết quả phân loại gần nhất ngay trên giao diện.

## Công nghệ sử dụng

  * **Giao diện Web:** [Streamlit](https://streamlit.io/)
  * **Model NLP:** [Transformers](https://huggingface.co/transformers) (sử dụng mô hình `vinai/phobert-base-v2` đã được fine-tune)
  * **Cơ sở dữ liệu:** [SQLite 3](https://www.sqlite.org/index.html)
  * **Ngôn ngữ:** Python 3

## Hướng dẫn cài đặt và Chạy ứng dụng

### 1\. Yêu cầu

  * Python 3.13.1
  * `pip` (Python package installer)

### 2\. Cài đặt

1.  **Giải nén dự án:**
    Giải nén file `.zip` của dự án vào một thư mục (ví dụ: `SentimentApp`).

2.  **Di chuyển vào thư mục dự án:**

    ```bash
    cd SentimentApp
    ```

3.  **(Khuyến nghị) Tạo và kích hoạt môi trường ảo (virtual environment):**

    ```bash
    # Dành cho Windows
    python -m venv venv
    .\venv\Scripts\activate

    # Dành cho macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

4.  **Cài đặt các thư viện cần thiết:**
    Tất cả thư viện đã được liệt kê trong file `requirements.txt`. Chạy lệnh:

    ```bash
    pip install -r requirements.txt
    ```

### 3\. Chạy ứng dụng

Sau khi cài đặt thành công, chạy lệnh sau từ terminal:

```bash
streamlit run app.py
```

Streamlit sẽ tự động mở một tab mới trên trình duyệt của bạn (thường là `http://localhost:8501`) để bạn bắt đầu sử dụng ứng dụng.

## Cấu trúc Thư mục (Dự kiến)

```
SentimentApp/
│
├── model/                        # Thư mục chứa model PhoBERT đã fine-tune
│   └── phobert-sentiment-vietnamese-best/
│       ├── config.json
│       ├── pytorch_model.bin
│       └── ... (các file tokenizer)
│
├── app.py                        # File chính, chứa giao diện Streamlit
├── sentiment_analysis.py         # File chứa logic tiền xử lý và gọi model
├── db.py                         # File chứa logic khởi tạo và truy vấn CSDL SQLite
├── requirements.txt              # Danh sách các thư viện Python
└── README.md                     # File hướng dẫn này
```

## Bộ 10 Test Cases (Theo yêu cầu)

Đây là 10 test case cơ bản dùng để đánh giá độ chính xác của mô hình:

| STT | Đầu vào (Câu tiếng Việt) | Đầu ra mong đợi (Sentiment) |
|:--- |:---|:---|
| 1 | Hôm nay tôi rất vui | POSITIVE |
| 2 | Món ăn này dở quá | NEGATIVE |
| 3 | Thời tiết bình thường | NEUTRAL |
| 4 | Rat vui hom nay | POSITIVE |
| 5 | Công việc ổn định | NEUTRAL |
| 6 | Phim này hay lắm | POSITIVE |
| 7 | Tôi buồn vì thất bại | NEGATIVE |
| 8 | Ngày mai đi học | NEUTRAL |
| 9 | Cảm ơn bạn rất nhiều | POSITIVE |
| 10 | Mệt mỏi quá hôm nay | NEGATIVE |

-----