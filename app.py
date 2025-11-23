import streamlit as st
import time # Dùng để giả lập thời gian pipeline chạy
from sentiment_analysis import analyze_text, preload_model
from db import save_sentiment, load_sentiments

st.set_page_config(page_title="Trợ lý Phân loại Cảm xúc", layout="wide")
st.title("Trợ lý Phân loại Cảm xúc Tiếng Việt")

# ========== PRELOAD MODEL ON APP STARTUP ==========
if 'model_loaded' not in st.session_state:
    with st.spinner("Đang tải mô hình PhoBERT từ HuggingFace..."):
        st.session_state.model_loaded = preload_model()
    if st.session_state.model_loaded:
        st.success("Mô hình đã được tải thành công!")
    else:
        st.error("Không thể tải mô hình. Sẽ sử dụng chế độ giả lập.")

st.header("Nhập câu bạn muốn phân tích:")

with st.form(key="sentiment_form"):
    user_input = st.text_input("Nhập câu tiếng Việt (ví dụ: Hôm nay tôi rất vui)", key="user_input")
    submit_button = st.form_submit_button(label="Phân loại")

if submit_button:
    if len(user_input) < 5:
        st.error("Câu quá ngắn! Vui lòng nhập câu có ý nghĩa (>= 5 ký tự).")
    else:
        with st.spinner("Mô hình Transformer đang phân tích..."):
            result = analyze_text(user_input)

            label = result.get("sentiment") or "NEUTRAL"
            score = result.get("score")
            fallback_note = ""

            # Some models may not return a numeric score; handle that gracefully
            try:
                score_val = float(score) if score is not None else None
            except Exception:
                score_val = None

            display_reason = f"(Điểm tin cậy: {score_val:.2f}) {fallback_note}" if score_val is not None else f"(Nhãn: {label}) {fallback_note}"

            # 4.4. Hiển thị kết quả
            st.subheader("Kết quả phân loại:")
            if label == "POSITIVE":
                st.success(f"Tích cực (POSITIVE)")
            elif label == "NEGATIVE":
                st.error(f"Tiêu cực (NEGATIVE)")
            else:
                st.info(f"Trung tính (NEUTRAL)")

            st.caption(f"Câu gốc: '{user_input}' | {display_reason}")

            # 4.5. Lưu vào CSDL
            try:
                save_sentiment(user_input, label)
                try:
                    st.toast("Đã lưu kết quả vào lịch sử.")
                except Exception:
                    st.caption("(Đã lưu kết quả vào lịch sử)")
            except Exception as e:
                st.error(f"Lỗi khi lưu vào CSDL: {e}")


# --- 5. Hiển thị Lịch sử  ---
st.header("Lịch sử phân loại")
st.caption("Hiển thị 50 kết quả mới nhất (Yêu cầu ở ảnh 6)")

import pandas as pd

# Load up to 50 most recent records from the SQLite DB
records = []
try:
    records = load_sentiments(limit=50)
except Exception as e:
    st.error(f"Lỗi khi tải lịch sử: {e}")

if records:
    df = pd.DataFrame(records)
    # Ensure ordering newest first
    df = df[["timestamp", "text", "sentiment"]]
else:
    df = pd.DataFrame(columns=["timestamp", "text", "sentiment"])

st.dataframe(df, use_container_width=True)