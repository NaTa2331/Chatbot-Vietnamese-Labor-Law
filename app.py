import os
import faiss
import pickle
import torch
import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
import google.generativeai as genai

# Thiết lập để tránh xung đột OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Tạo client Groq
client = Groq(api_key='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')

# Cấu hình API key của Gemini
genai.configure(api_key="AIzaSyAZt_jcRAviU7um0EUiegKFu6UouAPFNc0")

# Load mô hình nhúng
embedding_model = SentenceTransformer('intfloat/multilingual-e5-large')

# Load FAISS index
faiss_directory = "faiss"
faiss_index_file = os.path.join(faiss_directory, "faiss_index.index")
index = faiss.read_index(faiss_index_file)

with open("document_chunks.pkl", "rb") as f:
    document_chunks = pickle.load(f)

# Hàm tìm kiếm FAISS
def search_with_faiss(query_embedding, index, top_k=5):
    distances, indices = index.search(query_embedding.reshape(1, -1), top_k)
    return [(document_chunks[i], distances[0][j]) for j, i in enumerate(indices[0])]

# Hàm hỏi Groq
def ask_groq(query, context):
    messages = [
        {"role": "system", "content": "Bạn là trợ lý pháp luật, chỉ trả lời theo dữ liệu."},
        {"role": "user", "content": f"Context: {context}\nQuestion: {query}"}
    ]
    response = client.chat.completions.create(
        messages=messages, model="llama3-70b-8192"
    )
    return response.choices[0].message.content

# Hàm hỏi Gemini
def ask_gemini(query, context):
    prompt = f"""
    Bạn là chuyên gia pháp luật. Dựa vào thông tin sau, hãy trả lời:
    Context: {context}
    Question: {query}
    - Trả lời chính xác dựa trên luật pháp Việt Nam.
    - So sánh nếu có sự thay đổi giữa các bộ luật.
    - Sử dụng markdown để định dạng rõ ràng.
    - Trả lời bằng Tiếng Việt.
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text

# Giao diện Streamlit
st.set_page_config(page_title="Chatbot Pháp Luật", layout="wide")

# Khởi tạo session_state
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = []  # Lưu danh sách các hội thoại

# Sidebar: Lịch sử hội thoại
# Sidebar: Lịch sử hội thoại
st.sidebar.header("📜 Lịch sử hội thoại")
for i, session in enumerate(st.session_state.chat_sessions):
    # Kiểm tra nếu hội thoại có ít nhất một câu hỏi
    if session["conversation"]:
        # Hiển thị tên hội thoại nếu có câu hỏi
        if st.sidebar.button(f"📌 Hội thoại {i+1}: {session['conversation'][0]['question'][:30]}...", key=f"history_{i}"):
            st.session_state.selected_session = session  # Chọn hội thoại để xem lại
            # Hiển thị các câu hỏi đã hỏi trong hội thoại
            st.sidebar.subheader("Câu hỏi trong hội thoại này:")
            for q in session["conversation"][-3:]:
                st.sidebar.write(f"- {q['question']}")
    else:
        # Nếu hội thoại trống, hiển thị dòng "Chưa có câu hỏi"
        if st.sidebar.button(f"📌 Hội thoại {i+1}: Chưa có câu hỏi", key=f"history_empty_{i}"):
            st.session_state.selected_session = session


# Nút tạo hội thoại mới
if st.sidebar.button("➕ Tạo hội thoại mới", key="create_new_session"):
    new_session = {
        "conversation": []
    }
    st.session_state.chat_sessions.append(new_session)
    st.session_state.selected_session = new_session  # Chọn hội thoại mới để bắt đầu

# Nút xóa tất cả hội thoại
if st.sidebar.button("🗑️ Xóa toàn bộ hội thoại", key="clear_all_sessions"):
    st.session_state.chat_sessions.clear()
    st.session_state.selected_session = None
    st.toast("🗑️ Đã xoá toàn bộ hội thoại!", icon="✅")

# Tiêu đề chatbot
st.title("💬 Chatbot Hỗ trợ Pháp Luật")

# Giao diện chat giống ChatGPT
chat_container = st.container()

# Hiển thị hội thoại từ lịch sử (nếu có)
if "selected_session" in st.session_state and st.session_state.selected_session:
    session = st.session_state.selected_session
    with chat_container:
        for chat in session['conversation']:
            st.chat_message("user").write(chat["question"])
            st.chat_message("assistant").write(f"**🔷 Trả lời từ Groq:**\n{chat['answer_groq']}")
            st.chat_message("assistant").write(f"**🔶 Trả lời từ Gemini:**\n{chat['answer_gemini']}")

        # Nút xóa hội thoại này
        if st.button("🗑️ Xóa hội thoại này"):
            st.session_state.chat_sessions.remove(session)
            st.session_state.selected_session = None
            st.toast("🗑️ Đã xóa hội thoại này!", icon="✅")

# Ô nhập câu hỏi
query = st.chat_input("Nhập câu hỏi pháp luật của bạn...")

# Nếu có câu hỏi
if query:
    with st.spinner("🔍 Đang tìm kiếm dữ liệu..."):
        query_embedding = embedding_model.encode(query, convert_to_numpy=True)
        context_chunks = search_with_faiss(query_embedding, index, top_k=5)

    if not context_chunks:
        st.warning("Không tìm thấy thông tin phù hợp!")
    else:
        context_text = "\n".join([chunk for chunk, _ in context_chunks])

        with st.spinner("🤖 Đang tạo câu trả lời..."):
            answer_groq = ask_groq(query, context_text)
            answer_gemini = ask_gemini(query, context_text)

        # Lưu câu hỏi và câu trả lời vào hội thoại hiện tại
        if "selected_session" not in st.session_state or st.session_state.selected_session is None:
            # Nếu chưa có hội thoại nào, tạo một hội thoại mới
            session = {
                "conversation": [
                    {"question": query, "answer_groq": answer_groq, "answer_gemini": answer_gemini}
                ]
            }
            st.session_state.chat_sessions.append(session)
            st.session_state.selected_session = session
        else:
            # Thêm câu hỏi mới vào hội thoại hiện tại
            st.session_state.selected_session['conversation'].append(
                {"question": query, "answer_groq": answer_groq, "answer_gemini": answer_gemini}
            )

        # Hiển thị câu trả lời
        with chat_container:
            st.chat_message("user").write(query)
            st.chat_message("assistant").write(f"**🔷 Trả lời từ Groq:**\n{answer_groq}")
            st.chat_message("assistant").write(f"**🔶 Trả lời từ Gemini:**\n{answer_gemini}")
