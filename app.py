import os
import faiss
import pickle
import torch
import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
import google.generativeai as genai

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

client = Groq(api_key='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')
genai.configure(api_key="AIzaSyApnSpZVliqfTKmhfEOu66kczAbsvyPslQ")

# Load embedding model
embedding_model = SentenceTransformer('./saved_Embedding_model')

faiss_directory = "faiss"
faiss_index_file = os.path.join(faiss_directory, "faiss_index.index")
index = faiss.read_index(faiss_index_file)

with open("document_chunks.pkl", "rb") as f:
    document_chunks = pickle.load(f)

def search_with_faiss(query_embedding, index, top_k=10):
    distances, indices = index.search(query_embedding.reshape(1, -1), top_k)
    return [(document_chunks[i], distances[0][j]) for j, i in enumerate(indices[0])]

def ask_groq(query, context):
    messages = [
        {"role": "system", "content": "Bạn là một chuyên gia pháp luật, dựa vào thông tin truy xuất được hãy trả lời câu hỏi. Trả lời của bạn cần phải nêu đầy đủ và chính xác nội dung với thông tin được cung cấp từ Chương, điều, mục sau đó nêu câu trả lời rõ ràng. Nêu các điều luật liên quan và nội dung của điều đó qua các năm của các bộ luật, ưu tiên bộ luật mới nhất. Diễn giải lại câu trả lời cho người dùng nắm bắt tốt nhất. So sánh sự thay đổi của bộ luật qua các năm. Lưu ý bắt buộc trả lời bằng tiếng Việt"},
        {"role": "user", "content": f"Context: {context}\nQuestion: {query}"}
    ]
    response = client.chat.completions.create(
        messages=messages, model="llama3-70b-8192"
    )
    return response.choices[0].message.content

def ask_gemini(query, context):
    prompt = f"""
    Bạn là chuyên gia pháp luật, dựa vào thông tin đã RAG, hãy trả lời:
    Context: {context}
    Question: {query}
    - Câu trả lời của bạn cần phải nêu đầy đủ và chính xác với thông tin được cung cấp từ Chương, điều, mục sau đó nêu câu trả lời rõ ràng.
    - Nêu các điều luật liên quan và nội dung của điều đó qua các năm của các bộ luật. 
    - Trích dẫn các điều luật cụ thể phải chính xác theo dữ liệu được cung cấp, ưu tiên bộ luật mới nhất.
    - So sánh sự thay đổi của bộ luật qua các năm.
    - Trả lời bằng tiếng Việt.
    - Diễn giải lại câu trả lời cho người dùng nắm bắt tốt nhất.
    """
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text

st.set_page_config(page_title="Chatbot Pháp Luật", layout="wide")

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = []

st.sidebar.header("📜 Lịch sử hội thoại")
for i, session in enumerate(st.session_state.chat_sessions):
    if session["conversation"]:
        if st.sidebar.button(f"📌 Hội thoại {i+1}: {session['conversation'][0]['question'][:30]}...", key=f"history_{i}"):
            st.session_state.selected_session = session
            st.sidebar.subheader("Câu hỏi trong hội thoại này:")
            for q in session["conversation"][-3:]:
                st.sidebar.write(f"- {q['question']}")
    else:
        if st.sidebar.button(f"📌 Hội thoại {i+1}: Chưa có câu hỏi", key=f"history_empty_{i}"):
            st.session_state.selected_session = session

if st.sidebar.button("➕ Tạo hội thoại mới", key="create_new_session"):
    new_session = {
        "conversation": []
    }
    st.session_state.chat_sessions.append(new_session)
    st.session_state.selected_session = new_session 

if st.sidebar.button("🗑️ Xóa toàn bộ hội thoại", key="clear_all_sessions"):
    st.session_state.chat_sessions.clear()
    st.session_state.selected_session = None
    st.toast("🗑️ Đã xoá toàn bộ hội thoại!", icon="✅")

st.title("💬 Chatbot Hỗ trợ Pháp Luật")


chat_container = st.container()

if "selected_session" in st.session_state and st.session_state.selected_session:
    session = st.session_state.selected_session
    with chat_container:
        for chat in session['conversation']:
            st.chat_message("user").write(chat["question"])
            st.chat_message("assistant").write(f"**🔷 Trả lời từ Groq:**\n{chat['answer_groq']}")
            st.chat_message("assistant").write(f"**🔶 Trả lời từ Gemini:**\n{chat['answer_gemini']}")


        if st.button("🗑️ Xóa hội thoại này"):
            st.session_state.chat_sessions.remove(session)
            st.session_state.selected_session = None
            st.toast("🗑️ Đã xóa hội thoại này!", icon="✅")

query = st.chat_input("Nhập câu hỏi pháp luật của bạn...")


if query:
    with st.spinner("🔍 Đang tìm kiếm dữ liệu..."):
        query_embedding = embedding_model.encode(query, convert_to_numpy=True)
        context_chunks = search_with_faiss(query_embedding, index, top_k=5)

    if not context_chunks:
        st.warning("Không tìm thấy thông tin phù hợp!")
    else:
        context_text = "\n".join([chunk for chunk, _ in context_chunks])

        with st.spinner("Đang tạo câu trả lời..."):
            answer_groq = ask_groq(query, context_text)
            answer_gemini = ask_gemini(query, context_text)

        if "selected_session" not in st.session_state or st.session_state.selected_session is None:
            session = {
                "conversation": [
                    {"question": query, "answer_groq": answer_groq, "answer_gemini": answer_gemini}
                ]
            }
            st.session_state.chat_sessions.append(session)
            st.session_state.selected_session = session
        else:
            st.session_state.selected_session['conversation'].append(
                {"question": query, "answer_groq": answer_groq, "answer_gemini": answer_gemini}
            )

        # Display the answer
        with chat_container:
            st.chat_message("user").write(query)
            st.chat_message("assistant").write(f"**🔷 Trả lời từ Groq:**\n{answer_groq}")
            st.chat_message("assistant").write(f"**🔶 Trả lời từ Gemini:**\n{answer_gemini}")
