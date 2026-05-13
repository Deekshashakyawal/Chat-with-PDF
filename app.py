import streamlit as st
from rag_engine import ingest_pdf, query

st.set_page_config(page_title="Chat with PDF")

st.title("📄 Chat with your PDF")

# Always initialize state FIRST
if "history" not in st.session_state:
    st.session_state.history = []

if "ready" not in st.session_state:
    st.session_state.ready = False


# ----------------------------
# UPLOAD SECTION (always visible)
# ----------------------------
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None
if uploaded_file is not None:
    if uploaded_file.name.split(".")[-1].lower() != "pdf":
        st.error("Only PDF files are allowed!")
        st.stop()
    if uploaded_file.name != st.session_state.last_uploaded_file:

        # reset app for new PDF
        st.session_state.ready = False
        st.session_state.history = []
        st.session_state.last_uploaded_file = uploaded_file.name

        # save file
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        # rebuild vector DB
        with st.spinner("Processing new PDF..."):
            ingest_pdf("temp.pdf")

        st.session_state.ready = True
        st.success("PDF ready! Ask questions below 👇")
if uploaded_file:
    if not st.session_state.ready:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        ingest_pdf("temp.pdf")
        st.session_state.ready = True
        st.success("PDF ready! Ask questions below 👇")


# ----------------------------
# CHAT INPUT (always visible after upload)
# ----------------------------
if st.session_state.ready:
    user_question = st.chat_input("Ask a question about the PDF")

     # Render chat history
    for chat in st.session_state.history[:-1]:
        with st.chat_message("user"):
            st.write(chat["user"])

        with st.chat_message("assistant"):
            st.write(chat["bot"])
            st.caption(f"📚 Pages: {chat['pages']}")

    if user_question:
        with st.chat_message("user"):
            st.write(user_question)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            
            stream, pages = query(user_question, st.session_state.history)
            
            for chunk in stream:
                # Groq chunks have .content attribute
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                full_response += token
                placeholder.write(full_response + "▌")  # typing cursor
            
            placeholder.write(full_response)  # final clean render
            st.caption(f"📚 Pages: {pages}")

        st.session_state.history.append({
            "user": user_question,
            "bot": full_response,
            "pages": pages
        })

   
else:
    st.info("👆 Upload a PDF to start chatting")
