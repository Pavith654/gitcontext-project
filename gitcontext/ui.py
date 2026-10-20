import requests
import streamlit as st

API = "http://localhost:8000"
st.title("GitContext")

with st.sidebar:
    st.header("Repository")
    name = st.text_input("Name", "my-project")
    path = st.text_input("Local path to cloned repo")
    if st.button("Index repository") and path:
        with st.spinner("Indexing..."):
            r = requests.post(f"{API}/repositories", json={"name": name, "path": path})
        st.success(r.json()) if r.ok else st.error(r.text)
    repos = requests.get(f"{API}/repositories").json() if True else []
    repo = st.selectbox("Ask about", repos) if repos else None
    rerank = st.checkbox("Rerank results")

q = st.chat_input("Ask about the codebase...")
if q and repo:
    st.chat_message("user").write(q)
    r = requests.post(f"{API}/ask", json={"repository_id": repo, "question": q, "rerank": rerank})
    if r.ok:
        d = r.json()
        with st.chat_message("assistant"):
            st.write(d["answer"])
            with st.expander("Sources"):
                for s in d["sources"]:
                    st.write(f"[{s['ref']}] {s['label']}  ·  {s['type']}")
    else:
        st.error(r.text)
