import json
import uuid
import requests
import streamlit as st

# ─────────────────────────────────────────────────────────────
# API CONFIG
# ─────────────────────────────────────────────────────────────
API_BASE   = "http://localhost:8000/api/v1"
QUERY_URL  = f"{API_BASE}/query"
UPLOAD_URL = f"{API_BASE}/admin/upload"

# ─────────────────────────────────────────────────────────────
# PROFILE JSON TEMPLATES
# ─────────────────────────────────────────────────────────────
PROFILE_TEMPLATES = {
    "policy": {
        "type": "policy",
        "department": "Motor Claims",
        "grade": "Comprehensive"
    },
    "claim": {
        "type": "claim",
        "employee_id": "E12345",
        "claim_id": "CLM001",
        "claim_type": "Motor",
        "claim_amount": 300000
    },
    "insurance": {
        "type": "insurance",
        "employee_id": "E12345",
        "policy_number": "POL00234",
        "coverage_type": "Motor"
    }
}

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG & STYLES
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Policy Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

section[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}

/* Remove default top padding on main */
.block-container {
    padding-top: 1.8rem;
    padding-bottom: 2rem;
    max-width: 860px;
}

/* Clean chat bubbles */
[data-testid="stChatMessage"] {
    border: none;
    background: transparent;
    padding: 0.2rem 0;
}

/* Textarea clean */
textarea { 
    font-family: monospace !important; 
    font-size: 0.82rem !important; 
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SESSION INITIALIZATION
# ─────────────────────────────────────────────────────────────
def _init():
    defaults = {
        "session_id":       str(uuid.uuid4()),
        "messages":         [],          # {role, content, citation, page_no, doc, relevant_chunks}
        "customer_context": "",          # JSON string
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 Policy Assistant")
    st.divider()

    page = st.radio("", ["Chat", "Admin"], label_visibility="collapsed")
    st.divider()

    st.markdown("**Query as (JSON)**")

    template_choice = st.selectbox(
        "Insert profile template",
        ["None", "policy", "claim", "insurance"]
    )

    # If a template is chosen, populate the text area
    if template_choice != "None":
        st.session_state.customer_context = json.dumps(
            PROFILE_TEMPLATES[template_choice], indent=2
        )

    json_text = st.text_area(
        "Paste / edit JSON",
        value=st.session_state.customer_context,
        height=220
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Save profile", use_container_width=True):
            try:
                if json_text.strip():
                    json.loads(json_text) # Validate JSON
                    st.session_state.customer_context = json_text
                    st.success("Profile saved")
                else:
                    st.session_state.customer_context = ""
            except json.JSONDecodeError:
                st.error("Invalid JSON")

    with c2:
        if st.button("Clear profile", use_container_width=True):
            st.session_state.customer_context = ""
            st.rerun()

    if st.session_state.customer_context:
        try:
            ctx_dict = json.loads(st.session_state.customer_context)
            ident = ctx_dict.get("employee_id") or ctx_dict.get("policy_number") or ctx_dict.get("type", "").title()
            if ident:
                st.caption(f"👤 Active: {ident}")
        except:
            pass

    st.divider()

    if st.button("New session", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages   = []
        st.rerun()

    st.caption(f"Session `{st.session_state.session_id[:16]}…`")


# ─────────────────────────────────────────────────────────────
# CHAT PAGE
# ─────────────────────────────────────────────────────────────
if page == "Chat":

    st.markdown("### Ask a policy question")

    ctx_active = bool(st.session_state.customer_context.strip())
    if ctx_active:
        st.caption("Personalising answers using provided JSON profile.")
    else:
        st.caption("Add a customer profile in the sidebar to get personalised answers.")

    st.divider()

    # Render existing messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑‍💼" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])
            
            if msg["role"] == "assistant":
                # Render Citation
                if msg.get("citation"):
                    page_no = msg.get("page_no")
                    doc     = msg.get("doc", "")
                    meta    = " · ".join(filter(None, [doc, f"Page {page_no}" if page_no else None]))
                    with st.expander(f"📎 Source  —  {meta}" if meta else "📎 Source"):
                        st.markdown(f"{msg['citation']}")
                
                # Render Relevant Chunks (JSON)
                if msg.get("relevant_chunks"):
                    with st.expander("🧩 View Relevant Chunks (JSON)"):
                        st.json(msg["relevant_chunks"])

    # Chat input
    user_input = st.chat_input("Type your question…")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user", avatar="🧑‍💼"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Searching…"):
                try:
                    payload = {
                        "query":      user_input,
                        "session_id": st.session_state.session_id,
                    }
                    if st.session_state.customer_context:
                        payload["customer_context"] = json.loads(st.session_state.customer_context)

                    resp = requests.post(QUERY_URL, json=payload, timeout=60)
                    resp.raise_for_status()
                    data = resp.json()

                    answer   = data.get("answer", "No answer returned.")
                    citation = data.get("citation", "")
                    page_no  = data.get("page_no")
                    doc      = data.get("document_name", "")
                    chunks   = data.get("relevant_chunks", []) 

                    st.markdown(answer)

                    # Display Citation
                    if citation:
                        meta = " · ".join(filter(None, [doc, f"Page {page_no}" if page_no else None]))
                        with st.expander(f"📎 Source  —  {meta}" if meta else "📎 Source"):
                            st.markdown(f"{citation}")
                    
                    # Display Relevant Chunks
                    if chunks:
                        with st.expander("🧩 View Relevant Chunks (JSON)"):
                            st.json(chunks) 

                    # Save to session state
                    st.session_state.messages.append({
                        "role":            "assistant",
                        "content":         answer,
                        "citation":        citation,
                        "page_no":         page_no,
                        "doc":             doc,
                        "relevant_chunks": chunks,
                    })

                except requests.exceptions.ConnectionError:
                    st.error("Cannot reach the API. Is FastAPI running on port 8000?")
                except requests.exceptions.HTTPError:
                    st.error(f"API error {resp.status_code}: {resp.text}")
                except Exception as e:
                    st.error(str(e))


# ─────────────────────────────────────────────────────────────
# ADMIN PAGE
# ─────────────────────────────────────────────────────────────
elif page == "Admin":

    st.markdown("### Upload a policy document")
    st.caption("PDF files only. The document will be chunked and stored in PGVector.")
    st.divider()

    uploaded = st.file_uploader("Choose a PDF", type=["pdf"], label_visibility="collapsed")

    if uploaded:
        st.markdown(f"**{uploaded.name}** ·  {uploaded.size / 1024:.1f} KB")

        if st.button("Ingest document"):
            with st.spinner("Ingesting…"):
                try:
                    resp = requests.post(
                        UPLOAD_URL,
                        files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                        timeout=120,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    st.success(f"Done — {data.get('file', uploaded.name)} ingested.")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot reach the API. Is FastAPI running on port 8000?")
                except requests.exceptions.HTTPError:
                    st.error(f"Upload failed ({resp.status_code}): {resp.text}")
                except Exception as e:
                    st.error(str(e))
