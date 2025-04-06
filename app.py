import streamlit as st
import requests
from bs4 import BeautifulSoup
import torch

from search import search

# Patch for PyTorch bug
if hasattr(torch, "classes"):
    try:
        _ = torch.classes.__name__
    except Exception:
        pass

st.set_page_config(page_title="SHL Assessment Recommender", layout="centered")
st.title("📊 SHL Assessment Recommendation Tool")

st.markdown("### 🧩 Assessment Categories")
st.markdown("""
A - Ability & Aptitude  
B - Biodata & Situational Judgement  
C - Competencies  
D - Development & 360  
E - Assessment Exercises  
K - Knowledge & Skills  
P - Personality & Behavior  
S - Simulations
""")

st.markdown("Enter a **job description** or paste a **URL**. The system will recommend the most relevant assessments.")

input_type = st.radio("Choose input type:", ("Text", "URL"))
user_input = ""

if input_type == "Text":
    user_input = st.text_area("Paste your job description or requirement here:", height=200)

elif input_type == "URL":
    user_url = st.text_input("Enter a URL pointing to a job description:")
    if user_url:
        if not user_url.startswith(("http://", "https://")):
            st.warning("⚠️ Please enter a valid URL (with http:// or https://)")
        else:
            try:
                response = requests.get(user_url, timeout=5)
                soup = BeautifulSoup(response.text, "html.parser")
                user_input = soup.get_text(separator=" ", strip=True)
                st.success("✅ Extracted text from URL!")
            except Exception as e:
                st.error(f"❌ Failed to fetch text from URL: {e}")

# --- Options ---
st.markdown("### 🧠 Gemini AI Features")
col1, col2 = st.columns(2)
with col1:
    enable_rerank = st.checkbox("🔁 Re-rank with Gemini", value=True)
    enable_fallback = st.checkbox("🧩 Fallback via Gemini", value=True)
with col2:
    show_explanations = st.checkbox("💬 Show LLM Explanations", value=False)
    top_k = st.slider("🔝 Number of results", min_value=5, max_value=15, value=10)

# --- Trigger Search ---
if st.button("🔍 Recommend Assessments"):
    if not user_input.strip():
        st.warning("⚠️ Please enter valid input before searching.")
    else:
        with st.spinner("Rewriting, embedding and searching..."):
            try:
                search_response = search(
                    query=user_input,
                    top_k=top_k,
                    debug=False,
                    include_explanations=show_explanations,
                    do_rerank=enable_rerank
                )

                rewritten_query = search_response.get("rewritten_query", "")
                results = search_response.get("results", [])
                fallback_msg = search_response.get("fallback", None)

            except Exception as e:
                st.error(f"❌ Search failed: {e}")
                results = []
                rewritten_query = ""
                fallback_msg = None

        if rewritten_query:
            st.info(f"📝 Gemini Rewritten Query:\n\n{rewritten_query}")

        if results:
            st.success(f"🎯 Top {len(results)} relevant assessments:")
            for idx, item in enumerate(results[:top_k], 1):
                name = item.get('Assessment Name', 'Untitled')
                url = item.get('URL', '#')

                from streamlit.components.v1 import html

                assessment_name = item.get('Assessment Name', 'Untitled')
                assessment_url = item.get('URL', '#')

                # Inline button that looks like a link
                html(f"""
                    <div style="margin-bottom: 0.5em">
                        <a href="{assessment_url}" target="_blank" style="
                        text-decoration: none;
                        color: #1f77b4;
                        font-size: 20px;
                        font-weight: 600;
                        font-family: 'sans-serif';
                        ">{idx}. {assessment_name}</a>
                    </div>
                    """, height=30)


                st.markdown(f"- **Job Levels**: {item.get('Job Levels', 'N/A')}")

                test_type_raw = item.get('Test Type(s)', 'N/A')
                test_type_list = [t.strip() for t in test_type_raw.split(',')]
                unique_test_types = ', '.join(sorted(set(test_type_list)))
                st.markdown(f"- **Test Type(s)**: {unique_test_types}")

                st.markdown(f"- **Remote Testing Support**: {item.get('Remote Testing Support', 'N/A')}")
                st.markdown(f"- **Adaptive Support**: {item.get('Adaptive Support', 'N/A')}")
                st.markdown(f"- **IRT Support**: {item.get('IRT Support', 'N/A')}")
                st.markdown(f"- **Duration**: {item.get('Duration', 'N/A')}")
                st.markdown(f"- **Description**:\n> {item.get('Description', '')[:1000]}...\n")

                if show_explanations and "LLM Explanation" in item:
                    st.markdown(f"🧠 **Gemini Explanation:**\n> {item['LLM Explanation']}")
                st.markdown("---")


        elif fallback_msg and enable_fallback:
            st.warning(f"🤖 {fallback_msg}")

        else:
            st.warning("😕 No relevant assessments found. Try rephrasing or simplifying your input.")
