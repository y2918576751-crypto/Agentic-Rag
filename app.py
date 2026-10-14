"""Optional Streamlit UI. Run: python -m streamlit run app.py."""
import json

import streamlit as st

from agentsearch.agent import Agent, AgentFailure
from agentsearch.config import ROOT, load_config
from agentsearch.llm import APIClient
from agentsearch.retrieval import Retrieval

st.set_page_config(page_title="Skill-Augmented Search", layout="wide")
st.title("Skill-Augmented Search Agent")
st.caption("Answers are based on the indexed fixed document collection.")
config_path = st.sidebar.text_input("Configuration file", str(ROOT / "config.json"))
question = st.text_area("Question")
if st.button("Search", type="primary", disabled=not question.strip()):
    retrieval = None
    try:
        config = load_config(config_path)
        retrieval = Retrieval(config)
        with st.spinner("Searching and checking evidence…"):
            result = Agent(retrieval, APIClient(config["max_output_tokens"]), config).run(question)
        st.session_state["result"] = result
        st.session_state.pop("failure", None)
    except AgentFailure as exc:
        st.session_state.pop("result", None)
        st.session_state["failure"] = {"error": str(exc), "trajectory": exc.state.trajectory}
    except Exception as exc:
        st.session_state.pop("result", None)
        st.session_state["failure"] = {"error": str(exc)}
    finally:
        if retrieval is not None:
            retrieval.close()

result = st.session_state.get("result")
if result:
    st.subheader("Answer")
    st.write(result["answer"])
    a, b = st.columns(2)
    a.metric("Tool calls", result["tool_calls"])
    b.metric("API tokens", result["total_tokens"])
    final = result["trajectory"][-1]
    st.subheader("Selected skills")
    st.write(final["selected_skills"] or "No skill selected")
    st.subheader("Supporting evidence")
    for evidence in final["evidence"]:
        with st.expander(f"{evidence['evidence_id']} · {evidence['doc_id']} · offset {evidence['start']}"):
            st.text(evidence["text"])
            if evidence["source_url"].startswith(("https://", "http://")):
                st.link_button("Source URL", evidence["source_url"])
    st.subheader("Tool-use process")
    for event in result["trajectory"]:
        if event["type"] in {"agent_decision", "tool_call", "answer_verification"}:
            with st.expander(f"Step {event['step']}: {event['type']}"):
                st.json(event)
    st.download_button("Download full trajectory", json.dumps(result, ensure_ascii=False, indent=2),
                       file_name="interactive_trace.json", mime="application/json")
if "failure" in st.session_state:
    st.error(st.session_state["failure"]["error"])
    with st.expander("Failure trace"):
        st.json(st.session_state["failure"])
