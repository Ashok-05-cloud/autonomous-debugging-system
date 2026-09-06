"""
Streamlit Web UI for AI Debugger
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

st.set_page_config(
    page_title="AI Multi-Agent Debugger",
    page_icon="🐛",
    layout="wide",
)

from orchestrator.graph import run_pipeline
from utils.output_formatter import build_output, save_output


# ─── State helpers ────────────────────────────────────────────────────────────

def init_session():
    defaults = {
        "pipeline_result": None,
        "output": None,
        "running": False,
        "agent_logs": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─── UI ───────────────────────────────────────────────────────────────────────

init_session()

st.title("🐛 AI Multi-Agent Debugger")
st.caption("Upload a bug report, logs, and optionally a repo zip to get a full root-cause analysis.")

# ── Sidebar: Configuration ────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    ollama_url = st.text_input("Ollama URL", value="http://localhost:11434")
    os.environ["OLLAMA_BASE_URL"] = ollama_url

    ollama_model = st.text_input("Model", value="llama3.2")
    os.environ["OLLAMA_MODEL"] = ollama_model

    st.divider()
    st.markdown("**About**")
    st.markdown(
        "Multi-agent pipeline using LangGraph.\n\n"
        "Agents: Triage → Log Analyst → Repo Navigator → "
        "Reproduction → Fix Planner → Reviewer"
    )

# ── Main: Input Panel ─────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📥 Input Panel")

    bug_file = st.file_uploader("Bug Report (JSON or Markdown)", type=["json", "md", "txt"])
    log_file = st.file_uploader("Log File", type=["log", "txt"])
    repo_zip = st.file_uploader("Repository (ZIP, optional)", type=["zip"])

    st.markdown("**Or use the mock repo:**")
    use_mock = st.checkbox("Use built-in mock repo (BUG-2847)", value=True)

    run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

with col2:
    st.subheader("📝 Quick Preview")
    if bug_file:
        content = bug_file.read().decode("utf-8", errors="replace")
        bug_file.seek(0)
        st.text_area("Bug Report", content[:800], height=200, disabled=True)
    elif use_mock:
        mock_path = Path("mock_repo/bug_report.json")
        if mock_path.exists():
            st.text_area("Mock Bug Report", mock_path.read_text()[:800], height=200, disabled=True)

# ── Run Pipeline ──────────────────────────────────────────────────────────────

if run_btn and not st.session_state.running:
    st.session_state.running = True

    with st.spinner("Running multi-agent pipeline..."):
        # Load inputs
        bug_report = {}
        logs = ""
        repo_path = ""

        if use_mock and not bug_file:
            mock_bug = Path("mock_repo/bug_report.json")
            mock_log = Path("mock_repo/logs/production.log")
            if mock_bug.exists():
                bug_report = json.loads(mock_bug.read_text())
            if mock_log.exists():
                logs = mock_log.read_text()
            repo_path = str(Path("mock_repo").absolute()) if Path("mock_repo").is_dir() else ""

        else:
            if bug_file:
                raw = bug_file.read().decode("utf-8", errors="replace")
                if bug_file.name.endswith(".json"):
                    try:
                        bug_report = json.loads(raw)
                    except Exception:
                        bug_report = {"title": "Bug report", "description": raw[:2000]}
                else:
                    bug_report = {"title": Path(bug_file.name).stem, "description": raw[:2000]}

            if log_file:
                logs = log_file.read().decode("utf-8", errors="replace")

            if repo_zip:
                import zipfile
                tmpdir = tempfile.mkdtemp(prefix="ai_debugger_repo_")
                zf = zipfile.ZipFile(repo_zip)
                zf.extractall(tmpdir)
                repo_path = tmpdir

        if not bug_report:
            bug_report = {"title": "Analysis from logs", "description": "No bug report provided"}

        # Run pipeline
        start = time.time()
        final_state = run_pipeline(bug_report, logs, repo_path)
        elapsed = time.time() - start

        output = build_output(final_state)
        output["metadata"]["elapsed_seconds"] = round(elapsed, 1)

        st.session_state.pipeline_result = final_state
        st.session_state.output = output

    st.session_state.running = False
    st.rerun()

# ── Results Panels ────────────────────────────────────────────────────────────

if st.session_state.output:
    output = st.session_state.output
    state = st.session_state.pipeline_result
    meta = output.get("metadata", {})

    st.divider()

    # ── Metrics row ───────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    conf = meta.get("pipeline_confidence", 0)
    conf_color = "🟢" if conf > 0.7 else ("🟡" if conf > 0.4 else "🔴")
    m1.metric("Confidence", f"{conf:.0%}", delta=conf_color)
    m2.metric("Bug Confirmed", "✅ Yes" if output["repro"].get("bug_confirmed") else "❌ No")
    m3.metric("LLM Used", "Yes" if meta.get("llm_used") else "No (fallback)")
    m4.metric("Pipeline Time", f"{meta.get('elapsed_seconds', 0):.1f}s")

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Root Cause", "🧪 Reproduction", "🔧 Patch Plan", "📊 Agent Trace", "📄 Full JSON"
    ])

    with tab1:
        st.subheader("Root Cause Analysis")
        rc = output.get("root_cause", {})
        bs = output.get("bug_summary", {})

        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.markdown(f"**Hypothesis:**\n\n{rc.get('hypothesis', 'Not determined')}")
            if rc.get("mechanism"):
                st.markdown(f"**Mechanism:**\n\n{rc.get('mechanism')}")
            if rc.get("affected_component"):
                st.code(rc.get("affected_component"), language="text")
        with col_b:
            st.metric("Confidence", f"{rc.get('confidence', 0):.0%}")
            st.markdown(f"**Bug Type:** `{rc.get('bug_type', 'unknown')}`")

        st.divider()
        st.subheader("Evidence")
        for ev in output.get("evidence", [])[:6]:
            if ev.get("type") == "stack_trace":
                with st.expander(f"🔴 {ev.get('exception')} (×{ev.get('frequency', 1)})", expanded=True):
                    st.code(f"{ev.get('file')}:{ev.get('line')} in {ev.get('function')}", language="text")
                    st.caption(ev.get("message", ""))
            elif ev.get("type") == "anomaly":
                st.warning(f"⚠️ {ev.get('description')}")

        review = output.get("metadata", {})
        if output.get("metadata", {}).get("contradictions"):
            st.subheader("⚠️ Contradictions")
            for c in output["metadata"]["contradictions"]:
                st.error(c)

    with tab2:
        st.subheader("Reproduction Script")
        repro = output.get("repro", {})

        status_map = {"fail": "🔴 FAIL (bug confirmed)", "pass": "🟢 PASS", "error": "🟡 ERROR", "timeout": "⏱️ TIMEOUT"}
        st.markdown(f"**Status:** {status_map.get(repro.get('status', 'unknown'), repro.get('status', 'unknown'))}")

        if repro.get("file") and Path(repro["file"]).exists():
            repro_code = Path(repro["file"]).read_text()
            st.code(repro_code, language="python")
            st.download_button(
                "⬇️ Download Repro Script",
                data=repro_code,
                file_name=Path(repro["file"]).name,
                mime="text/plain",
            )

        if repro.get("output"):
            st.subheader("Execution Output")
            st.code(repro["output"], language="text")

    with tab3:
        st.subheader("Patch Plan")
        pp = output.get("patch_plan", {})
        vp = output.get("validation_plan", {})

        st.markdown(f"**Summary:** {pp.get('summary', 'N/A')}")

        if pp.get("code_change"):
            st.subheader("Code Change")
            st.code(pp.get("code_change"), language="python")

        if pp.get("files_to_change"):
            st.subheader("Files to Change")
            for f in pp["files_to_change"]:
                st.markdown(f"- `{f.get('file')}` lines {f.get('lines')}: {f.get('change', '')}")

        if vp.get("validation_steps"):
            st.subheader("Validation Steps")
            for step in vp["validation_steps"]:
                st.markdown(f"- ✅ {step}")

        if vp.get("edge_cases"):
            st.subheader("Edge Cases to Test")
            for ec in vp["edge_cases"]:
                st.markdown(f"- 🔲 {ec}")

        oq = output.get("open_questions", [])
        if oq:
            st.subheader("Open Questions")
            for q in oq:
                st.markdown(f"- ❓ {q}")

    with tab4:
        st.subheader("Agent Execution Trace")
        trace = state.get("agent_trace", [])
        for entry in trace:
            icon = "✅" if entry.get("status") == "success" else "❌"
            with st.expander(f"{icon} {entry.get('agent')} ({entry.get('duration_s', 0):.2f}s)"):
                st.json(entry)

        errors = state.get("errors", [])
        if errors:
            st.subheader("Pipeline Errors/Warnings")
            for e in errors:
                st.warning(e)

    with tab5:
        st.subheader("Full JSON Report")
        json_str = json.dumps(output, indent=2, default=str)
        st.download_button(
            "⬇️ Download Full Report",
            data=json_str,
            file_name="debug_report.json",
            mime="application/json",
        )
        st.json(output)
