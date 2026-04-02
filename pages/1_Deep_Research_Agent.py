from __future__ import annotations
from bwa_backend import bwa_openai_key_var, bwa_tavily_key_var, bwa_google_key_var
import json
import os
import re
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, List, Iterator, Tuple
import re
from pathlib import Path
import pandas as pd
import streamlit as st

# -----------------------------
# Import your compiled LangGraph app
# -----------------------------
from bwa_backend import app


# -----------------------------
# Helpers
# -----------------------------
def safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def bundle_zip(md_text: str, md_filename: str, images_dir: Path) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(md_filename, md_text.encode("utf-8"))

        if images_dir.exists() and images_dir.is_dir():
            for p in images_dir.rglob("*"):
                if p.is_file():
                    z.write(p, arcname=str(p))
    return buf.getvalue()


def images_zip(images_dir: Path) -> Optional[bytes]:
    if not images_dir.exists() or not images_dir.is_dir():
        return None
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in images_dir.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(p))
    return buf.getvalue()


def try_stream(graph_app, inputs: Dict[str, Any], config: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    """
    Stream graph progress and yield the final state WITHOUT running it twice.
    """
    try:
        # 1. Start with the initial inputs
        accumulated_state = dict(inputs) 
        
        # 2. Run the stream ONCE
        for step in graph_app.stream(inputs, config=config, stream_mode="updates"):
            yield ("updates", step)
            
            # 3. Apply the changes from each node to our running state
            if isinstance(step, dict):
                for node_name, node_state in step.items():
                    if isinstance(node_state, dict):
                        accumulated_state.update(node_state)
        
        # 4. Yield the final state (DO NOT call .invoke() here!)
        yield ("final", accumulated_state)
        return
        
    except Exception as e:
        # 🚨 THE FIX: Just yield the error and STOP. Do not run .invoke()!
        print(f"Graph execution failed: {e}")
        yield ("error", str(e))
        return

    # 5. ONLY use invoke as a backup if streaming completely crashes
    out = graph_app.invoke(inputs, config=config)
    yield ("final", out)


def extract_latest_state(current_state: Dict[str, Any], step_payload: Any) -> Dict[str, Any]:
    if isinstance(step_payload, dict):
        if len(step_payload) == 1 and isinstance(next(iter(step_payload.values())), dict):
            inner = next(iter(step_payload.values()))
            current_state.update(inner)
        else:
            current_state.update(step_payload)
    return current_state


# -----------------------------
# Markdown renderer that supports local images
# -----------------------------
_MD_IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")
_CAPTION_LINE_RE = re.compile(r"^\*(?P<cap>.+)\*$")


def _resolve_image_path(src: str) -> Path:
    src = src.strip().lstrip("./")
    return Path(src).resolve()


def render_markdown_with_local_images(md: str):
    matches = list(_MD_IMG_RE.finditer(md))
    if not matches:
        st.markdown(md, unsafe_allow_html=False)
        return

    parts: List[Tuple[str, str]] = []
    last = 0
    for m in matches:
        before = md[last : m.start()]
        if before:
            parts.append(("md", before))

        alt = (m.group("alt") or "").strip()
        src = (m.group("src") or "").strip()
        parts.append(("img", f"{alt}|||{src}"))
        last = m.end()

    tail = md[last:]
    if tail:
        parts.append(("md", tail))

    i = 0
    while i < len(parts):
        kind, payload = parts[i]

        if kind == "md":
            st.markdown(payload, unsafe_allow_html=False)
            i += 1
            continue

        alt, src = payload.split("|||", 1)

        caption = None
        if i + 1 < len(parts) and parts[i + 1][0] == "md":
            nxt = parts[i + 1][1].lstrip()
            if nxt.strip():
                first_line = nxt.splitlines()[0].strip()
                mcap = _CAPTION_LINE_RE.match(first_line)
                if mcap:
                    caption = mcap.group("cap").strip()
                    rest = "\n".join(nxt.splitlines()[1:])
                    parts[i + 1] = ("md", rest)

        if src.startswith("http://") or src.startswith("https://"):
            st.image(src, caption=caption or (alt or None), use_container_width=True)
        else:
            img_path = _resolve_image_path(src)
            if img_path.exists():
                st.image(str(img_path), caption=caption or (alt or None), use_container_width=True)
            else:
                st.warning(f"Image not found: `{src}` (looked for `{img_path}`)")

        i += 1


# -----------------------------
# ✅ NEW: Past blogs helpers
# -----------------------------
def list_past_blogs() -> List[Path]:
    """
    Returns .md files in current working directory, newest first.
    Filters out obvious non-blog markdown files if needed.
    """
    cwd = Path(".")
    files = [p for p in cwd.glob("*.md") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def read_md_file(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def extract_title_from_md(md: str, fallback: str) -> str:
    """
    Use first '# ' heading as title if present.
    """
    for line in md.splitlines():
        if line.startswith("# "):
            t = line[2:].strip()
            return t or fallback
    return fallback


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Deep Research Agent", layout="wide")

# ✅ Inject your custom CSS background here too!
try:
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass # Ignores if style.css isn't found

if "user_email" not in st.session_state:
    st.title("🔒 Access Denied")
    st.warning("Please sign in with Google on the main 'ReAct Agent' page to unlock the Deep Research tool.")
    st.stop() # Immediately halts execution of the rest of the page!

with st.sidebar:
    st.header("Start Deep Research")
    topic = st.text_area("Topic", height=120)
    as_of = st.date_input("As-of date", value=date.today())
    
    # ✅ NEW: The UI Toggle for cost control
    generate_images = st.toggle("🖼️ Generate AI Images (Costs Extra)", value=False)
    run_btn = st.button("🚀 Generate Research", type="primary")

    # ✅ NEW: Past blogs list (keeps everything else intact)
    st.divider()
    st.subheader("Past Research")

    st.title("Deep Research Agent") # ✅ Changed from Blog Writing Agent

    # 1. Grab the logged-in user's email securely
    current_email = st.session_state.get("user_email", "guest")

    # Make the email safe for folder paths (matching bwa_backend.py)
    safe_email = current_email.strip().lower()
    safe_email = re.sub(r"[^a-z0-9 _-]+", "", safe_email)
    safe_email = re.sub(r"\s+", "_", safe_email).strip("_") or "guest"

    # 2. Point directly to THEIR private folder
    user_dir = Path("blogs") / safe_email

    # 3. Safely load and sort only THEIR files (newest first)
    if user_dir.exists():
        past_files = sorted(list(user_dir.glob("*.md")), key=lambda x: x.stat().st_mtime, reverse=True)
    else:
        past_files = []

    if not past_files:
        st.caption(f"No saved research found for {current_email}.")
        selected_md_file = None
    else:
        # Build labels from file name + (optional) parsed title
        options: List[str] = []
        file_by_label: Dict[str, Path] = {}
        for p in past_files[:50]:
            try:
                md_text = read_md_file(p)
                title = extract_title_from_md(md_text, p.stem)
            except Exception:
                title = p.stem
            label = f"{title}"
            options.append(label)
            file_by_label[label] = p

        selected_label = st.radio(
            "Select a research to load",
            options=options,
            index=0,
            label_visibility="collapsed",
        )
        selected_md_file = file_by_label.get(selected_label)

        if st.button("📂 Load selected research"):
            if selected_md_file:
                md_text = read_md_file(selected_md_file)
                # Load into session_state as if it were a run output
                st.session_state["last_out"] = {
                    "plan": None,          # old files don't include plan
                    "evidence": [],        # old files don't include evidence
                    "image_specs": [],     # optional (not persisted)
                    "final": md_text,      # markdown body
                }
                # also update the topic input to the title (best-effort) without changing UI
                st.session_state["topic_prefill"] = extract_title_from_md(md_text, selected_md_file.stem)

    

# Keep your topic input as-is; optionally prefill for next run after loading a blog
if "topic_prefill" in st.session_state and isinstance(st.session_state["topic_prefill"], str):
    # Do not mutate widgets; just keep as a hint.
    pass

# Storage for latest run
if "last_out" not in st.session_state:
    st.session_state["last_out"] = None

# Layout
tab_plan, tab_evidence, tab_preview, tab_images, tab_logs = st.tabs(
    ["🧩 Plan", "🔎 Evidence", "📝 Markdown Preview", "🖼️ Images", "🧾 Logs"]
)

logs: List[str] = []


def log(msg: str):
    logs.append(msg)


if run_btn:
    if not topic.strip():
        st.warning("Please enter a topic.")
        st.stop()

    # ✅ SECURE KEY INJECTION: Grab keys from the main app's session state
    openai_key = st.session_state.get("openai_api_key", "")
    tavily_key = st.session_state.get("tavily_api_key", "")
    google_key = st.session_state.get("google_api_key", "") # Assuming you add this to your main sidebar

    if not openai_key:
        st.error("⚠️ Please enter your OpenAI API key in the main chat sidebar first!")
        st.stop()

    # ✅ OPTIONAL: Force them to provide a Tavily key
    if not tavily_key:
        st.error("⚠️ Deep Research requires a Tavily API key. Please add it in the main sidebar.")
        st.stop()

    # ✅ THE NEW FIX: Catch the missing Gemini key before the graph runs!
    if generate_images and not google_key:
        st.error("⚠️ You checked 'Generate AI Images', but didn't provide a Google/Gemini API key in the main sidebar. Please add it or turn off image generation.")
        st.stop()

    config = {
        "configurable": {
            "user_email": st.session_state.get("user_email", "guest")
        }
    }
    # Inject securely into RAM
    t_o = bwa_openai_key_var.set(openai_key)
    t_t = bwa_tavily_key_var.set(tavily_key)
    t_g = bwa_google_key_var.set(google_key)

    inputs: Dict[str, Any] = {
        "topic": topic.strip(),
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "as_of": as_of.isoformat(),
        "recency_days": 7,
        "sections": [],
        "merged_md": "",
        "md_with_placeholders": "",
        "image_specs": [],
        "final": "",
        "user_wants_images": generate_images,  # Pass the user's choice to the graph
    }

    status = st.status("Running graph…", expanded=True)
    progress_area = st.empty()

    current_state: Dict[str, Any] = {}
    last_node = None
    try:
        for kind, payload in try_stream(app, inputs, config):
            if kind in ("updates", "values"):
                node_name = None
                if isinstance(payload, dict) and len(payload) == 1 and isinstance(next(iter(payload.values())), dict):
                    node_name = next(iter(payload.keys()))
            if node_name and node_name != last_node:
                status.write(f"➡️ Node: `{node_name}`")
                last_node = node_name

            current_state = extract_latest_state(current_state, payload)

            # ✅ THE FIX: Bulletproof task counting
            plan_data = current_state.get("plan")
            tasks_len = 0
            try:
                if plan_data:
                    if hasattr(plan_data, "tasks"):
                        tasks_len = len(plan_data.tasks)
                    elif isinstance(plan_data, dict) and "tasks" in plan_data:
                        tasks_len = len(plan_data["tasks"])
            except Exception:
                pass

            summary = {
                "mode": current_state.get("mode"),
                "needs_research": current_state.get("needs_research"),
                "queries": current_state.get("queries", [])[:5] if isinstance(current_state.get("queries"), list) else [],
                "evidence_count": len(current_state.get("evidence", []) or []),
                "tasks": tasks_len if tasks_len > 0 else "Planning...",
                "images": len(current_state.get("image_specs", []) or []),
                "sections_done": len(current_state.get("sections", []) or []),
            }
            progress_area.json(summary)

            log(f"[{kind}] {json.dumps(payload, default=str)[:1200]}")

            if kind == "final":
                out = payload
                st.session_state["last_out"] = out
                status.update(label="✅ Done", state="complete", expanded=False)
                log("[final] received final state")

    finally:
        # Cleanup RAM
        bwa_openai_key_var.reset(t_o)
        bwa_tavily_key_var.reset(t_t)
        bwa_google_key_var.reset(t_g)

# Render last result (if any)
out = st.session_state.get("last_out")
if out:
    # --- Plan tab ---
    with tab_plan:
        st.subheader("Plan")
        plan_obj = out.get("plan")
        if not plan_obj:
            st.info("No plan found in output.")
        else:
            if hasattr(plan_obj, "model_dump"):
                plan_dict = plan_obj.model_dump()
            elif isinstance(plan_obj, dict):
                plan_dict = plan_obj
            else:
                plan_dict = json.loads(json.dumps(plan_obj, default=str))

            st.write("**Title:**", plan_dict.get("blog_title"))  # key unchanged, display is fine
            cols = st.columns(3)
            cols[0].write("**Audience:** " + str(plan_dict.get("audience")))
            cols[1].write("**Tone:** " + str(plan_dict.get("tone")))
            cols[2].write("**Research kind:** " + str(plan_dict.get("blog_kind", "")))

            tasks = plan_dict.get("tasks", [])
            if tasks:
                df = pd.DataFrame(
                    [
                        {
                            "id": t.get("id"),
                            "title": t.get("title"),
                            "target_words": t.get("target_words"),
                            "requires_research": t.get("requires_research"),
                            "requires_citations": t.get("requires_citations"),
                            "requires_code": t.get("requires_code"),
                            "tags": ", ".join(t.get("tags") or []),
                        }
                        for t in tasks
                    ]
                ).sort_values("id")
                st.dataframe(df, use_container_width=True, hide_index=True)

                with st.expander("Task details"):
                    st.json(tasks)

    # --- Evidence tab ---
    with tab_evidence:
        st.subheader("Evidence")
        evidence = out.get("evidence") or []
        if not evidence:
            st.info("No evidence returned (maybe closed_book mode or no Tavily key/results).")
        else:
            rows = []
            for e in evidence:
                if hasattr(e, "model_dump"):
                    e = e.model_dump()
                rows.append(
                    {
                        "title": e.get("title"),
                        "published_at": e.get("published_at"),
                        "source": e.get("source"),
                        "url": e.get("url"),
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # --- Preview tab ---
    with tab_preview:
        st.subheader("Markdown Preview")
        final_md = out.get("final") or ""
        if not final_md:
            st.warning("No final markdown found.")
        else:
            render_markdown_with_local_images(final_md)

            plan_obj = out.get("plan")
            if hasattr(plan_obj, "blog_title"):
                blog_title = plan_obj.blog_title
            elif isinstance(plan_obj, dict):
                blog_title = plan_obj.get("blog_title", "blog")
            else:
                # fallback: parse from markdown title
                blog_title = extract_title_from_md(final_md, "blog")

            md_filename = f"{safe_slug(blog_title)}.md"
            st.download_button(
                "⬇️ Download Markdown",
                data=final_md.encode("utf-8"),
                file_name=md_filename,
                mime="text/markdown",
            )

            bundle = bundle_zip(final_md, md_filename, Path("images"))
            st.download_button(
                "📦 Download Bundle (MD + images)",
                data=bundle,
                file_name=f"{safe_slug(blog_title)}_bundle.zip",
                mime="application/zip",
            )

    # --- Images tab ---
    with tab_images:
        st.subheader("Images")
        specs = out.get("image_specs") or []
        images_dir = Path("images")

        if not specs and not images_dir.exists():
            st.info("No images generated for this research.")
        else:
            if specs:
                st.write("**Image plan:**")
                st.json(specs)

            if images_dir.exists():
                files = [p for p in images_dir.iterdir() if p.is_file()]
                if not files:
                    st.warning("images/ exists but is empty.")
                else:
                    for p in sorted(files):
                        st.image(str(p), caption=p.name, use_container_width=True)

                z = images_zip(images_dir)
                if z:
                    st.download_button(
                        "⬇️ Download Images (zip)",
                        data=z,
                        file_name="images.zip",
                        mime="application/zip",
                    )

    # --- Logs tab ---
    with tab_logs:
        st.subheader("Logs")
        if "logs" not in st.session_state:
            st.session_state["logs"] = []
        if logs:
            st.session_state["logs"].extend(logs)

        st.text_area("Event log", value="\n\n".join(st.session_state["logs"][-80:]), height=520)
else:
    st.info("Enter a topic and click **Generate Research**.")