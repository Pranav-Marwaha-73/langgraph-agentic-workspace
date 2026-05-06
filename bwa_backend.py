from __future__ import annotations
import contextvars
from langchain_core.runnables import RunnableConfig
import operator
import os
import re
from datetime import date, timedelta
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated
from langsmith import traceable
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import time
load_dotenv()

# ============================================================
# Blog Writer (Router → (Research?) → Orchestrator → Workers → ReducerWithImages)
# Patches image capability using your 3-node reducer flow:
#   merge_content -> decide_images -> generate_and_place_images
# ============================================================

bwa_openai_key_var = contextvars.ContextVar('bwa_openai_key', default="")
bwa_tavily_key_var = contextvars.ContextVar('bwa_tavily_key', default="")
bwa_google_key_var = contextvars.ContextVar('bwa_google_key', default="")

# -----------------------------
# 1) Schemas
# -----------------------------
class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="One sentence describing what the reader should do/understand.")
    bullets: List[str] = Field(..., min_length=3, max_length=6)
    target_words: int = Field(..., description="Target words (120–550).")

    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None  # ISO "YYYY-MM-DD" preferred
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str
    queries: List[str] = Field(default_factory=list)
    max_results_per_query: int = Field(5)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


# ---- Image planning schema (ported from your image flow) ----
class ImageSpec(BaseModel):
    placeholder: str = Field(..., description="e.g. [[IMAGE_1]]")
    filename: str = Field(..., description="Save under images/, e.g. qkv_flow.png")
    alt: str
    caption: str
    prompt: str = Field(..., description="Prompt to send to the image model.")
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"


class GlobalImagePlan(BaseModel):
    images: List[ImageSpec] = Field(default_factory=list)

class State(TypedDict):
    topic: str

    # routing / research
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    # recency
    as_of: str
    recency_days: int

    # workers
    sections: Annotated[List[tuple[int, str]], operator.add]  # (task_id, section_md)

    # reducer/image
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]

    final: str

    # ✅ NEW: Human-in-the-loop cost control
    user_wants_images: bool

# -----------------------------
# 3) Router
# -----------------------------
ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false): evergreen concepts.
- hybrid (needs_research=true): evergreen + needs up-to-date examples/tools/models.
- open_book (needs_research=true): volatile weekly/news/"latest"/pricing/policy.

If needs_research=true:
- Output 3–10 high-signal, scoped queries.
- For open_book weekly roundup, include queries reflecting last 7 days.
"""

def router_node(state: State, config: RunnableConfig) -> dict:
    openai_key = bwa_openai_key_var.get()
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_key)
    decider = llm.with_structured_output(RouterDecision)
    decision = decider.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {state['topic']}\nAs-of date: {state['as_of']}"),
        ]
    )

    if decision.mode == "open_book":
        recency_days = 7
    elif decision.mode == "hybrid":
        recency_days = 45
    else:
        recency_days = 3650

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
        "recency_days": recency_days,
    }

def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"

# -----------------------------
# 4) Research (Tavily)
# -----------------------------
def _tavily_search(query: str, tavily_key: str, max_results: int = 5) -> List[dict]:

    if not tavily_key:
        return []
    try:
        from langchain_tavily import TavilySearch  
        tool = TavilySearch(max_results=max_results, tavily_api_key=tavily_key)
        raw_out = tool.invoke({"query": query})

        # ✅ THE FIX: Deep JSON parsing to handle nested dictionaries!
        results_list = []
        if isinstance(raw_out, list):
            results_list = raw_out
        elif isinstance(raw_out, dict):
            if "results" in raw_out:
                results_list = raw_out["results"]
            elif "output" in raw_out and isinstance(raw_out["output"], dict):
                results_list = raw_out["output"].get("results", [])
        
        out: List[dict] = []
        for r in results_list:
            out.append(
                {
                    "title": r.get("title") or "",
                    "url": r.get("url") or "",
                    "snippet": r.get("content") or r.get("snippet") or "",
                    "published_at": r.get("published_date") or r.get("published_at"),
                    "source": r.get("source"),
                }
            )
        return out
    except Exception as e:
        print(f"🚨 TAVILY PARSE ERROR: {e}")
        return []

def _iso_to_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None

def research_node(state: State, config: RunnableConfig) -> dict:
    # Notice we don't even need the OpenAI key here anymore!
    tavily_key = bwa_tavily_key_var.get()

    queries = (state.get("queries") or [])[:10]
    raw: List[dict] = []
    
    for q in queries:
        # Lowered max_results to 3 to protect your free tier credits!
        raw.extend(_tavily_search(q, tavily_key, max_results=3))

    if not raw:
        return {"evidence": []}

    # ✅ THE FIX: Zero-Token Python Mapping (Bypassing the LLM)
    evidence = []
    seen_urls = set()
    
    for r in raw:
        url = r.get("url")
        # Only add it if there is a valid URL we haven't seen yet
        if url and url not in seen_urls:
            seen_urls.add(url)
            evidence.append(
                EvidenceItem(
                    title=r.get("title") or "Unknown",
                    url=url,
                    published_at=r.get("published_at"),
                    snippet=r.get("snippet") or r.get("content") or ""
                )
            )

    # ✅ The "Relaxed Date Filter" you successfully added earlier
    if state.get("mode") == "open_book":
        as_of = date.fromisoformat(state["as_of"])
        cutoff = as_of - timedelta(days=int(state["recency_days"]))
        
        filtered_evidence = []
        for e in evidence:
            d = _iso_to_date(e.published_at)
            # Keep the URL if it's recent, OR if we simply don't know the date
            if d is None or d >= cutoff:
                filtered_evidence.append(e)
        
        evidence = filtered_evidence

    return {"evidence": evidence}

# -----------------------------
# 5) Orchestrator (Plan)
# -----------------------------
ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Produce a highly actionable outline for a technical blog post.

Requirements:
- 5–9 tasks, each with goal + 3–6 bullets + target_words.
- Tags are flexible; do not force a fixed taxonomy.

Grounding:
- closed_book: evergreen, no evidence dependence.
- hybrid: use evidence for up-to-date examples; mark those tasks requires_research=True and requires_citations=True.
- open_book: weekly/news roundup:
  - Set blog_kind="news_roundup"
  - No tutorial content unless requested
  - If evidence is weak, plan should explicitly reflect that (don’t invent events).

Output must match Plan schema.
"""

def orchestrator_node(state: State, config: RunnableConfig) -> dict:
    openai_key = bwa_openai_key_var.get()
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_key)
    planner = llm.with_structured_output(Plan)
    mode = state.get("mode", "closed_book")
    evidence = state.get("evidence", [])

    forced_kind = "news_roundup" if mode == "open_book" else None

    plan = planner.invoke(
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n"
                    f"As-of: {state['as_of']} (recency_days={state['recency_days']})\n"
                    f"{'Force blog_kind=news_roundup' if forced_kind else ''}\n\n"
                    f"Evidence:\n{[e.model_dump() for e in evidence][:16]}"
                )
            ),
        ]
    )
    if forced_kind:
        plan.blog_kind = "news_roundup"

    # ✅ THE FIX: Force Synchronization
    # If the router bypassed research, force the tasks to match!
    if mode == "closed_book":
        for task in plan.tasks:
            task.requires_research = False
            task.requires_citations = False

    return {"plan": plan}


# -----------------------------
# 6) Fanout
# -----------------------------
def fanout(state: State):
    assert state["plan"] is not None
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "as_of": state["as_of"],
                "recency_days": state["recency_days"],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in state["plan"].tasks
    ]

# -----------------------------
# 7) Worker
# -----------------------------
WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Constraints:
- Cover ALL bullets in order.
- Target words ±15%.
- Output only section markdown starting with "## <Section Title>".

Scope guard:
- If blog_kind=="news_roundup", do NOT drift into tutorials (scraping/RSS/how to fetch).
  Focus on events + implications.

Grounding:
- If mode=="open_book": do not introduce any specific event/company/model/funding/policy claim unless supported by provided Evidence URLs.
  For each supported claim, attach a Markdown link ([Source](URL)).
  If unsupported, write "Not found in provided sources."
- If requires_citations==true (hybrid tasks): cite Evidence URLs for external claims.

Code:
- If requires_code==true, include at least one minimal snippet.
"""

def worker_node(payload: dict, config: RunnableConfig) -> dict:
    openai_key = bwa_openai_key_var.get()
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_key)

    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]

    bullets_text = "\n- " + "\n- ".join(task.bullets)
    evidence_text = "\n".join(
        f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}"
        for e in evidence[:20]
    )

    section_md = llm.invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {payload['topic']}\n"
                    f"Mode: {payload.get('mode')}\n"
                    f"As-of: {payload.get('as_of')} (recency_days={payload.get('recency_days')})\n\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (ONLY cite these URLs):\n{evidence_text}\n"
                )
            ),
        ]
    ).content.strip()

    return {"sections": [(task.id, section_md)]}

# ============================================================
# 8) ReducerWithImages (subgraph)
#    merge_content -> decide_images -> generate_and_place_images
# ============================================================
def merge_content(state: State) -> dict:
    plan = state["plan"]
    if plan is None:
        raise ValueError("merge_content called without plan.")
    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    merged_md = f"# {plan.blog_title}\n\n{body}\n"
    return {"merged_md": merged_md}


DECIDE_IMAGES_SYSTEM = """You are an expert technical editor.
Decide if images/diagrams are needed for THIS blog.
Rules:
- Max 3 images total.
- Each image must materially improve understanding (architectural diagram, flowchart, concept illustration).
- NEVER request a table as an image. Tables must be handled via markdown text. 
- NEVER request text-heavy infographics or charts that compare detailed features/weaknesses. Image models cannot render paragraphs of text.
- Keep image prompts focused on abstract, visual representations of systems or workflows with minimal labels.
- If no images needed, return an empty list [].
- Avoid decorative images.
"""

def decide_images(state: State, config: RunnableConfig) -> dict:
    # ✅ NEW: Instantly bypass if user opted out of images to save money
    if not state.get("user_wants_images", False):
        return {
            "md_with_placeholders": state["merged_md"],
            "image_specs": []
        }
    openai_key = bwa_openai_key_var.get()
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_key)
    planner = llm.with_structured_output(GlobalImagePlan)
    merged_md = state["merged_md"]
    plan = state["plan"]
    assert plan is not None

    image_plan = planner.invoke(
        [
            SystemMessage(content=DECIDE_IMAGES_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Topic: {state['topic']}\n\n"
                    "Insert placeholders + propose image prompts.\n\n"
                    f"{merged_md}"
                )
            ),
        ]
    )

    return {
        "md_with_placeholders": state["merged_md"],
        "image_specs": [img.model_dump() for img in image_plan.images],
    }

@traceable(run_type="llm", name="Gemini_Image_Generation")
def _gemini_generate_image_bytes(prompt: str) -> bytes:
    """
    Returns raw image bytes generated by Gemini.
    """
    from google import genai
    from google.genai import types

    # ✅ SECURE: Grab the key from isolated RAM inside the function
    google_key = bwa_google_key_var.get()
    
    if not google_key:
        raise RuntimeError("GOOGLE_API_KEY is missing.")
    
    client = genai.Client(api_key=google_key)

    resp = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_ONLY_HIGH",
                )
            ],
        ),
    )

    # Depending on SDK version, parts may hang off resp.candidates[0].content.parts
    parts = getattr(resp, "parts", None)
    if not parts and getattr(resp, "candidates", None):
        try:
            parts = resp.candidates[0].content.parts
        except Exception:
            parts = None

    if not parts:
        raise RuntimeError("No image content returned (safety/quota/SDK change).")

    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            return inline.data

    raise RuntimeError("No inline image bytes found in response.")


def _safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"

def generate_and_place_images(state: State, config: RunnableConfig) -> dict:
    google_key = bwa_google_key_var.get()

    plan = state["plan"]
    assert plan is not None

    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = state.get("image_specs", []) or []

    # If no images requested, just write merged markdown
    if not image_specs:
        filename = f"{_safe_slug(plan.blog_title)}.md"
        Path(filename).write_text(md, encoding="utf-8")
        return {"final": md}

    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)

    for i, spec in enumerate(image_specs):
        # ✅ FIX 1: Prevent Gemini Rate Limit Blocks (Wait 5 seconds between images)
        if i > 0:
            time.sleep(5) 

        placeholder = spec["placeholder"]
        filename = spec["filename"]
        out_path = images_dir / filename

        # generate only if needed
        if not out_path.exists():
            try:
                img_bytes = _gemini_generate_image_bytes(spec["prompt"])
                out_path.write_bytes(img_bytes)
            except Exception as e:
                # graceful fallback: keep doc usable
                prompt_block = (
                    f"> **[IMAGE GENERATION FAILED]** {spec.get('caption','')}\n>\n"
                    f"> **Alt:** {spec.get('alt','')}\n>\n"
                    f"> **Prompt:** {spec.get('prompt','')}\n>\n"
                    f"> **Error:** {e}\n"
                )
                md = md.replace(placeholder, prompt_block)
                continue

        img_md = f"![{spec['alt']}](images/{filename})\n*{spec['caption']}*"
        # ✅ THE FIX: Defensive Replacement Logic
        if placeholder in md:
            # If the LLM did its job, replace the placeholder
            md = md.replace(placeholder, img_md)
        else:
            # 🛡️ SMART FALLBACK: The LLM was lazy. Let's inject it into the middle!
            paragraphs = md.split('\n\n')
            
            # Mathematically space the images out across the total number of paragraphs
            # e.g., if there are 2 images and 10 paragraphs, insert at index 3 and 6
            spacing = max(1, len(paragraphs) // (len(image_specs) + 1))
            insert_idx = spacing * (i + 1)
            
            # Ensure we don't accidentally insert past the end of the text
            insert_idx = min(insert_idx, len(paragraphs) - 1)
            
            # Inject the image into the paragraph list
            paragraphs.insert(insert_idx, img_md)
            
            # Rebuild the markdown string
            md = '\n\n'.join(paragraphs)

    # Grab the user's email from the config
    user_email = config.get("configurable", {}).get("user_email", "guest")
    safe_email = _safe_slug(user_email)
    
    # Create a private folder for this user (e.g., blogs/pranav_gmail_com/)
    user_dir = Path("blogs") / safe_email
    user_dir.mkdir(parents=True, exist_ok=True)

    # Save the file INSIDE their private folder
    filename = user_dir / f"{_safe_slug(plan.blog_title)}.md"
    filename.write_text(md, encoding="utf-8")
    
    return {"final": md}

# build reducer subgraph
reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)
reducer_subgraph = reducer_graph.compile()

# -----------------------------
# 9) Build main graph
# -----------------------------
g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")

g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()
app
