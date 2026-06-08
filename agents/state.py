from typing import TypedDict, List, Optional, Annotated
import operator


class ResearchState(TypedDict):
    """
    The state object that flows through every node in the LangGraph.

    Think of it like a baton in a relay race.
    Each node receives it, does its work, updates some fields, passes it on.

    Why TypedDict and not a regular dict?
    TypedDict gives you type hints. If you try to access a field that
    doesn't exist, your editor warns you immediately instead of crashing
    at runtime after a 2 minute research run.

    Why Annotated[List, operator.add] on some fields?
    Because web_agent runs in PARALLEL for each sub-question.
    If 4 agents all try to write to raw_sources at the same time,
    the last one would overwrite the others without operator.add.
    With operator.add, LangGraph APPENDS each agent's results instead.
    This is the key to making parallel fan-out work correctly.
    """

    # Set at the start, never changed
    topic: str                 # "latest advances in quantum computing"
    recipient_email: str       # who gets the final report

    # Set by decompose node
    sub_questions: List[str]   # ["What are key milestones?", "Who are leaders?"]
    run_id: str                # unique ID, used as ChromaDB collection name

    # Set by web agents - operator.add means APPEND not REPLACE
    # This is what makes parallel agents work
    raw_sources: Annotated[List[dict], operator.add]
    errors: Annotated[List[str], operator.add]

    # Set by embed node
    chunks_embedded: int

    # Set by synthesis node
    report_markdown: str

    # Set by deliver node
    email_sent: bool
    email_error: Optional[str]