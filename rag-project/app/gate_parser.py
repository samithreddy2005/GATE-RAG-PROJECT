"""
Subject-agnostic GATE paper parser.

GATE papers across all 30 subjects follow the same conventions (confirmed
against the official 2023 CS paper): questions are marked "Q.<n>", grouped
under banners like "Q.1 - Q.5 Carry ONE mark Each", multiple-choice options
are lettered (A)-(D), and Numerical Answer Type (NAT) questions have no
options and end in a blank ("____" or "."). Because the convention is
uniform, this parser works on any subject's paper -- only the topic
taxonomy in TOPIC_KEYWORDS below is subject-specific and needs a new
dictionary per subject family.

Real-world messiness handled here (seen directly in the fetched official
PDF text): repeated page headers/footers interleaved mid-question,
inconsistent spacing, and marks-banners that apply to a *range* of
question numbers rather than one.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

Q_START_RE = re.compile(r"(?m)^Q\.(\d+)\b")
MARKS_BANNER_RE = re.compile(
    r"Q\.(\d+)\s*[-–]\s*Q\.(\d+)\s+Carry\s+(ONE|TWO)\s+marks?\s+each",
    re.IGNORECASE,
)
OPTION_RE = re.compile(r"(?m)^\(([A-D])\)\s*(.+)$")
NOISE_LINE_RES = [
    re.compile(r"^Page \d+ of \d+$"),
    re.compile(r"^Organizing Institute:.*$"),
    re.compile(r"^\w{2,3} Page \d+ of \d+$"),
    re.compile(r"^GATE \d{4} .+$"),
]

# Minimal example taxonomy for GATE CS -- one dictionary per subject family.
# Extend with more subjects (EC, ME, ...) by adding a new keyword map and
# selecting it based on the `subject` field during ingestion.
TOPIC_KEYWORDS_CS = {
    "Data Structures": ["heap", "linked list", "stack", "queue", "array", "tree", "binary tree"],
    "Algorithms": ["time complexity", "worst-case", "recurrence", "sorting", "greedy", "dynamic programming", "o(", "θ(", "ω("],
    "DBMS": ["relation", "sql", "select", "primary key", "schema", "database", "tuple", "index file"],
    "Operating Systems": ["thread", "process", "scheduling", "semaphore", "page fault", "context switch", "deadlock", "page table"],
    "Computer Networks": ["tcp", "dns", "http", "router", "subnet", "routing", "rtt", "ospf", "ip address"],
    "Theory of Computation": ["dfa", "nfa", "regular expression", "pushdown automaton", "context-free", "turing", "finite-state"],
    "Compiler Design": ["lexical", "parser", "grammar", "front-end", "back-end", "syntax directed", "activation tree"],
    "Digital Logic": ["multiplexer", "flip-flop", "boolean", "logic gate", "cache", "adder"],
    "Computer Organization": ["pipeline", "cache", "memory", "instruction", "assembly", "addressing", "ieee-754"],
    "Discrete Mathematics": ["graph", "eigenvalue", "permutation", "set", "function", "group", "probability", "bfs"],
    "Programming & C": ["printf", "int main", "struct", "pointer", "recursion", "c program"],
    "General Aptitude": [],  # fallback bucket, matched by subject code GA
}

# Topic taxonomy for GATE DA (Data Science and AI), built directly from
# the official DA 2024 syllabus areas and confirmed against real fetched
# paper content: Probability/Stats, Linear Algebra, Calculus/Optimization,
# Programming, Data Structures & Algorithms, DBMS, Machine Learning, and AI.
TOPIC_KEYWORDS_DA = {
    "Probability & Statistics": ["probability", "random variable", "distribution", "variance", "expectation",
                                  "covariance", "mean", "standard deviation", "bayes", "poisson", "normal",
                                  "z-score", "independent", "conditional"],
    "Linear Algebra": ["matrix", "eigenvalue", "eigenvector", "determinant", "vector", "subspace",
                        "null space", "rank", "singular value", "trace"],
    "Calculus & Optimization": ["derivative", "limit", "differentiable", "local minimum", "local maximum",
                                 "continuous", "integral", "gradient", "convex"],
    "Programming": ["python", "def ", "for i", "code", "function(", "recursion", "pseudocode", "int func"],
    "Data Structures & Algorithms": ["stack", "queue", "sort", "binary search", "hash", "tree", "graph",
                                      "dfs", "bfs", "complexity", "quicksort", "linked list", "swap"],
    "DBMS & Data Warehousing": ["relation", "sql", "select", "schema", "functional dependenc", "foreign key",
                                 "index", "database", "b+ tree", "normalization", "relational algebra"],
    "Machine Learning": ["classifier", "regression", "svm", "k-means", "clustering", "naive bayes",
                          "neural network", "overfitting", "cross validation", "decision tree", "knn",
                          "principal component", "roc", "support vector", "information gain"],
    "Artificial Intelligence": ["search tree", "a* search", "admissible heuristic", "alpha-beta", "minimax",
                                 "adversarial", "first order logic", "tautology", "bayesian network",
                                 "iddfs", "heuristic"],
    "General Aptitude": [],
}


@dataclass
class GateQuestion:
    subject: str
    year: int
    q_no: int
    marks: int
    q_type: str          # MCQ, MSQ, NAT
    stem: str
    options: dict[str, str] = field(default_factory=dict)
    official_answer: str = ""
    topics: list[str] = field(default_factory=list)

    def to_chunk_text(self) -> str:
        """Flattened text used for embedding / BM25 indexing."""
        parts = [self.stem]
        for letter, text in self.options.items():
            parts.append(f"({letter}) {text}")
        return "\n".join(parts)


def _strip_noise_lines(text: str) -> str:
    lines = text.split("\n")
    kept = []
    for line in lines:
        stripped = line.strip()
        if any(r.match(stripped) for r in NOISE_LINE_RES):
            continue
        kept.append(line)
    return "\n".join(kept)


def _marks_lookup(text: str) -> dict[int, int]:
    """Map each question number to its marks value based on banner lines
    like 'Q.1 - Q.5 Carry ONE mark Each'."""
    lookup: dict[int, int] = {}
    for m in MARKS_BANNER_RE.finditer(text):
        start, end, word = int(m.group(1)), int(m.group(2)), m.group(3).upper()
        marks = 1 if word == "ONE" else 2
        for q in range(start, end + 1):
            lookup[q] = marks
    return lookup


def parse_paper(raw_text: str, subject: str, year: int) -> list[GateQuestion]:
    text = _strip_noise_lines(raw_text)
    marks_lookup = _marks_lookup(text)
    # Remove banner lines entirely so they can't be mistaken for a
    # question start (they begin with "Q.<n>" too, e.g. "Q.6 - Q.10
    # Carry TWO marks Each").
    text = MARKS_BANNER_RE.sub("", text)

    starts = list(Q_START_RE.finditer(text))
    by_q_no: dict[int, GateQuestion] = {}

    for i, m in enumerate(starts):
        q_no = int(m.group(1))
        block_start = m.end()
        block_end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
        block = text[block_start:block_end].strip()

        if len(block) < 15:
            continue  # too short to be a real question -> stray/duplicate marker

        options = {letter: opt.strip() for letter, opt in OPTION_RE.findall(block)}
        if options:
            stem = block[:block.find("(A)")].strip() if "(A)" in block else block.strip()
            q_type = "MSQ" if _looks_like_msq(stem) else "MCQ"
        else:
            stem = block.strip()
            q_type = "NAT"

        marks = marks_lookup.get(q_no, 1)
        candidate = GateQuestion(
            subject=subject, year=year, q_no=q_no, marks=marks,
            q_type=q_type, stem=stem, options=options,
        )

        # A question number can appear more than once due to PDF
        # page-break artifacts (confirmed in the real fetched paper).
        # Keep whichever candidate is more complete: has options, or
        # has the longer stem.
        existing = by_q_no.get(q_no)
        if existing is None or _is_more_complete(candidate, existing):
            by_q_no[q_no] = candidate

    return [by_q_no[k] for k in sorted(by_q_no)]


def _is_more_complete(a: GateQuestion, b: GateQuestion) -> bool:
    if bool(a.options) != bool(b.options):
        return bool(a.options)
    return len(a.stem) > len(b.stem)


def _looks_like_msq(stem: str) -> bool:
    return bool(re.search(r"one or more|which of the following.+is/are", stem, re.IGNORECASE))


def attach_answer_keys(questions: list[GateQuestion], answer_key_csv: Path) -> None:
    answers: dict[int, str] = {}
    types: dict[int, str] = {}
    with open(answer_key_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            q_no = int(row["q_no"])
            answers[q_no] = row["key"]
            if row.get("type"):
                types[q_no] = row["type"]
    for q in questions:
        q.official_answer = answers.get(q.q_no, "")
        # Answer key type is authoritative -- overrides the text-based
        # heuristic guess, since phrasing like "one or more" doesn't
        # always mean the official type is MSQ.
        if q.q_no in types:
            q.q_type = types[q.q_no]


SUBJECT_TAXONOMIES = {
    "CS": TOPIC_KEYWORDS_CS,
    "DA": TOPIC_KEYWORDS_DA,
}


def tag_topics(questions: list[GateQuestion], keyword_map: dict[str, list[str]] | None = None) -> None:
    """Word-boundary matching, not plain substring: naive `kw in text`
    checks caused real false positives during testing -- e.g. the
    keyword 'mean' matched inside 'meaning', mistagging a vocabulary
    question as Probability & Statistics. Keywords are compiled to
    regex with \\b boundaries where safe (alphanumeric on both ends);
    hyphens/spaces inside a keyword are treated as interchangeable
    (e.g. 'decision tree' also matches 'decision-tree', which appears
    in real GATE phrasing). Keywords ending in symbols like 'o(' fall
    back to plain substring matching since \\b doesn't apply after a
    non-word character.
    """
    for q in questions:
        active_map = keyword_map or SUBJECT_TAXONOMIES.get(q.subject, TOPIC_KEYWORDS_CS)
        text_lower = q.stem.lower()
        matched = []
        for topic, keywords in active_map.items():
            for kw in keywords:
                flexible = re.escape(kw).replace(r"\ ", "[ -]")
                leading = r"\b" if kw[0].isalnum() else ""
                trailing = r"\b" if kw[-1].isalnum() else ""
                pattern = leading + flexible + trailing
                if re.search(pattern, text_lower):
                    matched.append(topic)
                    break
        q.topics = matched or ["Uncategorized"]


def parse_full_paper(raw_text_path: Path, answer_key_csv: Path, subject: str, year: int) -> list[GateQuestion]:
    raw_text = raw_text_path.read_text(encoding="utf-8")
    questions = parse_paper(raw_text, subject=subject, year=year)
    attach_answer_keys(questions, answer_key_csv)
    tag_topics(questions)
    return questions
