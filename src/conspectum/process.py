import asyncio
from dataclasses import dataclass
import io
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import typing
import unicodedata

import openai

from .logger import Logger
from .summary import create_chat_completion
from .summary import detect_language_from_text
from .summary import LANGUAGE_NAMES
from .summary import make_summary_from_transcript
from .summary import map_transcription_language_to_output_language
from .summary import normalize_latex_text
from .summary import postprocess_summary
from .summary import transcribe_audio_with_metadata

# Поддержка macOS, если pdflatex лежит в стандартной папке TeX
DETAIL_LEVEL_GUIDANCE = {
    "brief": (
        "Produce a genuinely brief, high-density conspectus. Keep only the core definitions, named results, "
        "essential formulas, and final takeaways. Skip secondary examples, routine derivations, long proofs, "
        "historical context, and repeated explanations."
    ),
    "standard": (
        "Produce full study notes with the main argument, definitions, formulas, examples, and key implications."
    ),
    "detailed": (
        "Produce deep lecture notes with careful explanations, more subsections, definitions, derivations, examples, "
        "and explicit links between concepts."
    ),
}

DEFAULT_CHUNK_TARGET_CHARS = {
    "brief": 4800,
    "standard": 3000,
    "detailed": 2400,
}
MIN_CHUNK_TARGET_CHARS = 1200
MAX_CHUNK_TARGET_CHARS = 7000
DEFAULT_CHUNK_MAX_TOKENS = {
    "brief": 850,
    "standard": 2800,
    "detailed": 3800,
}
MIN_CHUNK_MAX_TOKENS = 700
MAX_CHUNK_MAX_TOKENS = 6000
MAX_PREVIOUS_CHUNK_CONTEXT_CHARS = 900
MAX_ADJACENT_TRANSCRIPT_CONTEXT_CHARS = 700
DEFAULT_CHUNK_PROCESS_CONCURRENCY = 4
DEFAULT_BRIEF_CHUNK_PROCESS_CONCURRENCY = 2
MAX_CHUNK_PROCESS_CONCURRENCY = 8
DEFAULT_CHUNK_AI_TIMEOUT_SECONDS = 90.0
MIN_CHUNK_AI_TIMEOUT_SECONDS = 30.0
MAX_CHUNK_AI_TIMEOUT_SECONDS = 300.0
DEFAULT_CHUNK_AI_MODEL_ATTEMPTS = 3
MAX_CHUNK_AI_MODEL_ATTEMPTS = 5
CHUNK_PROGRESS_HEARTBEAT_SECONDS = 30
ENABLE_LLM_POSTPROCESS_VALUES = {"1", "true", "yes", "on"}
BOX_ENVIRONMENTS = (
    "thmbox",
    "defbox",
    "lembox",
    "propbox",
    "corbox",
    "exbox",
    "rembox",
)
TRANSCRIPT_ARTIFACT_PATTERNS = (
    r"\bПродолжение следует\.{0,3}",
    r"\bСубтитры делал[^\n.]*\.?",
    r"\bSubtitles by[^\n.]*\.?",
)
REPEATED_ELLIPSIS_PATTERN = re.compile(r"(?:\s*\.\.\.\s*){2,}")

PROTECTED_AMPERSAND_ENVIRONMENTS = {
    "tabular",
    "tabular*",
    "array",
    "align",
    "align*",
    "aligned",
    "eqnarray",
    "eqnarray*",
    "split",
    "matrix",
    "pmatrix",
    "bmatrix",
    "vmatrix",
    "Vmatrix",
    "smallmatrix",
    "cases",
}

PROOF_ENVIRONMENT_FALLBACK = r"""\makeatletter
\@ifundefined{proof}{%
  \newenvironment{proof}[1][Proof]{%
    \par\noindent\textit{#1.}\ %
  }{%
    \hfill\textit{QED}\par
  }%
}{}
\makeatother"""
QED_COMMAND_FALLBACK = r"\providecommand{\qed}{\hfill\rule{0.5em}{0.5em}}"

MATH_ENVIRONMENTS = (
    "equation",
    "equation*",
    "align",
    "align*",
    "aligned",
    "gather",
    "gather*",
    "multline",
    "multline*",
    "displaymath",
    "array",
    "matrix",
    "pmatrix",
    "bmatrix",
    "vmatrix",
    "Vmatrix",
    "smallmatrix",
    "cases",
    "split",
)

DISPLAY_MATH_ENVIRONMENTS = (
    "equation",
    "equation*",
    "align",
    "align*",
    "gather",
    "gather*",
    "multline",
    "multline*",
    "displaymath",
)

MATH_BLOCK_BOUNDARY_ENVIRONMENTS = (
    "thmbox",
    "defbox",
    "lembox",
    "propbox",
    "corbox",
    "exbox",
    "rembox",
    "document",
    "itemize",
    "enumerate",
    "quote",
    "center",
    "customtable",
    "table",
    "tabular",
)

MATH_BLOCK_BOUNDARY_PATTERN = re.compile(
    r"(\\end\{(?:"
    + "|".join(re.escape(environment) for environment in MATH_BLOCK_BOUNDARY_ENVIRONMENTS)
    + r")\}|\\(?:section|subsection|subsubsection)\*?\{)"
)
DISPLAY_MATH_TOKEN_PATTERN = re.compile(
    r"(?<!\\)\\\[|(?<!\\)\\\]|\$\$|\\begin\{("
    + "|".join(re.escape(environment) for environment in DISPLAY_MATH_ENVIRONMENTS)
    + r")\}|\\end\{("
    + "|".join(re.escape(environment) for environment in DISPLAY_MATH_ENVIRONMENTS)
    + r")\}"
)
LATEX_COMMENT_PATTERN = re.compile(r"(?<!\\)%.*$")
LATEX_MISSING_CHARACTER_PATTERN = re.compile(r"Missing character:\s+There is no\s+.")
UNGUARDED_LMODERN_IFEXISTS_PATTERN = re.compile(
    r"\\IfFileExists\{lmodern\.sty\}\{\s*\\usepackage\{lmodern\}\s*\}\{\s*\}"
)
UNGUARDED_LMODERN_USEPACKAGE_PATTERN = re.compile(r"\\usepackage\{lmodern\}")
GUARDED_LMODERN_PATTERN = re.compile(
    r"\\ifPDFTeX\s*"
    r"\\IfFileExists\{lmodern\.sty\}\{\s*\\usepackage\{lmodern\}\s*\}\{\s*\}\s*"
    r"\\fi\s*"
)
BOX_ENVIRONMENT_PATTERN = "|".join(re.escape(environment) for environment in BOX_ENVIRONMENTS)
BOX_BEGIN_BRACED_TITLE_PATTERN = re.compile(
    rf"\\begin\{{(?P<environment>{BOX_ENVIRONMENT_PATTERN})\}}\{{(?P<title>[^{{}}\n]+)\}}"
)
BOX_COMMAND_BRACED_BODY_START_PATTERN = re.compile(
    rf"^(?P<indent>[^\S\n]*)\\(?P<environment>{BOX_ENVIRONMENT_PATTERN})"
    rf"\{{(?P<title>[^{{}}\n]+)\}}[^\S\n]*\{{[^\S\n]*$"
)
BOX_COMMAND_SINGLE_BRACED_BODY_START_PATTERN = re.compile(
    rf"^(?P<indent>[^\S\n]*)\\(?P<environment>{BOX_ENVIRONMENT_PATTERN})[^\S\n]*\{{[^\S\n]*$"
)
BOX_COMMAND_PREFIX_PATTERN = re.compile(rf"^(?P<indent>[^\S\n]*)\\(?P<environment>{BOX_ENVIRONMENT_PATTERN})")
BOX_COMMAND_INLINE_BODY_PATTERN = re.compile(
    rf"(?m)^(?P<indent>[^\S\n]*)\\(?P<environment>{BOX_ENVIRONMENT_PATTERN})"
    rf"\{{(?P<title>[^{{}}\n]+)\}}[^\S\n]+(?P<body>(?!\{{).+?)[^\S\n]*$"
)
BOX_COMMAND_SHORTHAND_PATTERN = re.compile(
    rf"(?m)^(?P<indent>[^\S\n]*)\\(?P<environment>{BOX_ENVIRONMENT_PATTERN})"
    rf"\{{(?P<title>[^{{}}\n]+)\}}[^\S\n]*$"
)
INLINE_BOX_COMMAND_PATTERN = re.compile(
    rf"(?<!\\)\\(?P<environment>{BOX_ENVIRONMENT_PATTERN})"
    rf"\{{(?P<body>[^{{}}\n]+)\}}"
)
BOX_ENVIRONMENT_TOKEN_PATTERN = re.compile(rf"\\(?P<kind>begin|end)\{{(?P<environment>{BOX_ENVIRONMENT_PATTERN})\}}")
BOX_BEGIN_OPTIONAL_TITLE_PREFIX_PATTERN = re.compile(rf"\\begin\{{(?P<environment>{BOX_ENVIRONMENT_PATTERN})\}}\[")
BOX_TITLE_SIMPLE_TEXT_COMMAND_PATTERN = re.compile(r"\\text\{(?P<text>[^{}\n]*)\}")
BOX_TITLE_SCRIPT_EXPRESSION_PATTERN = re.compile(r"(?<![\\A-Za-z0-9])(?P<expression>[A-Za-z](?:[_^]\{[^{}\n]+\})+)")
BOX_TITLE_MATH_COMMAND_PATTERN = re.compile(
    r"(?P<expression>\\(?:mathbb|mathbf|mathcal|mathrm|mathsf|mathtt)"
    r"\{[^{}\n]+\}(?:[_^]\{[^{}\n]+\})*)"
)
BOX_ENVIRONMENT_BOUNDARY_PATTERN = re.compile(
    rf"\\(?:section|subsection|subsubsection)\*?\{{|\\end\{{document\}}|\\begin\{{(?:{BOX_ENVIRONMENT_PATTERN})\}}"
)
LIST_ENVIRONMENTS = ("enumerate", "itemize", "description")
LIST_ENVIRONMENT_PATTERN = "|".join(re.escape(environment) for environment in LIST_ENVIRONMENTS)
LIST_ENVIRONMENT_TOKEN_PATTERN = re.compile(rf"\\(?P<kind>begin|end)\{{(?P<environment>{LIST_ENVIRONMENT_PATTERN})\}}")
ORPHAN_LIST_ITEM_PATTERN = re.compile(r"\\item\b")
LIST_ENVIRONMENT_BOUNDARY_PATTERN = re.compile(
    rf"\\(?:section|subsection|subsubsection)\*?\{{|"
    rf"\\(?:begin|end)\{{(?:{BOX_ENVIRONMENT_PATTERN})\}}|\\end\{{document\}}"
)
REPAIRABLE_LATEX_ENVIRONMENTS = tuple(
    dict.fromkeys(
        BOX_ENVIRONMENTS
        + MATH_ENVIRONMENTS
        + tuple(PROTECTED_AMPERSAND_ENVIRONMENTS)
        + (
            "abstract",
            "center",
            "customtable",
            "description",
            "document",
            "enumerate",
            "itemize",
            "proof",
            "quote",
            "table",
        )
    )
)
REPAIRABLE_LATEX_ENVIRONMENT_PATTERN = "|".join(re.escape(environment) for environment in REPAIRABLE_LATEX_ENVIRONMENTS)
ESCAPED_LATEX_ENVIRONMENT_COMMAND_PATTERN = re.compile(
    rf"(?m)^(?P<indent>[^\S\n]*)\\\\(?P<kind>begin|end)\{{(?P<environment>{REPAIRABLE_LATEX_ENVIRONMENT_PATTERN})\}}"
)
TCOLORBOX_TITLE_PLACEHOLDER_PATTERN = re.compile(r"(title\s*=\s*)#1(?=\s*(?:,|\]))")
ESCAPED_INLINE_MATH_DELIMITER_PATTERN = re.compile(r"\\\\\((?P<body>[^\n]{1,500}?)\\\\\)")
INCOMPLETE_INLINE_MATH_COMMAND_LINE_PATTERN = re.compile(r"^[^\S\n]*\\\([^\S\n]*\\[A-Za-z@]{1,32}\*?[^\S\n]*$")

UNICODE_LATEX_REPLACEMENTS = {
    "\u00a0": " ",
    "\u00ab": '"',
    "\u00bb": '"',
    "\u2013": "--",
    "\u2014": "---",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2026": "...",
    "\u2116": "No. ",
    "\u2212": "-",
    "\u2260": r"\ensuremath{\neq}",
    "\u2264": r"\ensuremath{\leq}",
    "\u2265": r"\ensuremath{\geq}",
    "\u2190": r"\ensuremath{\leftarrow}",
    "\u2192": r"\ensuremath{\to}",
    "\u221e": r"\ensuremath{\infty}",
    "\u2208": r"\ensuremath{\in}",
    "\u00b1": r"\ensuremath{\pm}",
    "\u00d7": r"\ensuremath{\times}",
    "\u00f7": r"\ensuremath{\div}",
    "\u00b0": r"\ensuremath{^\circ}",
    "\u2248": r"\ensuremath{\approx}",
    "\u2261": r"\ensuremath{\equiv}",
    "\u2194": r"\ensuremath{\leftrightarrow}",
    "\u21d2": r"\ensuremath{\Rightarrow}",
    "\u21d4": r"\ensuremath{\Leftrightarrow}",
    "\u2202": r"\ensuremath{\partial}",
    "\u2207": r"\ensuremath{\nabla}",
    "\u2211": r"\ensuremath{\sum}",
    "\u220f": r"\ensuremath{\prod}",
    "\u222b": r"\ensuremath{\int}",
    "\u221a": r"\ensuremath{\sqrt{}}",
    "\u226a": r"\ensuremath{\ll}",
    "\u226b": r"\ensuremath{\gg}",
    "\u2282": r"\ensuremath{\subset}",
    "\u2286": r"\ensuremath{\subseteq}",
    "\u2209": r"\ensuremath{\notin}",
}

CYRILLIC_TO_ASCII = {
    "\u0410": "A",
    "\u0411": "B",
    "\u0412": "V",
    "\u0413": "G",
    "\u0414": "D",
    "\u0415": "E",
    "\u0401": "E",
    "\u0416": "Zh",
    "\u0417": "Z",
    "\u0418": "I",
    "\u0419": "I",
    "\u041a": "K",
    "\u041b": "L",
    "\u041c": "M",
    "\u041d": "N",
    "\u041e": "O",
    "\u041f": "P",
    "\u0420": "R",
    "\u0421": "S",
    "\u0422": "T",
    "\u0423": "U",
    "\u0424": "F",
    "\u0425": "Kh",
    "\u0426": "Ts",
    "\u0427": "Ch",
    "\u0428": "Sh",
    "\u0429": "Shch",
    "\u042a": "",
    "\u042b": "Y",
    "\u042c": "",
    "\u042d": "E",
    "\u042e": "Yu",
    "\u042f": "Ya",
    "\u0430": "a",
    "\u0431": "b",
    "\u0432": "v",
    "\u0433": "g",
    "\u0434": "d",
    "\u0435": "e",
    "\u0451": "e",
    "\u0436": "zh",
    "\u0437": "z",
    "\u0438": "i",
    "\u0439": "i",
    "\u043a": "k",
    "\u043b": "l",
    "\u043c": "m",
    "\u043d": "n",
    "\u043e": "o",
    "\u043f": "p",
    "\u0440": "r",
    "\u0441": "s",
    "\u0442": "t",
    "\u0443": "u",
    "\u0444": "f",
    "\u0445": "kh",
    "\u0446": "ts",
    "\u0447": "ch",
    "\u0448": "sh",
    "\u0449": "shch",
    "\u044a": "",
    "\u044b": "y",
    "\u044c": "",
    "\u044d": "e",
    "\u044e": "yu",
    "\u044f": "ya",
}

UNICODE_GREEK_MAP = {
    "Α": "A",
    "Β": "B",
    "Γ": r"\Gamma",
    "Δ": r"\Delta",
    "Ε": "E",
    "Ζ": "Z",
    "Η": "H",
    "Θ": r"\Theta",
    "Ι": "I",
    "Κ": "K",
    "Λ": r"\Lambda",
    "Μ": "M",
    "Ν": "N",
    "Ξ": r"\Xi",
    "Ο": "O",
    "Π": r"\Pi",
    "Ρ": "P",
    "Σ": r"\Sigma",
    "Τ": "T",
    "Υ": r"\Upsilon",
    "Φ": r"\Phi",
    "Χ": "X",
    "Ψ": r"\Psi",
    "Ω": r"\Omega",
    "α": r"\alpha",
    "β": r"\beta",
    "γ": r"\gamma",
    "δ": r"\delta",
    "ε": r"\epsilon",
    "ζ": r"\zeta",
    "η": r"\eta",
    "θ": r"\theta",
    "ι": r"\iota",
    "κ": r"\kappa",
    "λ": r"\lambda",
    "μ": r"\mu",
    "ν": r"\nu",
    "ξ": r"\xi",
    "ο": "o",
    "π": r"\pi",
    "ρ": r"\rho",
    "ς": r"\varsigma",
    "σ": r"\sigma",
    "τ": r"\tau",
    "υ": r"\upsilon",
    "φ": r"\phi",
    "χ": r"\chi",
    "ψ": r"\psi",
    "ω": r"\omega",
    "ϑ": r"\vartheta",
    "ϕ": r"\varphi",
    "ϖ": r"\varpi",
    "ϵ": r"\varepsilon",
    "ϱ": r"\varrho",
}

UNICODE_MATH_TEXT_MAP = {
    "ℝ": r"\mathbb{R}",
    "ℂ": r"\mathbb{C}",
    "ℕ": r"\mathbb{N}",
    "ℤ": r"\mathbb{Z}",
    "ℚ": r"\mathbb{Q}",
    "ℓ": r"\ell",
    "∅": r"\varnothing",
}

UNICODE_MATH_OPERATOR_MAP = {
    "−": "-",
    "≤": r"\leq",
    "≥": r"\geq",
    "≠": r"\neq",
    "≈": r"\approx",
    "≡": r"\equiv",
    "±": r"\pm",
    "×": r"\times",
    "÷": r"\div",
    "·": r"\cdot",
    "⋅": r"\cdot",
    "∞": r"\infty",
    "∈": r"\in",
    "∉": r"\notin",
    "⊂": r"\subset",
    "⊆": r"\subseteq",
    "→": r"\to",
    "←": r"\leftarrow",
    "↔": r"\leftrightarrow",
    "⇒": r"\Rightarrow",
    "⇔": r"\Leftrightarrow",
    "∂": r"\partial",
    "∇": r"\nabla",
    "∑": r"\sum",
    "∏": r"\prod",
    "∫": r"\int",
    "°": r"^\circ",
}

UNICODE_SUBSCRIPT_MAP = {
    "₀": "0",
    "₁": "1",
    "₂": "2",
    "₃": "3",
    "₄": "4",
    "₅": "5",
    "₆": "6",
    "₇": "7",
    "₈": "8",
    "₉": "9",
    "₊": "+",
    "₋": "-",
    "₌": "=",
    "₍": "(",
    "₎": ")",
}

UNICODE_SUPERSCRIPT_MAP = {
    "⁰": "0",
    "¹": "1",
    "²": "2",
    "³": "3",
    "⁴": "4",
    "⁵": "5",
    "⁶": "6",
    "⁷": "7",
    "⁸": "8",
    "⁹": "9",
    "⁺": "+",
    "⁻": "-",
    "⁼": "=",
    "⁽": "(",
    "⁾": ")",
    "ⁿ": "n",
}

_LATEX_BASE_COMMANDS: dict[str, list[str]] = {}
_PDF_FALLBACK_FONT_NAME: typing.Optional[str] = None
LANGUAGE_PREAMBLE_START = "% <CONSPECTUM LANGUAGE SETUP START>"
LANGUAGE_PREAMBLE_END = "% <CONSPECTUM LANGUAGE SETUP END>"
HYPERREF_FALLBACKS = {
    "texorpdfstring": r"\providecommand{\texorpdfstring}[2]{#1}",
    "cref": r"\providecommand{\cref}[1]{\ref{#1}}",
    "Cref": r"\providecommand{\Cref}[1]{\ref{#1}}",
    "autoref": r"\providecommand{\autoref}[1]{\ref{#1}}",
}
TEXT_FALLBACKS = {
    "enquote": r"\providecommand{\enquote}[1]{``#1''}",
}
MATH_FALLBACKS = {
    "Beta": r"\providecommand{\Beta}{\mathrm{B}}",
    "bigsqcap": r"\providecommand{\bigsqcap}{\bigwedge}",
    "bigsqcup": r"\providecommand{\bigsqcup}{\bigvee}",
    "jump": r"\providecommand{\jump}[2]{\Delta #1\vert_{#2}}",
}
TEXT_MATH_TOKEN_PATTERN = re.compile(
    r"(?<!\\)(?P<base>[A-Za-z0-9]+|[Α-Ωα-ωϐ-ϖ])(?P<sub>[₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎]+)?(?P<sup>[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ]+)?"
)


@dataclass
class ProcessResult:
    transcript: str
    language: str
    title: str
    abstract: str
    tex: str
    pdf: typing.Optional[bytes]
    pdf_warning: typing.Optional[str] = None


@dataclass
class LatexPreparationResult:
    tex: str
    notes: list[str]


@dataclass
class LatexCompilationError(RuntimeError):
    engine: str
    summary: str
    diagnostics: str

    def __str__(self) -> str:
        return self.summary


if not shutil.which("pdflatex"):
    _tex_bin = "/Library/TeX/texbin"
    if os.path.isdir(_tex_bin):
        os.environ["PATH"] = _tex_bin + os.pathsep + os.environ.get("PATH", "")


LANG_CONFIG = {
    "en": {
        "fontenc": "T1",
        "babel": "english",
        "polyglossia": "english",
        "other_language": "russian",
        "abstract_label": "Abstract",
        "theorem": "Theorem",
        "definition": "Definition",
        "lemma": "Lemma",
        "proposition": "Proposition",
        "corollary": "Corollary",
        "example": "Example",
        "remark": "Remark",
    },
    "ru": {
        "fontenc": "T2A",
        "babel": "russian",
        "polyglossia": "russian",
        "other_language": "english",
        "abstract_label": "\u0410\u043d\u043d\u043e\u0442\u0430\u0446\u0438\u044f",
        "theorem": "Теорема",
        "definition": "Определение",
        "lemma": "Лемма",
        "proposition": "Утверждение",
        "corollary": "Следствие",
        "example": "Пример",
        "remark": "Замечание",
    },
}


def localize_template(tex_template: str, language: str) -> str:
    config = LANG_CONFIG[language]

    replacements = {
        "<FONTENC>": config["fontenc"],
        "<BABEL_LANG>": config["babel"],
        "<POLYGLOSSIA_LANG>": config["polyglossia"],
        "<OTHER_LANG>": config["other_language"],
        "<THEOREM_NAME>": config["theorem"],
        "<DEFINITION_NAME>": config["definition"],
        "<LEMMA_NAME>": config["lemma"],
        "<PROPOSITION_NAME>": config["proposition"],
        "<COROLLARY_NAME>": config["corollary"],
        "<EXAMPLE_NAME>": config["example"],
        "<REMARK_NAME>": config["remark"],
        "<ABSTRACT_LABEL>": config["abstract_label"],
    }

    for placeholder, value in replacements.items():
        tex_template = tex_template.replace(placeholder, value)

    return tex_template


def build_language_setup_block(language: str) -> str:
    config = LANG_CONFIG[language]
    return "\n".join(
        [
            LANGUAGE_PREAMBLE_START,
            r"\usepackage{iftex}",
            r"\ifPDFTeX",
            r"  \usepackage[utf8]{inputenc}",
            rf"  \usepackage[{config['fontenc']}]{{fontenc}}",
            rf"  \usepackage[{config['babel']}]{{babel}}",
            r"\else",
            r"  \usepackage{fontspec}",
            r"  \defaultfontfeatures{Ligatures=TeX,Scale=MatchLowercase}",
            r"  \IfFontExistsTF{Times New Roman}{\setmainfont{Times New Roman}}{%",
            r"    \IfFontExistsTF{Noto Serif}{\setmainfont{Noto Serif}}{%",
            r"      \IfFontExistsTF{DejaVu Serif}{\setmainfont{DejaVu Serif}}{%",
            r"        \IfFontExistsTF{Liberation Serif}{\setmainfont{Liberation Serif}}{%",
            r"          \IfFontExistsTF{Arial}{\setmainfont{Arial}}{%",
            r"            \setmainfont{Latin Modern Roman}%",
            r"          }%",
            r"        }%",
            r"      }%",
            r"    }%",
            r"  }",
            r"  \IfFontExistsTF{Arial}{\setsansfont{Arial}}{%",
            r"    \IfFontExistsTF{Noto Sans}{\setsansfont{Noto Sans}}{%",
            r"      \IfFontExistsTF{DejaVu Sans}{\setsansfont{DejaVu Sans}}{%",
            r"        \setsansfont{Latin Modern Sans}%",
            r"      }%",
            r"    }%",
            r"  }",
            r"  \IfFontExistsTF{Consolas}{\setmonofont{Consolas}}{%",
            r"    \IfFontExistsTF{DejaVu Sans Mono}{\setmonofont{DejaVu Sans Mono}}{%",
            r"      \setmonofont{Latin Modern Mono}%",
            r"    }%",
            r"  }",
            r"\fi",
            LANGUAGE_PREAMBLE_END,
        ]
    )


def ensure_multilingual_latex_preamble(latex_content: str, language: str) -> str:
    language_setup = build_language_setup_block(language)
    marked_setup_pattern = re.compile(
        re.escape(LANGUAGE_PREAMBLE_START) + r".*?" + re.escape(LANGUAGE_PREAMBLE_END),
        flags=re.DOTALL,
    )
    if marked_setup_pattern.search(latex_content):
        return marked_setup_pattern.sub(lambda _match: language_setup, latex_content, count=1)

    preamble_cleanup_patterns = [
        r"\\usepackage\{iftex\}\s*",
        r"\\ifPDFTeX\b.*?\\fi\s*",
        r"\\usepackage\[[^\]]*\]\{inputenc\}\s*",
        r"\\usepackage\[[^\]]*\]\{fontenc\}\s*",
        r"\\usepackage\[[^\]]*\]\{babel\}\s*",
        r"\\usepackage\{fontspec\}\s*",
        r"\\setmainlanguage\{[^{}]+\}\s*",
        r"\\setotherlanguage\{[^{}]+\}\s*",
        r"\\defaultfontfeatures\{[^{}]*\}\s*",
        r"\\setmainfont\{[^{}]+\}\s*",
        r"\\setsansfont\{[^{}]+\}\s*",
        r"\\setmonofont\{[^{}]+\}\s*",
        r"\\newfontfamily\\[A-Za-z@]+\{[^{}]+\}\s*",
    ]

    preamble_ready = latex_content
    for pattern in preamble_cleanup_patterns:
        preamble_ready = re.sub(pattern, "", preamble_ready, flags=re.DOTALL)

    documentclass_match = re.search(
        r"(\\documentclass(?:\[[^\]]*\])?\{[^{}]+\})",
        preamble_ready,
    )
    if not documentclass_match:
        return language_setup + "\n" + preamble_ready

    insert_at = documentclass_match.end()
    return preamble_ready[:insert_at] + "\n\n" + language_setup + "\n" + preamble_ready[insert_at:]


def latex_engine_available(engine: str) -> bool:
    return shutil.which(engine) is not None


def any_latex_engine_available() -> bool:
    return any(latex_engine_available(engine) for engine in ("pdflatex", "xelatex", "lualatex"))


def get_latex_base_command(engine: str = "pdflatex") -> list[str]:
    if engine not in _LATEX_BASE_COMMANDS:
        command = [engine]
        try:
            version = subprocess.run(
                [engine, "--version"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
            )
            if "MiKTeX" in f"{version.stdout}\n{version.stderr}":
                command.append("--disable-installer")
        except Exception:
            pass

        _LATEX_BASE_COMMANDS[engine] = command

    return list(_LATEX_BASE_COMMANDS[engine])


def contains_non_ascii_characters(text: str) -> bool:
    return any(ord(char) > 127 for char in text)


def contains_cyrillic_characters(text: str) -> bool:
    return any("\u0400" <= char <= "\u04ff" for char in text)


def contains_unicode_math_characters(text: str) -> bool:
    math_chars = (
        set(UNICODE_GREEK_MAP)
        | set(UNICODE_MATH_OPERATOR_MAP)
        | set(UNICODE_MATH_TEXT_MAP)
        | set(UNICODE_SUBSCRIPT_MAP)
        | set(UNICODE_SUPERSCRIPT_MAP)
    )
    return any(char in math_chars for char in text)


def get_preferred_latex_engines(latex_content: str, language: str) -> list[str]:
    prefer_unicode = (
        latex_engine_available("xelatex")
        or latex_engine_available("lualatex")
        or language == "ru"
        or contains_cyrillic_characters(latex_content)
        or contains_unicode_math_characters(latex_content)
        or contains_non_ascii_characters(latex_content)
    )
    engines = ["xelatex", "lualatex", "pdflatex"] if prefer_unicode else ["pdflatex"]
    return [engine for engine in engines if latex_engine_available(engine)]


def make_ascii_safe_latex(latex_content: str) -> str:
    ascii_safe = latex_content

    for source, replacement in UNICODE_LATEX_REPLACEMENTS.items():
        ascii_safe = ascii_safe.replace(source, replacement)

    converted: list[str] = []
    for char in ascii_safe:
        if char in CYRILLIC_TO_ASCII:
            converted.append(CYRILLIC_TO_ASCII[char])
        elif ord(char) <= 127:
            converted.append(char)
        else:
            converted.append("?")

    return "".join(converted)


def _consume_latex_command(text: str, start_index: int) -> tuple[str, int]:
    if start_index >= len(text) or text[start_index] != "\\":
        return "", start_index

    if start_index + 1 >= len(text):
        return "\\", start_index + 1

    next_char = text[start_index + 1]
    if next_char.isalpha() or next_char == "@":
        end_index = start_index + 2
        while end_index < len(text) and (text[end_index].isalpha() or text[end_index] == "@"):
            end_index += 1
        if end_index < len(text) and text[end_index] == "*":
            end_index += 1
        return text[start_index:end_index], end_index

    return text[start_index : start_index + 2], start_index + 2


def _normalize_script_sequence(sequence: str, mapping: dict[str, str]) -> str:
    return "".join(mapping.get(char, char) for char in sequence)


def _normalize_math_base(base: str) -> str:
    if len(base) == 1 and base in UNICODE_GREEK_MAP:
        return UNICODE_GREEK_MAP[base]
    if len(base) == 1 and base in UNICODE_MATH_TEXT_MAP:
        return UNICODE_MATH_TEXT_MAP[base]
    return base


def _replace_text_math_tokens(text_segment: str) -> str:
    def replacer(match: re.Match[str]) -> str:
        base = match.group("base")
        sub = match.group("sub") or ""
        sup = match.group("sup") or ""
        is_greek_or_math_symbol = len(base) == 1 and (base in UNICODE_GREEK_MAP or base in UNICODE_MATH_TEXT_MAP)

        if not sub and not sup and not is_greek_or_math_symbol:
            return match.group(0)

        math_token = _normalize_math_base(base)
        if sub:
            math_token += f"_{{{_normalize_script_sequence(sub, UNICODE_SUBSCRIPT_MAP)}}}"
        if sup:
            math_token += f"^{{{_normalize_script_sequence(sup, UNICODE_SUPERSCRIPT_MAP)}}}"
        return rf"\ensuremath{{{math_token}}}"

    return TEXT_MATH_TOKEN_PATTERN.sub(replacer, text_segment)


def normalize_text_unicode_segment(text_segment: str) -> str:
    normalized = _replace_text_math_tokens(text_segment)
    output: list[str] = []
    index = 0

    while index < len(normalized):
        char = normalized[index]

        if char == "\\":
            command, index = _consume_latex_command(normalized, index)
            output.append(command)
            continue

        if char in UNICODE_MATH_TEXT_MAP:
            output.append(rf"\ensuremath{{{UNICODE_MATH_TEXT_MAP[char]}}}")
        elif char in UNICODE_GREEK_MAP:
            output.append(rf"\ensuremath{{{UNICODE_GREEK_MAP[char]}}}")
        elif char in UNICODE_MATH_OPERATOR_MAP:
            output.append(rf"\ensuremath{{{UNICODE_MATH_OPERATOR_MAP[char]}}}")
        elif char in UNICODE_SUBSCRIPT_MAP:
            start = index
            while index < len(normalized) and normalized[index] in UNICODE_SUBSCRIPT_MAP:
                index += 1
            output.append(
                rf"\ensuremath{{_{{{_normalize_script_sequence(normalized[start:index], UNICODE_SUBSCRIPT_MAP)}}}}}"
            )
            continue
        elif char in UNICODE_SUPERSCRIPT_MAP:
            start = index
            while index < len(normalized) and normalized[index] in UNICODE_SUPERSCRIPT_MAP:
                index += 1
            output.append(
                rf"\ensuremath{{^{{{_normalize_script_sequence(normalized[start:index], UNICODE_SUPERSCRIPT_MAP)}}}}}"
            )
            continue
        elif char in UNICODE_LATEX_REPLACEMENTS:
            output.append(UNICODE_LATEX_REPLACEMENTS[char])
        else:
            output.append(char)
        index += 1

    return "".join(output)


def normalize_math_unicode_segment(math_segment: str) -> str:
    output: list[str] = []
    index = 0

    while index < len(math_segment):
        char = math_segment[index]

        if char == "\\":
            command, index = _consume_latex_command(math_segment, index)
            output.append("*" if command == r"\*" else command)
            continue

        if char in UNICODE_MATH_TEXT_MAP:
            output.append(UNICODE_MATH_TEXT_MAP[char])
        elif char in UNICODE_GREEK_MAP:
            output.append(UNICODE_GREEK_MAP[char])
        elif char in UNICODE_MATH_OPERATOR_MAP:
            output.append(UNICODE_MATH_OPERATOR_MAP[char])
        elif char in UNICODE_SUBSCRIPT_MAP:
            start = index
            while index < len(math_segment) and math_segment[index] in UNICODE_SUBSCRIPT_MAP:
                index += 1
            output.append(f"_{{{_normalize_script_sequence(math_segment[start:index], UNICODE_SUBSCRIPT_MAP)}}}")
            continue
        elif char in UNICODE_SUPERSCRIPT_MAP:
            start = index
            while index < len(math_segment) and math_segment[index] in UNICODE_SUPERSCRIPT_MAP:
                index += 1
            output.append(f"^{{{_normalize_script_sequence(math_segment[start:index], UNICODE_SUPERSCRIPT_MAP)}}}")
            continue
        elif char in UNICODE_LATEX_REPLACEMENTS:
            replacement = UNICODE_LATEX_REPLACEMENTS[char]
            if replacement.startswith(r"\ensuremath{") and replacement.endswith("}"):
                replacement = replacement[len(r"\ensuremath{") : -1]
            output.append(replacement)
        else:
            output.append(char)
        index += 1

    return "".join(output)


def split_latex_math_segments(latex_content: str) -> list[tuple[bool, str]]:
    math_patterns = [
        r"\\\(.+?\\\)",
        r"\\\[.+?\\\]",
        r"\$\$.*?\$\$",
        r"(?<!\$)\$(?:\\.|[^$\\])+\$",
    ]
    math_patterns.extend(
        rf"\\begin\{{{re.escape(environment)}\}}.*?\\end\{{{re.escape(environment)}\}}"
        for environment in MATH_ENVIRONMENTS
    )
    math_pattern = re.compile("(" + "|".join(math_patterns) + ")", flags=re.DOTALL)

    segments: list[tuple[bool, str]] = []
    last_index = 0
    for match in math_pattern.finditer(latex_content):
        if match.start() > last_index:
            segments.append((False, latex_content[last_index : match.start()]))
        segments.append((True, match.group(0)))
        last_index = match.end()

    if last_index < len(latex_content):
        segments.append((False, latex_content[last_index:]))

    return segments


def normalize_unicode_latex_document(latex_content: str) -> LatexPreparationResult:
    normalized = unicodedata.normalize("NFC", latex_content)
    notes: list[str] = []
    if normalized != latex_content:
        notes.append("Applied Unicode NFC normalization.")

    rebuilt_segments: list[str] = []
    changed_segments = False
    for is_math, segment in split_latex_math_segments(normalized):
        normalized_segment = (
            normalize_math_unicode_segment(segment) if is_math else normalize_text_unicode_segment(segment)
        )
        if normalized_segment != segment:
            changed_segments = True
        rebuilt_segments.append(normalized_segment)

    rebuilt = "".join(rebuilt_segments)
    if changed_segments:
        notes.append("Normalized raw Unicode math symbols, escaped math operators, Greek letters, and script digits.")

    return LatexPreparationResult(tex=rebuilt, notes=notes)


def remove_lmodern_package(latex_content: str) -> tuple[str, bool]:
    repaired, guarded_replacements = GUARDED_LMODERN_PATTERN.subn("", latex_content)
    repaired, ifexists_replacements = UNGUARDED_LMODERN_IFEXISTS_PATTERN.subn("", repaired)
    repaired, direct_replacements = UNGUARDED_LMODERN_USEPACKAGE_PATTERN.subn("", repaired)
    return repaired, bool(guarded_replacements or ifexists_replacements or direct_replacements)


def repair_tcolorbox_title_placeholder(latex_content: str) -> tuple[str, bool]:
    repaired, replacements = TCOLORBOX_TITLE_PLACEHOLDER_PATTERN.subn(r"\1{#1}", latex_content)
    return repaired, bool(replacements)


def repair_escaped_inline_math_delimiters(latex_content: str) -> tuple[str, bool]:
    def replace_escaped_inline_math(match: re.Match[str]) -> str:
        return r"\(" + match.group("body") + r"\)"

    repaired, replacements = ESCAPED_INLINE_MATH_DELIMITER_PATTERN.subn(
        replace_escaped_inline_math,
        latex_content,
    )
    return repaired, bool(replacements)


def repair_escaped_latex_environment_commands(latex_content: str) -> tuple[str, bool]:
    def replace_escaped_environment(match: re.Match[str]) -> str:
        return f"{match.group('indent')}\\{match.group('kind')}{{{match.group('environment')}}}"

    repaired, replacements = ESCAPED_LATEX_ENVIRONMENT_COMMAND_PATTERN.subn(
        replace_escaped_environment,
        latex_content,
    )
    return repaired, bool(replacements)


def remove_incomplete_inline_math_command_lines(latex_content: str) -> tuple[str, bool]:
    repaired_lines: list[str] = []
    changed = False

    for line in latex_content.splitlines():
        code_line = _strip_latex_comment(line)
        if INCOMPLETE_INLINE_MATH_COMMAND_LINE_PATTERN.match(code_line):
            changed = True
            continue
        repaired_lines.append(line)

    return "\n".join(repaired_lines), changed


def find_latex_optional_argument_end(text: str, start_index: int) -> int | None:
    brace_depth = 0
    bracket_depth = 0
    index = start_index

    while index < len(text):
        if text.startswith(r"\(", index):
            math_end = text.find(r"\)", index + 2)
            if math_end != -1:
                index = math_end + 2
                continue
        if text.startswith(r"\[", index):
            math_end = text.find(r"\]", index + 2)
            if math_end != -1:
                index = math_end + 2
                continue

        char = text[index]
        if char == "\\":
            index += 2
            continue

        if char == "$":
            math_end = index + 1
            while math_end < len(text):
                if text[math_end] == "$" and text[math_end - 1] != "\\":
                    break
                math_end += 1
            if math_end < len(text):
                index = math_end + 1
                continue

        if char == "{":
            brace_depth += 1
        elif char == "}" and brace_depth:
            brace_depth -= 1
        elif brace_depth == 0:
            if char == "[":
                bracket_depth += 1
            elif char == "]":
                if bracket_depth == 0:
                    return index
                bracket_depth -= 1

        index += 1

    return None


def protect_latex_box_optional_titles(latex_content: str) -> tuple[str, bool]:
    def normalize_title_math(title: str) -> tuple[str, bool]:
        normalized_segments: list[str] = []
        changed = False

        for is_math, segment in split_latex_math_segments(title):
            if is_math:
                normalized_segments.append(segment)
                continue

            segment, text_replacements = BOX_TITLE_SIMPLE_TEXT_COMMAND_PATTERN.subn(
                lambda match: match.group("text"),
                segment,
            )
            segment, script_replacements = BOX_TITLE_SCRIPT_EXPRESSION_PATTERN.subn(
                lambda match: rf"\({match.group('expression')}\)",
                segment,
            )
            segment, command_replacements = BOX_TITLE_MATH_COMMAND_PATTERN.subn(
                lambda match: rf"\({match.group('expression')}\)",
                segment,
            )
            changed = changed or bool(text_replacements or script_replacements or command_replacements)
            normalized_segments.append(segment)

        return "".join(normalized_segments), changed

    chunks: list[str] = []
    last_index = 0
    changed = False
    search_index = 0

    while True:
        match = BOX_BEGIN_OPTIONAL_TITLE_PREFIX_PATTERN.search(latex_content, search_index)
        if match is None:
            break

        title_start = match.end()
        title_end = find_latex_optional_argument_end(latex_content, title_start)
        if title_end is None:
            search_index = title_start
            continue

        title = latex_content[title_start:title_end]
        title, title_math_changed = normalize_title_math(title)
        stripped_title = title.strip()
        already_braced = stripped_title.startswith("{") and stripped_title.endswith("}")
        needs_braces = not already_braced and ("[" in title or "]" in title)

        if needs_braces or title_math_changed:
            chunks.append(latex_content[last_index:title_start])
            chunks.append("{" + title + "}" if needs_braces else title)
            last_index = title_end
            changed = True

        search_index = title_end + 1

    if not changed:
        return latex_content, False

    chunks.append(latex_content[last_index:])
    return "".join(chunks), True


def ensure_xelatex_magic_comment(latex_content: str, language: str) -> tuple[str, bool]:
    if language != "ru":
        return latex_content, False

    first_lines = "\n".join(latex_content.splitlines()[:5])
    if "TeX program" in first_lines:
        return latex_content, False

    return "% !TeX program = xelatex\n" + latex_content.lstrip(), True


def normalize_latex_box_syntax(latex_content: str) -> tuple[str, bool]:
    def replace_braced_title(match: re.Match[str]) -> str:
        return rf"\begin{{{match.group('environment')}}}[{match.group('title').strip()}]"

    def read_latex_braced_group(text: str, start_index: int) -> tuple[str, int] | None:
        if start_index >= len(text) or text[start_index] != "{":
            return None

        group_chars: list[str] = []
        depth = 1
        index = start_index + 1
        while index < len(text):
            char = text[index]

            if char == "\\" and index + 1 < len(text) and text[index + 1] in "{}":
                group_chars.append(text[index : index + 2])
                index += 2
                continue

            if char == "{":
                depth += 1
                group_chars.append(char)
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return "".join(group_chars), index + 1
                group_chars.append(char)
            else:
                group_chars.append(char)

            index += 1

        return None

    def normalize_inline_braced_command_box(line: str) -> str | None:
        match = BOX_COMMAND_PREFIX_PATTERN.match(line)
        if not match:
            return None

        index = match.end()
        while index < len(line) and line[index].isspace():
            index += 1

        title_group = read_latex_braced_group(line, index)
        if title_group is None:
            return None
        title, index = title_group

        while index < len(line) and line[index].isspace():
            index += 1

        body_group = read_latex_braced_group(line, index)
        if body_group is None:
            trailing = line[index:].strip()
            if trailing and not trailing.startswith("%"):
                return None

            body_or_title = title.strip()
            if not body_or_title.endswith((".", "!", "?", ":", ";")):
                return None

            indent = match.group("indent")
            environment = match.group("environment")
            return (
                f"{indent}"
                rf"\begin{{{environment}}}"
                "\n"
                f"{indent}{body_or_title}"
                "\n"
                f"{indent}"
                rf"\end{{{environment}}}" + (f" {trailing}" if trailing else "")
            )
        body, index = body_group

        trailing = line[index:].strip()
        if trailing and not trailing.startswith("%"):
            return None

        indent = match.group("indent")
        environment = match.group("environment")
        return (
            f"{indent}"
            rf"\begin{{{environment}}}[{title.strip()}]"
            "\n"
            f"{indent}{body.strip()}"
            "\n"
            f"{indent}"
            rf"\end{{{environment}}}" + (f" {trailing}" if trailing else "")
        )

    def normalize_command_braced_body(content: str) -> tuple[str, bool]:
        repaired_lines: list[str] = []
        command_box_stack: list[str] = []
        changed = False

        for line in content.splitlines():
            inline_braced_command = normalize_inline_braced_command_box(line)
            if inline_braced_command is not None:
                repaired_lines.append(inline_braced_command)
                changed = True
                continue

            match = BOX_COMMAND_SINGLE_BRACED_BODY_START_PATTERN.match(line)
            if match:
                repaired_lines.append(
                    f"{match.group('indent')}"
                    rf"\begin{{{match.group('environment')}}}"
                )
                command_box_stack.append(match.group("environment"))
                changed = True
                continue

            match = BOX_COMMAND_BRACED_BODY_START_PATTERN.match(line)
            if match:
                repaired_lines.append(
                    f"{match.group('indent')}"
                    rf"\begin{{{match.group('environment')}}}[{match.group('title').strip()}]"
                )
                command_box_stack.append(match.group("environment"))
                changed = True
                continue

            if command_box_stack and line.strip() == "}":
                environment = command_box_stack.pop()
                indent = line[: len(line) - len(line.lstrip())]
                repaired_lines.append(f"{indent}" rf"\end{{{environment}}}")
                changed = True
                continue

            repaired_lines.append(line)

        return "\n".join(repaired_lines), changed

    def replace_inline_body(match: re.Match[str]) -> str:
        return (
            f"{match.group('indent')}"
            rf"\begin{{{match.group('environment')}}}[{match.group('title').strip()}]"
            "\n"
            f"{match.group('indent')}{match.group('body').strip()}"
        )

    def replace_command_shorthand(match: re.Match[str]) -> str:
        return (
            f"{match.group('indent')}"
            rf"\begin{{{match.group('environment')}}}[{match.group('title').strip()}]"
        )

    def replace_inline_box_commands(content: str) -> tuple[str, bool]:
        repaired_lines: list[str] = []
        changed = False

        def replace_match(match: re.Match[str]) -> str:
            nonlocal changed
            changed = True
            return rf"\textbf{{{match.group('body').strip()}}}"

        for line in content.splitlines():
            if BOX_COMMAND_PREFIX_PATTERN.match(line):
                repaired_lines.append(line)
                continue
            repaired_lines.append(INLINE_BOX_COMMAND_PATTERN.sub(replace_match, line))

        return "\n".join(repaired_lines), changed

    repaired, braced_replacements = BOX_BEGIN_BRACED_TITLE_PATTERN.subn(replace_braced_title, latex_content)
    repaired, command_braced_body_changed = normalize_command_braced_body(repaired)
    repaired, inline_body_replacements = BOX_COMMAND_INLINE_BODY_PATTERN.subn(replace_inline_body, repaired)
    repaired, command_replacements = BOX_COMMAND_SHORTHAND_PATTERN.subn(replace_command_shorthand, repaired)
    repaired, inline_command_replacements = replace_inline_box_commands(repaired)
    return repaired, bool(
        braced_replacements
        or command_braced_body_changed
        or inline_body_replacements
        or command_replacements
        or inline_command_replacements
    )


def _update_latex_box_stack(line: str, box_stack: list[str]) -> None:
    for match in BOX_ENVIRONMENT_TOKEN_PATTERN.finditer(line):
        kind = match.group("kind")
        environment = match.group("environment")
        if kind == "begin":
            box_stack.append(environment)
        elif environment in box_stack:
            while box_stack:
                open_environment = box_stack.pop()
                if open_environment == environment:
                    break


def remove_unmatched_latex_box_endings(latex_content: str) -> tuple[str, bool]:
    repaired_lines: list[str] = []
    box_stack: list[str] = []
    changed = False

    for line in latex_content.splitlines():
        code_line = _strip_latex_comment(line)
        repaired_line = line
        offset = 0

        for match in BOX_ENVIRONMENT_TOKEN_PATTERN.finditer(code_line):
            kind = match.group("kind")
            environment = match.group("environment")

            if kind == "begin":
                box_stack.append(environment)
                continue

            if environment in box_stack:
                while box_stack:
                    open_environment = box_stack.pop()
                    if open_environment == environment:
                        break
                continue

            start = match.start() + offset
            end = match.end() + offset
            repaired_line = repaired_line[:start] + repaired_line[end:]
            offset -= match.end() - match.start()
            changed = True

        repaired_lines.append(repaired_line)

    return "\n".join(repaired_lines), changed


def close_unclosed_latex_box_environments(latex_content: str) -> tuple[str, bool]:
    repaired_lines: list[str] = []
    box_stack: list[str] = []
    changed = False

    for line in latex_content.splitlines():
        code_line = _strip_latex_comment(line)
        boundary_match = BOX_ENVIRONMENT_BOUNDARY_PATTERN.search(code_line)

        if box_stack and boundary_match is not None:
            stack_at_boundary = box_stack.copy()
            _update_latex_box_stack(code_line[: boundary_match.start()], stack_at_boundary)

            if stack_at_boundary:
                for open_environment in reversed(stack_at_boundary):
                    repaired_lines.append(rf"\end{{{open_environment}}}")
                repaired_lines.append(line)
                box_stack.clear()
                _update_latex_box_stack(code_line[boundary_match.start() :], box_stack)
                changed = True
                continue

        repaired_lines.append(line)
        _update_latex_box_stack(code_line, box_stack)

    if box_stack:
        for open_environment in reversed(box_stack):
            repaired_lines.append(rf"\end{{{open_environment}}}")
        changed = True

    return "\n".join(repaired_lines), changed


def repair_latex_list_environments(latex_content: str) -> tuple[str, bool]:
    repaired_lines: list[str] = []
    list_stack: list[str] = []
    inside_document = False
    changed = False

    for line in latex_content.splitlines():
        code_line = _strip_latex_comment(line)

        if not inside_document:
            repaired_lines.append(line)
            if r"\begin{document}" in code_line:
                inside_document = True
            continue

        boundary_match = LIST_ENVIRONMENT_BOUNDARY_PATTERN.search(code_line)
        token_region = code_line[: boundary_match.start()] if boundary_match else code_line
        repaired_line = line
        offset = 0
        closures_before_line: list[str] = []

        for match in LIST_ENVIRONMENT_TOKEN_PATTERN.finditer(token_region):
            kind = match.group("kind")
            environment = match.group("environment")

            if kind == "begin":
                list_stack.append(environment)
                continue

            if list_stack and list_stack[-1] == environment:
                list_stack.pop()
                continue

            if environment in list_stack:
                while list_stack and list_stack[-1] != environment:
                    closures_before_line.append(rf"\end{{{list_stack.pop()}}}")
                list_stack.pop()
                changed = True
                continue

            start = match.start() + offset
            end = match.end() + offset
            repaired_line = repaired_line[:start] + repaired_line[end:]
            offset -= match.end() - match.start()
            changed = True

        repaired_lines.extend(closures_before_line)

        if ORPHAN_LIST_ITEM_PATTERN.search(token_region) and not list_stack:
            repaired_lines.append(r"\begin{itemize}")
            list_stack.append("itemize")
            changed = True

        if boundary_match and list_stack:
            for open_environment in reversed(list_stack):
                repaired_lines.append(rf"\end{{{open_environment}}}")
            list_stack.clear()
            changed = True

        repaired_lines.append(repaired_line)
        if r"\end{document}" in code_line:
            inside_document = False

    if list_stack:
        for open_environment in reversed(list_stack):
            repaired_lines.append(rf"\end{{{open_environment}}}")
        changed = True

    return "\n".join(repaired_lines), changed


def _update_open_display_math_block(line: str, open_block: str | None) -> str | None:
    for match in DISPLAY_MATH_TOKEN_PATTERN.finditer(line):
        token = match.group(0)
        begin_environment = match.group(1)
        end_environment = match.group(2)

        if token == r"\[":
            if open_block is None:
                open_block = r"\]"
        elif token == r"\]":
            if open_block == r"\]":
                open_block = None
        elif token == "$$":
            open_block = None if open_block == "$$" else "$$"
        elif begin_environment:
            if open_block is None:
                open_block = rf"\end{{{begin_environment}}}"
        elif end_environment and open_block == rf"\end{{{end_environment}}}":
            open_block = None

    return open_block


def _strip_latex_comment(line: str) -> str:
    return LATEX_COMMENT_PATTERN.sub("", line)


def close_unclosed_latex_display_math(latex_content: str) -> tuple[str, bool]:
    repaired_lines: list[str] = []
    open_block: str | None = None
    changed = False

    for line in latex_content.splitlines():
        code_line = _strip_latex_comment(line)
        boundary_match = MATH_BLOCK_BOUNDARY_PATTERN.search(code_line)
        if open_block is not None and boundary_match is not None:
            before_boundary = code_line[: boundary_match.start()]
            open_after_prefix = _update_open_display_math_block(before_boundary, open_block)

            if open_after_prefix is not None:
                boundary_start = boundary_match.start()
                prefix = line[:boundary_start].rstrip()
                if prefix:
                    repaired_lines.append(prefix)
                repaired_lines.append(open_after_prefix)
                line = line[boundary_start:]
                code_line = _strip_latex_comment(line)
                open_block = None
                changed = True

        repaired_lines.append(line)
        open_block = _update_open_display_math_block(code_line, open_block)

    if open_block is not None:
        repaired_lines.append(open_block)
        changed = True

    return "\n".join(repaired_lines), changed


def remove_unmatched_latex_display_math_endings(latex_content: str) -> tuple[str, bool]:
    repaired_lines: list[str] = []
    open_block: str | None = None
    changed = False

    for line in latex_content.splitlines():
        code_line = _strip_latex_comment(line)
        stripped = code_line.strip()

        if stripped == r"\]" and open_block != r"\]":
            changed = True
            continue
        if stripped == r"\end{displaymath}" and open_block != r"\end{displaymath}":
            changed = True
            continue

        repaired_lines.append(line)
        open_block = _update_open_display_math_block(code_line, open_block)

    return "\n".join(repaired_lines), changed


def simplify_latex_math(math_content: str) -> str:
    simplified = math_content

    while True:
        previous = simplified
        simplified = re.sub(
            r"\\(?:text|textbf|textit|mathrm|mathbf|mathit|operatorname|emph)\{([^{}]*)\}",
            r"\1",
            simplified,
        )
        simplified = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1) / (\2)", simplified)
        simplified = re.sub(
            r"\\sqrt(?:\[[^\]]*\])?\{([^{}]*)\}",
            r"sqrt(\1)",
            simplified,
        )
        if simplified == previous:
            break

    replacements = {
        r"\cdot": "*",
        r"\times": "x",
        r"\to": "->",
        r"\rightarrow": "->",
        r"\leftarrow": "<-",
        r"\Rightarrow": "=>",
        r"\Leftarrow": "<=",
        r"\neq": "!=",
        r"\leq": "<=",
        r"\geq": ">=",
        r"\approx": "~",
        r"\pm": "+/-",
        r"\mp": "-/+",
        r"\infty": "infinity",
        r"\sum": "sum",
        r"\prod": "prod",
        r"\int": "int",
        r"\log": "log",
        r"\ln": "ln",
        r"\sin": "sin",
        r"\cos": "cos",
        r"\tan": "tan",
        r"\ldots": "...",
        r"\dots": "...",
        r"\quad": " ",
        r"\qquad": " ",
        r"\left": "",
        r"\right": "",
        r"\,": " ",
        r"\;": " ",
        r"\:": " ",
        r"\!": "",
    }
    for source, replacement in replacements.items():
        simplified = simplified.replace(source, replacement)

    simplified = re.sub(r"\\(?:label|ref|cref|eqref|cite)\{[^{}]*\}", "", simplified)
    simplified = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", "", simplified)
    simplified = simplified.replace(r"\{", "{")
    simplified = simplified.replace(r"\}", "}")
    simplified = re.sub(r"[{}]", "", simplified)
    simplified = re.sub(r"\s+", " ", simplified)
    return simplified.strip()


def latex_to_readable_text(latex_content: str) -> str:
    document_match = re.search(
        r"\\begin\{document\}(.*?)\\end\{document\}",
        latex_content,
        flags=re.DOTALL,
    )
    text = document_match.group(1) if document_match else latex_content

    text = re.sub(r"(?<!\\)%.*", "", text)
    text = re.sub(
        r"\\begin\{center\}(.*?)\\end\{center\}",
        lambda match: "\n" + match.group(1) + "\n",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"(?<!\\)\\\[(.*?)\\\]",
        lambda match: f"\n{simplify_latex_math(match.group(1))}\n",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"\\\((.*?)\\\)",
        lambda match: simplify_latex_math(match.group(1)),
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"\$(.+?)\$",
        lambda match: simplify_latex_math(match.group(1)),
        text,
        flags=re.DOTALL,
    )
    text = re.sub(r"\\\\(?:\[[^\]]*\])?", "\n", text)
    text = text.replace(r"\par", "\n")
    text = re.sub(r"\\item\b", "\n- ", text)

    heading_prefixes = {
        "section": "## ",
        "subsection": "### ",
        "subsubsection": "#### ",
    }
    for command, prefix in heading_prefixes.items():
        text = re.sub(
            rf"\\{command}\*?\{{([^{{}}]*)\}}",
            rf"\n\n{prefix}\1\n",
            text,
        )

    for environment in (
        "equation",
        "equation*",
        "align",
        "align*",
        "gather",
        "gather*",
        "multline",
        "multline*",
        "displaymath",
    ):
        text = re.sub(rf"\\begin\{{{environment}\}}", "\n", text)
        text = re.sub(rf"\\end\{{{environment}\}}", "\n", text)

    for environment in ("enumerate", "itemize", "quote", "center", "customtable", "table", "tabular"):
        text = re.sub(rf"\\begin\{{{environment}\}}(\[[^\]]*\])?(\{{[^{{}}]*\}})*", "\n", text)
        text = re.sub(rf"\\end\{{{environment}\}}", "\n", text)

    box_defaults = {
        "thmbox": "Теорема",
        "defbox": "Определение",
        "lembox": "Лемма",
        "propbox": "Утверждение",
        "corbox": "Следствие",
        "exbox": "Пример",
        "rembox": "Замечание",
    }
    for environment, fallback_title in box_defaults.items():
        text = re.sub(
            rf"\\begin\{{{environment}\}}(?:\[([^\]]*)\])?",
            lambda match: f"\n\n> {match.group(1) or fallback_title}\n",
            text,
        )
        text = re.sub(rf"\\end\{{{environment}\}}", "\n", text)

    text = re.sub(r"\\href\{[^{}]*\}\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\(?:label|ref|cref|eqref|cite)\{[^{}]*\}", "", text)

    while True:
        previous = text
        text = re.sub(
            r"\\(?:text|textbf|textit|texttt|textsf|textrm|emph|underline|mathrm|mathbf|mathit|operatorname|boxed|url)\{([^{}]*)\}",
            r"\1",
            text,
        )
        if text == previous:
            break

    replacements = {
        r"\&": "&",
        r"\%": "%",
        r"\_": "_",
        r"\#": "#",
        r"\$": "$",
        r"\{": "{",
        r"\}": "}",
    }
    for source, replacement in replacements.items():
        text = text.replace(source, replacement)

    text = re.sub(
        r"\b(?:thmbox|defbox|lembox|propbox|corbox|exbox|rembox)\[([^\]]*)\]",
        r"\n\n> \1\n",
        text,
    )
    text = re.sub(r"(?m)^\s*1em\]\s*", "", text)
    text = text.replace("&", " | ")
    text = re.sub(r"\\begin\{[^{}]+\}", "\n", text)
    text = re.sub(r"\\end\{[^{}]+\}", "\n", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", "", text)
    text = re.sub(r"\\.", "", text)
    text = re.sub(r"(?m)^\s*(?:center|itemize|enumerate|quote|tabular|table)\s*$", "", text)
    text = re.sub(r"[{}]", "", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def get_pdf_fallback_font_name() -> str:
    global _PDF_FALLBACK_FONT_NAME

    if _PDF_FALLBACK_FONT_NAME is not None:
        return _PDF_FALLBACK_FONT_NAME

    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
    except Exception:
        _PDF_FALLBACK_FONT_NAME = "Helvetica"
        return _PDF_FALLBACK_FONT_NAME

    font_candidates = [
        os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "arial.ttf"),
        os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "calibri.ttf"),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/Library/Fonts/Arial.ttf",
    ]

    for font_path in font_candidates:
        if not os.path.exists(font_path):
            continue
        try:
            pdfmetrics.registerFont(TTFont("ConspectumFallback", font_path))
            _PDF_FALLBACK_FONT_NAME = "ConspectumFallback"
            return _PDF_FALLBACK_FONT_NAME
        except Exception:
            continue

    _PDF_FALLBACK_FONT_NAME = "Helvetica"
    return _PDF_FALLBACK_FONT_NAME


def text_to_pdf_bytes(text: str, title: str | None = None) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import simpleSplit
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    page_width, page_height = A4
    left_margin = 54
    right_margin = 54
    top_margin = 56
    bottom_margin = 48
    max_width = page_width - left_margin - right_margin
    body_font = get_pdf_fallback_font_name()
    title_font_size = 18
    body_font_size = 11
    line_height = 15
    y = page_height - top_margin

    def new_page() -> None:
        nonlocal y
        pdf.showPage()
        pdf.setFont(body_font, body_font_size)
        y = page_height - top_margin

    if title:
        pdf.setFont(body_font, title_font_size)
        for line in simpleSplit(title, body_font, title_font_size, max_width):
            if y < bottom_margin + line_height:
                new_page()
                pdf.setFont(body_font, title_font_size)
            pdf.drawString(left_margin, y, line)
            y -= 24
        y -= 8

    def draw_paragraph(
        paragraph: str,
        *,
        font_size: int = body_font_size,
        indent: int = 0,
        gap_after: int = 4,
    ) -> None:
        nonlocal y

        wrapped_lines = simpleSplit(
            paragraph,
            body_font,
            font_size,
            max_width - indent,
        ) or [paragraph]

        pdf.setFont(body_font, font_size)
        for line in wrapped_lines:
            if y < bottom_margin + line_height:
                new_page()
                pdf.setFont(body_font, font_size)
            pdf.drawString(left_margin + indent, y, line)
            y -= max(line_height, font_size + 3)
        y -= gap_after

    for raw_line in text.splitlines():
        paragraph = raw_line.strip()
        if not paragraph:
            y -= 8
            if y < bottom_margin + line_height:
                new_page()
            continue

        if paragraph.startswith("## "):
            draw_paragraph(paragraph[3:].strip(), font_size=16, gap_after=8)
            continue
        if paragraph.startswith("### "):
            draw_paragraph(paragraph[4:].strip(), font_size=14, gap_after=6)
            continue
        if paragraph.startswith("#### "):
            draw_paragraph(paragraph[5:].strip(), font_size=12, gap_after=4)
            continue
        if paragraph.startswith("> "):
            draw_paragraph(paragraph[2:].strip(), font_size=12, indent=10, gap_after=4)
            continue
        if paragraph.startswith("- "):
            draw_paragraph(paragraph, indent=12, gap_after=2)
            continue

        draw_paragraph(paragraph)

    pdf.save()
    return buffer.getvalue()


def latex_to_fallback_pdf(latex_content: str, title: str | None = None) -> bytes:
    readable_text = latex_to_readable_text(latex_content)
    return text_to_pdf_bytes(readable_text, title=title)


def latex_output_has_missing_characters(output: str) -> bool:
    return LATEX_MISSING_CHARACTER_PATTERN.search(output) is not None


def latex_to_pdf(latex_content: str, engine: str = "pdflatex") -> tuple[bytes, str]:
    with tempfile.TemporaryDirectory() as temp_dir:
        tex_path = os.path.join(temp_dir, "file.tex")
        pdf_path = os.path.join(temp_dir, "file.pdf")
        diagnostics_lines = [
            f"engine={engine}",
            f"temp_dir={temp_dir}",
            f"tex_path={tex_path}",
        ]

        with open(tex_path, "w", encoding="utf-8", errors="replace") as f:
            f.write(latex_content)

        compiler_command = get_latex_base_command(engine) + [
            "-halt-on-error",
            "-interaction=nonstopmode",
            "-file-line-error",
            "file.tex",
        ]
        diagnostics_lines.append(f"command={' '.join(compiler_command)}")

        last_result: subprocess.CompletedProcess[str] | None = None
        for pass_index in (1, 2):
            result = subprocess.run(
                compiler_command,
                cwd=temp_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=45,
            )
            last_result = result
            diagnostics_lines.append(f"pass={pass_index} returncode={result.returncode}")
            combined_output = f"{result.stdout}\n{result.stderr}".strip()
            if combined_output:
                diagnostics_lines.append(combined_output[-4000:])

            if latex_output_has_missing_characters(combined_output):
                raise LatexCompilationError(
                    engine=engine,
                    summary=(
                        "PDF compilation produced missing glyphs. "
                        "The selected LaTeX font cannot render some document characters."
                    ),
                    diagnostics="\n\n".join(diagnostics_lines),
                )

            if result.returncode != 0 or not os.path.exists(pdf_path):
                raise LatexCompilationError(
                    engine=engine,
                    summary=format_latex_error(result),
                    diagnostics="\n\n".join(diagnostics_lines),
                )

        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        if last_result is not None:
            diagnostics_lines.append(f"pdf_size={len(pdf_bytes)}")

        return pdf_bytes, "\n\n".join(diagnostics_lines)


def sanitize_generated_latex(latex_content: str) -> str:
    sanitized = normalize_latex_text(latex_content)

    replacements = [
        (r"\\begin\{section\}\{([^{}]+)\}", r"\\section{\1}"),
        (r"\\end\{section\}", ""),
        (r"\\begin\{subsection\}\{([^{}]+)\}", r"\\subsection{\1}"),
        (r"\\end\{subsection\}", ""),
        (r"\\begin\{subsubsection\}\{([^{}]+)\}", r"\\subsubsection{\1}"),
        (r"\\end\{subsubsection\}", ""),
        (r"\\begin\{remark\}", r"\\begin{rembox}"),
        (r"\\end\{remark\}", r"\\end{rembox}"),
        (r"\\begin\{definition\}", r"\\begin{defbox}"),
        (r"\\end\{definition\}", r"\\end{defbox}"),
        (r"\\begin\{example\}", r"\\begin{exbox}"),
        (r"\\end\{example\}", r"\\end{exbox}"),
    ]
    for pattern, replacement in replacements:
        sanitized = re.sub(pattern, replacement, sanitized)

    sanitized = re.sub(r"\\begin\{rembox\}\s*\n\s*([^\\\n][^\n]*)\n", r"\\begin{rembox}[Note]\n\1\n", sanitized)
    sanitized = re.sub(r"\\begin\{defbox\}\s*\n\s*([^\\\n][^\n]*)\n", r"\\begin{defbox}[Definition]\n\1\n", sanitized)
    sanitized = re.sub(r"\\begin\{exbox\}\s*\n\s*([^\\\n][^\n]*)\n", r"\\begin{exbox}[Example]\n\1\n", sanitized)

    sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)
    return sanitized.strip()


def escape_unescaped_ampersands(tex_content: str) -> str:
    document_start = tex_content.find(r"\begin{document}")
    document_end = tex_content.rfind(r"\end{document}")

    if document_start == -1 or document_end == -1 or document_end <= document_start:
        return tex_content

    prefix = tex_content[:document_start]
    body = tex_content[document_start:document_end]
    suffix = tex_content[document_end:]

    env_stack: list[str] = []
    escaped_lines: list[str] = []

    for line in body.splitlines():
        begin_envs = re.findall(r"\\begin\{([^{}]+)\}", line)
        end_envs = re.findall(r"\\end\{([^{}]+)\}", line)

        protected = any(env in PROTECTED_AMPERSAND_ENVIRONMENTS for env in env_stack) or any(
            env in PROTECTED_AMPERSAND_ENVIRONMENTS for env in begin_envs
        )

        if not protected:
            line = re.sub(r"(?<!\\)&", r"\\&", line)

        escaped_lines.append(line)

        for env in begin_envs:
            env_stack.append(env)
        for env in end_envs:
            if env in env_stack:
                env_stack.reverse()
                env_stack.remove(env)
                env_stack.reverse()

    escaped_body = "\n".join(escaped_lines)
    body_suffix_separator = "" if not escaped_body or suffix.startswith(("\n", "\r")) else "\n"
    return prefix + escaped_body + body_suffix_separator + suffix


def count_tabular_spec_columns(spec: str) -> int:
    def expand_repeat(match: re.Match[str]) -> str:
        return match.group(2) * int(match.group(1))

    expanded = re.sub(r"\\\*\{(\d+)\}\{([^{}]+)\}", expand_repeat, spec)
    expanded = re.sub(r"[@!><]\{[^{}]*\}", "", expanded)

    count = 0
    index = 0
    while index < len(expanded):
        char = expanded[index]
        if char in "lcrX":
            count += 1
        elif char in "pmb":
            count += 1
            if index + 1 < len(expanded) and expanded[index + 1] == "{":
                depth = 0
                index += 1
                while index < len(expanded):
                    if expanded[index] == "{":
                        depth += 1
                    elif expanded[index] == "}":
                        depth -= 1
                        if depth == 0:
                            break
                    index += 1
        index += 1
    return count


def count_tabular_row_columns(row: str) -> int:
    code_row = _strip_latex_comment(row).strip()
    if not code_row or "&" not in code_row:
        return 0
    return len(re.findall(r"(?<!\\)&", code_row)) + 1


def expand_tabular_spec_to_columns(spec: str, target_columns: int) -> tuple[str, bool]:
    current_columns = count_tabular_spec_columns(spec)
    if current_columns <= 0 or target_columns <= current_columns:
        return spec, False

    missing_columns = target_columns - current_columns
    trailing_rule_match = re.search(r"\|*$", spec)
    trailing_rules = trailing_rule_match.group(0) if trailing_rule_match else ""
    insert_at = len(spec) - len(trailing_rules)
    expanded = spec[:insert_at] + ("c" * missing_columns) + trailing_rules
    return expanded, True


def repair_latex_tabular_column_specs(tex_content: str) -> tuple[str, bool]:
    document_start = tex_content.find(r"\begin{document}")
    document_end = tex_content.rfind(r"\end{document}")
    if document_start == -1 or document_end == -1 or document_end <= document_start:
        return tex_content, False

    prefix = tex_content[:document_start]
    body = tex_content[document_start:document_end]
    suffix = tex_content[document_end:]
    changed = False

    tabular_pattern = re.compile(
        r"\\begin\{tabular\}(\[[^\]]*\])?\{([^{}]+)\}(.*?)\\end\{tabular\}",
        flags=re.DOTALL,
    )

    def repair_match(match: re.Match[str]) -> str:
        nonlocal changed
        options = match.group(1) or ""
        spec = match.group(2)
        tabular_body = match.group(3)
        rows = re.split(r"(?<!\\)\\\\(?:\[[^\]]*\])?", tabular_body)
        target_columns = max((count_tabular_row_columns(row) for row in rows), default=0)
        expanded_spec, spec_changed = expand_tabular_spec_to_columns(spec, target_columns)
        if spec_changed:
            changed = True
        return rf"\begin{{tabular}}{options}{{{expanded_spec}}}{tabular_body}\end{{tabular}}"

    repaired_body = tabular_pattern.sub(repair_match, body)
    return prefix + repaired_body + suffix, changed


def remove_fragile_enumerate_options(tex_content: str) -> tuple[str, bool]:
    repaired, replacements = re.subn(
        r"\\begin\{enumerate\}\[[^\]\n]*\]",
        r"\\begin{enumerate}",
        tex_content,
    )
    return repaired, replacements > 0


def ensure_latex_proof_environment(tex_content: str) -> tuple[str, bool]:
    uses_proof_environment = r"\begin{proof}" in tex_content
    uses_qed_command = r"\qed" in tex_content
    if not uses_proof_environment and not uses_qed_command:
        return tex_content, False

    proof_environment_defined = (
        r"\@ifundefined{proof}" in tex_content
        or r"\newenvironment{proof}" in tex_content
        or r"\renewenvironment{proof}" in tex_content
    )
    qed_command_defined = re.search(r"\\(?:providecommand|newcommand|renewcommand)\{\\qed\}", tex_content) is not None

    fallback_blocks: list[str] = []
    if uses_proof_environment and not proof_environment_defined:
        fallback_blocks.append(PROOF_ENVIRONMENT_FALLBACK)
    if uses_qed_command and not qed_command_defined:
        fallback_blocks.append(QED_COMMAND_FALLBACK)

    if not fallback_blocks:
        return tex_content, False

    fallback = "\n".join(fallback_blocks)
    document_start = tex_content.find(r"\begin{document}")
    if document_start == -1:
        return tex_content.rstrip() + "\n\n" + fallback + "\n", True

    return (
        tex_content[:document_start].rstrip() + "\n\n" + fallback + "\n\n" + tex_content[document_start:].lstrip()
    ), True


def ensure_latex_hyperref_fallback_commands(tex_content: str) -> tuple[str, bool]:
    fallback_blocks: list[str] = []

    for command_name, fallback in HYPERREF_FALLBACKS.items():
        command = "\\" + command_name
        if command not in tex_content:
            continue
        if re.search(rf"\\(?:providecommand|newcommand|renewcommand)\{{\\{command_name}\}}", tex_content):
            continue
        fallback_blocks.append(fallback)

    if not fallback_blocks:
        return tex_content, False

    fallback = "\n".join(fallback_blocks)
    document_start = tex_content.find(r"\begin{document}")
    if document_start == -1:
        return tex_content.rstrip() + "\n\n" + fallback + "\n", True

    return (
        tex_content[:document_start].rstrip() + "\n\n" + fallback + "\n\n" + tex_content[document_start:].lstrip()
    ), True


def ensure_latex_text_fallback_commands(tex_content: str) -> tuple[str, bool]:
    fallback_blocks: list[str] = []

    for command_name, fallback in TEXT_FALLBACKS.items():
        command = "\\" + command_name
        if command not in tex_content:
            continue
        if re.search(rf"\\(?:providecommand|newcommand|renewcommand)\{{\\{command_name}\}}", tex_content):
            continue
        fallback_blocks.append(fallback)

    if not fallback_blocks:
        return tex_content, False

    fallback = "\n".join(fallback_blocks)
    document_start = tex_content.find(r"\begin{document}")
    if document_start == -1:
        return tex_content.rstrip() + "\n\n" + fallback + "\n", True

    return (
        tex_content[:document_start].rstrip() + "\n\n" + fallback + "\n\n" + tex_content[document_start:].lstrip()
    ), True


def ensure_latex_math_fallback_commands(tex_content: str) -> tuple[str, bool]:
    fallback_blocks: list[str] = []

    for command_name, fallback in MATH_FALLBACKS.items():
        command = "\\" + command_name
        if command not in tex_content:
            continue
        if re.search(rf"\\(?:providecommand|newcommand|renewcommand)\{{\\{command_name}\}}", tex_content):
            continue
        fallback_blocks.append(fallback)

    if not fallback_blocks:
        return tex_content, False

    fallback = "\n".join(fallback_blocks)
    document_start = tex_content.find(r"\begin{document}")
    if document_start == -1:
        return tex_content.rstrip() + "\n\n" + fallback + "\n", True

    return (
        tex_content[:document_start].rstrip() + "\n\n" + fallback + "\n\n" + tex_content[document_start:].lstrip()
    ), True


def prepare_latex_document(tex_content: str, language: str) -> LatexPreparationResult:
    notes: list[str] = []
    repaired = sanitize_generated_latex(tex_content)
    repaired, lmodern_removed = remove_lmodern_package(repaired)
    if lmodern_removed:
        notes.append("Removed Latin Modern font package so Cyrillic fonts remain available.")
    repaired, tcolorbox_title_repaired = repair_tcolorbox_title_placeholder(repaired)
    if tcolorbox_title_repaired:
        notes.append("Protected tcolorbox titles that contain commas or nested brackets.")
    repaired, escaped_inline_math_repaired = repair_escaped_inline_math_delimiters(repaired)
    if escaped_inline_math_repaired:
        notes.append("Repaired double-escaped inline math delimiters.")
    repaired, escaped_environment_repaired = repair_escaped_latex_environment_commands(repaired)
    if escaped_environment_repaired:
        notes.append("Repaired double-escaped LaTeX environment commands.")
    repaired, incomplete_inline_math_removed = remove_incomplete_inline_math_command_lines(repaired)
    if incomplete_inline_math_removed:
        notes.append("Removed incomplete inline math command lines.")
    repaired, box_syntax_normalized = normalize_latex_box_syntax(repaired)
    if box_syntax_normalized:
        notes.append("Normalized malformed lecture-note box syntax.")
    repaired, box_titles_protected = protect_latex_box_optional_titles(repaired)
    if box_titles_protected:
        notes.append("Protected lecture-note box titles and normalized fragile title math.")
    repaired, unmatched_display_math_endings_removed = remove_unmatched_latex_display_math_endings(repaired)
    if unmatched_display_math_endings_removed:
        notes.append("Removed unmatched LaTeX display math endings.")
    repaired, display_math_repaired = close_unclosed_latex_display_math(repaired)
    if display_math_repaired:
        notes.append("Closed unbalanced LaTeX display math blocks before environment boundaries.")
    repaired, list_environments_repaired = repair_latex_list_environments(repaired)
    if list_environments_repaired:
        notes.append("Repaired unbalanced LaTeX list environments before structural boundaries.")
    repaired, unmatched_box_endings_removed = remove_unmatched_latex_box_endings(repaired)
    if unmatched_box_endings_removed:
        notes.append("Removed unmatched lecture-note box endings.")
    repaired, box_environments_repaired = close_unclosed_latex_box_environments(repaired)
    if box_environments_repaired:
        notes.append("Closed unbalanced lecture-note box environments before section boundaries.")
    unicode_prepared = normalize_unicode_latex_document(repaired)
    repaired = escape_unescaped_ampersands(unicode_prepared.tex)
    repaired, tabular_specs_repaired = repair_latex_tabular_column_specs(repaired)
    repaired, enumerate_options_removed = remove_fragile_enumerate_options(repaired)
    repaired = ensure_multilingual_latex_preamble(repaired, language)
    repaired, proof_environment_added = ensure_latex_proof_environment(repaired)
    repaired, hyperref_fallback_added = ensure_latex_hyperref_fallback_commands(repaired)
    repaired, text_fallback_added = ensure_latex_text_fallback_commands(repaired)
    repaired, math_fallback_added = ensure_latex_math_fallback_commands(repaired)
    repaired, magic_comment_added = ensure_xelatex_magic_comment(repaired, language)
    notes.extend(unicode_prepared.notes)
    if tabular_specs_repaired:
        notes.append("Expanded underspecified LaTeX tabular column definitions.")
    if enumerate_options_removed:
        notes.append("Removed fragile LaTeX enumerate options.")
    if proof_environment_added:
        notes.append("Added fallback LaTeX proof environment.")
    if hyperref_fallback_added:
        notes.append("Added fallback LaTeX hyperref compatibility commands.")
    if text_fallback_added:
        notes.append("Added fallback LaTeX text compatibility commands.")
    if math_fallback_added:
        notes.append("Added fallback LaTeX math compatibility commands.")
    if magic_comment_added:
        notes.append("Added XeLaTeX magic comment for Russian PDF compilation.")
    notes.append(f"Ensured multilingual LaTeX preamble for {language}.")
    return LatexPreparationResult(tex=repaired.strip(), notes=notes)


def repair_latex_document(tex_content: str, language: str = "en") -> str:
    return prepare_latex_document(tex_content, language).tex


def validate_complete_latex(tex_content: str) -> None:
    required_elements = [
        r"\documentclass",
        r"\begin{document}",
        r"\end{document}",
    ]
    missing = [element for element in required_elements if element not in tex_content]
    if missing:
        raise RuntimeError(f"Incomplete LaTeX document: missing {', '.join(missing)}")


def format_latex_error(result: subprocess.CompletedProcess[str]) -> str:
    combined_output = f"{result.stdout}\n{result.stderr}"
    relevant_lines = []

    for line in combined_output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("!"):
            relevant_lines.append(stripped)
        elif "LaTeX Error:" in stripped or "Fatal error occurred" in stripped:
            relevant_lines.append(stripped)
        elif re.search(r"file\.tex:\d+:", stripped):
            relevant_lines.append(stripped)

    if not relevant_lines:
        tail = "\n".join(combined_output.splitlines()[-15:])
        return f"PDF was not generated.\n{tail}".strip()

    return "PDF was not generated.\n" + "\n".join(relevant_lines[:10])


async def compile_latex_pdf(
    latex_content: str,
    language: str,
    logger: Logger,
) -> tuple[typing.Optional[bytes], typing.Optional[str]]:
    engines = get_preferred_latex_engines(latex_content, language)
    if not engines:
        return None, (
            "PDF generation skipped: no LaTeX engine (pdflatex, xelatex, or lualatex) was found in PATH. "
            "Returning only .tex file."
        )

    errors: list[str] = []
    primary_engine = engines[0]
    await logger.partial_result(f"Selected {primary_engine} as the primary LaTeX engine for this document.")

    for attempt_index, engine in enumerate(engines):
        is_primary = attempt_index == 0
        stage_code = "pdf" if is_primary else "pdf_retry"
        stage_progress = 25 if is_primary else min(95, 45 + attempt_index * 18)
        await logger.stage(stage_code, stage_progress)
        await logger.partial_result(f"Compiling PDF with {engine}...")

        try:
            pdf_bytes, diagnostics = await asyncio.to_thread(latex_to_pdf, latex_content, engine)
            await logger.file(
                f"latex_compile_{engine}_success",
                diagnostics,
                Logger.FileType.TEXT,
            )
            await logger.stage(stage_code, 100)
            await logger.partial_result(f"PDF compilation succeeded with {engine}.")
            return pdf_bytes, None
        except LatexCompilationError as exc:
            await logger.file(
                f"latex_compile_{engine}_failure",
                exc.diagnostics,
                Logger.FileType.TEXT,
            )
            errors.append(f"{engine}: {exc.summary}")
            if attempt_index < len(engines) - 1:
                await logger.partial_result(f"{engine} could not compile the document. Trying the next LaTeX engine...")
            else:
                await logger.partial_result(f"{engine} could not compile the document.")

    return None, "\n".join(errors)


def get_chunk_target_chars(detail_level: str) -> int:
    configured = os.environ.get("BRIEF_CHUNK_TARGET_CHARS") if detail_level == "brief" else None
    if not configured and detail_level != "brief":
        configured = os.environ.get("CHUNK_TARGET_CHARS")
    if configured:
        try:
            value = int(configured)
        except ValueError:
            value = DEFAULT_CHUNK_TARGET_CHARS.get(detail_level, DEFAULT_CHUNK_TARGET_CHARS["standard"])
    else:
        value = DEFAULT_CHUNK_TARGET_CHARS.get(detail_level, DEFAULT_CHUNK_TARGET_CHARS["standard"])

    return max(MIN_CHUNK_TARGET_CHARS, min(MAX_CHUNK_TARGET_CHARS, value))


def get_chunk_max_tokens(detail_level: str) -> int:
    configured = os.environ.get("BRIEF_CHUNK_MAX_TOKENS") if detail_level == "brief" else None
    if not configured and detail_level != "brief":
        configured = os.environ.get("CHUNK_MAX_TOKENS")
    if configured:
        try:
            value = int(configured.strip())
        except ValueError:
            value = DEFAULT_CHUNK_MAX_TOKENS.get(detail_level, DEFAULT_CHUNK_MAX_TOKENS["standard"])
    else:
        value = DEFAULT_CHUNK_MAX_TOKENS.get(detail_level, DEFAULT_CHUNK_MAX_TOKENS["standard"])

    return max(MIN_CHUNK_MAX_TOKENS, min(MAX_CHUNK_MAX_TOKENS, value))


def llm_postprocess_enabled() -> bool:
    configured = os.environ.get("ENABLE_LLM_POSTPROCESS")
    if configured is None or not configured.strip():
        return True
    return configured.strip().lower() in ENABLE_LLM_POSTPROCESS_VALUES


def get_chunk_process_concurrency(total_chunks: int, detail_level: str = "standard") -> int:
    if total_chunks <= 1:
        return 1

    default_value = (
        DEFAULT_BRIEF_CHUNK_PROCESS_CONCURRENCY if detail_level == "brief" else DEFAULT_CHUNK_PROCESS_CONCURRENCY
    )
    configured = os.environ.get("BRIEF_CHUNK_PROCESS_CONCURRENCY") if detail_level == "brief" else None
    if not configured and detail_level != "brief":
        configured = os.environ.get("CHUNK_PROCESS_CONCURRENCY")
    if configured:
        try:
            value = int(configured.strip())
        except ValueError:
            value = default_value
    else:
        value = default_value

    return max(1, min(MAX_CHUNK_PROCESS_CONCURRENCY, total_chunks, value))


def get_chunk_ai_timeout_seconds() -> float:
    configured = os.environ.get("CHUNK_AI_TIMEOUT_SECONDS", "").strip()
    try:
        value = float(configured) if configured else DEFAULT_CHUNK_AI_TIMEOUT_SECONDS
    except ValueError:
        value = DEFAULT_CHUNK_AI_TIMEOUT_SECONDS
    return max(MIN_CHUNK_AI_TIMEOUT_SECONDS, min(MAX_CHUNK_AI_TIMEOUT_SECONDS, value))


def get_chunk_ai_model_attempts() -> int:
    configured = os.environ.get("CHUNK_AI_MODEL_ATTEMPTS", "").strip()
    try:
        value = int(configured) if configured else DEFAULT_CHUNK_AI_MODEL_ATTEMPTS
    except ValueError:
        value = DEFAULT_CHUNK_AI_MODEL_ATTEMPTS
    return max(1, min(MAX_CHUNK_AI_MODEL_ATTEMPTS, value))


def build_chunk_formatting_rules(language: str, detail_level: str = "standard") -> str:
    core_rules = (
        "Transform each transcript chunk into a mergeable, exam-ready LaTeX lecture-note fragment.\n"
        f"Write all prose and headings in {LANGUAGE_NAMES[language]}; keep LaTeX commands standard.\n"
        "Return only LaTeX that belongs between \\begin{document} and \\end{document}; "
        "never include a preamble, \\begin{document}, \\end{document}, markdown fences, "
        "title, abstract, or references.\n"
        "The transcript may contain ASR noise, repeated phrases, mixed languages, and mistranscribed terms. "
        "Ignore filler and subtitle artifacts; repair only obvious terminology when the surrounding math supports it. "
        "Do not invent facts, sources, examples, or equations.\n"
    )

    if detail_level == "brief":
        detail_rules = (
            "Create a compact conspectus, not full lecture notes. Aim for roughly 55--65% of the detailed-mode length "
            "for the same transcript while preserving the exam-critical substance.\n"
            "For each chunk, keep at most 1--2 heading levels, 2--4 short blocks, and only formulas/results that are "
            "central to the lecture. Prefer dense paragraphs and short lists over many boxes.\n"
            "Use defbox/thmbox only for indispensable definitions or theorems. Use exbox only when a single example "
            "is essential for understanding; otherwise omit examples. Do not add introductions or conclusions for "
            "intermediate chunks.\n"
        )
    else:
        detail_rules = (
            "Create real notes, not a terse recap: include definitions, formulas, assumptions, examples, comparisons, "
            "and consequences present in the chunk. Use sections/subsections to mirror lecture flow and "
            "defbox/thmbox/exbox/rembox for key ideas.\n"
            "For chunk 1, add a short introduction if useful. "
            "For the final chunk, add a brief conclusion if the transcript supports it.\n"
        )

    closing_rules = (
        "Use math mode for formulas and close every LaTeX environment. "
        "Escape literal %, &, _, #, $, {, and } in text.\n"
    )

    return core_rules + detail_rules + closing_rules


def clean_transcript_for_notes(transcript: str) -> str:
    cleaned = transcript
    for pattern in TRANSCRIPT_ARTIFACT_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)

    cleaned = REPEATED_ELLIPSIS_PATTERN.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or transcript.strip()


def extract_previous_chunk_context(previous_chunk_result: str | None) -> str | None:
    if not previous_chunk_result:
        return None

    last_section_start = max(
        previous_chunk_result.rfind("\\section"),
        previous_chunk_result.rfind("\\subsection"),
        previous_chunk_result.rfind("\\subsubsection"),
    )
    context = previous_chunk_result[last_section_start:] if last_section_start >= 0 else previous_chunk_result
    context = context.strip()
    if len(context) > MAX_PREVIOUS_CHUNK_CONTEXT_CHARS:
        context = context[-MAX_PREVIOUS_CHUNK_CONTEXT_CHARS:].lstrip()
    return context or None


def extract_adjacent_transcript_context(chunks: typing.Sequence[str], chunk_index: int) -> str | None:
    context_parts: list[str] = []

    if chunk_index > 0:
        previous_chunk = chunks[chunk_index - 1].strip()
        if len(previous_chunk) > MAX_ADJACENT_TRANSCRIPT_CONTEXT_CHARS:
            previous_chunk = previous_chunk[-MAX_ADJACENT_TRANSCRIPT_CONTEXT_CHARS:].lstrip()
        if previous_chunk:
            context_parts.append(f"Previous transcript tail:\n{previous_chunk}")

    if chunk_index + 1 < len(chunks):
        next_chunk = chunks[chunk_index + 1].strip()
        if len(next_chunk) > MAX_ADJACENT_TRANSCRIPT_CONTEXT_CHARS:
            next_chunk = next_chunk[:MAX_ADJACENT_TRANSCRIPT_CONTEXT_CHARS].rstrip()
        if next_chunk:
            context_parts.append(f"Next transcript head:\n{next_chunk}")

    return "\n\n".join(context_parts) or None


def split_long_sentence(sentence: str, target_chars: int) -> list[str]:
    words = sentence.split()
    pieces: list[str] = []
    current = ""

    for word in words:
        separator = " " if current else ""
        candidate = current + separator + word
        if len(candidate) > target_chars and current:
            pieces.append(current)
            current = word
        else:
            current = candidate

    if current:
        pieces.append(current)
    return pieces or [sentence]


async def split_into_chunks(
    transcript: str,
    logger: Logger = Logger(),
    target_chars: int = DEFAULT_CHUNK_TARGET_CHARS["standard"],
) -> typing.List[str]:
    """Split transcript into text chunks instead of audio chunks."""
    sentences = [sentence for sentence in re.split(r"(?<=[.!?])\s+", transcript.strip()) if sentence]
    chunks = []
    current_chunk = ""

    for raw_sentence in sentences:
        sentence_parts = (
            split_long_sentence(raw_sentence, target_chars) if len(raw_sentence) > target_chars else [raw_sentence]
        )

        for sentence in sentence_parts:
            separator = " " if current_chunk else ""
            candidate = current_chunk + separator + sentence

            if len(candidate) > target_chars:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    chunks.append(sentence.strip())
            else:
                current_chunk = candidate

    if current_chunk:
        chunks.append(current_chunk.strip())

    for i, chunk in enumerate(chunks):
        await logger.file(f"chunk_{i + 1}_text", chunk, Logger.FileType.TEXT)

    return chunks


async def process_chunk(
    text_chunk: str,
    chunk_num: int,
    total_chunks: int,
    chunk_rules: str,
    ai: openai.AsyncOpenAI,
    language: str,
    detail_level: str,
    previous_chunk_result: typing.Optional[str] = None,
    adjacent_transcript_context: typing.Optional[str] = None,
):
    with open(
        pathlib.Path(__file__).parent / "prompts/system_prompt.txt",
        encoding="utf-8",
        errors="replace",
    ) as prompt_file:
        system_prompt = prompt_file.read()

    system_prompt = system_prompt.replace("<OUTPUT_LANGUAGE>", LANGUAGE_NAMES[language])

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": chunk_rules},
        {
            "role": "system",
            "content": (
                f"Target detail level: {detail_level}. "
                f"{DETAIL_LEVEL_GUIDANCE.get(detail_level, DETAIL_LEVEL_GUIDANCE['standard'])}"
            ),
        },
        {
            "role": "user",
            "content": f"This is chunk {chunk_num}/{total_chunks} of the lecture transcript:\n\n{text_chunk}",
        },
    ]

    previous_context = extract_previous_chunk_context(previous_chunk_result)
    if previous_context is not None:
        messages.append(
            {
                "role": "system",
                "content": f"Use this short previous-ending context only for continuity:\n\n{previous_context}",
            }
        )

    if adjacent_transcript_context is not None:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Adjacent transcript context for continuity only. "
                    "Do not summarize it; summarize only the current chunk:\n\n"
                    f"{adjacent_transcript_context}"
                ),
            }
        )

    response = await create_chat_completion(
        ai,
        messages=messages,
        temperature=0.3,
        max_tokens=get_chunk_max_tokens(detail_level),
        timeout=get_chunk_ai_timeout_seconds(),
        max_model_attempts=get_chunk_ai_model_attempts(),
    )

    content = response.choices[0].message.content

    if content is None:
        raise RuntimeError("The model returned an empty response for a chunk.")

    return content


async def process_chunks(
    chunks: typing.Sequence[str],
    chunk_rules: str,
    ai: openai.AsyncOpenAI,
    language: str,
    detail_level: str,
    logger: Logger,
) -> list[str]:
    total_chunks = len(chunks)
    if total_chunks == 0:
        return []

    concurrency = get_chunk_process_concurrency(total_chunks, detail_level)
    await logger.stage("sections", 0)
    await logger.progress(0, total_chunks)
    await logger.partial_result(
        f"Generating {total_chunks} section chunks with up to {concurrency} parallel AI requests."
    )

    results: list[str | None] = [None] * total_chunks
    completed = 0

    async def report_activity() -> None:
        while completed < total_chunks:
            await asyncio.sleep(CHUNK_PROGRESS_HEARTBEAT_SECONDS)
            if completed < total_chunks:
                await logger.partial_result(
                    f"Section generation is active: {completed}/{total_chunks} ready. "
                    "Waiting for the AI provider to finish the current requests."
                )

    heartbeat_task = asyncio.create_task(report_activity())

    try:
        if concurrency == 1:
            previous_result: str | None = None
            for i, chunk in enumerate(chunks):
                content = await process_chunk(
                    text_chunk=chunk,
                    chunk_num=i + 1,
                    total_chunks=total_chunks,
                    chunk_rules=chunk_rules,
                    ai=ai,
                    language=language,
                    detail_level=detail_level,
                    previous_chunk_result=previous_result,
                )

                await logger.file(f"chunk_{i + 1}", content, Logger.FileType.TEXT)
                previous_result = sanitize_generated_latex(content)
                results[i] = previous_result
                completed += 1
                await logger.progress(completed, total_chunks)

            return [result or "" for result in results]

        semaphore = asyncio.Semaphore(concurrency)

        async def process_one(index: int, chunk: str) -> tuple[int, str]:
            async with semaphore:
                content = await process_chunk(
                    text_chunk=chunk,
                    chunk_num=index + 1,
                    total_chunks=total_chunks,
                    chunk_rules=chunk_rules,
                    ai=ai,
                    language=language,
                    detail_level=detail_level,
                    adjacent_transcript_context=extract_adjacent_transcript_context(chunks, index),
                )
                await logger.file(f"chunk_{index + 1}", content, Logger.FileType.TEXT)
                return index, sanitize_generated_latex(content)

        tasks = [asyncio.create_task(process_one(i, chunk)) for i, chunk in enumerate(chunks)]
        try:
            for task in asyncio.as_completed(tasks):
                index, sanitized_content = await task
                results[index] = sanitized_content
                completed += 1
                await logger.progress(completed, total_chunks)
        except Exception:
            for task in tasks:
                task.cancel()
            raise

        return [result or "" for result in results]
    finally:
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass


async def process(
    audio_file: bytes,
    ai: openai.AsyncOpenAI,
    logger: Logger = Logger(),
    language: typing.Optional[str] = None,
    detail_level: str = "standard",
    audio_filename: str | None = None,
    audio_mime_type: str | None = None,
) -> ProcessResult:
    if language is not None and language not in ("ru", "en"):
        raise ValueError(f"Unsupported language: {language}. Must be 'ru' or 'en'.")
    if detail_level not in DETAIL_LEVEL_GUIDANCE:
        raise ValueError(
            f"Unsupported detail level: {detail_level}. Must be one of {', '.join(DETAIL_LEVEL_GUIDANCE)}."
        )

    await logger.stage("starting", 40)
    await logger.partial_result("Starting transcription...")
    transcription = await transcribe_audio_with_metadata(
        audio_file,
        logger,
        filename=audio_filename,
        mime_type=audio_mime_type,
        language=language,
    )
    transcript = transcription.text

    if language is None:
        await logger.stage("detect_language", 20)
        language = map_transcription_language_to_output_language(transcription.language)
        if language is None:
            language = await detect_language_from_text(transcript, ai)
        await logger.stage("detect_language", 100)
        await logger.partial_result(f"Detected language: {language}")

    note_transcript = clean_transcript_for_notes(transcript)
    if note_transcript != transcript.strip():
        await logger.file("transcript_for_notes", note_transcript, Logger.FileType.TEXT)
        await logger.partial_result("Removed repeated subtitle/ASR artifacts before note generation.")

    summary = await make_summary_from_transcript(
        note_transcript,
        ai,
        language,
        logger,
        detail_level=detail_level,
    )

    with open(
        pathlib.Path(__file__).parent / "prompts/template.tex",
        encoding="utf-8",
        errors="replace",
    ) as tex_template_file:
        tex_template = tex_template_file.read()

    tex_template = localize_template(tex_template, language)
    tex_template = tex_template.replace("<INSERT TITLE HERE>", summary.title)
    tex_template = tex_template.replace("<INSERT ABSTRACT HERE>", summary.abstract)

    await logger.file("tex_template", tex_template, Logger.FileType.TEX)

    chunk_target_chars = get_chunk_target_chars(detail_level)
    chunks = await split_into_chunks(note_transcript, logger, target_chars=chunk_target_chars)
    chunk_rules = build_chunk_formatting_rules(language, detail_level)
    await logger.partial_result(f"Transcript split into {len(chunks)} chunks (~{chunk_target_chars} chars each).")
    results = await process_chunks(
        chunks=chunks,
        chunk_rules=chunk_rules,
        ai=ai,
        language=language,
        detail_level=detail_level,
        logger=logger,
    )

    tex = tex_template.replace("%% <INSERT CONTENT HERE>", "\n\n".join(results))
    prepared_initial = prepare_latex_document(normalize_latex_text(tex), language)
    tex = prepared_initial.tex
    validate_complete_latex(tex)
    await logger.file(
        "latex_preparation_before_postprocess",
        "\n".join(prepared_initial.notes) or "No preparation changes were needed.",
        Logger.FileType.TEXT,
    )
    if prepared_initial.notes:
        await logger.partial_result("Normalized Unicode text and formulas for multilingual LaTeX compilation.")

    await logger.file("lecture_before_postprocess", tex, Logger.FileType.TEX)

    tex_postprocessed = tex
    if llm_postprocess_enabled():
        try:
            tex_postprocessed_raw = await postprocess_summary(
                tex,
                ai,
                language,
                logger,
                source_transcript=note_transcript,
            )
            prepared_postprocessed = prepare_latex_document(tex_postprocessed_raw, language)
            tex_postprocessed = prepared_postprocessed.tex
            validate_complete_latex(tex_postprocessed)
            await logger.file(
                "latex_preparation_after_postprocess",
                "\n".join(prepared_postprocessed.notes) or "No preparation changes were needed.",
                Logger.FileType.TEXT,
            )
            await logger.file("lecture", tex_postprocessed, Logger.FileType.TEX)
        except Exception as e:
            await logger.stage("postprocess", 100)
            await logger.partial_result(f"Postprocessing failed: {e}. Continuing with original version...")
            tex_postprocessed = tex
            await logger.file("lecture", tex, Logger.FileType.TEX)
    else:
        await logger.stage("postprocess", 100)
        await logger.partial_result("Fast postprocessing complete: local LaTeX cleanup and validation were applied.")
        await logger.file("lecture", tex, Logger.FileType.TEX)

    if not any_latex_engine_available():
        await logger.stage("tex_only", 100)
        warning_message = (
            "PDF generation skipped: no LaTeX engine (pdflatex, xelatex, or lualatex) was found in PATH. "
            "Returning only .tex file."
        )
        await logger.partial_result(warning_message)
        return ProcessResult(
            transcript=transcript,
            language=language,
            title=summary.title,
            abstract=summary.abstract,
            tex=tex_postprocessed,
            pdf=None,
            pdf_warning=warning_message,
        )

    pdf_source_tex = tex_postprocessed
    pdf, error_msg = await compile_latex_pdf(pdf_source_tex, language, logger)
    if pdf is not None:
        await logger.file("lecture", pdf, Logger.FileType.PDF)
        return ProcessResult(
            transcript=transcript,
            language=language,
            title=summary.title,
            abstract=summary.abstract,
            tex=pdf_source_tex,
            pdf=pdf,
        )

    fallback_warning = (
        "PDF was generated in a readable fallback layout because LaTeX compilation failed. "
        "The original UTF-8 TEX file is still available."
    )
    try:
        await logger.stage("pdf_retry", 85)
        await logger.partial_result("Retrying PDF generation with a readable fallback layout...")
        pdf = await asyncio.to_thread(latex_to_fallback_pdf, pdf_source_tex, title=summary.title)
        await logger.stage("pdf_retry", 100)
        await logger.file("lecture_fallback", pdf, Logger.FileType.PDF)
        await logger.partial_result(fallback_warning)
        return ProcessResult(
            transcript=transcript,
            language=language,
            title=summary.title,
            abstract=summary.abstract,
            tex=pdf_source_tex,
            pdf=pdf,
            pdf_warning=fallback_warning,
        )
    except Exception as fallback_pdf_error:
        error_msg = f"Readable PDF fallback also failed: {fallback_pdf_error}"
        await logger.partial_result(error_msg)

    if contains_non_ascii_characters(pdf_source_tex):
        ascii_fallback_tex = make_ascii_safe_latex(pdf_source_tex)
        if ascii_fallback_tex != pdf_source_tex:
            transliteration_warning = (
                "PDF was generated with an ASCII/transliterated fallback because the local "
                "LaTeX installation cannot typeset this document's Unicode characters. "
                "The original UTF-8 TEX file is still available."
            )
            try:
                await logger.stage("pdf_retry", 92)
                await logger.partial_result("Retrying PDF generation with an ASCII-safe transliteration fallback...")
                pdf, diagnostics = await asyncio.to_thread(latex_to_pdf, ascii_fallback_tex)
                await logger.stage("pdf_retry", 100)
                await logger.file(
                    "lecture_ascii_fallback",
                    ascii_fallback_tex,
                    Logger.FileType.TEX,
                )
                await logger.file(
                    "latex_compile_ascii_fallback_success",
                    diagnostics,
                    Logger.FileType.TEXT,
                )
                await logger.file("lecture", pdf, Logger.FileType.PDF)
                await logger.partial_result(transliteration_warning)
                return ProcessResult(
                    transcript=transcript,
                    language=language,
                    title=summary.title,
                    abstract=summary.abstract,
                    tex=pdf_source_tex,
                    pdf=pdf,
                    pdf_warning=transliteration_warning,
                )
            except LatexCompilationError as ascii_retry_error:
                await logger.file(
                    "latex_compile_ascii_fallback_failure",
                    ascii_retry_error.diagnostics,
                    Logger.FileType.TEXT,
                )
                error_msg = f"ASCII-safe PDF retry also failed: {ascii_retry_error}"
                await logger.partial_result(error_msg)

    return ProcessResult(
        transcript=transcript,
        language=language,
        title=summary.title,
        abstract=summary.abstract,
        tex=pdf_source_tex,
        pdf=None,
        pdf_warning=error_msg,
    )
