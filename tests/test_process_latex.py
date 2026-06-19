from pathlib import Path
import sys
import types
import unittest

if "faster_whisper" not in sys.modules:
    fake_faster_whisper = types.ModuleType("faster_whisper")

    class DummyWhisperModel:
        def __init__(self, *args, **kwargs):
            return

    fake_faster_whisper.WhisperModel = DummyWhisperModel
    sys.modules["faster_whisper"] = fake_faster_whisper

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from conspectum.process import get_preferred_latex_engines  # noqa: E402
from conspectum.process import latex_engine_available  # noqa: E402
from conspectum.process import latex_output_has_missing_characters  # noqa: E402
from conspectum.process import latex_to_pdf  # noqa: E402
from conspectum.process import localize_template  # noqa: E402
from conspectum.process import prepare_latex_document  # noqa: E402

TEMPLATE_TEXT = (SRC_DIR / "conspectum" / "prompts" / "template.tex").read_text(encoding="utf-8")


def build_prepared_document(language: str, title: str, abstract: str, body: str):
    tex = localize_template(TEMPLATE_TEXT, language)
    tex = tex.replace("<INSERT TITLE HERE>", title)
    tex = tex.replace("<INSERT ABSTRACT HERE>", abstract)
    tex = tex.replace("%% <INSERT CONTENT HERE>", body)
    return prepare_latex_document(tex, language)


class ProcessLatexTests(unittest.TestCase):
    def test_unicode_engines_are_preferred_when_available(self):
        prepared = build_prepared_document(
            "en",
            "Study of Time and Space",
            "We compare τ₂ ≤ 5 in a short English abstract.",
            "\\section{Overview}\nEnglish text.\n",
        )
        engines = get_preferred_latex_engines(prepared.tex, "en")

        if latex_engine_available("xelatex"):
            self.assertEqual(engines[0], "xelatex")
        elif latex_engine_available("lualatex"):
            self.assertEqual(engines[0], "lualatex")
        else:
            self.assertEqual(engines[0], "pdflatex")

    def test_prepare_latex_document_normalizes_russian_unicode_math_tokens(self):
        prepared = build_prepared_document(
            "ru",
            "\u0418\u0437\u0443\u0447\u0435\u043d\u0438\u0435 \u0432\u0440\u0435\u043c\u0435\u043d\u0438 \u0438 \u043f\u0440\u043e\u0441\u0442\u0440\u0430\u043d\u0441\u0442\u0432\u0430",
            "\u0420\u0430\u0441\u0441\u043c\u0430\u0442\u0440\u0438\u0432\u0430\u0435\u043c \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440 \u03c4 \u0438 \u043c\u043e\u043b\u0435\u043a\u0443\u043b\u0443 CO\u2082, \u0433\u0434\u0435 \u03c4\u2082 \u2264 5.",
            "\\section{\u041e\u0441\u043d\u043e\u0432\u043d\u0430\u044f \u0438\u0434\u0435\u044f}\n"
            "\u041f\u0443\u0441\u0442\u044c \u03c4\u2082 \u2264 5 \u0438 x\u2082 = 3. "
            "\u0422\u043e\u0433\u0434\u0430 \u0440\u0430\u0441\u0441\u043c\u043e\u0442\u0440\u0438\u043c $τ₂ + 1 \\\\ge 0$ "
            "\u0438 \u043c\u043d\u043e\u0436\u0435\u0441\u0442\u0432\u043e ℝ.\n",
        )

        document_body = prepared.tex.split(r"\begin{document}", 1)[1]
        self.assertIn(r"\tau", document_body)
        self.assertIn(r"CO_{2}", document_body)
        self.assertIn(r"\leq", document_body)
        self.assertIn(r"\mathbb{R}", document_body)
        self.assertNotIn("\u03c4", document_body)
        self.assertNotIn("\u2082", document_body)
        self.assertNotIn("\u2264", document_body)
        self.assertNotIn("\u211d", document_body)

    def test_prepare_latex_document_repairs_escaped_math_star(self):
        prepared = build_prepared_document(
            "ru",
            "\u0424\u0438\u043a\u0441\u0438\u0440\u043e\u0432\u0430\u043d\u043d\u0430\u044f \u0442\u043e\u0447\u043a\u0430",
            "\u041a\u0440\u0430\u0442\u043a\u0430\u044f \u0430\u043d\u043d\u043e\u0442\u0430\u0446\u0438\u044f.",
            "\\section{\u0422\u0435\u043e\u0440\u0435\u043c\u0430}\n"
            "\u0422\u043e\u0447\u043a\u0430 \\(x^\\*\\in X\\) \u0443\u0434\u043e\u0432\u043b\u0435\u0442\u0432\u043e\u0440\u044f\u0435\u0442 \\(Tx^\\*=x^\\*\\).\n",
        )

        self.assertIn(r"x^*\in X", prepared.tex)
        self.assertIn(r"Tx^*=x^*", prepared.tex)
        self.assertNotIn(r"x^\*", prepared.tex)

        engines_available = get_preferred_latex_engines(prepared.tex, "ru")
        if not engines_available:
            self.skipTest("No LaTeX engine is available in PATH.")
        pdf_bytes, diagnostics = latex_to_pdf(prepared.tex, engines_available[0])

        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertNotIn("Missing { inserted", diagnostics)
        self.assertFalse(latex_output_has_missing_characters(diagnostics))

    def test_prepare_latex_document_removes_incomplete_inline_math_command_line(self):
        prepared = build_prepared_document(
            "ru",
            "\u041f\u0440\u043e\u0441\u0442\u0440\u0430\u043d\u0441\u0442\u0432\u043e",
            "\u041a\u0440\u0430\u0442\u043a\u0430\u044f \u0430\u043d\u043d\u043e\u0442\u0430\u0446\u0438\u044f.",
            "\\section{\u041e\u0431\u0437\u043e\u0440}\n"
            "\\begin{defbox}[\u041f\u0440\u043e\u0441\u0442\u0440\u0430\u043d\u0441\u0442\u0432\u043e \\(\\widetilde{R}(E)\\)]\n"
            "\\(\\wid\n"
            "\\end{defbox}\n"
            "\u0421\u043b\u0435\u0434\u0443\u044e\u0449\u0438\u0439 \u0442\u0435\u043a\u0441\u0442.\n",
        )

        self.assertIn(r"\(\widetilde{R}(E)\)", prepared.tex)
        self.assertNotIn("\\(\\wid\n", prepared.tex)
        self.assertIn("Removed incomplete inline math command lines.", "\n".join(prepared.notes))

        engines_available = get_preferred_latex_engines(prepared.tex, "ru")
        if not engines_available:
            self.skipTest("No LaTeX engine is available in PATH.")
        pdf_bytes, diagnostics = latex_to_pdf(prepared.tex, engines_available[0])

        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertNotIn("Undefined control sequence", diagnostics)
        self.assertFalse(latex_output_has_missing_characters(diagnostics))

    def test_prepare_latex_document_keeps_cyrillic_text(self):
        prepared = build_prepared_document(
            "ru",
            "\u0418\u0437\u0443\u0447\u0435\u043d\u0438\u0435 \u0432\u0440\u0435\u043c\u0435\u043d\u0438",
            "\u041a\u0440\u0430\u0442\u043a\u0430\u044f \u0430\u043d\u043d\u043e\u0442\u0430\u0446\u0438\u044f.",
            "\\section{\u041e\u0431\u0437\u043e\u0440}\n"
            "\u042d\u0442\u043e \u0440\u0443\u0441\u0441\u043a\u0438\u0439 \u0442\u0435\u043a\u0441\u0442 \u0431\u0435\u0437 \u043f\u043e\u0442\u0435\u0440\u0438 \u043a\u0438\u0440\u0438\u043b\u043b\u0438\u0446\u044b.\n",
        )

        self.assertIn("\u0410\u043d\u043d\u043e\u0442\u0430\u0446\u0438\u044f", prepared.tex)
        self.assertIn(
            "\u042d\u0442\u043e \u0440\u0443\u0441\u0441\u043a\u0438\u0439 \u0442\u0435\u043a\u0441\u0442", prepared.tex
        )

    def test_prepare_latex_document_closes_display_math_before_box_end(self):
        prepared = build_prepared_document(
            "ru",
            "\u0421\u043a\u043e\u0440\u0438\u043d\u0433",
            "\u041a\u0440\u0430\u0442\u043a\u0430\u044f \u0430\u043d\u043d\u043e\u0442\u0430\u0446\u0438\u044f.",
            "\\section{\u041a\u043e\u043d\u0432\u0435\u0440\u0441\u0438\u044f}\n"
            "\\begin{exbox}[\u041f\u0440\u0438\u043c\u0435\u0440]\n"
            "\u0415\u0441\u043b\u0438 \u0438\u0437 100 \u0437\u0430\u044f\u0432\u043e\u043a "
            "\u043e\u0434\u043e\u0431\u0440\u0435\u043d\u043e 60:\n"
            "\\[\n"
            "\\text{\u041a\u043e\u043d\u0432\u0435\u0440\u0441\u0438\u044f} = "
            "\\frac{60}{100} \\times 100\\% = 60\\%\n"
            "\\end{exbox}\n",
        )

        self.assertIn("\\]\n\\end{exbox}", prepared.tex)
        self.assertIn("Closed unbalanced LaTeX display math blocks", "\n".join(prepared.notes))

    def test_prepare_latex_document_removes_unmatched_display_math_end(self):
        prepared = build_prepared_document(
            "ru",
            "\u0418\u043d\u0442\u0435\u0433\u0440\u0430\u043b",
            "\u041a\u0440\u0430\u0442\u043a\u0430\u044f \u0430\u043d\u043d\u043e\u0442\u0430\u0446\u0438\u044f.",
            "\\section{\u0424\u043e\u0440\u043c\u0443\u043b\u0430}\n"
            "\\begin{defbox}[\u041a\u0430\u043d\u043e\u043d\u0438\u0447\u0435\u0441\u043a\u0438\u0439 \u0438\u043d\u0442\u0435\u0433\u0440\u0430\u043b]\n"
            "\\[\n"
            "J(\\lambda)=\\int_{0}^{\\infty} e^{-\\lambda x}x^{\\alpha-1}\\,dx\n"
            "\\]\n"
            "\\]\n"
            "\\end{defbox}\n",
        )

        self.assertEqual(prepared.tex.count("\\]\n\\end{defbox}"), 1)
        self.assertIn("Removed unmatched LaTeX display math endings.", "\n".join(prepared.notes))

        engines_available = get_preferred_latex_engines(prepared.tex, "ru")
        if not engines_available:
            self.skipTest("No LaTeX engine is available in PATH.")
        pdf_bytes, diagnostics = latex_to_pdf(prepared.tex, engines_available[0])

        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertNotIn("Bad math environment delimiter", diagnostics)
        self.assertFalse(latex_output_has_missing_characters(diagnostics))

    def test_prepare_latex_document_repairs_command_style_box(self):
        prepared = build_prepared_document(
            "ru",
            "\u042d\u043d\u0442\u0440\u043e\u043f\u0438\u044f",
            "\u041a\u0440\u0430\u0442\u043a\u0430\u044f \u0430\u043d\u043d\u043e\u0442\u0430\u0446\u0438\u044f.",
            "\\section{\u041e\u0431\u0437\u043e\u0440}\n"
            "\\defbox{\u041e\u043f\u0440\u0435\u0434\u0435\u043b\u0435\u043d\u0438\u0435 \u044d\u043d\u0442\u0440\u043e\u043f\u0438\u0438}\n"
            "\u042d\u043d\u0442\u0440\u043e\u043f\u0438\u044f \u0438\u0437\u043c\u0435\u0440\u044f\u0435\u0442 "
            "\u043d\u0435\u043e\u043f\u0440\u0435\u0434\u0435\u043b\u0435\u043d\u043d\u043e\u0441\u0442\u044c.\n"
            "\\subsection{\u0414\u0430\u043b\u044c\u0448\u0435}\n"
            "\u0422\u0435\u043a\u0441\u0442.\n",
        )

        self.assertIn(
            "\\begin{defbox}[\u041e\u043f\u0440\u0435\u0434\u0435\u043b\u0435\u043d\u0438\u0435 "
            "\u044d\u043d\u0442\u0440\u043e\u043f\u0438\u0438]",
            prepared.tex,
        )
        self.assertIn("\\end{defbox}\n\\subsection", prepared.tex)
        self.assertIn("Normalized malformed lecture-note box syntax", "\n".join(prepared.notes))
        self.assertIn("Closed unbalanced lecture-note box environments", "\n".join(prepared.notes))

    def test_prepare_latex_document_repairs_command_box_with_braced_body(self):
        prepared = build_prepared_document(
            "ru",
            "Relations",
            "Short abstract.",
            "\\section{Overview}\n"
            "\\defbox{Reflexivity}{\n"
            "Every element relates to itself.\n"
            "}\n"
            "\\subsection{Next}\n"
            "Text.\n",
        )

        self.assertIn("\\begin{defbox}[Reflexivity]\nEvery element relates to itself.", prepared.tex)
        self.assertIn("Every element relates to itself.\n\\end{defbox}\n\\subsection", prepared.tex)
        self.assertNotIn("\\defbox{Reflexivity}{", prepared.tex)
        self.assertIn("Normalized malformed lecture-note box syntax", "\n".join(prepared.notes))

    def test_prepare_latex_document_repairs_one_line_command_box_with_braced_body(self):
        prepared = build_prepared_document(
            "ru",
            "Relations",
            "Short abstract.",
            "\\section{Overview}\n"
            "\\thmbox{Transitivity}{If \\( A \\mathrel{R} B \\), then \\textbf{nested} braces stay valid.}\n"
            "\\subsection{Next}\n"
            "Text.\n",
        )

        self.assertIn(
            "\\begin{thmbox}[Transitivity]\n"
            "If \\( A \\mathrel{R} B \\), then \\textbf{nested} braces stay valid.\n"
            "\\end{thmbox}",
            prepared.tex,
        )
        self.assertNotIn("\\thmbox{Transitivity}{", prepared.tex)
        self.assertIn("Normalized malformed lecture-note box syntax", "\n".join(prepared.notes))

    def test_prepare_latex_document_repairs_inline_command_style_box(self):
        prepared = build_prepared_document(
            "ru",
            "Relations",
            "Short abstract.",
            "\\section{Overview}\n"
            "\\defbox{Definition:} A relation is asymmetric when reversal is impossible.\n"
            "\\[\n"
            "A \\mathrel{R} B \\implies \\neg(B \\mathrel{R} A)\n"
            "\\]\n"
            "\\subsection{Next}\n"
            "Text.\n",
        )

        self.assertIn(
            "\\begin{defbox}[Definition:]\nA relation is asymmetric when reversal is impossible.",
            prepared.tex,
        )
        self.assertIn("\\]\n\\end{defbox}\n\\subsection", prepared.tex)
        self.assertNotIn("\\defbox{Definition:}", prepared.tex)
        self.assertIn("Normalized malformed lecture-note box syntax", "\n".join(prepared.notes))
        self.assertIn("Closed unbalanced lecture-note box environments", "\n".join(prepared.notes))

    def test_prepare_latex_document_repairs_inline_box_command_inside_sentence(self):
        prepared = build_prepared_document(
            "en",
            "Lazy classifiers",
            "Short abstract.",
            "\\section{Overview}\n"
            "A \\defbox{lazy classifier} is applied directly to the dataset.\n"
            "\\subsection{Next}\n"
            "Text.\n",
        )

        self.assertIn("A \\textbf{lazy classifier} is applied directly", prepared.tex)
        self.assertNotIn("\\defbox{lazy classifier}", prepared.tex)
        self.assertIn("Normalized malformed lecture-note box syntax", "\n".join(prepared.notes))

    def test_prepare_latex_document_repairs_single_argument_command_box(self):
        prepared = build_prepared_document(
            "en",
            "Further remarks",
            "Short abstract.",
            "\\section{Overview}\n"
            "\\rembox{\n"
            "All solution details are provided on the accompanying slides.\n"
            "Ask questions before the break.\n"
            "}\n",
        )

        self.assertIn(
            "\\begin{rembox}\n"
            "All solution details are provided on the accompanying slides.\n"
            "Ask questions before the break.\n"
            "\\end{rembox}",
            prepared.tex,
        )
        self.assertNotIn("\\rembox{", prepared.tex)
        self.assertIn("Normalized malformed lecture-note box syntax", "\n".join(prepared.notes))

        engines_available = get_preferred_latex_engines(prepared.tex, "en")
        if not engines_available:
            self.skipTest("No LaTeX engine is available in PATH.")
        pdf_bytes, diagnostics = latex_to_pdf(prepared.tex, engines_available[0])

        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertNotIn("tcb@savebox", diagnostics)
        self.assertFalse(latex_output_has_missing_characters(diagnostics))

    def test_prepare_latex_document_repairs_one_line_single_argument_command_box(self):
        prepared = build_prepared_document(
            "en",
            "Further remarks",
            "Short abstract.",
            "\\section{Overview}\n\\rembox{All solution details are provided on the slides.}\n",
        )

        self.assertIn(
            "\\begin{rembox}\nAll solution details are provided on the slides.\n\\end{rembox}",
            prepared.tex,
        )
        self.assertNotIn("\\rembox{", prepared.tex)
        self.assertIn("Normalized malformed lecture-note box syntax", "\n".join(prepared.notes))

    def test_prepare_latex_document_removes_unmatched_box_end_before_document_end(self):
        tex = (
            "\\documentclass{article}\n"
            "\\newenvironment{rembox}{}{}\n"
            "\\begin{document}\n"
            "\\begin{rembox}\n"
            "A note.\n"
            "\\end{rembox}\n"
            "\\end{rembox}\\end{document}\n"
        )
        prepared = prepare_latex_document(tex, "ru")

        self.assertEqual(prepared.tex.count("\\end{rembox}"), 1)
        self.assertIn("\\end{rembox}\n\\end{document}", prepared.tex)
        self.assertNotIn("\\end{rembox}\\end{document}", prepared.tex)
        self.assertIn("Removed unmatched lecture-note box endings", "\n".join(prepared.notes))

    def test_prepare_latex_document_does_not_duplicate_box_end_before_document_end(self):
        tex = (
            "\\documentclass{article}\n"
            "\\newenvironment{rembox}{}{}\n"
            "\\begin{document}\n"
            "\\begin{rembox}\n"
            "A note.\n"
            "\\end{rembox}\\end{document}\n"
        )
        prepared = prepare_latex_document(tex, "ru")

        self.assertEqual(prepared.tex.count("\\end{rembox}"), 1)
        self.assertNotIn("\\end{rembox}\n\\end{rembox}", prepared.tex)
        self.assertIn("\\end{rembox}\n\\end{document}", prepared.tex)

    def test_prepare_latex_document_normalizes_braced_box_title(self):
        prepared = build_prepared_document(
            "ru",
            "\u041e\u0446\u0435\u043d\u043a\u0438",
            "\u041a\u0440\u0430\u0442\u043a\u0430\u044f \u0430\u043d\u043d\u043e\u0442\u0430\u0446\u0438\u044f.",
            "\\section{\u041e\u0431\u0437\u043e\u0440}\n"
            "\\begin{defbox}{\u0421\u0432\u043e\u0439\u0441\u0442\u0432\u0430}\n"
            "\u0422\u0435\u043a\u0441\u0442.\n"
            "\\end{defbox}\n",
        )

        self.assertIn("\\begin{defbox}[\u0421\u0432\u043e\u0439\u0441\u0442\u0432\u0430]", prepared.tex)
        self.assertNotIn("\\begin{defbox}{", prepared.tex)

    def test_prepare_latex_document_braces_tcolorbox_title_placeholder(self):
        old_template = TEMPLATE_TEXT.replace("title={#1}", "title=#1")
        tex = localize_template(old_template, "ru")
        tex = tex.replace("<INSERT TITLE HERE>", "Lp metrics")
        tex = tex.replace("<INSERT ABSTRACT HERE>", "Short abstract.")
        tex = tex.replace(
            "%% <INSERT CONTENT HERE>",
            "\\section{Overview}\n"
            "\\begin{defbox}[\\(L^{p}\\) metric on \\([a,b]\\)]\n"
            "For \\(p\\ge 1\\), define \\(d_p(f,g)\\).\n"
            "\\end{defbox}\n",
        )

        prepared = prepare_latex_document(tex, "ru")

        self.assertIn("title={#1}", prepared.tex)
        self.assertNotIn("title=#1,", prepared.tex)
        self.assertIn("\\begin{defbox}[{\\(L^{p}\\) metric on \\([a,b]\\)}]", prepared.tex)
        self.assertIn("Protected tcolorbox titles", "\n".join(prepared.notes))
        self.assertIn("Protected lecture-note box titles", "\n".join(prepared.notes))

    def test_prepare_latex_document_repairs_double_escaped_inline_math(self):
        prepared = build_prepared_document(
            "en",
            "Lp metrics",
            "Short abstract.",
            "\\section{Overview}\n"
            "\\begin{defbox}\n"
            "\\textbf{Average quadratic (\\\\(L^{2}\\\\)) metric.}\n"
            "\\end{defbox}\n",
        )

        self.assertIn(r"\(L^{2}\)", prepared.tex)
        self.assertNotIn(r"\\(L^{2}\\)", prepared.tex)
        self.assertIn("Repaired double-escaped inline math delimiters.", "\n".join(prepared.notes))

    def test_prepare_latex_document_repairs_double_escaped_environment_command(self):
        prepared = build_prepared_document(
            "en",
            "Parameter integral",
            "Short abstract.",
            "\\section{Overview}\n"
            "\\begin{defbox}[Parametric multiple integral]\n"
            "\\[\n"
            "I(\\lambda)=\\iint_D f(x,y,\\lambda)\\,dx\\,dy\n"
            "\\]\n"
            "\\\\end{defbox}\n"
            "\\begin{rembox}[Next]\n"
            "Text.\n"
            "\\end{rembox}\n",
        )

        self.assertIn("\n\\end{defbox}\n\\begin{rembox}", prepared.tex)
        self.assertNotIn("\n\\\\end{defbox}", prepared.tex)
        self.assertIn("Repaired double-escaped LaTeX environment commands.", "\n".join(prepared.notes))

        engines_available = get_preferred_latex_engines(prepared.tex, "en")
        if not engines_available:
            self.skipTest("No LaTeX engine is available in PATH.")
        pdf_bytes, diagnostics = latex_to_pdf(prepared.tex, engines_available[0])

        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertNotIn("tcb@savebox", diagnostics)
        self.assertFalse(latex_output_has_missing_characters(diagnostics))

    def test_prepare_latex_document_adds_fallback_proof_environment(self):
        tex = (
            "\\documentclass{article}\n"
            "\\begin{document}\n"
            "\\section{Claim}\n"
            "\\begin{proof}\n"
            "Assume both relations hold. Therefore the claim follows.\n"
            "\\end{proof}\n"
            "\\end{document}\n"
        )

        prepared = prepare_latex_document(tex, "en")

        self.assertIn("\\@ifundefined{proof}", prepared.tex)
        self.assertIn("\\begin{proof}", prepared.tex)
        self.assertIn("Added fallback LaTeX proof environment.", "\n".join(prepared.notes))

        engines_available = get_preferred_latex_engines(prepared.tex, "en")
        if not engines_available:
            self.skipTest("No LaTeX engine is available in PATH.")
        pdf_bytes, diagnostics = latex_to_pdf(prepared.tex, engines_available[0])

        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertFalse(latex_output_has_missing_characters(diagnostics))

    def test_prepare_latex_document_adds_qed_fallback(self):
        tex = (
            "\\documentclass{article}\n\\begin{document}\nThis implication is vacuously true. \\qed\n\\end{document}\n"
        )

        prepared = prepare_latex_document(tex, "en")

        self.assertIn("\\providecommand{\\qed}", prepared.tex)
        self.assertIn("Added fallback LaTeX proof environment.", "\n".join(prepared.notes))

        engines_available = get_preferred_latex_engines(prepared.tex, "en")
        if not engines_available:
            self.skipTest("No LaTeX engine is available in PATH.")
        pdf_bytes, diagnostics = latex_to_pdf(prepared.tex, engines_available[0])

        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertNotIn("Undefined control sequence", diagnostics)
        self.assertFalse(latex_output_has_missing_characters(diagnostics))

    def test_prepare_latex_document_adds_texorpdfstring_fallback(self):
        tex = (
            "\\documentclass{article}\n"
            "\\begin{document}\n"
            "\\section{Option 1: Neural \\texorpdfstring{$\\mathcal{A}$}{A}---Concept Lattice Networks}\n"
            "Text.\n"
            "\\end{document}\n"
        )

        prepared = prepare_latex_document(tex, "en")

        self.assertIn("\\providecommand{\\texorpdfstring}[2]{#1}", prepared.tex)
        self.assertIn("Added fallback LaTeX hyperref compatibility commands.", "\n".join(prepared.notes))

        engines_available = get_preferred_latex_engines(prepared.tex, "en")
        if not engines_available:
            self.skipTest("No LaTeX engine is available in PATH.")
        pdf_bytes, diagnostics = latex_to_pdf(prepared.tex, engines_available[0])

        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertNotIn("Undefined control sequence", diagnostics)
        self.assertFalse(latex_output_has_missing_characters(diagnostics))

    def test_prepare_latex_document_adds_reference_fallbacks(self):
        tex = (
            "\\documentclass{article}\n"
            "\\begin{document}\n"
            "See \\Cref{thm:report-requirements} and \\autoref{sec:appendix}.\n"
            "\\end{document}\n"
        )

        prepared = prepare_latex_document(tex, "en")

        self.assertIn("\\providecommand{\\Cref}[1]{\\ref{#1}}", prepared.tex)
        self.assertIn("\\providecommand{\\autoref}[1]{\\ref{#1}}", prepared.tex)
        self.assertIn("Added fallback LaTeX hyperref compatibility commands.", "\n".join(prepared.notes))

        engines_available = get_preferred_latex_engines(prepared.tex, "en")
        if not engines_available:
            self.skipTest("No LaTeX engine is available in PATH.")
        pdf_bytes, diagnostics = latex_to_pdf(prepared.tex, engines_available[0])

        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertNotIn("Undefined control sequence", diagnostics)
        self.assertFalse(latex_output_has_missing_characters(diagnostics))

    def test_prepare_latex_document_adds_enquote_fallback(self):
        tex = (
            "\\documentclass{article}\n"
            "\\begin{document}\n"
            "This form represents \\enquote{flux} through a surface.\n"
            "\\end{document}\n"
        )

        prepared = prepare_latex_document(tex, "en")

        self.assertIn("\\providecommand{\\enquote}[1]{``#1''}", prepared.tex)
        self.assertIn("Added fallback LaTeX text compatibility commands.", "\n".join(prepared.notes))

        engines_available = get_preferred_latex_engines(prepared.tex, "en")
        if not engines_available:
            self.skipTest("No LaTeX engine is available in PATH.")
        pdf_bytes, diagnostics = latex_to_pdf(prepared.tex, engines_available[0])

        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertNotIn("Undefined control sequence", diagnostics)
        self.assertFalse(latex_output_has_missing_characters(diagnostics))

    def test_prepare_latex_document_adds_math_operator_fallbacks(self):
        tex = (
            "\\documentclass{article}\n"
            "\\begin{document}\n"
            "\\[\n"
            "A^{\\uparrow}=\\bigsqcap_{g\\in A}\\delta(g)\n"
            "\\]\n"
            "\\end{document}\n"
        )

        prepared = prepare_latex_document(tex, "en")

        self.assertIn("\\providecommand{\\bigsqcap}{\\bigwedge}", prepared.tex)
        self.assertIn("Added fallback LaTeX math compatibility commands.", "\n".join(prepared.notes))

        engines_available = get_preferred_latex_engines(prepared.tex, "en")
        if not engines_available:
            self.skipTest("No LaTeX engine is available in PATH.")
        pdf_bytes, diagnostics = latex_to_pdf(prepared.tex, engines_available[0])

        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertNotIn("Undefined control sequence", diagnostics)
        self.assertFalse(latex_output_has_missing_characters(diagnostics))

    def test_prepare_latex_document_adds_jump_fallback(self):
        tex = "\\documentclass{article}\n\\begin{document}\n\\[\n\\jump{f}{x}\n\\]\n\\end{document}\n"

        prepared = prepare_latex_document(tex, "en")

        self.assertIn("\\providecommand{\\jump}[2]{\\Delta #1\\vert_{#2}}", prepared.tex)
        self.assertIn("Added fallback LaTeX math compatibility commands.", "\n".join(prepared.notes))

        engines_available = get_preferred_latex_engines(prepared.tex, "en")
        if not engines_available:
            self.skipTest("No LaTeX engine is available in PATH.")
        pdf_bytes, diagnostics = latex_to_pdf(prepared.tex, engines_available[0])

        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertNotIn("Undefined control sequence", diagnostics)
        self.assertFalse(latex_output_has_missing_characters(diagnostics))

    def test_prepare_latex_document_adds_beta_fallback(self):
        tex = (
            "\\documentclass{article}\n"
            "\\begin{document}\n"
            "\\[\n"
            "\\Beta(\\alpha,\\beta)=\\int_0^1 t^{\\alpha-1}(1-t)^{\\beta-1}\\,dt\n"
            "\\]\n"
            "\\end{document}\n"
        )

        prepared = prepare_latex_document(tex, "en")

        self.assertIn("\\providecommand{\\Beta}{\\mathrm{B}}", prepared.tex)
        self.assertIn("Added fallback LaTeX math compatibility commands.", "\n".join(prepared.notes))

        engines_available = get_preferred_latex_engines(prepared.tex, "en")
        if not engines_available:
            self.skipTest("No LaTeX engine is available in PATH.")
        pdf_bytes, diagnostics = latex_to_pdf(prepared.tex, engines_available[0])

        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertNotIn("Undefined control sequence", diagnostics)
        self.assertFalse(latex_output_has_missing_characters(diagnostics))

    def test_prepare_latex_document_removes_fragile_enumerate_options(self):
        tex = (
            "\\documentclass{article}\n"
            "\\begin{document}\n"
            "\\begin{enumerate}[label=\\alph*)]\n"
            "\\item First property.\n"
            "\\item Second property.\n"
            "\\end{enumerate}\n"
            "\\end{document}\n"
        )

        prepared = prepare_latex_document(tex, "en")

        self.assertIn("\\begin{enumerate}", prepared.tex)
        self.assertNotIn("\\begin{enumerate}[label=\\alph*)]", prepared.tex)
        self.assertIn("Removed fragile LaTeX enumerate options.", "\n".join(prepared.notes))

    def test_prepare_latex_document_closes_unfinished_list_before_box_end(self):
        prepared = build_prepared_document(
            "ru",
            "Списки",
            "Проверка восстановления структуры.",
            "\\begin{thmbox}[Содержание]\n"
            "\\begin{enumerate}\n"
            "\\item Первый раздел\n"
            "\\begin{itemize}\n"
            "\\item Подраздел\n"
            "\\end{itemize}\n"
            "\\end{thmbox}\n",
        )

        expected = "\\end{itemize}\n\\end{enumerate}\n\\end{thmbox}"
        self.assertIn(expected, prepared.tex)
        self.assertIn(
            "Repaired unbalanced LaTeX list environments before structural boundaries.",
            prepared.notes,
        )

    def test_prepare_latex_document_repairs_math_in_box_title(self):
        prepared = build_prepared_document(
            "ru",
            "Метрические пространства",
            "Определения метрик.",
            "\\begin{defbox}[L^{p}\\text{-метрика в }\\mathbb{R}^{n}]\nОпределение.\n\\end{defbox}\n",
        )

        self.assertIn(
            "\\begin{defbox}[\\(L^{p}\\)-метрика в \\(\\mathbb{R}^{n}\\)]",
            prepared.tex,
        )
        self.assertNotIn("L^{p}\\text", prepared.tex)

    def test_prepare_latex_document_wraps_orphan_items(self):
        prepared = build_prepared_document(
            "ru",
            "Структура лекции",
            "Восстановление списка.",
            "\\subsection*{Первый раздел}\n"
            "Вводный текст.\n"
            "\\item Первый пункт.\n"
            "\\item Второй пункт.\n"
            "\\subsection*{Следующий раздел}\n"
            "Продолжение.\n",
        )

        expected = (
            "Вводный текст.\n"
            "\\begin{itemize}\n"
            "\\item Первый пункт.\n"
            "\\item Второй пункт.\n"
            "\\end{itemize}\n"
            "\\subsection*{Следующий раздел}"
        )
        self.assertIn(expected, prepared.tex)

        engines_available = get_preferred_latex_engines(prepared.tex, "en")
        if not engines_available:
            self.skipTest("No LaTeX engine is available in PATH.")
        pdf_bytes, diagnostics = latex_to_pdf(prepared.tex, engines_available[0])

        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertNotIn("Counter too large", diagnostics)
        self.assertFalse(latex_output_has_missing_characters(diagnostics))

    def test_prepare_latex_document_expands_short_tabular_column_spec(self):
        prepared = build_prepared_document(
            "en",
            "Relation Table",
            "Short abstract.",
            "\\section{Table}\n"
            "\\begin{tabular}{lcc}\n"
            "Property & Reflexive & Symmetric & Transitive \\\\\n"
            "R & Yes & No & Yes \\\\\n"
            "\\end{tabular}\n",
        )

        self.assertIn("\\begin{tabular}{lccc}", prepared.tex)
        self.assertIn("Expanded underspecified LaTeX tabular column definitions.", "\n".join(prepared.notes))

        engines_available = get_preferred_latex_engines(prepared.tex, "en")
        if not engines_available:
            self.skipTest("No LaTeX engine is available in PATH.")
        pdf_bytes, diagnostics = latex_to_pdf(prepared.tex, engines_available[0])

        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertFalse(latex_output_has_missing_characters(diagnostics))

    def test_english_document_with_unicode_math_compiles(self):
        engines_available = get_preferred_latex_engines("plain", "en")
        if not engines_available:
            self.skipTest("No LaTeX engine is available in PATH.")

        prepared = build_prepared_document(
            "en",
            "Study of Time and Space",
            "We compare \u03c4\u2082 \u2264 5 while keeping $x_2$ intact.",
            "\\section{Overview}\nEnglish text with τ and CO₂ outside math, plus $τ₂ + 1$ inside math.\n",
        )
        engine = get_preferred_latex_engines(prepared.tex, "en")[0]
        pdf_bytes, diagnostics = latex_to_pdf(prepared.tex, engine)

        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertGreater(len(pdf_bytes), 1000)
        self.assertFalse(latex_output_has_missing_characters(diagnostics))

    def test_russian_document_with_unicode_math_compiles(self):
        engines_available = get_preferred_latex_engines("plain", "ru")
        if not engines_available:
            self.skipTest("No LaTeX engine is available in PATH.")

        prepared = build_prepared_document(
            "ru",
            "\u0418\u0437\u0443\u0447\u0435\u043d\u0438\u0435 \u0432\u0440\u0435\u043c\u0435\u043d\u0438 \u0438 \u043f\u0440\u043e\u0441\u0442\u0440\u0430\u043d\u0441\u0442\u0432\u0430",
            "\u0420\u0430\u0441\u0441\u043c\u0430\u0442\u0440\u0438\u0432\u0430\u0435\u043c \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440 \u03c4 \u0438 \u043c\u043e\u043b\u0435\u043a\u0443\u043b\u0443 CO\u2082, \u0433\u0434\u0435 \u03c4\u2082 \u2264 5.",
            "\\section{\u041e\u0441\u043d\u043e\u0432\u043d\u0430\u044f \u0438\u0434\u0435\u044f}\n"
            "\u041f\u0443\u0441\u0442\u044c \u03c4\u2082 \u2264 5 \u0438 x\u2082 = 3. "
            "\u0422\u043e\u0433\u0434\u0430 \u0440\u0430\u0441\u0441\u043c\u043e\u0442\u0440\u0438\u043c $τ₂ + 1 \\\\ge 0$ "
            "\u0438 \u043c\u043d\u043e\u0436\u0435\u0441\u0442\u0432\u043e ℝ.\n",
        )
        engine = get_preferred_latex_engines(prepared.tex, "ru")[0]
        pdf_bytes, diagnostics = latex_to_pdf(prepared.tex, engine)

        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertGreater(len(pdf_bytes), 1000)
        self.assertFalse(latex_output_has_missing_characters(diagnostics))


if __name__ == "__main__":
    unittest.main()
