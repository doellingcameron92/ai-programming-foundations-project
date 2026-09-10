"""Render module_summary.md to module_summary.pdf.

Keeping the report in Markdown under version control and generating the PDF with
a script means the submitted PDF can always be rebuilt from the tracked source.
Run: python make_pdf.py
"""

import html
import re
from pathlib import Path

from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "module_summary.md"
OUTPUT = ROOT / "module_summary.pdf"

FIGURES = [
    ("figures/figure1_quality_distribution.png",
     "Figure 1: Distribution of wine quality scores by wine colour."),
    ("figures/figure2_correlation_heatmap.png",
     "Figure 2: Correlation between chemical properties and quality."),
    ("figures/figure3_alcohol_vs_quality.png",
     "Figure 3: Alcohol content rises with wine quality score."),
    ("figures/figure4_alcohol_vs_volatile_acidity.png",
     "Figure 4: Higher alcohol and lower volatile acidity accompany better scores."),
]

styles = getSampleStyleSheet()
BODY = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=10.5, leading=15,
                      spaceAfter=9, alignment=TA_JUSTIFY)
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=17, leading=21, spaceAfter=12)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, leading=17,
                    spaceBefore=14, spaceAfter=7)
REF = ParagraphStyle("Ref", parent=BODY, leftIndent=22, firstLineIndent=-22, alignment=0)
CAPTION = ParagraphStyle("Caption", parent=BODY, fontSize=9.5, alignment=1,
                         textColor="#444444", spaceBefore=4, spaceAfter=16)


def inline(text: str) -> str:
    """Convert inline Markdown to the mini-HTML reportlab understands."""
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"`([^`]+?)`", r'<font face="Courier" size="9.5">\1</font>', text)
    text = re.sub(r"(https?://\S+?)(?=[.,)]?(?:\s|$))",
                  r'<link href="\1" color="#1a4f8a">\1</link>', text)
    return text


def build_story():
    lines = SOURCE.read_text().splitlines()
    story, paragraph, in_references = [], [], False

    def flush():
        if paragraph:
            text = inline(" ".join(paragraph))
            story.append(Paragraph(text, REF if in_references else BODY))
            paragraph.clear()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush()
        elif stripped.startswith("## "):
            flush()
            in_references = stripped[3:].strip().lower() == "references"
            story.append(Paragraph(inline(stripped[3:]), H2))
        elif stripped.startswith("# "):
            flush()
            story.append(Paragraph(inline(stripped[2:]), H1))
        else:
            paragraph.append(stripped)
    flush()

    story.append(PageBreak())
    story.append(Paragraph("Appendix: Figures Referenced in This Report", H2))
    story.append(Spacer(1, 6))
    for path, caption in FIGURES:
        image = Image(str(ROOT / path))
        scale = min(6.2 * inch / image.imageWidth, 4.0 * inch / image.imageHeight)
        image.drawWidth = image.imageWidth * scale
        image.drawHeight = image.imageHeight * scale
        image.hAlign = "CENTER"
        story.append(image)
        story.append(Paragraph(inline(caption), CAPTION))
    return story


def main() -> None:
    doc = SimpleDocTemplate(
        str(OUTPUT), pagesize=LETTER,
        leftMargin=1 * inch, rightMargin=1 * inch,
        topMargin=0.9 * inch, bottomMargin=0.9 * inch,
        title="Module Summary - Wine Quality Data Workflow",
        author="Cameron Doelling",
    )
    doc.build(build_story())
    print(f"wrote {OUTPUT.name} ({OUTPUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
