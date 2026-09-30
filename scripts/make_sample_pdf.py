"""Convert data/samples/Kebijakan_SDM_2025.txt into a simple PDF (dev-only script)."""

from pathlib import Path

from fpdf import FPDF

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"
SRC = SAMPLES / "Kebijakan_SDM_2025.txt"
DST = SAMPLES / "Kebijakan_SDM_2025.pdf"


def main() -> None:
    """Write each line of the source text into the PDF, preserving line breaks."""
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_margins(20, 20, 20)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)

    for line in SRC.read_text(encoding="utf-8").splitlines():
        # Core fonts are latin-1 only; replace anything outside it.
        safe = line.replace("–", "-").encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, 6, safe or " ", new_x="LMARGIN", new_y="NEXT")

    pdf.output(str(DST))
    print(f"Wrote {DST}")


if __name__ == "__main__":
    main()
