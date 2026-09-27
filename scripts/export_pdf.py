"""Export the self-contained report with consistent math and private metadata."""
import json
from pathlib import Path
import re

import pypandoc
from pypdf import PdfReader, PdfWriter
from pypdf.generic import NameObject
import typst


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/vaegan_mnist_report.md"
OUTPUT = ROOT / "output/pdf/vaegan_mnist_report.pdf"
SCRATCH = ROOT / "tmp/pdfs"

STYLE = r'''
#set document(title: "基于二维 VAE-GAN 的手写数字生成", author: (), date: none)
#set page(paper: "a4", margin: (top: 19mm, bottom: 19mm, left: 22mm, right: 22mm),
  numbering: "1", number-align: center)
#set text(font: ("Cambria", "SimSun"), size: 10.5pt, lang: "zh", region: "CN")
#set par(justify: true, leading: 0.62em, spacing: 0.85em)
#set heading(numbering: none, outlined: true)
#show heading.where(level: 1): it => {
  set text(font: ("Cambria", "SimHei"), size: 18pt, weight: "bold")
  align(center, it)
  v(5pt)
}
#show heading.where(level: 2): set text(font: ("Cambria", "SimHei"), size: 13pt)
#show heading.where(level: 3): set text(font: ("Cambria", "SimHei"), size: 11.5pt)
#show math.equation: set text(font: "Cambria Math", size: 1em, weight: "regular")
#show math.equation.where(block: true): set block(above: 0.85em, below: 0.85em)
#show link: set text(fill: black)
#set table(inset: (x: 5pt, y: 5pt), stroke: (x: none, y: 0.35pt + luma(75%)))
#show table: set text(size: 9.5pt)
#show table: set par(leading: 0.5em)
#show table.cell.where(y: 0): set text(weight: "bold")
#show figure.where(kind: table): set block(breakable: false)
#set raw(theme: none)
#show raw: set text(font: "Consolas")
#show raw.where(block: false): set text(size: 0.9em)
#show raw.where(block: true): it => {
  set text(size: 8.4pt)
  set par(justify: false, leading: 0.35em, spacing: 0pt)
  block(breakable: true, inset: (y: 5pt), it)
}
'''


def convert(value, source_format="json"):
    result = pypandoc.convert_text(value, "typst", format=source_format,
                                  extra_args=["--wrap=none"])
    # Normalize Windows subprocess newlines before writing and compiling.
    return result.replace("\r\n", "\n").replace("\r", "\n")


def build():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    report = SOURCE.read_text(encoding="utf-8")
    embedded = report.split("```python\n")[1].split("```")[0]
    assert embedded == (ROOT / "src/vaegan.py").read_text(encoding="utf-8")
    assert len(embedded.splitlines()) <= 800
    assert not re.search(r"(?<![A-Za-z])[A-Za-z]:[\\/]", report)
    doc = json.loads(pypandoc.convert_text(report, "json", format="markdown"))

    def render(blocks):
        return convert(json.dumps({**doc, "blocks": blocks}))

    blocks = doc["blocks"]
    output = []
    i = 0
    table_index = 0
    widths = [(0.22, 0.78), (0.25, 0.29, 0.46),
              (0.43, 0.19, 0.19, 0.19), (0.30, 0.175, 0.175, 0.175, 0.175),
              (0.12, 0.22, 0.22, 0.22, 0.22)]
    while i < len(blocks):
        block = blocks[i]
        if block["t"] == "Figure":
            # The report supplies numbered captions in the following paragraph.
            image = block["c"][2][0]["c"][0]
            target = image["c"][2][0]
            filename = Path(target).name
            caption = render([blocks[i + 1]]).strip()
            limits = {"latent_grid.png": "151mm", "latent_scatter.png": "112mm"}
            height = limits.get(filename)
            width = ("80%" if filename in {"ablation_comparison.png", "prior_comparison.png",
                                          "reconstructions.png"} else "100%")
            dimensions = (f'width: 100%, height: {height}, fit: "contain"'
                          if height else f'width: {width}')
            figure = ('#block(breakable: false)[\n'
                      f'#align(center, image("../../results/{filename}", '
                      f'{dimensions}))\n'
                      '#v(3pt)\n#text(size: 9.5pt)[' + caption + ']\n]\n')
            output.append({"t": "RawBlock", "c": ["typst", figure]})
            i += 2
            continue
        if block["t"] == "Table":
            for col, width in zip(block["c"][2], widths[table_index]):
                col[1] = {"t": "ColWidth", "c": width}
                if table_index < 2:
                    col[0] = {"t": "AlignLeft"}
            table_index += 1
        if block["t"] == "Header":
            title = "".join(x.get("c", "") for x in block["c"][2] if x["t"] == "Str")
            if title.startswith(("附录", "3.3")):
                output.append({"t": "RawBlock", "c": ["typst", "#pagebreak()"]})
        if block["t"] == "CodeBlock" and "python" in block["c"][0][1]:
            output.append({"t": "RawBlock", "c": ["typst", "=== 完整代码"]})
        output.append(block)
        i += 1
    body = render(output)
    # Pandoc's Typst writer maps TeX negative thin spaces to an entire em.
    body = body.replace("#h(-1em)", "#h(-0.1667em)")
    body = body.replace("dot.circle", "⊙")
    typ_path = SCRATCH / "report.typ"
    typ_path.write_text(STYLE + body, encoding="utf-8", newline="\n")
    raw_pdf = SCRATCH / "report_raw.pdf"
    try:
        typst.compile(str(typ_path), output=str(raw_pdf), root=str(ROOT))
    except typst.TypstError as error:
        diagnostic = getattr(error, "diagnostic", str(error))
        print(diagnostic.replace(str(ROOT), ".").encode("ascii", "backslashreplace").decode())
        raise SystemExit("PDF compilation failed.") from None
    reader = PdfReader(raw_pdf)
    writer = PdfWriter()
    writer.clone_document_from_reader(reader)
    # Keep only the non-identifying document title. Remove XMP and document IDs.
    writer.metadata = {"/Title": "基于二维 VAE-GAN 的手写数字生成"}
    writer.root_object.pop(NameObject("/Metadata"), None)
    writer._ID = None
    writer.page_mode = "/UseOutlines"
    with OUTPUT.open("wb") as stream:
        writer.write(stream)
    final = PdfReader(OUTPUT)
    assert final.outline and final.outline[0].title == final.metadata.title
    assert set(final.metadata) == {"/Title"}
    assert "/Metadata" not in final.root_object
    assert "/EmbeddedFiles" not in final.root_object.get("/Names", {})
    print(f"PDF exported: {len(final.pages)} pages; title bookmark and metadata verified.")


if __name__ == "__main__":
    build()
