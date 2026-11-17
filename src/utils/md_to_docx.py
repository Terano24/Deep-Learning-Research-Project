"""
Build the Elsevier submission files from docs/MANUSCRIPT.md.

Computational Biology and Chemistry's Guide for authors does not accept a PDF as the
manuscript source: it asks for an editable .doc/.docx in a single-column layout, with the
Highlights and the title page as separate files. This writes all three, plus the figures
under the Figure_N names the guide asks for.

Run:  python -m src.utils.md_to_docx            (writes docs/submission/)
"""
import re
import shutil
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

BASE = Path(__file__).resolve().parent.parent.parent
MD = BASE / 'docs' / 'MANUSCRIPT.md'
OUT = BASE / 'docs' / 'submission'
FIG = BASE / 'docs' / 'figures'

# What the submission system and the title page need that the manuscript body does not carry.
AUTHORS = [
    ('I Nyoman Agung Ananda Terano', 'a'),
    ('Mirrabell Frederica Hadiwijono', 'b'),
]
AFFILIATIONS = {
    'a': 'Department of Information Systems, Faculty of Intelligent Electrical and Informatics '
         'Technology, Institut Teknologi Sepuluh Nopember, Kampus ITS Sukolilo, Jl. Raya ITS, '
         'Keputih, Surabaya 60111, Indonesia',
    'b': 'Faculty of Applied Science and Engineering, University of Toronto, Toronto, '
         'Ontario M5S 2L6, Canada',
}
CORRESPONDING = ('I Nyoman Agung Ananda Terano', 'teranoace@gmail.com')

FIGURES = {1: 'fig1_dataset_composition.png', 2: 'fig2_ceiling_and_performance.png'}

TOKEN = re.compile(r'(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`)')


def base_doc(spacing=2.0):
    d = Document()
    s = d.sections[0]
    s.page_width, s.page_height = Cm(21.0), Cm(29.7)
    s.left_margin = s.right_margin = Cm(3.0)     # guide: wide (3 cm) margins
    s.top_margin = s.bottom_margin = Cm(2.5)
    n = d.styles['Normal']
    n.font.name = 'Times New Roman'
    n.font.size = Pt(12)
    n.element.rPr.rFonts.set(qn('w:eastAsia'), 'Times New Roman')
    n.paragraph_format.line_spacing = spacing    # guide: double spacing
    n.paragraph_format.space_after = Pt(0)
    for name, size in (('Heading 1', 14), ('Heading 2', 13), ('Heading 3', 12)):
        h = d.styles[name]
        h.font.name = 'Times New Roman'
        h.font.size = Pt(size)
        h.font.bold = True
        h.font.color.rgb = None
        h.element.rPr.rFonts.set(qn('w:eastAsia'), 'Times New Roman')
        h.paragraph_format.space_before = Pt(12)
        h.paragraph_format.space_after = Pt(4)
    add_page_numbers(s)
    return d


def add_page_numbers(section):
    """Guide: number all pages consecutively."""
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run()
    for kind, text in (('begin', None), (None, 'PAGE'), ('end', None)):
        if kind:
            el = OxmlElement('w:fldChar')
            el.set(qn('w:fldCharType'), kind)
        else:
            el = OxmlElement('w:instrText')
            el.set(qn('xml:space'), 'preserve')
            el.text = text
        r._r.append(el)


def runs(par, text, bold=False, italic=False):
    """Inline markdown -> runs. Line breaks inside a paragraph become spaces."""
    text = re.sub(r'\s*\n\s*', ' ', text)
    for tok in TOKEN.split(text):
        if not tok:
            continue
        if tok.startswith('**') and tok.endswith('**'):
            r = par.add_run(tok[2:-2]); r.bold = True; r.italic = italic
        elif tok.startswith('`') and tok.endswith('`'):
            r = par.add_run(tok[1:-1]); r.font.name = 'Courier New'; r.font.size = Pt(10)
            r.bold = bold; r.italic = italic
        elif tok.startswith('*') and tok.endswith('*') and len(tok) > 2:
            r = par.add_run(tok[1:-1]); r.italic = True; r.bold = bold
        else:
            r = par.add_run(tok); r.bold = bold; r.italic = italic


def add_table(d, rows):
    t = d.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = 'Table Grid'
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            c = t.cell(i, j)
            p = c.paragraphs[0]
            p.paragraph_format.line_spacing = 1.0
            runs(p, cell.strip(), bold=(i == 0))
            for r in p.runs:
                r.font.size = Pt(9)
    d.add_paragraph().paragraph_format.line_spacing = 1.0


def split_row(line):
    return [c for c in line.strip().strip('|').split('|')]


def blocks(lines):
    """Yield (kind, payload) for headings, tables, bullets, numbered items and paragraphs."""
    i, buf = 0, []

    def flush():
        nonlocal buf
        if buf:
            yield ('p', ' '.join(s.strip() for s in buf))
            buf = []

    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            yield from flush(); i += 1; continue
        m = re.match(r'^(#{1,3}) (.*)', ln)
        if m:
            yield from flush(); yield ('h%d' % len(m.group(1)), m.group(2).strip()); i += 1; continue
        if ln.lstrip().startswith('|'):
            yield from flush()
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith('|'):
                if not re.match(r'^\s*\|[\s:|-]+\|\s*$', lines[i]):
                    rows.append(split_row(lines[i]))
                i += 1
            yield ('table', rows); continue
        m = re.match(r'^\s*[-*] (.*)', ln)
        if m:
            yield from flush()
            item = [m.group(1)]; i += 1
            while i < len(lines) and lines[i].startswith('  ') and lines[i].strip() \
                    and not re.match(r'^\s*[-*] ', lines[i]):
                item.append(lines[i].strip()); i += 1
            yield ('bullet', ' '.join(item)); continue
        m = re.match(r'^\s*(\d+)\. (.*)', ln)
        if m:
            yield from flush()
            item = [m.group(2)]; i += 1
            while i < len(lines) and lines[i].startswith('   ') and lines[i].strip():
                item.append(lines[i].strip()); i += 1
            yield ('num', (m.group(1), ' '.join(item))); continue
        buf.append(ln); i += 1
    yield from flush()


def section_of(md_text, title):
    m = re.search(r'^## %s\s*$(.*?)(?=^## |\Z)' % re.escape(title), md_text, re.S | re.M)
    return m.group(1).strip() if m else ''


def build_manuscript(md_text, title):
    d = base_doc()
    body = md_text.split('\n', 1)[1]                      # drop the '# title' line
    # Highlights go in their own file; the byline and affiliations go on the title page.
    body = re.sub(r'^## Highlights\s*$.*?(?=^## )', '', body, flags=re.S | re.M)
    body = re.sub(r'^\*\*I Nyoman.*?\n', '', body, flags=re.M)
    body = re.sub(r'^\^¹.*?\n', '', body, flags=re.M)
    d.add_paragraph().add_run(title).bold = True
    d.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    in_figures = False
    for kind, p in blocks(body.split('\n')):
        if kind in ('h1', 'h2', 'h3'):
            level = int(kind[1])
            in_figures = (p == 'Figures')
            d.add_heading(p, level=level)
        elif kind == 'table':
            add_table(d, p)
        elif kind == 'bullet':
            par = d.add_paragraph(style='List Bullet'); runs(par, p)
        elif kind == 'num':
            par = d.add_paragraph(style='List Number'); runs(par, p[1])
        else:
            par = d.add_paragraph(); runs(par, p)
            if in_figures:
                m = re.match(r'\*\*Figure (\d)\.', p)
                if m and FIGURES.get(int(m.group(1))):
                    img = d.add_paragraph(); img.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    img.add_run().add_picture(str(FIG / FIGURES[int(m.group(1))]), width=Cm(15))
    return d


def build_highlights(md_text):
    items = re.findall(r'^- (.*)$', section_of(md_text, 'Highlights'), re.M)
    assert 3 <= len(items) <= 5 and all(len(x) <= 85 for x in items), items
    d = base_doc(spacing=1.15)
    d.add_paragraph().add_run('Highlights').bold = True
    for x in items:
        d.add_paragraph(x, style='List Bullet')
    return d, items


def build_title_page(title):
    d = base_doc(spacing=1.5)
    p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(title); r.bold = True; r.font.size = Pt(16)
    p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for k, (name, sup) in enumerate(AUTHORS):
        if k:
            p.add_run(', ')
        p.add_run(name)
        p.add_run(sup + (',*' if name == CORRESPONDING[0] else '')).font.superscript = True
    for sup, text in AFFILIATIONS.items():
        p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run(sup).font.superscript = True
        p.add_run(' ' + text)
    d.add_paragraph()
    p = d.add_paragraph()
    p.add_run('*').font.superscript = True
    p.add_run(' Corresponding author: %s. E-mail: %s' % CORRESPONDING)
    return d


# Wording follows the publisher's sample cover letter (sample_cover_letter2018.doc)
# sentence for sentence; only its blanks and placeholders are filled in. Each entry is
# one paragraph; "\n" is a line break inside it, as in the sample.
COVER_LETTER = [
    'Dear Editor,',
    'We wish to submit an original research article entitled \u201c{title}\u201d for '
    'consideration by Computational Biology and Chemistry.\n'
    'We confirm that this work is original and has not been published elsewhere, nor is it '
    'currently under consideration for publication elsewhere.',
    'In this paper, we show that the accuracy published predictors report for T-cell epitope '
    'immunogenicity falls sharply when the test data keep each publication\u2019s results '
    'together, whereas holding out individual peptides, the usual safeguard, barely changes it. '
    'This is significant because researchers who rely on the reported scores may overestimate '
    'how well these tools will work on new peptides.\n'
    'We believe that this manuscript is appropriate for publication by Computational Biology '
    'and Chemistry because it evaluates machine learning models for biological data, one of the '
    'areas named in the journal\u2019s aims and scope, and recommends how such models should be '
    'tested.',
    'The problem is that benchmark scores in this field are hard to compare, because they '
    'depend on how the data were split and on labels that different laboratories do not '
    'reproduce consistently. We measure how far the labels agree between laboratories, and we '
    'find that a baseline that sees only the HLA allele, and never the peptide, beats the '
    'published tools we tested. Readers who build or rely on epitope predictors will want to '
    'know this, and we release the curated benchmark, data splits, predictions and code '
    '(doi:10.5281/zenodo.23936440) so that others can check and reuse the work.',
    'We have no conflicts of interest to disclose.',
    'Please address all correspondence concerning this manuscript to me at teranoace@gmail.com.',
    'Thank you for your consideration of this manuscript.',
    'Sincerely,',
    'I Nyoman Agung Ananda Terano',
]


def build_cover_letter(title):
    import datetime
    d = base_doc(spacing=1.0)
    for sec in d.sections:
        sec.top_margin = sec.bottom_margin = Cm(2.0)
    date = datetime.date.today().strftime('%d %B %Y').lstrip('0')
    # Sender block, right-aligned, as in the sample: name, affiliation, address.
    sender = ['I Nyoman Agung Ananda Terano',
              'Department of Information Systems, Faculty of Intelligent Electrical and '
              'Informatics Technology, Institut Teknologi Sepuluh Nopember',
              'Kampus ITS Sukolilo, Jl. Raya ITS, Keputih, Surabaya 60111, Indonesia']
    for line in sender:
        p = d.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        p.paragraph_format.space_after = Pt(0)
        p.add_run(line)
    d.add_paragraph().paragraph_format.space_after = Pt(6)
    p = d.add_paragraph()
    p.paragraph_format.space_after = Pt(9)
    p.add_run(date)
    for t in COVER_LETTER:
        p = d.add_paragraph()
        p.paragraph_format.space_after = Pt(9)
        lines = t.format(title=title).split('\n')
        for k, line in enumerate(lines):
            r = p.add_run(line)
            if k < len(lines) - 1:
                r.add_break()
    return d


def main():
    md = MD.read_text(encoding='utf-8')
    title = re.sub(r'[*_`]', '', md.split('\n', 1)[0][2:]).strip()
    OUT.mkdir(exist_ok=True)
    build_manuscript(md, title).save(OUT / 'Manuscript.docx')
    h, items = build_highlights(md)
    h.save(OUT / 'Highlights.docx')
    build_title_page(title).save(OUT / 'Title_page.docx')
    build_cover_letter(title).save(OUT / 'Cover_letter.docx')
    for n, f in FIGURES.items():
        shutil.copy(FIG / f, OUT / f'Figure_{n}.png')
    shutil.copy(FIG / 'graphical_abstract.tif', OUT / 'Graphical_abstract.tif')
    print('wrote', sorted(p.name for p in OUT.iterdir()))
    print('highlight lengths', [len(x) for x in items])


if __name__ == '__main__':
    sys.exit(main())
