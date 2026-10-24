"""
Render a Markdown manuscript to PDF.

Written against reportlab only, because pandoc / weasyprint are not available here.
Handles the subset of Markdown the manuscript uses: ATX headings, paragraphs, pipe
tables, bullet and numbered lists, bold/italic/code spans, horizontal rules and
blockquotes. A Unicode TrueType font is registered so Greek letters, superscripts and
arrows (kappa, CD8+, ->, x, ~) survive; reportlab's built-in Helvetica is WinAnsi and
would drop them.

Usage:
    python -m src.utils.md_to_pdf docs/MANUSCRIPT.md docs/MANUSCRIPT.pdf
"""
import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, HRFlowable, PageBreak,
                                PageTemplate, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

FONT_CANDIDATES = [
    ('DejaVuSans', 'DejaVuSans-Bold', 'DejaVuSans-Oblique',
     ['C:/Windows/Fonts/DejaVuSans.ttf']),
    ('Arial', 'Arial-Bold', 'Arial-Italic',
     ['C:/Windows/Fonts/arial.ttf']),
    ('Calibri', 'Calibri-Bold', 'Calibri-Italic',
     ['C:/Windows/Fonts/calibri.ttf']),
]


def register_font():
    """Return (regular, bold, italic) font names, preferring a Unicode TTF.

    Times New Roman first: the manuscript is set in a serif face for submission.
    """
    pairs = [
        ('TimesNewRoman', 'C:/Windows/Fonts/times.ttf', 'C:/Windows/Fonts/timesbd.ttf',
         'C:/Windows/Fonts/timesi.ttf'),
        ('Arial', 'C:/Windows/Fonts/arial.ttf', 'C:/Windows/Fonts/arialbd.ttf',
         'C:/Windows/Fonts/ariali.ttf'),
        ('DejaVuSans', 'C:/Windows/Fonts/DejaVuSans.ttf',
         'C:/Windows/Fonts/DejaVuSans-Bold.ttf', 'C:/Windows/Fonts/DejaVuSans-Oblique.ttf'),
    ]
    for name, reg, bold, ital in pairs:
        if Path(reg).exists():
            try:
                pdfmetrics.registerFont(TTFont(name, reg))
                b = f'{name}-Bold' if Path(bold).exists() else name
                i = f'{name}-Italic' if Path(ital).exists() else name
                if Path(bold).exists():
                    pdfmetrics.registerFont(TTFont(b, bold))
                if Path(ital).exists():
                    pdfmetrics.registerFont(TTFont(i, ital))
                pdfmetrics.registerFontFamily(name, normal=name, bold=b, italic=i, boldItalic=b)
                return name, b, i
            except Exception:
                continue
    return 'Helvetica', 'Helvetica-Bold', 'Helvetica-Oblique'


REG, BOLD, ITAL = register_font()


def esc(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def inline(s):
    """Markdown inline spans -> reportlab mini-HTML."""
    s = esc(s)
    s = re.sub(r'\*\*\*(.+?)\*\*\*', r'<b><i>\1</i></b>', s)
    s = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', s)
    s = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<i>\1</i>', s)
    # inline code set in the document's own italic, so no second font family is embedded
    s = re.sub(r'`(.+?)`', r'<i>\1</i>', s)
    # links stay black: the manuscript is printed monochrome
    s = re.sub(r'\[(.+?)\]\((.+?)\)', r'<link href="\2" color="black">\1</link>', s)
    return s


def styles(compact=False):
    """compact=True tightens type for two-column layout, where measure is ~8 cm."""
    ss = getSampleStyleSheet()
    f = 8.4 if compact else 9.6
    base = dict(fontName=REG, leading=f * 1.32, fontSize=f)
    return {
        'title': ParagraphStyle('t', parent=ss['Title'], fontName=BOLD,
                                fontSize=15 if compact else 16,
                                leading=20, spaceAfter=10, alignment=TA_CENTER),
        'authors': ParagraphStyle('a', fontName=REG, fontSize=11, leading=15,
                                  alignment=TA_CENTER, spaceAfter=4),
        'h1': ParagraphStyle('h1', fontName=BOLD, fontSize=10.5 if compact else 13,
                             leading=13 if compact else 16,
                             spaceBefore=16, spaceAfter=7, textColor=colors.black),
        'h2': ParagraphStyle('h2', fontName=BOLD, fontSize=9.4 if compact else 11,
                             leading=12 if compact else 14,
                             spaceBefore=12, spaceAfter=5, textColor=colors.black),
        'h3': ParagraphStyle('h3', fontName=BOLD, fontSize=8.8 if compact else 10,
                             leading=11.5 if compact else 13,
                             spaceBefore=9, spaceAfter=4),
        'body': ParagraphStyle('b', alignment=TA_JUSTIFY, spaceAfter=6, **base),
        'li': ParagraphStyle('li', alignment=TA_JUSTIFY, leftIndent=14, bulletIndent=4,
                             spaceAfter=3, bulletFontName=REG, bulletFontSize=9.6, **base),
        'quote': ParagraphStyle('q', leftIndent=16, rightIndent=10, fontName=ITAL,
                                fontSize=9.4, leading=13, spaceAfter=6,
                                textColor=colors.black),
        'affil': ParagraphStyle('af', fontName=REG, fontSize=6.6 if compact else 7.6,
                                leading=8.4 if compact else 9.6, spaceBefore=3,
                                textColor=colors.black),
        'cell': ParagraphStyle('c', fontName=REG, fontSize=6.8 if compact else 8.2,
                               leading=8.8 if compact else 10.5,
                               textColor=colors.black),
        'cellh': ParagraphStyle('ch', fontName=BOLD, fontSize=6.8 if compact else 8.2,
                                leading=8.8 if compact else 10.5,
                                textColor=colors.black),
    }


def split_row(line):
    return [c.strip() for c in line.strip().strip('|').split('|')]


def build_table(rows, st, width):
    header, body = rows[0], rows[1:]
    ncol = max(len(r) for r in rows)
    data = []
    for i, r in enumerate(rows):
        r = r + [''] * (ncol - len(r))
        sty = st['cellh'] if i == 0 else st['cell']
        data.append([Paragraph(inline(c), sty) for c in r])
    t = Table(data, colWidths=[width / ncol] * ncol, repeatRows=1, hAlign='LEFT')
    # monochrome: the header is separated by a rule, not a fill
    t.setStyle(TableStyle([
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.black),
        ('LINEBELOW', (0, 0), (-1, 0), 1.0, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    return t


class DuplexDoc(BaseDocTemplate):
    """Mirrored margins for double-sided printing.

    Odd (right-hand) pages get the wider margin on the left, even pages on the right, so
    the binding edge is consistent once the sheets are bound. Folios sit on the outer
    corner, as in a printed book.
    """

    def __init__(self, filename, pagesize=A4, inner=2.6 * cm, outer=1.9 * cm,
                 vmargin=2.0 * cm, title=None):
        BaseDocTemplate.__init__(self, filename, pagesize=pagesize, title=title,
                                 leftMargin=inner, rightMargin=outer,
                                 topMargin=vmargin, bottomMargin=vmargin)
        w = pagesize[0] - inner - outer
        h = pagesize[1] - 2 * vmargin
        self.addPageTemplates([
            PageTemplate(id='odd', frames=[Frame(inner, vmargin, w, h, id='f')],
                         onPage=self._folio),
            PageTemplate(id='even', frames=[Frame(outer, vmargin, w, h, id='f')],
                         onPage=self._folio),
        ])

    def _folio(self, canvas, doc):
        n = canvas.getPageNumber()
        if n <= 1:
            return                                   # no folio on the title page
        canvas.saveState()
        canvas.setFont(REG, 8.5)
        pw = self.pagesize[0]
        x = pw - self.rightMargin if n % 2 else self.leftMargin
        canvas.drawRightString(x, self.bottomMargin * 0.55, str(n)) if n % 2 else \
            canvas.drawString(x, self.bottomMargin * 0.55, str(n))
        canvas.restoreState()

    def handle_pageBegin(self):
        self._handle_pageBegin()
        # page 1 is odd; alternate from there
        self._handle_nextPageTemplate('even' if self.page % 2 else 'odd')


class TwoColumnDoc(BaseDocTemplate):
    """Two-column layout, as used by machine-learning preprints.

    Text fills the left column, then continues in the right. The title block spans the
    full width, so the first page uses a one-frame template and the rest use two.
    Optionally mirrors margins for double-sided printing.
    """

    def __init__(self, filename, pagesize=A4, vmargin=2.0 * cm, gutter=0.7 * cm,
                 inner=2.0 * cm, outer=2.0 * cm, duplex=False, title=None,
                 banner_h=3.4 * cm):
        BaseDocTemplate.__init__(self, filename, pagesize=pagesize, title=title,
                                 leftMargin=inner, rightMargin=outer,
                                 topMargin=vmargin, bottomMargin=vmargin)
        self._mirror = duplex
        pw, ph = pagesize
        h = ph - 2 * vmargin
        total = pw - inner - outer
        colw = (total - gutter) / 2.0
        self.colw = colw

        def cols(left):
            return [Frame(left, vmargin, colw, h, id='c1', leftPadding=0, rightPadding=0),
                    Frame(left + colw + gutter, vmargin, colw, h, id='c2',
                          leftPadding=0, rightPadding=0)]

        # first page: a full-width banner for title/authors/abstract, then two columns
        first = [Frame(inner, ph - vmargin - banner_h, total, banner_h, id='banner',
                       leftPadding=0, rightPadding=0),
                 Frame(inner, vmargin, colw, h - banner_h, id='c1',
                       leftPadding=0, rightPadding=0),
                 Frame(inner + colw + gutter, vmargin, colw, h - banner_h, id='c2',
                       leftPadding=0, rightPadding=0)]
        self.addPageTemplates([
            PageTemplate(id='first', frames=first, onPage=self._folio),
            PageTemplate(id='odd', frames=cols(inner), onPage=self._folio),
            PageTemplate(id='even', frames=cols(outer), onPage=self._folio),
        ])

    def _folio(self, canvas, doc):
        n = canvas.getPageNumber()
        if n <= 1:
            return
        canvas.saveState()
        canvas.setFont(REG, 8.5)
        canvas.drawCentredString(self.pagesize[0] / 2.0, self.bottomMargin * 0.55, str(n))
        canvas.restoreState()

    def handle_pageBegin(self):
        self._handle_pageBegin()
        if not self._mirror:
            self._handle_nextPageTemplate('odd')
        else:
            self._handle_nextPageTemplate('even' if self.page % 2 else 'odd')


def convert(md_path, pdf_path, vmargin=2.0, duplex=False, twocol=False):
    st = styles(compact=twocol)
    lines = Path(md_path).read_text(encoding='utf-8').split('\n')
    if twocol:
        doc = TwoColumnDoc(str(pdf_path), vmargin=vmargin * cm, duplex=duplex,
                           title=Path(md_path).stem)
    elif duplex:
        doc = DuplexDoc(str(pdf_path), vmargin=vmargin * cm, title=Path(md_path).stem)
    else:
        doc = SimpleDocTemplate(str(pdf_path), pagesize=A4,
                                leftMargin=2.1 * cm, rightMargin=2.1 * cm,
                                topMargin=vmargin * cm, bottomMargin=vmargin * cm,
                                title=Path(md_path).stem)
    avail = doc.width
    flow = []
    i = 0
    first_h1 = True
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()

        if not s:
            i += 1
            continue

        if re.match(r'^(---|\*\*\*|___)$', s):
            flow.append(Spacer(1, 4))
            flow.append(HRFlowable(width='100%', thickness=0.6,
                                   color=colors.black))
            flow.append(Spacer(1, 4))
            i += 1
            continue

        if s.startswith('|') and i + 1 < len(lines) and re.match(r'^\|[\s:|-]+\|?$',
                                                                 lines[i + 1].strip()):
            rows = [split_row(s)]
            i += 2
            while i < len(lines) and lines[i].strip().startswith('|'):
                rows.append(split_row(lines[i]))
                i += 1
            flow.append(Spacer(1, 3))
            flow.append(build_table(rows, st, avail))
            flow.append(Spacer(1, 8))
            continue

        m = re.match(r'^(#{1,6})\s+(.*)$', s)
        if m:
            level, text = len(m.group(1)), m.group(2)
            if level == 1 and first_h1:
                flow.append(Paragraph(inline(text), st['title']))
                first_h1 = False
            else:
                flow.append(Paragraph(inline(text), st[f'h{min(level,3)}']))
            i += 1
            continue

        if s.startswith('>'):
            buf = []
            while i < len(lines) and lines[i].strip().startswith('>'):
                buf.append(lines[i].strip().lstrip('>').strip())
                i += 1
            flow.append(Paragraph(inline(' '.join(buf)), st['quote']))
            continue

        if s.startswith('^'):
            buf = []
            while i < len(lines) and lines[i].strip().startswith('^'):
                buf.append(lines[i].strip().lstrip('^').strip())
                i += 1
            flow.append(Spacer(1, 5))
            flow.append(HRFlowable(width='38%', thickness=0.6, color=colors.black,
                                   hAlign='LEFT', spaceAfter=2))
            for line in buf:
                flow.append(Paragraph(inline(line), st['affil']))
            continue

        m = re.match(r'^[-*+]\s+(.*)$', s)
        if m:
            flow.append(Paragraph(inline(m.group(1)), st['li'], bulletText='\u2022'))
            i += 1
            continue

        m = re.match(r'^(\d+)[.)]\s+(.*)$', s)
        if m:
            flow.append(Paragraph(inline(m.group(2)), st['li'],
                                  bulletText=f'{m.group(1)}.'))
            i += 1
            continue

        buf = []
        while i < len(lines) and lines[i].strip() and not re.match(
                r'^(#{1,6}\s|\||>|[-*+]\s|\d+[.)]\s|---$)', lines[i].strip()):
            buf.append(lines[i].strip())
            i += 1
        para = ' '.join(buf)
        # a centred author/affiliation block sits right under the title
        style = st['authors'] if len(flow) < 4 and para.startswith('**') else st['body']
        flow.append(Paragraph(inline(para), style))

    doc.build(flow)
    return pdf_path


if __name__ == '__main__':
    FLAGS = {'--duplex', '--twocol'}
    duplex = '--duplex' in sys.argv          # mirrored margins for double-sided printing
    twocol = '--twocol' in sys.argv          # two-column layout, as used by ML preprints
    args = [a for a in sys.argv[1:] if a not in FLAGS]
    src = args[0] if args else 'docs/MANUSCRIPT.md'
    dst = args[1] if len(args) > 1 else str(Path(src).with_suffix('.pdf'))
    # optional third positional argument: top/bottom margin in cm (default 2.0)
    vm = float(args[2]) if len(args) > 2 else 2.0
    out = convert(src, dst, vmargin=vm, duplex=duplex, twocol=twocol)
    tags = ' '.join(t for t, on in [('[two-column]', twocol),
                                    ('[double-sided]', duplex)] if on)
    print('wrote', out, f'({Path(out).stat().st_size:,} bytes)', tags)
