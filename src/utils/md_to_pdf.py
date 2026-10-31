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

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4, LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, FrameBreak, HRFlowable,
                                Image as RLImage, KeepTogether, PageBreak, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle)

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
    base = dict(fontName=REG, leading=f * 1.32, fontSize=f, allowWidows=0, allowOrphans=0)
    return {
        'title': ParagraphStyle('t', parent=ss['Title'], fontName=BOLD,
                                fontSize=15 if compact else 16,
                                leading=20, spaceAfter=10, alignment=TA_CENTER),
        'authors': ParagraphStyle('a', fontName=REG, fontSize=11, leading=15,
                                  alignment=TA_CENTER, spaceAfter=4),
        'h1': ParagraphStyle('h1', fontName=BOLD, fontSize=10.5 if compact else 13,
                             leading=13 if compact else 16,
                             spaceBefore=11 if compact else 14, spaceAfter=4 if compact else 6,
                             keepWithNext=True, textColor=colors.black),
        'h2': ParagraphStyle('h2', fontName=BOLD, fontSize=9.4 if compact else 11,
                             leading=12 if compact else 14,
                             spaceBefore=8 if compact else 10, spaceAfter=3.5 if compact else 4.5,
                             keepWithNext=True, textColor=colors.black),
        'h3': ParagraphStyle('h3', fontName=BOLD, fontSize=8.8 if compact else 10,
                             leading=11.5 if compact else 13,
                             spaceBefore=6 if compact else 8, spaceAfter=2.5 if compact else 3.5,
                             keepWithNext=True, textColor=colors.black),
        'body': ParagraphStyle('b', alignment=TA_JUSTIFY, spaceAfter=4.5 if compact else 5.5, **base),
        'li': ParagraphStyle('li', alignment=TA_JUSTIFY, leftIndent=14, bulletIndent=4,
                             spaceAfter=2 if compact else 3, bulletFontName=REG, bulletFontSize=9.6, **base),
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
        'caption': ParagraphStyle('cap', fontName=REG, fontSize=6.8 if compact else 8.2,
                                  leading=8.5 if compact else 10.5,
                                  spaceBefore=2 if compact else 3,
                                  spaceAfter=2 if compact else 3, alignment=TA_JUSTIFY,
                                  textColor=colors.black),
        'tcaption': ParagraphStyle('tcap', fontName=REG, fontSize=6.8 if compact else 8.5,
                                   leading=8.8 if compact else 11.0,
                                   spaceBefore=2 if compact else 3,
                                   spaceAfter=3 if compact else 4, alignment=TA_JUSTIFY,
                                   keepWithNext=True, textColor=colors.black),
    }


def split_row(line):
    return [c.strip() for c in line.strip().strip('|').split('|')]


def list_item_text(lines, i, first):
    """Join a list item with its indented continuation lines.

    Without this a wrapped item's tail falls through to the paragraph branch and is
    typeset as an unindented paragraph of its own, below the item it belongs to.
    """
    parts, i = [first], i + 1
    while i < len(lines) and lines[i].strip() and lines[i][:1].isspace() \
            and not re.match(r'^\s*([-*+]|\d{1,2}[.)])\s', lines[i]):
        parts.append(lines[i].strip())
        i += 1
    return ' '.join(parts), i


def compute_col_widths(rows, total_width, font_name, font_size, padding=8):
    ncol = max(len(r) for r in rows)
    min_widths = []
    text_lens = []
    for c in range(ncol):
        words = []
        full_texts = []
        for r in rows:
            if c < len(r):
                clean = re.sub(r'[*_`#\[\]]|<[^>]+>', '', r[c]).strip()
                words.extend(clean.split())
                full_texts.append(clean)
        max_word_w = max([pdfmetrics.stringWidth(w, font_name, font_size) for w in words] or [10])
        min_widths.append(max_word_w + padding + 4)
        max_len = max([len(t) for t in full_texts] or [1])
        text_lens.append(max_len)

    rem_width = total_width - sum(min_widths)
    if rem_width > 0:
        tot_len = sum(text_lens)
        return [min_w + rem_width * (l / tot_len) for min_w, l in zip(min_widths, text_lens)]
    else:
        scale = total_width / sum(min_widths)
        return [w * scale for w in min_widths]


def build_table(rows, st, width):
    header, body = rows[0], rows[1:]
    ncol = max(len(r) for r in rows)
    data = []
    for i, r in enumerate(rows):
        r = r + [''] * (ncol - len(r))
        sty = st['cellh'] if i == 0 else st['cell']
        data.append([Paragraph(inline(c), sty) for c in r])
    col_widths = compute_col_widths(rows, width, st['cellh'].fontName, st['cellh'].fontSize)
    t = Table(data, colWidths=col_widths, repeatRows=1, hAlign='LEFT')
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


class FloatBlock(KeepTogether):
    """Atomic figure or table+caption block that floats to the top of the next frame
    when it does not fit in the remaining space of the current frame, allowing
    subsequent prose to fill the current page or column first."""
    is_float = True

    def __init__(self, content, *args, **kwargs):
        for c in content:
            c._in_float = True
        KeepTogether.__init__(self, content, *args, **kwargs)


class LayoutDoc(BaseDocTemplate):
    """Base document template with LaTeX-like [ht] floating for figures/tables and
    orphan-free section headings."""

    def __init__(self, *args, **kwargs):
        BaseDocTemplate.__init__(self, *args, **kwargs)
        self._deferred = []

    def _rem_h(self):
        return self.frame._y - self.frame._y1

    def _fits(self, fl):
        rem = self._rem_h()
        s = 0 if self.frame._atTop else fl.getSpaceBefore()
        _, h = fl.wrapOn(self.canv, self.frame._aW, rem)
        true_h = getattr(fl, '_H', h)
        return (true_h + s) <= (rem - 1.0)

    def handle_flowable(self, flowables):
        if getattr(self, 'frame', None) is None or flowables is getattr(self, '_hanging', None):
            BaseDocTemplate._handle_flowable(self, flowables)
            return

        if flowables and getattr(flowables[0], '_in_float', False):
            BaseDocTemplate._handle_flowable(self, flowables)
            if not flowables and self._deferred:
                flowables.append(FrameBreak())
                flowables.extend(self._deferred)
                self._deferred.clear()
            return

        if self._deferred and (self.frame._atTop or self._fits(self._deferred[0])):
            for fl in flowables[:3]:
                if hasattr(fl, '_postponed'):
                    delattr(fl, '_postponed')
            flowables.insert(0, self._deferred.pop(0))
            BaseDocTemplate._handle_flowable(self, flowables)
            if not flowables and self._deferred:
                flowables.append(FrameBreak())
                flowables.extend(self._deferred)
                self._deferred.clear()
            return

        if not flowables:
            return

        f = flowables[0]

        if getattr(f, 'is_barrier', False) and self._deferred:
            flowables[0:0] = self._deferred
            self._deferred.clear()
            if not self.frame._atTop and not self._fits(flowables[0]):
                flowables.insert(0, FrameBreak())
            BaseDocTemplate._handle_flowable(self, flowables)
            return

        if getattr(f, 'is_float', False) and not self.frame._atTop and not self._fits(f):
            k = 0
            while k < len(flowables) and getattr(flowables[k], 'is_float', False):
                k += 1
            if k < len(flowables) and not getattr(flowables[k], 'is_barrier', False):
                self._deferred.extend(flowables[:k])
                del flowables[:k]
                f = flowables[0]
            else:
                if self._deferred:
                    flowables[0:0] = self._deferred
                    self._deferred.clear()
                flowables.insert(0, FrameBreak())
                BaseDocTemplate._handle_flowable(self, flowables)
                return

        if getattr(f, 'is_heading', False) and not self.frame._atTop:
            rem = self._rem_h()
            req_h = 0
            idx = 0
            while idx < len(flowables) and getattr(flowables[idx], 'is_heading', False):
                fl = flowables[idx]
                s = fl.getSpaceBefore() if (idx > 0 or not self.frame._atTop) else 0
                _, h = fl.wrapOn(self.canv, self.frame._aW, rem)
                req_h += s + h + fl.getSpaceAfter()
                idx += 1
            req_h += 28
            if req_h > rem:
                flowables.insert(0, FrameBreak())
                BaseDocTemplate._handle_flowable(self, flowables)
                return

        BaseDocTemplate._handle_flowable(self, flowables)
        if not flowables and self._deferred:
            flowables.append(FrameBreak())
            flowables.extend(self._deferred)
            self._deferred.clear()


class SingleColumnDoc(LayoutDoc):
    """Single-column layout with centred page numbers on pages 2+."""

    def __init__(self, filename, pagesize=A4, hmargin=2.1 * cm, vmargin=2.0 * cm, title=None,
                author=None):
        LayoutDoc.__init__(self, filename, pagesize=pagesize, title=title, author=author,
                           leftMargin=hmargin, rightMargin=hmargin,
                           topMargin=vmargin, bottomMargin=vmargin)
        w = pagesize[0] - 2 * hmargin
        h = pagesize[1] - 2 * vmargin
        self.addPageTemplates([
            PageTemplate(id='main', frames=[Frame(hmargin, vmargin, w, h, id='f',
                                                  leftPadding=0, rightPadding=0,
                                                  topPadding=0, bottomPadding=0)],
                         onPage=self._folio),
        ])

    def _folio(self, canvas, doc):
        n = canvas.getPageNumber()
        if n <= 1:
            return
        canvas.saveState()
        canvas.setFont(REG, 8.5)
        canvas.drawCentredString(self.pagesize[0] / 2.0, self.bottomMargin * 0.55, str(n))
        canvas.restoreState()


class DuplexDoc(LayoutDoc):
    """Mirrored margins for double-sided printing.

    Odd (right-hand) pages get the wider margin on the left, even pages on the right, so
    the binding edge is consistent once the sheets are bound. Folios sit on the outer
    corner, as in a printed book.
    """

    def __init__(self, filename, pagesize=A4, inner=2.6 * cm, outer=1.9 * cm,
                 vmargin=2.0 * cm, title=None, author=None):
        LayoutDoc.__init__(self, filename, pagesize=pagesize, title=title, author=author,
                           leftMargin=inner, rightMargin=outer,
                           topMargin=vmargin, bottomMargin=vmargin)
        w = pagesize[0] - inner - outer
        h = pagesize[1] - 2 * vmargin
        self.addPageTemplates([
            PageTemplate(id='odd', frames=[Frame(inner, vmargin, w, h, id='f',
                                                 leftPadding=0, rightPadding=0,
                                                 topPadding=0, bottomPadding=0)],
                         onPage=self._folio),
            PageTemplate(id='even', frames=[Frame(outer, vmargin, w, h, id='f',
                                                  leftPadding=0, rightPadding=0,
                                                  topPadding=0, bottomPadding=0)],
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


class TwoColumnDoc(LayoutDoc):
    """Two-column layout, as used by machine-learning preprints.

    Text fills the left column, then continues in the right. The title block spans the
    full width, so the first page uses a one-frame template and the rest use two.
    Optionally mirrors margins for double-sided printing.
    """

    def __init__(self, filename, pagesize=A4, vmargin=2.0 * cm, gutter=0.7 * cm,
                 inner=2.0 * cm, outer=2.0 * cm, duplex=False, title=None,
                 banner_h=3.4 * cm, author=None):
        LayoutDoc.__init__(self, filename, pagesize=pagesize, title=title, author=author,
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


def convert(md_path, pdf_path, vmargin=2.0, duplex=False, twocol=False, with_figures=False,
           letter=False, preprint=False):
    # A4 is the default everywhere: it is what the target journal and most non-US
    # venues expect. --letter exists only for destinations that ask for US Letter
    # specifically (bioRxiv's own upload tips name it), and only the single-column
    # path honours it, since that is the only layout a preprint upload uses.
    page = LETTER if letter else A4
    st = styles(compact=twocol)
    lines = Path(md_path).read_text(encoding='utf-8').split('\n')
    # The PDF's own internal metadata (Title/Author), as distinct from what the
    # submission form asks for. Left unset, reportlab writes the literal strings
    # "(unspecified)"/"(anonymous)", and title falls back to the source filename
    # ("MANUSCRIPT") rather than the paper's real title. Pulled from the markdown
    # itself so the two can never drift apart.
    doc_title = next((re.sub(r'[*_`]', '', l[2:]).strip()
                      for l in lines if l.startswith('# ')), Path(md_path).stem)
    author_line = next((l for l in lines if re.search(r'\*\*[A-Z][\w .\'-]+\*\*', l)), '')
    doc_author = '; '.join(re.sub(r'[¹-⁹⁰]+$', '', a).strip()
                           for a in re.findall(r'\*\*([^*]+)\*\*', author_line)) or None
    if twocol:
        doc = TwoColumnDoc(str(pdf_path), vmargin=vmargin * cm, duplex=duplex,
                           title=doc_title, author=doc_author)
    elif duplex:
        doc = DuplexDoc(str(pdf_path), vmargin=vmargin * cm, title=doc_title, author=doc_author)
    else:
        doc = SingleColumnDoc(str(pdf_path), pagesize=page,
                              hmargin=2.1 * cm, vmargin=vmargin * cm,
                              title=doc_title, author=doc_author)
    avail = doc.colw if twocol else doc.width

    captions = {}
    fig_files = {}
    inserted_figs = set()
    if with_figures:
        in_figures = False
        current_fig = None
        caption_buf = []
        for ln in lines:
            s = ln.strip()
            if s == '## Figures':
                in_figures = True
                continue
            if in_figures and s.startswith('## '):
                break
            if in_figures:
                m = re.match(r'^\*\*Figure\s+(\d+)\.\s*(.*?)\*\*(.*)$', s)
                if m:
                    if current_fig:
                        captions[current_fig] = ' '.join(caption_buf).strip()
                    current_fig = int(m.group(1))
                    caption_buf = [f'**Figure {current_fig}. {m.group(2)}**' + m.group(3)]
                elif current_fig and s:
                    caption_buf.append(s)
        if current_fig:
            captions[current_fig] = ' '.join(caption_buf).strip()

        # Two variants are rendered per figure: the full-width one, and a '_col'
        # version typeset for a single column. A 7-inch figure scaled into a 3-inch
        # column has 4pt type, so the two-column build takes the narrow one.
        fig_dir = Path(md_path).parent / 'figures'
        for num in range(1, 10):
            matches = [p for p in sorted(fig_dir.glob(f'fig{num}_*.png'))
                       if p.stem.endswith('_col') == twocol]
            if not matches:
                matches = sorted(fig_dir.glob(f'fig{num}_*.png'))
            if matches:
                fig_files[num] = matches[0]

    def make_fig_flow(f_num):
        if f_num not in fig_files or f_num not in captions:
            return []
        p = fig_files[f_num]
        im = PILImage.open(p)
        w, h = im.size
        aspect = h / w
        fig_w = avail if twocol else avail * (0.90 if f_num == 1 else 0.88)
        img_flow = RLImage(str(p), width=fig_w, height=fig_w * aspect)
        img_flow.hAlign = 'CENTER'
        # a caption that flows into the next column is worse than a short column:
        # the image and its caption travel as one float block
        return [
            FloatBlock([
                Spacer(1, 2 if twocol else 3),
                img_flow,
                Paragraph(inline(captions[f_num]), st['caption']),
                Spacer(1, 4 if twocol else 6),
            ])
        ]

    flow = []
    pending_figs = []
    pending_table_caption = None

    def flush_pending_figs():
        """Emit queued FloatBlocks for figures cited in the preceding subsection."""
        while pending_figs:
            flow.extend(make_fig_flow(pending_figs.pop(0)))

    i = 0
    first_h1 = True
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()

        if not s:
            i += 1
            continue

        if with_figures and s == '## Figures':
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('## '):
                i += 1
            continue

        # Highlights are an Elsevier submission item, not a preprint convention: they
        # appear in about 4% of bioRxiv preprints. The journal build keeps them; the
        # preprint build drops the section and its bullets.
        if preprint and s.lower().startswith('## highlights'):
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('## '):
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

            # Tables are never swapped out for a figure. Where a result is adequately
            # stated by its table there is no figure to swap in: the figure set is
            # restricted to displays a table cannot replace (see make_figures.py).
            tbl = build_table(rows, st, avail)
            if pending_table_caption is not None:
                flow.append(FloatBlock([
                    pending_table_caption,
                    tbl,
                    Spacer(1, 5 if twocol else 7),
                ]))
                pending_table_caption = None
            else:
                flow.append(FloatBlock([
                    Spacer(1, 2),
                    tbl,
                    Spacer(1, 5 if twocol else 7),
                ]))
            continue

        if pending_table_caption is not None:
            flow.append(pending_table_caption)
            pending_table_caption = None

        m = re.match(r'^(#{1,6})\s+(.*)$', s)
        if m:
            level, text = len(m.group(1)), m.group(2)
            flush_pending_figs()
            if level == 1 and first_h1:
                flow.append(Paragraph(inline(text), st['title']))
                first_h1 = False
            else:
                p_head = Paragraph(inline(text), st[f'h{min(level,3)}'])
                p_head.is_heading = True
                if text.lower().startswith(('data and code', 'references', 'credit ')):
                    p_head.is_barrier = True
                flow.append(p_head)
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
            text, i = list_item_text(lines, i, m.group(1))
            flow.append(Paragraph(inline(text), st['li'], bulletText='\u2022'))
            continue

        m = re.match(r'^(\d{1,2})[.)]\s+(.*)$', s)
        if m:
            text, i = list_item_text(lines, i, m.group(2))
            flow.append(Paragraph(inline(text), st['li'],
                                  bulletText=f'{m.group(1)}.'))
            continue

        buf = []
        while i < len(lines) and lines[i].strip() and not re.match(
                r'^(#{1,6}\s|\||>|[-*+]\s|\d{1,2}[.)]\s|---$)', lines[i].strip()):
            buf.append(lines[i].strip())
            i += 1
        para = ' '.join(buf)

        if re.match(r'^\*\*Table\s+\d+\.', para):
            j = i
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and lines[j].strip().startswith('|'):
                pending_table_caption = Paragraph(inline(para), st['tcaption'])
                continue

        # a centred author/affiliation block sits right under the title
        style = st['authors'] if len(flow) < 4 and para.startswith('**') else st['body']
        flow.append(Paragraph(inline(para), style))

        if with_figures:
            for f_num in sorted(fig_files):
                if f'**Figure {f_num}**' in para and f_num not in inserted_figs:
                    pending_figs.append(f_num)
                    inserted_figs.add(f_num)

    flush_pending_figs()
    doc.build(flow)
    return pdf_path


if __name__ == '__main__':
    FLAGS = {'--duplex', '--twocol', '--figures', '--letter', '--preprint'}
    duplex = '--duplex' in sys.argv          # mirrored margins for double-sided printing
    twocol = '--twocol' in sys.argv          # two-column layout, as used by ML preprints
    with_figures = '--figures' in sys.argv    # embed figures inline
    letter = '--letter' in sys.argv          # US Letter instead of A4 (single-column only)
    preprint = '--preprint' in sys.argv      # drop Elsevier-only Highlights for bioRxiv
    args = [a for a in sys.argv[1:] if a not in FLAGS]
    src = args[0] if args else 'docs/MANUSCRIPT.md'
    # Without a suffix every variant would default to MANUSCRIPT.pdf and silently
    # overwrite the journal build, so the layout flags name themselves.
    suffix = ''.join(s for s, on in [('_2col', twocol),
                                     ('_figures', with_figures and twocol),
                                     ('_preprint', preprint)] if on)
    dst = args[1] if len(args) > 1 else str(
        Path(src).with_name(Path(src).stem + suffix).with_suffix('.pdf'))
    # optional third positional argument: top/bottom margin in cm (default 2.0)
    vm = float(args[2]) if len(args) > 2 else 2.0
    out = convert(src, dst, vmargin=vm, duplex=duplex, twocol=twocol, with_figures=with_figures,
                 letter=letter, preprint=preprint)
    tags = ' '.join(t for t, on in [('[preprint]', preprint),
                                    ('[two-column]', twocol),
                                    ('[double-sided]', duplex),
                                    ('[with-figures]', with_figures),
                                    ('[US Letter]', letter)] if on)
    print('wrote', out, f'({Path(out).stat().st_size:,} bytes)', tags)
