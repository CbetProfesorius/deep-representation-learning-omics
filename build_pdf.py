#!/usr/bin/env python
"""Build the thesis as a proper two-column scientific PDF (Bioinformatics-journal
style) from thesis.md, via XeLaTeX. This replaces the fragile Word route: LaTeX
handles two columns with full-width spanning figures/tables natively as floats.

Pipeline:
  1. strip the YAML header; split off the Abstract and the body (Introduction on).
  2. pandoc each part -> LaTeX fragments.
  3. promote figures to figure* and convert pandoc longtables -> full-width table*
     (tabular) so wide content spans both columns.
  4. assemble a master .tex: UVic/FMP title page (single column) -> Abstract + TOC
     (single column) -> two-column body.
  5. compile twice with xelatex (TOC + references).
Outputs Thesis.pdf.
"""
import re, subprocess
from pathlib import Path

ROOT = Path(__file__).parent
TITLE = 'Deep Representation Learning for High-Dimensional Omics Data'

md = (ROOT / 'thesis.md').read_text()
md = re.sub(r'^---\n.*?\n---\n', '', md, count=1, flags=re.S)        # drop YAML
i = md.index('# 1. Introduction')
head, body_md = md[:i], md[i:]
abstract_md = re.search(r'#\s*Abstract\s*(.*)', head, re.S).group(1).strip()

# figures are cited in prose by filename stem ("fig20"); renumber to proper
# contiguous "Figure N" matching the order they appear (= LaTeX's auto-numbering).
# Skip filename/code mentions (those are followed by '_', '`' or '/').
files = re.findall(r'\]\((data/[^)]*\.png)\)', body_md)
fmap = {}
for idx, f in enumerate(files, 1):
    m = re.match(r'(fig\d+)', f.rsplit('/', 1)[-1])
    if m:
        fmap[m.group(1)] = idx


def _figref(m):
    stem = 'fig' + m.group(1)
    return ('Figure ' + str(fmap[stem])) if stem in fmap else m.group(0)


body_md = re.sub(r'\bfig(\d+)\b(?![`/])', _figref, body_md)

(ROOT / '_abstract.md').write_text(abstract_md)
(ROOT / '_body.md').write_text(body_md)


def pandoc(src, out):
    subprocess.run(['pandoc', str(src), '-t', 'latex', '-o', str(out)], check=True)


pandoc(ROOT / '_abstract.md', ROOT / '_abstract.tex')
pandoc(ROOT / '_body.md', ROOT / '_body.tex')

body = (ROOT / '_body.tex').read_text()
body = body.replace('\\begin{figure}', '\\begin{figure*}[tp]').replace('\\end{figure}', '\\end{figure*}')


def brace_group(s, i):
    assert s[i] == '{'
    d = 0
    for j in range(i, len(s)):
        d += (s[j] == '{') - (s[j] == '}')
        if d == 0:
            return s[i:j + 1]
    raise ValueError('unbalanced')


def conv_block(block):
    colspec = brace_group(block, block.index('{', block.index('[]')))
    cap = re.search(r'\\caption\{(.*?)\}\\tabularnewline', block, re.S)
    caption = cap.group(1) if cap else ''
    head = re.search(r'\\toprule\\noalign\{\}\s*(.*?)\\midrule\\noalign\{\}\s*\\endfirsthead', block, re.S)
    header = head.group(1).strip() if head else ''
    rows = re.search(r'\\endlastfoot\s*(.*?)\\end\{longtable\}', block, re.S).group(1).strip()
    return ('\\begin{table*}[tp]\n\\centering\n\\footnotesize\n'
            + (('\\caption{' + caption + '}\n') if caption else '')
            + '\\begin{tabular}' + colspec + '\n\\toprule\n' + header
            + '\n\\midrule\n' + rows + '\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n')


def convert_longtables(s):
    out, i = [], 0
    while True:
        b = s.find('\\begin{longtable}', i)
        if b < 0:
            out.append(s[i:]); break
        e = s.find('\\end{longtable}', b) + len('\\end{longtable}')
        out.append(s[i:b]); out.append(conv_block(s[b:e])); i = e
    return ''.join(out)


body = convert_longtables(body)
abstract_tex = (ROOT / '_abstract.tex').read_text()

PREAMBLE = r'''\documentclass[twocolumn,a4paper,10pt]{article}
\usepackage[a4paper,top=2.23cm,bottom=2.23cm,left=1.93cm,right=2.44cm]{geometry}
\usepackage{fontspec}
\setmainfont{Times New Roman}
\setsansfont{Helvetica Neue}
\usepackage{amsmath,amssymb}
\usepackage{array,booktabs,calc,longtable,graphicx,xcolor}
\usepackage{caption}
\captionsetup{font=small,labelfont=bf,labelsep=period}
\captionsetup[table]{labelformat=empty,labelsep=none}
\usepackage{sectsty}
\allsectionsfont{\sffamily\bfseries}
\sectionfont{\large\sffamily\bfseries}
\subsectionfont{\normalsize\sffamily\bfseries}
\subsubsectionfont{\small\sffamily\bfseries}
\setcounter{secnumdepth}{0}
\setlength{\columnsep}{0.7cm}
\setlength{\parindent}{1em}
\usepackage{microtype}
\usepackage[hidelinks]{hyperref}
\hypersetup{colorlinks=true,linkcolor=black,citecolor=black,urlcolor=blue}
% generous float settings for many spanning floats
\extrafloats{80}
\renewcommand{\topfraction}{0.92}
\renewcommand{\floatpagefraction}{0.75}
\renewcommand{\dbltopfraction}{0.92}
\renewcommand{\dblfloatpagefraction}{0.75}
\setcounter{topnumber}{4}
\setcounter{dbltopnumber}{4}
\setcounter{totalnumber}{6}
% pandoc fragment helpers
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\makeatletter\providecommand{\pandocbounded}[1]{#1}\makeatother
'''

TITLEPAGE = r'''\begin{document}
\begin{titlepage}
\noindent\includegraphics[width=0.46\textwidth]{assets/uvic_logo.png}
\vspace{2.4cm}
\begin{center}
{\large Master of Science in Omics Data Analysis}\\[1.1cm]
{\large Master Thesis}\\[1.4cm]
{\LARGE\bfseries ''' + TITLE + r'''}\\[1.3cm]
{\large by}\\[0.4cm]
{\large\bfseries Arijus Skaisgirys}\\[1.8cm]
\begin{minipage}{0.82\textwidth}\centering
Supervisor: Prof. Dr. Andrius Stasiukynas, Kazimieras Simonavičius University\\[0.3cm]
Co-supervisor: Francesco Strati, PhD, Lithuanian University of Health Sciences\\[0.3cm]
Academic tutor: Jordi Vill\`a i Freixa, Department of Biosciences, Faculty of Sciences, Technology and Engineering, University of Vic -- Central University of Catalonia
\end{minipage}\\[1.6cm]
June 2026
\end{center}
\end{titlepage}
'''

# Abstract and contents each get their own page; the contents lists sections
# and subsections. \twocolumn then starts the body on a fresh page.
FRONTMATTER = ('\\onecolumn\n\\section*{Abstract}\n' + abstract_tex
               + '\n\\newpage\n{\\small\\tableofcontents}\n\\twocolumn\n')

master = PREAMBLE + TITLEPAGE + FRONTMATTER + body + '\n\\end{document}\n'
(ROOT / 'Thesis.tex').write_text(master)

for tmp in ['_abstract.md', '_body.md', '_abstract.tex', '_body.tex']:
    (ROOT / tmp).unlink(missing_ok=True)

for _ in range(2):
    r = subprocess.run(['xelatex', '-interaction=nonstopmode', '-halt-on-error', 'Thesis.tex'],
                       cwd=ROOT, capture_output=True, text=True)
if r.returncode != 0:
    print(r.stdout[-3000:])
    raise SystemExit('xelatex failed')
print('built Thesis.pdf')
