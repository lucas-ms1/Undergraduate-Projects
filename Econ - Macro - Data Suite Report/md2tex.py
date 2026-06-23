"""Convert main.txt (markdown) to LaTeX body for main.tex."""
import re
import sys

def escape_tex(s):
    # Escape LaTeX special chars when not in verbatim/code
    s = s.replace("\\", "\\textbackslash{}")
    s = s.replace("&", "\\&")
    s = s.replace("%", "\\%")
    s = s.replace("#", "\\#")
    s = s.replace("$", "\\$")
    s = s.replace("{", "\\{")
    s = s.replace("}", "\\}")
    s = s.replace("_", "\\_")
    s = s.replace("~", "\\textasciitilde{}")
    s = s.replace("^", "\\textasciicircum{}")
    return s

def code_to_tex(m):
    """Convert `...` to \texttt{...} with _ escaped."""
    inner = m.group(1).replace("_", "\\_").replace("\\", "\\textbackslash{}").replace("{", "\\{").replace("}", "\\}")
    return "\\texttt{" + inner + "}"

def code_to_placeholder(m):
    """Convert `...` to placeholder so escape_tex won't mangle it."""
    inner = m.group(1)
    inner = inner.replace("\\", "\\textbackslash{}").replace("{", "\\{").replace("}", "\\}").replace("_", "\\_").replace("#", "\\#")
    return "@C@" + inner + "@/C@"

def protect_code_then_escape(s):
    """Replace `code` with placeholder; escape only the plain parts, then output \\texttt{content}."""
    s = re.sub(r"`([^`]+)`", code_to_placeholder, s)
    out = []
    i = 0
    while i < len(s):
        if s[i:i + 3] == "@C@":
            j = s.find("@/C@", i + 3)
            if j == -1:
                out.append(escape_tex(s[i:]))
                break
            content = s[i + 3:j]
            out.append("\\texttt{" + content + "}")
            i = j + 4
        else:
            j = s.find("@C@", i)
            if j == -1:
                out.append(escape_tex(s[i:]))
                break
            out.append(escape_tex(s[i:j]))
            i = j
    return "".join(out)

def bold_to_tex(m):
    """Convert **...** to \textbf{...}, handling nested backticks."""
    content = m.group(1)
    # Replace `x` inside bold with \texttt{x\_} form
    def sub_code(c):
        inner = c.group(1).replace("_", "\\_").replace("\\", "\\textbackslash{}").replace("{", "\\{").replace("}", "\\}")
        return "\\texttt{" + inner + "}"
    content = re.sub(r"`([^`]+)`", sub_code, content)
    content = escape_tex(content)
    return "\\textbf{" + content + "}"

def main():
    with open("main.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()

    out = []
    in_verbatim = False
    i = 0
    while i < len(lines):
        line = lines[i]
        raw = line

        # Code block
        if line.strip() == "```mermaid" or line.strip() == "```":
            if in_verbatim:
                out.append("\\end{verbatim}\n")
                in_verbatim = False
            else:
                out.append("\\begin{verbatim}\n")
                in_verbatim = True
            i += 1
            continue

        if in_verbatim:
            out.append(line)  # verbatim: pass through unchanged
            i += 1
            continue

        # Section headers (use protect so backticked code in title isn't escaped)
        if line.startswith("#### "):
            title = line[5:].strip()
            out.append("\\subsubsection{" + protect_code_then_escape(title) + "}\n\n")
            i += 1
            continue
        if line.startswith("### "):
            title = line[4:].strip()
            out.append("\\subsection{" + protect_code_then_escape(title) + "}\n\n")
            i += 1
            continue
        if line.startswith("## "):
            title = line[3:].strip()
            out.append("\\section{" + protect_code_then_escape(title) + "}\n\n")
            i += 1
            continue

        # Horizontal rule
        if line.strip() == "---":
            out.append("\\vspace{0.5em}\\noindent\\rule{\\textwidth}{0.4pt}\\vspace{0.5em}\n\n")
            i += 1
            continue

        # List items: "  - " (nested) or "- " (top-level)
        stripped = line.lstrip()
        if stripped.startswith("- **`") and "**" in stripped[5:]:
            # - **`name()`** or similar (escape \ first so \_ doesn't become \textbackslash{}_)
            match = re.match(r"^- \*\*`([^`]+)`\*\*\s*$", stripped)
            if match:
                code = match.group(1).replace("\\", "\\textbackslash{}").replace("{", "\\{").replace("}", "\\}").replace("_", "\\_")
                out.append("\\item \\textbf{\\texttt{" + code + "}}\n")
                i += 1
                continue
        if re.match(r"^  - \*\*Technical\*\*:", stripped):
            rest = stripped[18:].strip()
            out.append("\\item \\textbf{Technical}: " + protect_code_then_escape(rest) + "\n")
            i += 1
            continue
        if re.match(r"^  - \*\*Intuitive\*\*:", stripped):
            rest = stripped[18:].strip()
            out.append("\\item \\textbf{Intuitive}: " + protect_code_then_escape(rest) + "\n")
            i += 1
            continue
        if re.match(r"^  - \*\*Gotchas\*\*:", stripped):
            rest = stripped[17:].strip()
            out.append("\\item \\textbf{Gotchas}: " + protect_code_then_escape(rest) + "\n")
            i += 1
            continue
        if stripped.startswith("- **Inner "):
            rest = stripped[2:]  # "**Inner `def _merge_job_fn(ctx)`**: logs..."
            if "**: " in rest:
                idx = rest.index("**: ")
                label_part = rest[9:idx]   # skip "**Inner " (9 chars)
                content_part = rest[idx + 4:]  # skip "**: "
                out.append("\\item \\textbf{Inner " + protect_code_then_escape(label_part) + "}}: " + protect_code_then_escape(content_part) + "\n")
            else:
                out.append("\\item \\textbf{Inner " + protect_code_then_escape(rest[6:]) + "}\n")
            i += 1
            continue
        if stripped.startswith("- **"):
            # Generic - **Label**: content or - **Label**
            rest = stripped[4:].strip()
            if "**: " in rest:
                label, content = rest.split("**: ", 1)
                out.append("\\item \\textbf{" + escape_tex(label) + "}: " + protect_code_then_escape(content) + "\n")
            else:
                out.append("\\item \\textbf{" + protect_code_then_escape(rest) + "}\n")
            i += 1
            continue
        if stripped.startswith("- "):
            rest = stripped[2:]
            out.append("\\item " + protect_code_then_escape(rest) + "\n")
            i += 1
            continue

        # Empty line
        if not line.strip():
            out.append("\n")
            i += 1
            continue

        # Paragraph line (no leading - or ##): use placeholders so escape_tex won't mangle \textbf/\texttt
        rest = line.strip()
        rest = re.sub(r"\*\*([^*]+)\*\*", lambda m: "@B@" + escape_tex(m.group(1)) + "@/B@", rest)
        rest = re.sub(r"`([^`]+)`", code_to_placeholder, rest)
        rest = escape_tex(rest)
        rest = rest.replace("@B@", "\\textbf{").replace("@/B@", "}").replace("@C@", "\\texttt{").replace("@/C@", "}")
        out.append(rest + "\n\n")
        i += 1

    if in_verbatim:
        out.append("\\end{verbatim}\n")

    # Now wrap consecutive \item in itemize; wrap section content
    result = []
    preamble = [
        "\\documentclass[11pt]{article}",
        "",
        "\\usepackage[utf8]{inputenc}",
        "\\usepackage[T1]{fontenc}",
        "\\usepackage[margin=1in]{geometry}",
        "\\usepackage{enumitem}",
        "\\usepackage{parskip}",
        "\\usepackage{hyperref}",
        "\\usepackage{newunicodechar}",
        "\\newunicodechar{\\u2265}{\\ensuremath{\\geq}}  % ≥",
        "\\newunicodechar{\\u2248}{\\ensuremath{\\approx}}  % ≈",
        "",
        "\\title{Macro Project 1 --- Data Suite\\\\Study Sheet}",
        "\\author{}",
        "\\date{}",
        "",
        "\\begin{document}",
        "\\maketitle",
        "",
    ]
    result.append("\n".join(preamble) + "\n\n")

    j = 0
    body = "".join(out)
    # Wrap \item blocks in \begin{itemize}\end{itemize}
    lines_out = body.split("\n")
    i = 0
    while i < len(lines_out):
        line = lines_out[i]
        if line.strip().startswith("\\item"):
            result.append("\\begin{itemize}[leftmargin=*,nosep]\n")
            while i < len(lines_out) and lines_out[i].strip().startswith("\\item"):
                result.append(lines_out[i] + "\n")
                i += 1
            result.append("\\end{itemize}\n\n")
            continue
        result.append(line + "\n")
        i += 1

    result.append("\n\\end{document}\n")
    full = "".join(result)
    # Fix double-escaped underscore (placeholder content wrongly escaped)
    full = full.replace("\\textbackslash{\\{\\}\\_", "\\_")
    with open("main.tex", "w", encoding="utf-8") as f:
        f.write(full)

if __name__ == "__main__":
    main()
