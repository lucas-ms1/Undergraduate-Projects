p = r"c:\Users\freew\Dropbox\ECO 317 Project 1\md2tex.py"
with open(p, "r", encoding="utf-8") as f:
    lines = f.readlines()
for i, line in enumerate(lines):
    if "Intuitive" in line and "rest.replace" in line and "out.append" in line:
        lines[i] = "            rest_quoted = rest.replace(chr(0x201c), \"``\").replace(chr(0x201d), \"''\")\n"
        lines.insert(i + 1, "            out.append(\"\\\\item \\\\textbf{Intuitive}: \" + escape_tex(rest_quoted) + \"\\\\n\")\n")
        break
with open(p, "w", encoding="utf-8") as f:
    f.writelines(lines)
print("Fixed")
