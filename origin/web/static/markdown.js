// Minimal, safe Markdown for chat replies: everything is HTML-escaped first, then a small
// subset (code blocks, inline code, bold, italic, links, headings, lists, rules,
// paragraphs) is applied.

const escapeHtml = (text) =>
  text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

function inline(text) {
  return text
    .replace(/`([^`\n]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(
      /\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>',
    )
    // Bare URLs (the sources list), minus trailing punctuation; not those already in a link.
    .replace(/(^|[\s(])(https?:\/\/[^\s<)]*[^\s<).,;:!?])/g, (_, lead, url) =>
      `${lead}<a href="${url}" target="_blank" rel="noopener noreferrer">${url}</a>`,
    );
}

const HEADING = /^\s*(#{1,6})\s+(.*?)\s*#*\s*$/;
const ITEM = /^(\s*)([-*+]|\d+[.)])\s+(.*)$/;
const RULE = /^\s*([-*_])(\s*\1){2,}\s*$/;
// A table's second line: | --- | :---: | ---: |
const TABLE_RULE = /^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$/;

const indentOf = (spaces) => spaces.replace(/\t/g, "    ").length;

function cells(line) {
  return line
    .trim()
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split("|")
    .map((cell) => cell.trim());
}

// A header row, then a rule with as many columns (GFM): "x | y" over "---" is a paragraph
// and a horizontal rule, not a table.
const isTable = (line, next = "") =>
  line.includes("|") && TABLE_RULE.test(next) && next.includes("-") && cells(next).length === cells(line).length;

function table(lines) {
  const align = cells(lines[1]).map((c) =>
    c.startsWith(":") && c.endsWith(":") ? "center" : c.endsWith(":") ? "right" : "",
  );
  const row = (line, tag) => {
    const values = cells(line);
    // Every row has the header's columns: missing cells are blank, extra ones dropped.
    return `<tr>${align
      .map((a, i) => `<${tag}${a ? ` style="text-align:${a}"` : ""}>${inline(values[i] ?? "")}</${tag}>`)
      .join("")}</tr>`;
  };
  const body = lines.slice(2).map((line) => row(line, "td")).join("");
  // Wide tables scroll inside the bubble instead of stretching it.
  return `<div class="table-wrap"><table><thead>${row(lines[0], "th")}</thead><tbody>${body}</tbody></table></div>`;
}

// Line by line: models put a heading right above its list ("### 3. Valores\n- ..."), so
// headings, list items, rules and tables are recognized on any line, not only as whole
// paragraphs. Lists nest by indentation.
function blocks(text) {
  const out = [];
  const lists = []; // open lists, outermost first: {kind, indent}
  let paragraph = [];
  const closeList = () => {
    const { kind } = lists.pop();
    out.push(`</li></${kind}>`);
  };
  const closeLists = () => {
    while (lists.length) closeList();
  };
  const closeParagraph = () => {
    if (paragraph.length) out.push(`<p>${paragraph.map(inline).join("<br>")}</p>`);
    paragraph = [];
  };
  const lines = text.split("\n");
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const heading = line.match(HEADING);
    const item = !heading && line.match(ITEM);
    if (!line.trim()) {
      closeParagraph(); // a list stays open: items split by blank lines are still one list
    } else if (isTable(line, lines[i + 1])) {
      closeParagraph();
      closeLists();
      const rows = [line, lines[i + 1]];
      for (i += 2; i < lines.length && lines[i].includes("|") && lines[i].trim(); i++) rows.push(lines[i]);
      i--; // the for loop steps past the last row
      out.push(table(rows));
    } else if (heading) {
      closeParagraph();
      closeLists();
      const level = Math.min(heading[1].length + 2, 6); // chat headings stay small: h3..h6
      out.push(`<h${level}>${inline(heading[2])}</h${level}>`);
    } else if (RULE.test(line)) {
      closeParagraph();
      closeLists();
      out.push("<hr>");
    } else if (item) {
      closeParagraph();
      const indent = indentOf(item[1]);
      const kind = /^\d/.test(item[2]) ? "ol" : "ul";
      while (lists.length && lists.at(-1).indent > indent) closeList();
      const top = lists.at(-1);
      if (top && top.indent === indent && top.kind !== kind) closeList();
      if (!lists.length || indent > lists.at(-1).indent) {
        // A new list; deeper than the open one, it nests inside its current item.
        const start = kind === "ol" ? parseInt(item[2], 10) : 1;
        out.push(start > 1 ? `<ol start="${start}">` : `<${kind}>`);
        lists.push({ kind, indent });
      } else {
        out.push("</li>");
      }
      out.push(`<li>${inline(item[3])}`);
    } else if (lists.length && /^\s{2,}\S/.test(line)) {
      out.push(`<br>${inline(line.trim())}`); // a wrapped continuation of the item
    } else {
      closeLists();
      paragraph.push(line);
    }
  }
  closeParagraph();
  closeLists();
  return out.join("");
}

export function renderMarkdown(source) {
  const escaped = escapeHtml(source);
  // Split out fenced code blocks (an unterminated fence while streaming counts as code).
  const parts = escaped.split(/```/);
  return parts
    .map((part, i) => {
      if (i % 2 === 0) return blocks(part.trim());
      const code = part.replace(/^[\w-]*\n/, "");
      return `<pre><code>${code.replace(/\n$/, "")}</code></pre>`;
    })
    .join("");
}
