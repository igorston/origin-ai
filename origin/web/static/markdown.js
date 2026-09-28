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
const ITEM = /^\s*([-*+]|\d+[.)])\s+(.*)$/;
const RULE = /^\s*([-*_])(\s*\1){2,}\s*$/;

// Line by line: models put a heading right above its list ("### 3. Valores\n- ..."), so
// headings, list items and rules are recognized on any line, not only as whole paragraphs.
function blocks(text) {
  const out = [];
  let list = null;
  let paragraph = [];
  const closeList = () => {
    if (list) out.push(`</${list}>`);
    list = null;
  };
  const closeParagraph = () => {
    if (paragraph.length) out.push(`<p>${paragraph.map(inline).join("<br>")}</p>`);
    paragraph = [];
  };
  for (const line of text.split("\n")) {
    const heading = line.match(HEADING);
    const item = !heading && line.match(ITEM);
    if (!line.trim()) {
      closeParagraph(); // a list stays open: items split by blank lines are still one list
    } else if (heading) {
      closeParagraph();
      closeList();
      const level = Math.min(heading[1].length + 2, 6); // chat headings stay small: h3..h6
      out.push(`<h${level}>${inline(heading[2])}</h${level}>`);
    } else if (RULE.test(line)) {
      closeParagraph();
      closeList();
      out.push("<hr>");
    } else if (item) {
      closeParagraph();
      const kind = /^\d/.test(item[1]) ? "ol" : "ul";
      if (list !== kind) {
        closeList();
        const start = kind === "ol" ? parseInt(item[1], 10) : 1;
        out.push(start > 1 ? `<ol start="${start}">` : `<${kind}>`);
        list = kind;
      }
      out.push(`<li>${inline(item[2])}</li>`);
    } else if (list && /^\s{2,}\S/.test(line)) {
      // A wrapped continuation of the last item.
      out[out.length - 1] = out[out.length - 1].replace(/<\/li>$/, `<br>${inline(line.trim())}</li>`);
    } else {
      closeList();
      paragraph.push(line);
    }
  }
  closeParagraph();
  closeList();
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
