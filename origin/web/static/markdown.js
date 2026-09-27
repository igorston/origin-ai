// Minimal, safe Markdown for chat replies: everything is HTML-escaped first, then a small
// subset (code blocks, inline code, bold, italic, links, lists, paragraphs) is applied.

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
    );
}

function blocks(text) {
  const out = [];
  let list = null;
  const closeList = () => {
    if (list) out.push(`</${list}>`);
    list = null;
  };
  for (const paragraph of text.split(/\n{2,}/)) {
    const lines = paragraph.split("\n");
    const isList = lines.every((l) => /^\s*([-*]|\d+\.)\s+/.test(l));
    if (isList) {
      const kind = /^\s*\d+\./.test(lines[0]) ? "ol" : "ul";
      if (list !== kind) {
        closeList();
        out.push(`<${kind}>`);
        list = kind;
      }
      for (const l of lines) out.push(`<li>${inline(l.replace(/^\s*([-*]|\d+\.)\s+/, ""))}</li>`);
    } else {
      closeList();
      const heading = paragraph.match(/^(#{1,3})\s+(.*)$/);
      out.push(
        heading
          ? `<h${heading[1].length + 2}>${inline(heading[2])}</h${heading[1].length + 2}>`
          : `<p>${lines.map(inline).join("<br>")}</p>`,
      );
    }
  }
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
