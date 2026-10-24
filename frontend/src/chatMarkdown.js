import MarkdownIt from 'markdown-it';

const md = new MarkdownIt({ html: false, linkify: false, breaks: true });
md.disable('image');
const defaultText = md.renderer.rules.text;
md.renderer.rules.text = (tokens, index, options, env, renderer) => {
  const text = tokens[index].content;
  if (!env?.citations) return defaultText(tokens, index, options, env, renderer);
  let linkDepth = 0;
  for (let i = 0; i < index; i += 1) {
    if (tokens[i].type === 'link_open') linkDepth += 1;
    if (tokens[i].type === 'link_close') linkDepth -= 1;
  }
  if (linkDepth) return defaultText(tokens, index, options, env, renderer);
  return text.split(/([[【]\s*来源\s*\d+\s*[\]】])/g).map(part => {
    const match = part.match(/^[[【]\s*来源\s*(\d+)\s*[\]】]$/);
    const id = match ? Number(match[1]) : 0;
    return env.citations.has(id)
      ? `<button type="button" class="citation-link" data-citation="${id}" aria-label="查看来源 ${id}">${md.utils.escapeHtml(part)}</button>`
      : md.utils.escapeHtml(part);
  }).join('');
};

export function renderAnswer(content, sources = []) {
  const citations = new Map();
  for (const source of sources) {
    for (const entry of source.evidence || []) {
      if (Number.isInteger(entry.id) && entry.id > 0) citations.set(entry.id, entry);
    }
  }
  return { html: md.render(String(content || ''), { citations }), citations };
}
