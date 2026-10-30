const escapeLabel = value => String(value || '').replace(/[\\`*_{}[\]<>#|]/g, '\\$&').replace(/[\r\n]+/g, ' ');

export function sessionMarkdown(messages, title, knowledgeBase) {
  if (!messages.length) return '';
  const sections = [`# ${escapeLabel(title || '会话')}\n\n知识库：${escapeLabel(knowledgeBase)}`];
  messages.forEach(message => {
    const heading = message.role === 'user' ? '问题' : '回答';
    sections.push(`## ${heading}\n\n${message.content || ''}`);
    if (message.imageUrl || message.attachment) sections.push('附图：[图片附件未包含在导出文件中]');
    if (message.role !== 'assistant') return;
    const sources = Array.isArray(message.sources) ? message.sources : [];
    if (!sources.length) return;
    sections.push('### 本轮引用来源');
    sources.flatMap(source => source.evidence?.length ? source.evidence : [source]).forEach(source => {
      const label = Number.isInteger(source.id) && source.id > 0 ? source.id : '';
      const location = [source.kb_name, source.source, source.page, source.context_title].filter(Boolean).map(escapeLabel).join(' / ');
      sections.push(`- [来源 ${label}] ${location}${source.url ? ` <${String(source.url).replace(/[<>\s]/g, encodeURIComponent)}>` : ''}`);
      if (source.content) sections.push(`> ${String(source.content).split(/\r?\n/).join('\n> ')}`);
    });
  });
  return sections.join('\n\n') + '\n';
}

export function exportFilename(title) {
  const clean = Array.from(String(title || '会话'), char => char.charCodeAt(0) < 32 ? '_' : char).join('');
  const name = clean.replace(/[<>:"/\\|?*]/g, '_').replace(/[. ]+$/g, '').slice(0, 100);
  return `${name || '会话'}.md`;
}
