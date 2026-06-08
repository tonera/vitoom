import DOMPurify from 'dompurify'
import { toHtml } from 'hast-util-to-html'
import { common, createLowlight } from 'lowlight'
import MarkdownIt from 'markdown-it'
import { translate } from './translate'
// @ts-expect-error — markdown-it-katex 包未附带类型声明（无需改 env.d.ts）
import mk from 'markdown-it-katex'

const lowlight = createLowlight(common)

/** 围栏语言标识 → lowlight(common) 内已注册的 highlight.js 语言名 */
const FENCE_LANG_ALIASES: Record<string, string> = {
  js: 'javascript',
  jsx: 'javascript',
  mjs: 'javascript',
  cjs: 'javascript',
  ts: 'typescript',
  tsx: 'typescript',
  py: 'python',
  python3: 'python',
  sh: 'shell',
  zsh: 'shell',
  yml: 'yaml',
  md: 'markdown',
  mkd: 'markdown',
  rs: 'rust',
  rb: 'ruby',
  kt: 'kotlin',
  kts: 'kotlin',
  html: 'xml',
  htm: 'xml',
  vue: 'xml',
  svg: 'xml',
}

function resolveFenceLanguage(raw: string): string {
  const key = raw.trim().toLowerCase()
  if (!key) return 'plaintext'
  const mapped = FENCE_LANG_ALIASES[key] ?? key
  return lowlight.registered(mapped) ? mapped : 'plaintext'
}

function highlightFenceCode(src: string, lang: string): string {
  try {
    const language = resolveFenceLanguage(lang || '')
    const value = src.endsWith('\n') ? src : `${src}\n`
    return toHtml(lowlight.highlight(language, value))
  } catch {
    return ''
  }
}

const md = new MarkdownIt({
  /*
   * html:true：GFM 表格单元格内的 `<br>` 等内联 HTML 才能按标签渲染（html:false 会转成 `&lt;br&gt;` 字面量）。
   * XSS 与危险标签仍由下方 sanitizeChatHtml(DOMPurify) 处理。
   */
  html: true,
  breaks: true,
  linkify: true,
  highlight: (str, lang) => highlightFenceCode(str, lang),
})

md.use(mk, {
  throwOnError: false,
  errorColor: 'var(--agent-md-katex-error, #fca5a5)',
})

function buildCopyButtonHtml(): string {
  const label = translate('common.copy')
  return `<button type="button" class="agent-md-copy-btn group absolute right-1 top-1 z-10 inline-flex h-8 w-8 cursor-pointer items-center justify-center border-0 bg-transparent p-0 text-gray-400 shadow-none ring-0 outline-none transition-colors hover:text-gray-200 focus-visible:text-gray-100" title="${label}" aria-label="${label}"><span class="absolute inset-0 flex items-center justify-center opacity-100 transition-opacity group-[.is-copied]:pointer-events-none group-[.is-copied]:opacity-0" aria-hidden="true"><svg xmlns="http://www.w3.org/2000/svg" class="h-3.5 w-3.5 shrink-0" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg></span><span class="absolute inset-0 flex items-center justify-center text-emerald-400 opacity-0 transition-opacity group-[.is-copied]:opacity-100" aria-hidden="true"><svg xmlns="http://www.w3.org/2000/svg" class="h-3.5 w-3.5 shrink-0" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg></span></button>`
}

function wrapWithCopyUi(inner: string): string {
  return `<div class="agent-md-copy-wrap relative my-2">${buildCopyButtonHtml()}<div class="agent-md-copy-slot [&>blockquote]:m-0 [&>blockquote]:pr-1 [&>blockquote]:pt-7 [&>pre]:m-0 [&>pre]:pt-8">${inner}</div></div>`
}

const defaultFence = md.renderer.rules.fence!
md.renderer.rules.fence = (tokens, idx, options, env, self) => {
  const html = defaultFence(tokens, idx, options, env, self)
  const withHljs = html.includes('<code class="')
    ? html.replace('<code class="', '<code class="hljs ')
    : html.replace('<pre><code>', '<pre><code class="hljs">')
  return wrapWithCopyUi(withHljs)
}

md.renderer.rules.blockquote_open = (tokens, idx, options, _env, self) => {
  const open = self.renderToken(tokens, idx, options)
  return `<div class="agent-md-copy-wrap relative my-2">${buildCopyButtonHtml()}<div class="agent-md-copy-slot [&>blockquote]:m-0 [&>blockquote]:pr-1 [&>blockquote]:pt-7 [&>pre]:m-0 [&>pre]:pt-8">${open}`
}

md.renderer.rules.blockquote_close = (tokens, idx, options, _env, self) => {
  const close = self.renderToken(tokens, idx, options)
  return `${close}</div></div>\n`
}

const defaultLinkOpen = md.renderer.rules.link_open ?? ((tokens, idx, options, _env, self) => self.renderToken(tokens, idx, options))
md.renderer.rules.link_open = (tokens, idx, options, env, self) => {
  const token = tokens[idx]
  if (!token) return defaultLinkOpen(tokens, idx, options, env, self)
  if (token.attrIndex('target') < 0) {
    token.attrPush(['target', '_blank'])
    token.attrPush(['rel', 'noopener noreferrer'])
  }
  return defaultLinkOpen(tokens, idx, options, env, self)
}

/** renderer.link 之外的裸 <a>（如内联 HTML）也统一新标签打开。 */
function patchAnchorsWithoutTarget(html: string): string {
  return html.replace(/<a\s(?![^>]*\btarget\s*=)/gi, '<a target="_blank" rel="noopener noreferrer" ')
}

/** U+200B：CommonMark 下 `)**汉字` 等紧挨字母时，结束定界 `**` 常无法闭合，字面量 `**` 会残留；在括号与 `**` 间插入 ZWSP 不改变视觉。 */
const ZWSP = '\u200b'

const INLINE_CODE_PLACEHOLDER_PREFIX = '\uE000agent-md-code-'
const INLINE_CODE_PLACEHOLDER_SUFFIX = '\uE001'

const MATH_VARIABLE_TOKEN = '[A-Za-z\u0370-\u03FF](?:\\s*[_^]\\s*(?:\\{[^{}]+\\}|[A-Za-z0-9]+))?'
const MATH_VARIABLE_LIST_RE = new RegExp(`^${MATH_VARIABLE_TOKEN}(?:\\s*[,，]\\s*${MATH_VARIABLE_TOKEN})*$`)
const BARE_LATEX_MATH_LINE_RE = /^[\s0-9A-Za-z\\{}()[\]^_+\-*/=<>.,:×÷·±≤≥≠≈→←↔⇒⇔]+$/

function withInlineCodePlaceholders(src: string, transform: (text: string) => string): string {
  const inlineCodes: string[] = []
  const protectedSrc = src.replace(/(`+)([\s\S]*?)\1/g, (match) => {
    const idx = inlineCodes.push(match) - 1
    return `${INLINE_CODE_PLACEHOLDER_PREFIX}${idx}${INLINE_CODE_PLACEHOLDER_SUFFIX}`
  })
  const transformed = transform(protectedSrc)
  return transformed.replace(
    new RegExp(`${INLINE_CODE_PLACEHOLDER_PREFIX}(\\d+)${INLINE_CODE_PLACEHOLDER_SUFFIX}`, 'g'),
    (_match, idx: string) => inlineCodes[Number(idx)] ?? ''
  )
}

function normalizeMathBody(src: string): string {
  return src
    .trim()
    .replace(/\\\\\s+(?=\\?[A-Za-z])/g, ' ')
    .replace(/\\\\([A-Za-z]+)/g, (_match, command: string) => `\\${command}`)
    .replace(/\\implies\b/g, '\\Rightarrow')
}

function normalizeDisplayMathBody(src: string): string {
  const body = src
    .trim()
    .replace(/\\\\([A-Za-z]+)/g, (_match, command: string) => `\\${command}`)
    .replace(/\\implies\b/g, '\\Rightarrow')

  if (!/\\\\/.test(body) || /\\begin\{(?:aligned|align|array|matrix|cases)\}/.test(body)) return body
  return `\\begin{aligned}\n${body}\n\\end{aligned}`
}

function isLikelyMathContent(src: string): boolean {
  const body = src.trim()
  if (!body) return false
  return (
    /\\[A-Za-z]+/.test(body) ||
    /[\^_{}=<>+\-*/×÷±≤≥≠≈→←↔⇒⇔]/.test(body) ||
    /[A-Za-z\u0370-\u03FF]\s*\d|\d\s*[A-Za-z\u0370-\u03FF]/.test(body) ||
    MATH_VARIABLE_LIST_RE.test(body)
  )
}

function isLikelyBareLatexMathLine(src: string): boolean {
  const line = src.trim()
  if (!line || line.includes('$')) return false
  if (/^(?:#{1,6}|>|[-+*]|\d+\.)\s/.test(line)) return false
  if (!/\\\\?[A-Za-z]+/.test(line)) return false
  return BARE_LATEX_MATH_LINE_RE.test(line) && isLikelyMathContent(line)
}

function isLikelyPartialMathLine(src: string): boolean {
  const line = src.trim()
  if (!line || !line.includes('$')) return false
  if (/^(?:#{1,6}|>|[-+*]|\d+\.)\s/.test(line)) return false
  if (/[\u4E00-\u9FFF]/.test(line)) return false

  const withoutDollars = line.replace(/\$/g, '').trim()
  return BARE_LATEX_MATH_LINE_RE.test(withoutDollars) && isLikelyMathContent(withoutDollars)
}

function normalizeMathLines(src: string): string {
  const lines = src.split('\n')
  let inDisplayMath = false

  return lines
    .map((line) => {
      const trimmed = line.trim()
      const displayDelimiterCount = (trimmed.match(/\$\$/g) ?? []).length
      const isDelimiterOnly = /^\$\$\s*$/.test(trimmed)
      const isSingleLineDisplayMath = displayDelimiterCount >= 2

      if (inDisplayMath) {
        if (isDelimiterOnly || displayDelimiterCount % 2 === 1) inDisplayMath = false
        return line
      }

      if (isSingleLineDisplayMath) return line

      if (isDelimiterOnly || displayDelimiterCount % 2 === 1) {
        inDisplayMath = true
        return line
      }

      if (isLikelyPartialMathLine(line)) return `$${normalizeMathBody(line.replace(/\$/g, ''))}$`
      if (isLikelyBareLatexMathLine(line)) return `$${normalizeMathBody(line)}$`
      return line
    })
    .join('\n')
}

function normalizeMathInMarkdownText(src: string): string {
  return withInlineCodePlaceholders(src, (text) => {
    const withLatexDelimiters = text
      .replace(/\\\[([\s\S]+?)\\\]/g, (_match, body: string) => `$$${normalizeDisplayMathBody(body)}$$`)
      .replace(/\\\(([\s\S]+?)\\\)/g, (_match, body: string) => `$${normalizeMathBody(body)}$`)
      .replace(/\$\$([\s\S]+?)\$\$/g, (_match, body: string) => `$$${normalizeDisplayMathBody(body)}$$`)

    const withNormalizedInlineMath = withLatexDelimiters.replace(
      /(^|[^$])\$([^\n$]*?\S[^\n$]*?)\$(?!\$)/g,
      (match, prefix: string, body: string) => {
        if (!isLikelyMathContent(body)) return match
        return `${prefix}$${normalizeMathBody(body)}$`
      }
    )

    return normalizeMathLines(withNormalizedInlineMath)
  })
}

/**
 * 在「非围栏代码」行内，将 `)`/`）` 后紧贴的 `**` 改为 `)\u200b**`，避免粗体无法闭合。
 * 仅按 ``` 围栏切换，避免改动代码块字面量。
 */
function transformOutsideFences(src: string, transform: (text: string) => string): string {
  const lines = src.split('\n')
  let inFence = false
  const chunks: string[] = []
  let current: string[] = []

  const flush = () => {
    if (current.length) {
      chunks.push(transform(current.join('\n')))
      current = []
    }
  }

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]!
    if (line.trimStart().startsWith('```')) {
      flush()
      inFence = !inFence
      chunks.push(line)
      continue
    }
    if (inFence) {
      chunks.push(line)
      continue
    }
    current.push(line)
  }
  flush()
  return chunks.join('\n')
}

function prepareMarkdownText(src: string): string {
  return transformOutsideFences(src, (text) =>
    normalizeMathInMarkdownText(text).replace(/([\)\）])(\*\*)/g, `$1${ZWSP}$2`)
  )
}

function sanitizeChatHtml(html: string): string {
  return DOMPurify.sanitize(html, {
    /* mathMl/svg：放行 KaTeX 输出的 MathML 与内联 SVG；再补 semantics/annotation（默认 mathMl 配置不含） */
    USE_PROFILES: { html: true, mathMl: true, svg: true },
    FORBID_TAGS: ['script', 'iframe', 'object', 'embed', 'style', 'base', 'link', 'meta', 'form'],
    ADD_TAGS: ['semantics', 'annotation'],
    ADD_ATTR: [
      'target',
      'rel',
      'viewBox',
      'xmlns',
      'fill',
      'stroke',
      'stroke-width',
      'stroke-linecap',
      'stroke-linejoin',
      'd',
      'width',
      'height',
      'aria-hidden',
      'x',
      'y',
      'rx',
      'ry',
      'fill-rule',
      'clip-rule',
    ],
  })
}

/** Agent 聊天气泡：Markdown → 安全 HTML（不含 Typography，样式由页面 scoped 控制）。 */
export function renderAssistantMarkdown(src: string): string {
  const pre = prepareMarkdownText(src || '')
  const raw = md.render(pre)
  return sanitizeChatHtml(patchAnchorsWithoutTarget(raw))
}
