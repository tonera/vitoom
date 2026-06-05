/**
 * TranslateGemma 上下文约 2K tokens；配合 max_new_tokens=768 与模板开销。
 * Markdown/代码文档的 tokenizer 开销远高于 CJK 粗算，需同时限制 token 与字符。
 */
export const TRANSLATE_MAX_NEW_TOKENS = 768
export const TRANSLATE_CONTEXT_TOKENS = 2048
export const TRANSLATE_TEMPLATE_TOKEN_BUDGET = 120
/** 单段输入 token 上限（留足生成与模板余量） */
export const TRANSLATE_MAX_INPUT_TOKENS =
  TRANSLATE_CONTEXT_TOKENS - TRANSLATE_MAX_NEW_TOKENS - TRANSLATE_TEMPLATE_TOKEN_BUDGET

const CJK_RE = /[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]/

function cjkRatio(text: string): number {
  if (!text) return 0
  let cjk = 0
  for (const ch of text) {
    if (CJK_RE.test(ch)) cjk += 1
  }
  return cjk / text.length
}

/** 粗算 token；Markdown/ASCII 偏保守，避免低估导致模型截断输入。 */
export function estimateTranslateTokens(text: string): number {
  let tokens = 0
  let inFence = false

  for (let i = 0; i < text.length; i += 1) {
    const ch = text.charAt(i)
    if (ch === '`' && text.charAt(i + 1) === '`' && text.charAt(i + 2) === '`') {
      inFence = !inFence
      tokens += 2
      i += 2
      continue
    }

    if (inFence) {
      tokens += 0.55
      continue
    }

    if (CJK_RE.test(ch)) tokens += 1.05
    else if (/\s/.test(ch)) tokens += 0.2
    else if (/[a-zA-Z0-9]/.test(ch)) tokens += 0.45
    else tokens += 0.65
  }
  return Math.ceil(tokens)
}

function maxCharsForChunk(text: string, maxTokens: number): number {
  const ratio = cjkRatio(text)
  if (ratio >= 0.35) return Math.min(950, maxTokens)
  if (ratio >= 0.1) return Math.min(1400, Math.floor(maxTokens * 1.15))
  return Math.min(1200, Math.floor(maxTokens * 0.95))
}

export function chunkWithinLimit(text: string, maxInputTokens: number = TRANSLATE_MAX_INPUT_TOKENS): boolean {
  const sample = String(text || '')
  if (!sample) return true
  return (
    estimateTranslateTokens(sample) <= maxInputTokens &&
    sample.length <= maxCharsForChunk(sample, maxInputTokens)
  )
}

function findPreferredSplitIndex(text: string, maxTokens: number): number {
  if (!text) return 0
  if (chunkWithinLimit(text, maxTokens)) return text.length

  let lo = 1
  let hi = text.length
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2)
    if (chunkWithinLimit(text.slice(0, mid), maxTokens)) lo = mid
    else hi = mid - 1
  }

  const idx = lo
  if (idx <= 0) return Math.min(text.length, 1)

  const searchStart = Math.max(1, Math.floor(idx * 0.5))
  const segment = text.slice(searchStart, idx)

  const breakPatterns = [
    /\n#{1,6}\s/g,
    /\n\n+/g,
    /\n/g,
    /[。！？!?](?=\s|$)/g,
    /[;；](?=\s|$)/g,
  ]
  for (const pattern of breakPatterns) {
    let last = -1
    const re = new RegExp(pattern.source, pattern.flags)
    let match: RegExpExecArray | null
    while ((match = re.exec(segment)) !== null) {
      last = match.index + match[0].length
    }
    if (last > 0) {
      const cut = searchStart + last
      if (cut > 0 && chunkWithinLimit(text.slice(0, cut), maxTokens)) return cut
    }
  }

  return idx
}

/** 按 Markdown 标题切成逻辑块，避免把「标题」和「正文」拆到不同策略边界时丢失尾部。 */
function splitMarkdownBlocks(text: string): string[] {
  const source = String(text || '')
  if (!source) return []
  if (!/(^|\n)#{1,6}\s/m.test(source)) return [source]

  const blocks: string[] = []
  const re = /(?=^#{1,6}\s)/gm
  let last = 0
  for (const match of source.matchAll(re)) {
    const idx = match.index ?? 0
    if (idx > last) blocks.push(source.slice(last, idx))
    last = idx
  }
  if (last < source.length) blocks.push(source.slice(last))
  return blocks.filter((b) => b.length > 0)
}

function hardSplitOversizedBlock(block: string, maxInputTokens: number): string[] {
  if (chunkWithinLimit(block, maxInputTokens)) return [block]

  const pieces: string[] = []
  let rest = block
  let guard = 0
  while (rest && guard < 50_000) {
    guard += 1
    if (chunkWithinLimit(rest, maxInputTokens)) {
      pieces.push(rest)
      rest = ''
      break
    }
    const cut = findPreferredSplitIndex(rest, maxInputTokens)
    if (cut <= 0) {
      pieces.push(rest.slice(0, 1))
      rest = rest.slice(1)
      continue
    }
    pieces.push(rest.slice(0, cut))
    rest = rest.slice(cut)
  }
  if (rest) pieces.push(rest)
  return pieces.filter((p) => p.length > 0)
}

function packBlocksIntoChunks(blocks: string[], maxInputTokens: number): string[] {
  const chunks: string[] = []
  let buf = ''

  const flush = () => {
    if (buf) {
      chunks.push(buf)
      buf = ''
    }
  }

  for (const block of blocks) {
    const subPieces = hardSplitOversizedBlock(block, maxInputTokens)
    for (const piece of subPieces) {
      if (!buf) {
        buf = piece
        continue
      }
      const merged = buf + piece
      if (chunkWithinLimit(merged, maxInputTokens)) {
        buf = merged
      } else {
        flush()
        buf = piece
      }
    }
  }
  flush()
  return chunks
}

/** 将长文本切成多段；保证 ``chunks.join('') === source``。 */
export function splitTextForTranslation(
  text: string,
  maxInputTokens: number = TRANSLATE_MAX_INPUT_TOKENS,
): string[] {
  const source = String(text ?? '')
  if (!source) return []
  if (chunkWithinLimit(source, maxInputTokens)) return [source]

  const blocks = splitMarkdownBlocks(source)
  let chunks = packBlocksIntoChunks(blocks, maxInputTokens)

  if (chunks.join('') !== source) {
    chunks = hardSplitOversizedBlock(source, maxInputTokens)
  }
  if (chunks.join('') !== source) {
    console.error('[translateChunk] split integrity check failed', {
      sourceLen: source.length,
      joinedLen: chunks.join('').length,
    })
    throw new Error('translate chunk split failed: content would be lost')
  }
  return chunks.filter((chunk) => chunk.length > 0)
}

export function countTranslateChunks(text: string): number {
  return splitTextForTranslation(text).length
}

const MD_HEADING_START = /^#{1,6}\s/

/**
 * 根据原文相邻分段的边界，推断译文拼接时应插入的分隔符。
 * 各段独立翻译时模型常会吞掉段尾/段首换行，需在拼接时补回。
 */
export function inferChunkJoinSeparator(leftChunk: string, rightChunk: string): string {
  const left = String(leftChunk || '')
  const right = String(rightChunk || '')
  if (!left || !right) return ''

  const rightTrimStart = right.trimStart()

  if (MD_HEADING_START.test(rightTrimStart)) {
    return '\n\n'
  }

  if (left.endsWith('\n\n') || right.startsWith('\n\n')) return '\n\n'
  if (left.endsWith('\n') && right.startsWith('\n')) return '\n\n'
  if (left.endsWith('\n') || right.startsWith('\n')) return '\n'

  return ''
}

function mergeTranslatedParts(left: string, right: string, separator: string): string {
  if (!separator) return left + right
  const base = left.replace(/\s+$/, '')
  const next = right.replace(/^\s+/, '')
  return `${base}${separator}${next}`
}

/** 将各段译文按原文边界拼接，避免标题/段落与上一段粘连。 */
export function joinTranslatedChunks(parts: string[], sourceChunks: string[]): string {
  if (!parts.length) return ''
  if (parts.length === 1) return parts[0] ?? ''
  if (parts.length !== sourceChunks.length) {
    return parts.join('')
  }

  let out = parts[0] ?? ''
  for (let i = 1; i < parts.length; i += 1) {
    const sep = inferChunkJoinSeparator(sourceChunks[i - 1] ?? '', sourceChunks[i] ?? '')
    out = mergeTranslatedParts(out, parts[i] ?? '', sep)
  }
  return out
}
