/**
 * Remove keys with value `undefined` from objects (deep), and remove `undefined`
 * entries from arrays. Keeps `null`, `false`, `0`, and empty strings.
 */
export function stripUndefinedDeep<T>(input: T): T {
  if (input === undefined) return input
  if (input === null) return input

  // Arrays: strip undefined elements, deep-clean children
  if (Array.isArray(input)) {
    const out = (input as any[])
      .map((x) => stripUndefinedDeep(x))
      .filter((x) => x !== undefined)
    return out as any as T
  }

  // Plain objects: strip undefined keys, deep-clean values
  if (typeof input === 'object') {
    const obj = input as Record<string, any>
    const out: Record<string, any> = {}
    for (const k of Object.keys(obj)) {
      const v = stripUndefinedDeep(obj[k])
      if (v !== undefined) out[k] = v
    }
    return out as any as T
  }

  // primitives
  return input
}

