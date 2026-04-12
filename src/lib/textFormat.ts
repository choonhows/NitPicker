export const normalizeLegacySymbols = (text: string): string =>
  text
    .replace(/\uF0AE|\uF0E0/g, '→')
    .replace(/\uF0DF/g, '←')
    .replace(/\uF0A3/g, '≤')
    .replace(/\uF0B3/g, '≥')
    .replace(/\uF0B8/g, '÷')
    .replace(/\uF0B4/g, '×')
    .replace(/\uF0B9/g, '≠')

export const normalizeMathDelimiters = (text: string): string =>
  text
    .replace(/\\\((.+?)\\\)/gs, (_, expression: string) => `$${expression}$`)
    .replace(/\\\[(.+?)\\\]/gs, (_, expression: string) => `$$${expression}$$`)
