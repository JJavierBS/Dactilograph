export const DYNAMIC_LETTERS: ReadonlySet<string> = new Set([
  "J",
  "LL",
  "Ñ",
  "RR",
  "V",
  "W",
  "X",
  "Y",
  "Z",
]);

export const ALPHABET: readonly string[] = [
  "A",
  "B",
  "C",
  "CH",
  "D",
  "E",
  "F",
  "G",
  "H",
  "I",
  "J",
  "K",
  "L",
  "LL",
  "M",
  "N",
  "Ñ",
  "O",
  "P",
  "Q",
  "R",
  "RR",
  "S",
  "T",
  "U",
  "V",
  "W",
  "X",
  "Y",
  "Z",
];

export function isDynamic(letter: string): boolean {
  return DYNAMIC_LETTERS.has(letter);
}
