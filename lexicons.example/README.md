# Lexicons template

These are minimal English seed lists demonstrating the format. They are **placeholders**, not a curated lexicon — the real measurements need many more words per category.

## Bootstrap

On a fresh clone:
```bash
cp -r lexicons.example lexicons
```

## Format

- One entry per line. Whitespace trimmed.
- Lines starting with `#` and blank lines are ignored.
- Entries are case-folded on load (write in any case).
- Multiple language files per category are merged into one set.

## Layout

```
lexicons/
  <category>/
    en.txt        # English (this template)
    <any>.txt     # any filename ending in .txt — file-stem labels in startup logs
  intents/
    <intent_name>/
      <lang>.txt
```

The lang-suffix is just a label for diagnostics; the loader merges any `.txt` file under a category, regardless of name.

## Adding a language

1. `cp lexicons/self_ref/en.txt lexicons/self_ref/<lang>.txt`
2. Translate, edit, save.
3. Re-run the binary. Startup log will show `lexicon: self_ref loaded (N entries from langs: en, <lang>)`.

Your local `lexicons/` is `.gitignored` — never pushed.
