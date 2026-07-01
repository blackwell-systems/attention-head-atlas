# Probe Texts

Fixed texts used to measure head behavior at each checkpoint. These never change
so measurements are comparable across training steps.

Each probe is designed to isolate one behavior type.

| File | Tests | Key signal |
|------|-------|-----------|
| `prose.txt` | Positional, content, P0 sink | Natural language, no structure |
| `code.txt` | Bracket matching, syntactic, delimiter (`;(){}`) | Nested function calls, scopes |
| `structured.txt` | Delimiter attention (pipes, commas, colons) | GCF-style tabular data |
| `induction.txt` | Induction heads (in-context copying) | Repeated patterns (A B ... A ?) |
| `duplicates.txt` | Duplicate token heads | Same tokens at multiple positions |
| `brackets.txt` | Bracket/paren matching specifically | Deep nesting, Dyck-like |
