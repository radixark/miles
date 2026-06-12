# Miles Documentation

Live site: https://miles.radixark.com/docs

## Layout

```
docs/
├── docs.json        # Mintlify config: navigation, theme, redirects
├── index.md         # Homepage
├── getting-started/ models/ user-guide/ advanced/
├── examples/ developer/ platforms/ blog/
└── assets/          # Images and stylesheets
```

## Previewing locally

```bash
npm i -g mint
cd docs
mint dev
```

Then open http://localhost:3000.

## Adding or editing a page

1. Add or edit a `.md` file (e.g. `models/qwen/qwen4.md`).
2. New pages need an entry in the `navigation` tree in `docs.json`, otherwise they won't
   show up in the sidebar.
3. When linking between pages, use absolute paths: `[Quick Start](/getting-started/quick-start)`.
   Drop the `.md` extension.
4. Images and other assets go in `assets/` and are referenced the same way:
   `/assets/images/arch.png`.
