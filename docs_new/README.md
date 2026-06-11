# Miles Documentation

This directory contains the Miles documentation site, built with [Mintlify](https://mintlify.com).

It was migrated from the standalone [radixark/miles-doc](https://github.com/radixark/miles-doc)
repo, with full commit history preserved.

## Layout

```
docs_new/
├── docs.json        # Mintlify config: navigation, theme, redirects
└── miles/docs/      # All page content (markdown) and assets
```

The `miles/docs/` nesting looks odd, but don't flatten it: `docs.json` and the 280+ internal
links across the pages are all root-relative paths like `/miles/docs/models/index`, resolved
against this directory as the Mintlify content root. Flattening would break every one of them.

## Previewing locally

```bash
npm i -g mint
cd docs_new
mint dev
```

Then open http://localhost:3000.

## Adding or editing a page

1. Add or edit a `.md` file under `miles/docs/` (e.g. `miles/docs/models/qwen/qwen4.md`).
2. New pages need an entry in the `navigation` tree in `docs.json`, otherwise they won't
   show up in the sidebar.
3. When linking between pages, use absolute paths: `[Quick Start](/miles/docs/getting-started/quick-start)`.
   Drop the `.md` extension.
4. Images and other assets go in `miles/docs/assets/` and are referenced the same way:
   `/miles/docs/assets/images/arch.png`.
