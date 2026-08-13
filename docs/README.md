# Miles Documentation

Live site: https://miles.radixark.com/docs

## Layout

```
docs/
├── docs.json        # Mintlify config: navigation, theme, redirects
├── index.md         # Homepage
├── getting-started/ models/ user-guide/ advanced/
├── examples/ developer/ ci/ blog/
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

1. Add or edit a `.md` file (e.g. `models/qwen/qwen4.md`). Every page needs frontmatter
   with a `title` and a `description` — the description becomes the meta description and
   the social preview text, so write one sentence that reads well on its own and stays
   under 160 characters. Mintlify renders `title` as the page's `h1`, so do not repeat it
   as a `#` heading in the body.
2. New pages need an entry in the `navigation` tree in `docs.json`, otherwise they won't
   show up in the sidebar — and, because indexing follows the navigation, they stay out of
   the sitemap and out of search results entirely.
3. When linking between pages, use absolute paths: `[Quick Start](/getting-started/quick-start)`.
   Drop the `.md` extension.
4. Do not edit anything under `examples/`. Those pages, and the Examples tab of
   `docs.json`, are generated from the `README.md` files under the repository's
   `examples/` directory, which is the single source of truth. Edit the README and run
   `python scripts/tools/sync_example_docs.py` — pre-commit runs it for you and fails if
   the two ever diverge.
5. Images and other assets go in `assets/` and are referenced the same way:
   `/assets/images/arch.png`. Group them into a subdirectory once a topic has more than
   one image, named after the page or area that uses them: `assets/images/dashboard/` for
   the dashboard screenshots, `assets/images/brand/` for the logo and favicon. A one-off
   image stays at the top level.
