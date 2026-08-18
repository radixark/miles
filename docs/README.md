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

   Convention: every landing page (a tab's `index.md` — including the site homepage —
   a model family's `index.md`, `user-guide/environments.md`) titles itself after the
   tab or group it fronts (`Welcome`, `User Guide`, `DeepSeek`, …): the `title` feeds
   the `h1`, browser tab, and search results, which have no sidebar context. It then
   adds `sidebarTitle: Overview` so the sidebar shows a short, uniform label. The
   `sidebarTitle` is a constant — never rename it in step with the title.
2. New pages need an entry in the `navigation` tree in `docs.json`, otherwise they won't
   show up in the sidebar — and, because indexing follows the navigation, they stay out of
   the sitemap and out of search results entirely.

   Never give a navigation group a `root:` (a pre-commit hook rejects it). A rooted
   group header doubles as a link, which makes some headers navigate and others merely
   toggle — indistinguishable until clicked. Instead, list the group's landing page as
   its first entry in `pages` (with `sidebarTitle: Overview`, per the convention above),
   so every header is label-only and every navigation target is an explicit row.
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
