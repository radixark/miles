# Codex Instructions

Before working in this repository, discover and read all Markdown rule files under `.claude/rules/`, including subdirectories. This path is relative to the repository root. Treat these files as repository instructions; newly added rules belong to the same set.

Apply each rule according to its `paths` frontmatter, matching globs against repository-relative file paths. Rules without `paths` apply repository-wide. Preserve any further applicability conditions in the rule body, such as applying only to new or substantially modified code.

Recheck which rules apply when the task expands to additional files or components.

## Code Review Rules

For pull request reviews, also read and follow `REVIEW.md` at the repository root.
