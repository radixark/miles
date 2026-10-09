# Review instructions

The repository rules live in `.claude/rules/*.md`; this file only says how to apply them in review.

- For each changed file, read every `.claude/rules/*.md` whose `paths:` frontmatter matches it, check the change against that rule, and name the rule file in the finding.
- A rule violation is a Nit unless it also causes a correctness bug.
- Report at most five Nits per review.
- Do not report what `.pre-commit-config.yaml` already enforces, or style on lines the PR did not change.
