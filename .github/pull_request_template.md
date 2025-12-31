## Description

<!-- Please provide a clear and concise description of what this PR does. -->

## Code Style Compliance

**⚠️ IMPORTANT: Please ensure your code follows the [Code Style Guide](../../docs/en/developer_guide/code_style.md) before submitting this PR.**

Key points to check:

- [ ] Performance: Minimized synchronization calls (`.item()`, `.cpu()`, `.tolist()`) in inference paths
- [ ] Architecture: No duplicate code > 5 lines; files < 2,000 lines
- [ ] Function Purity: Avoided in-place modification of input arguments (unless explicitly documented for memory optimization)
- [ ] Pythonic: Lean constructors, minimal dynamic attributes, proper type hints on public APIs
- [ ] Testing: Provided a test script that reviewers can copy & paste to run immediately

## Changes Made

<!-- List the main changes in this PR -->

## Testing

<!-- Please provide a test script that reviewers can copy & paste to run immediately -->

```bash
# Paste your test command here
```

## Related Issues

<!-- Link to related issues, if any -->

## Checklist

- [ ] Code follows the [Code Style Guide](docs/en/developer_guide/code_style.md)
- [ ] Tests pass locally
- [ ] Documentation updated (if applicable)
- [ ] Type hints added for public APIs
- [ ] No unnecessary synchronization calls in hot paths

