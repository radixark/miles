# Code Style Guide

## Performance First

Miles is a high-performance system where every millisecond matters.

- **Minimize Synchronization**: Strictly avoid frequent calls to `.item()`, `.cpu()`, or `.tolist()` within the model inference path.
- **Data Processing Vectorized**: Keep data processing vectorized on the GPU whenever possible.
- **Optimize Hot Paths**: Prioritize low-overhead implementations for critical function paths. [Example](https://github.com/radixark/miles/pull/260#discussion_r2654301156)

## Architecture & Decoupling

Maintain a clean codebase to facilitate collaboration and long-term maintenance. For features not yet widely accepted by the community, prioritize adding abstract base classes to `/miles` and implementing specific logic under `/examples`. [Example](https://github.com/THUDM/slime/pull/429) [Example 2](https://github.com/THUDM/slime/pull/1141)

- **Don't Repeat Yourself (DRY)**: Duplicate code snippets exceeding 5 lines must be extracted into shared functions.
- **Keep Files Concise**: If a single file exceeds 2,000 lines, it must be split into multiple smaller files (e.g., using Mixin patterns or sub-modules).

## Function Purity

Prioritize writing **Pure Functions**. Avoid in-place modification of input arguments.

> **Exception**: If an in-place operation is required for extreme memory optimization (e.g., in the forward pass), it must be accompanied by an explicit comment.

## Pythonic & Clean

- **Lean Constructors**: Keep `__init__` parameters concise. Avoid passing massive, complex configuration objects; instead, pass only the necessary parameters.
- **Avoid Dynamic Attributes**: Minimize the use of `getattr` or `setattr`. Code should be explicit for better traceability. [Example](https://github.com/THUDM/slime/pull/1141#discussion_r2654651150) [Example 2](https://github.com/THUDM/slime/pull/1141#discussion_r2658247124)
- **Ternary Operator Limits**: Use `a if condition else b` only for very simple and clear cases. Complex logic must use standard `if-else` blocks.
- **Extract Complex Logic**: In multi-branch conditionals, if a specific branch contains multiple lines of logic, encapsulate it into a standalone private function.
- **Type Hints**: All public APIs and function signatures must include type hints.
- **Access Permission Indicator**: Use `_private` style to indicate this is a function to this class or file only, otherwise, it's a public function that can be exposed outside.

## Testing Efficiency

PR merge speed depends on verification/testing speed. 

- Please provide a verification script in your PR description that a reviewer can **copy & paste** to run immediately. [Example](https://github.com/radixark/miles/pull/246#issuecomment-3701278030)
- For important features, we need to also add the CI unit test for the specific feature like LoRA, SFT etc.
