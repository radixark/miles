
[fill here]

- how regiser_*_ci(labels=...) work? 
    - The label is matched to run-ci-* PR labels, which can trigger CI run with the tag.
- Where are labels design? where includes all labels? 
    - files name starting with test_ are files to be run. must add at least one label. Otherwise fast crash.
        - question: what is the file scan scope ?
    - tests/fast are auto labeld with cpu, and ran on cpu-only ubuntu runner.
- 
