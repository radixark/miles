"""Write release-lock.json for a release branch cut.

Usage: write_release_lock.py <branch> <sglang_sha> <sglang_wheel> <megatron_sha> <ci_image_tag> <fp_cu130_x86> <fp_cu130_arm64>
"""

import json
import sys


def main() -> int:
    branch, sglang, sglang_wheel, megatron, image, fp_x86, fp_arm = sys.argv[1:]
    assert len(sglang) == 40 and len(megatron) == 40, "failed to resolve dependency SHAs"
    json.dump(
        {
            "release_branch": branch,
            "sglang_repo": "sgl-project/sglang",
            "sglang_branch": "sglang-miles",
            "sglang_commit": sglang,
            # the miles wheel built from this branch depends on exactly this sglang build (setup.py)
            "sglang_wheel": sglang_wheel,
            "megatron_repo": "radixark/Megatron-LM",
            "megatron_branch": "miles-main",
            "megatron_commit": megatron,
            "ci_image_tag": image,
            # Audit only: wheels rolling tags can have assets replaced in place,
            # so they cannot be pinned by name (see docker-build.yml).
            "wheels_fingerprint_cu130_x86_64": fp_x86,
            "wheels_fingerprint_cu130_aarch64": fp_arm,
        },
        open("release-lock.json", "w"),
        indent=2,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
