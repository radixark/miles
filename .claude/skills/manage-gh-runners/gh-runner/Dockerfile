# gh-runner — env-var-driven GitHub Actions runner image for radixark/miles CI.
# Built to be a drop-in for manage-runners.sh: it is launched with an
# initially-EMPTY, identity-bind-mounted RUNNER_DIR (/data/runner-setup/data/<name>)
# and must populate that dir from a baked-in runner distribution, then
# config.sh --replace / run.sh. Registration (.runner) persists in the host
# volume, so `docker restart` keeps the runner registered.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    RUNNER_ALLOW_RUNASROOT=1 \
    RUNNER_VERSION=2.334.0

# ca-certificates/curl/tar/git/jq: runner + job plumbing.
# docker.io: the CLI the runner uses to spawn sibling job containers through
#            the bind-mounted host docker socket (daemon is NOT run here).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl tar sudo git jq docker.io \
    && rm -rf /var/lib/apt/lists/*

# Stage the actions-runner distribution into a template dir; entrypoint copies
# it into the (empty) RUNNER_DIR on first start.
RUN mkdir -p /opt/actions-runner-dist \
    && curl -fsSL -o /tmp/runner.tar.gz \
        "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz" \
    && tar -xzf /tmp/runner.tar.gz -C /opt/actions-runner-dist \
    && rm /tmp/runner.tar.gz \
    && apt-get update \
    && /opt/actions-runner-dist/bin/installdependencies.sh \
    && rm -rf /var/lib/apt/lists/*

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
