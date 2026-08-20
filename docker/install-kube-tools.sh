#!/usr/bin/env bash
set -euxo pipefail

KUBECTL_VERSION="v1.36.3"
HELM_VERSION="v4.2.3"

arch="${1:-}"
[ -n "$arch" ] || arch="$(dpkg --print-architecture)"

curl -fsSL -o /tmp/kubectl "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/${arch}/kubectl"
curl -fsSL -o /tmp/kubectl.sha256 "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/${arch}/kubectl.sha256"
echo "$(cat /tmp/kubectl.sha256)  /tmp/kubectl" | sha256sum -c -
install -m 0755 /tmp/kubectl /usr/local/bin/kubectl

curl -fsSL -o /tmp/helm.tar.gz "https://get.helm.sh/helm-${HELM_VERSION}-linux-${arch}.tar.gz"
curl -fsSL -o /tmp/helm.tar.gz.sha256 "https://get.helm.sh/helm-${HELM_VERSION}-linux-${arch}.tar.gz.sha256sum"
echo "$(awk '{print $1}' /tmp/helm.tar.gz.sha256)  /tmp/helm.tar.gz" | sha256sum -c -
tar -xzf /tmp/helm.tar.gz -C /tmp
install -m 0755 "/tmp/linux-${arch}/helm" /usr/local/bin/helm

rm -rf /tmp/kubectl /tmp/kubectl.sha256 /tmp/helm.tar.gz /tmp/helm.tar.gz.sha256 "/tmp/linux-${arch}"

kubectl version --client
helm version --short
