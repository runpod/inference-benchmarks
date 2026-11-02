import json
import os
import time
from datetime import datetime, timezone

import httpx


class SupplyUnavailable(RuntimeError):
    pass


class ProvisionRejected(RuntimeError):
    """Definitive API rejection, distinct from an ambiguous network/create failure."""


class Runpod:
    def __init__(self, client=None):
        key = os.environ.get("RUNPOD_ORGS_API_KEY") or os.environ.get("runpod_orgs_api_key")
        if not key and client is None:
            raise ValueError("Set RUNPOD_ORGS_API_KEY (or source ~/.zshrc)")
        self.client = client or httpx.Client(
            headers={"Authorization": f"Bearer {key}", "User-Agent": "sls-bench/0.1"}, timeout=45
        )

    def rest(self, method, path, **kwargs):
        # Never automatically retry creation: an ambiguous timeout could create two billable Pods.
        response = self.client.request(method, f"https://rest.runpod.io/v1{path}", **kwargs)
        if method == "DELETE" and response.status_code == 404:
            return None
        if response.is_error:
            raise RuntimeError(f"Runpod {method} {path}: HTTP {response.status_code}: {response.text[:600]}")
        return response.json() if response.content else None

    def graphql(self, query, variables=None):
        response = self.client.post(
            "https://api.runpod.io/graphql", json={"query": query, "variables": variables or {}}
        )
        if response.is_error:
            error_type = ProvisionRejected if 400 <= response.status_code < 500 else RuntimeError
            raise error_type(f"Runpod GraphQL HTTP {response.status_code}: {response.text[:2000]}")
        value = response.json()
        if value.get("errors"):
            errors = value["errors"]
            message = "; ".join(e.get("message", "Unknown GraphQL error") for e in errors)
            if any(e.get("extensions", {}).get("code") == "SUPPLY_CONSTRAINT" for e in errors):
                raise SupplyUnavailable(message)
            raise ProvisionRejected(f"Runpod GraphQL: {message}")
        return value["data"]

    def gpus(self):
        return self.graphql(
            "query { gpuTypes { id displayName manufacturer gpuGeneration memoryInGb securePrice communityPrice } }"
        )["gpuTypes"]

    def pods(self, name=None):
        params = {"includeMachine": "true"}
        if name:
            params["name"] = name
        return self.rest("GET", "/pods", params=params)

    def pod(self, pod_id):
        return self.rest("GET", f"/pods/{pod_id}", params={"includeMachine": "true"})

    def delete(self, pod_id):
        for attempt in range(3):
            try:
                self.rest("DELETE", f"/pods/{pod_id}")
                return
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2**attempt)

    def create(self, name, target, public_key, ttl_seconds):
        # Only the SSH PUBLIC key is passed. Org API credentials never leave the controller.
        bootstrap = r"""set -eux
export DEBIAN_FRONTEND=noninteractive
if ! command -v sshd >/dev/null 2>&1; then
    apt-get update
    apt-get install -y --no-install-recommends openssh-server
fi
python3 -c 'import pandas; assert pandas.__version__ == "2.2.3"' || python3 -m pip install 'pandas==2.2.3'
# SSH sessions do not inherit Conda's PATH from the container entrypoint.
python_bin="$(command -v python3)"
if [ "$python_bin" != /usr/local/bin/python3 ] && [ ! -e /usr/local/bin/python3 ]; then
    ln -s "$python_bin" /usr/local/bin/python3
fi
mkdir -p /root/.ssh /run/sshd /workspace/bench
chmod 700 /root/.ssh
printf '%s\n' "$PUBLIC_KEY" > /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
ssh-keygen -A
printf '\nPasswordAuthentication no\nPermitRootLogin prohibit-password\n' >> /etc/ssh/sshd_config
(sleep "$BENCH_TTL_SECONDS"; kill -TERM 1) &
exec /usr/sbin/sshd -D -e
"""
        payload = {
            "name": name,
            "imageName": target["image"],
            "cloudType": target["cloud"],
            "computeType": "GPU",
            "gpuTypeId": target["gpu_type"],
            "gpuCount": target["gpu_count"],
            "containerDiskInGb": target["disk_gb"],
            "volumeInGb": 0,
            "minVcpuCount": target["min_vcpu"] * target["gpu_count"],
            "minMemoryInGb": target["min_ram_gb"] * target["gpu_count"],
            "ports": "22/tcp",
            "supportPublicIp": True,
            "dockerArgs": json.dumps({"entrypoint": ["/bin/bash"], "cmd": ["-lc", bootstrap]}),
            "env": [
                {"key": "PUBLIC_KEY", "value": public_key},
                {"key": "BENCH_TTL_SECONDS", "value": str(ttl_seconds)},
            ],
            "terminateAfter": datetime.fromtimestamp(time.time() + ttl_seconds, timezone.utc).isoformat(),
        }
        if target.get("container_registry_auth_id"):
            payload["containerRegistryAuthId"] = target["container_registry_auth_id"]
        if target.get("min_cuda_version"):
            payload["minCudaVersion"] = target["min_cuda_version"]
        if target["data_centers"]:
            payload["dataCenterIds"] = target["data_centers"]
        # GraphQL accepts the current catalog's string SKUs; REST v1 has a stale GPU enum.
        return self.graphql(
            "mutation Deploy($input: PodFindAndDeployOnDemandInput!) { "
            "podFindAndDeployOnDemand(input: $input) { id name imageName machineId costPerHr } }",
            {"input": payload},
        )["podFindAndDeployOnDemand"]


def safe_metadata(pod):
    """Allowlist: existing Pod env may contain unrelated secrets."""
    return {
        k: pod.get(k)
        for k in (
            "id",
            "name",
            "image",
            "gpu",
            "machineId",
            "machine",
            "costPerHr",
            "adjustedCostPerHr",
            "memoryInGb",
            "vcpuCount",
            "desiredStatus",
            "lastStartedAt",
            "publicIp",
            "portMappings",
        )
    }
