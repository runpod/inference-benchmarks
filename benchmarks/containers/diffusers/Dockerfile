FROM --platform=linux/amd64 pytorch/pytorch@sha256:7b324d212a4450795b49edba9949b7cdc72429148a64e974334bfe5774d51385

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/opt/conda/bin:${PATH}" \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends openssh-server \
    && rm -rf /var/lib/apt/lists/* /etc/ssh/ssh_host_*
COPY requirements.txt verify_runtime.py /opt/sls-bench/
RUN python3 -m pip install --no-cache-dir -r /opt/sls-bench/requirements.txt pandas==2.2.3 \
    && python3 /opt/sls-bench/verify_runtime.py \
    && ln -s /opt/conda/bin/python3 /usr/local/bin/python3 \
    && mkdir -p /root/.ssh /run/sshd /workspace/bench \
    && chmod 700 /root/.ssh
COPY --chmod=755 bootstrap.sh /opt/sls-bench/bootstrap.sh
EXPOSE 22
ENTRYPOINT ["/opt/sls-bench/bootstrap.sh"]
