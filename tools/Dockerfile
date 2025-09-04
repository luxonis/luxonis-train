FROM python:3.10-slim

ARG OLD_VERSION=latest
ARG NEW_VERSION=main
ENV OLD_VERSION=${OLD_VERSION}
ENV NEW_VERSION=${NEW_VERSION}

WORKDIR /app


RUN <<EOF
    set -e

    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-dev \
        python3-pip \
        libglib2.0-0 \
        libgl1 \
        git

EOF

RUN <<EOF
    set -e

    python3 -m venv old
    . ./old/bin/activate
    pip install -U pip
    if [ "${OLD_VERSION}" = "latest" ]; then
        pip install luxonis-train
    else
        pip install luxonis-train==${OLD_VERSION}
    fi

EOF

RUN <<EOF
    set -e

    python3 -m venv new
    . ./new/bin/activate
    pip install -U pip

    pip install "luxonis-train @ git+https://github.com/luxonis/luxonis-train.git@${NEW_VERSION}"

EOF


COPY migrate.sh migrate_weights.py generate_execution_order.py tools/
RUN chmod +x tools/migrate.sh

ENTRYPOINT ["/app/tools/migrate.sh"]
