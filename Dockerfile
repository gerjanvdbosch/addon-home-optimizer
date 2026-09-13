ARG BUILD_FROM=ghcr.io/hassio-addons/debian-base:8.1.4
FROM ${BUILD_FROM}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-minimal \
        python3-pip \
        python3-venv && \
    python3 -m venv /opt/venv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/*

WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/

RUN /opt/venv/bin/pip install --no-cache-dir \
    --extra-index-url https://www.piwheels.org/simple \
    .

COPY addon.sh /addon.sh
RUN chmod a+x /addon.sh

ARG BUILD_ARCH
ARG BUILD_DATE
ARG BUILD_DESCRIPTION
ARG BUILD_NAME
ARG BUILD_REF
ARG BUILD_REPOSITORY
ARG BUILD_VERSION
LABEL \
    io.hass.name="${BUILD_NAME}" \
    io.hass.description="${BUILD_DESCRIPTION}" \
    io.hass.arch="${BUILD_ARCH}" \
    io.hass.type="addon" \
    io.hass.version="${BUILD_VERSION}" \
    maintainer="gerjanvandenbosch"

CMD ["/addon.sh"]
