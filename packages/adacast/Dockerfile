FROM alpine AS base

# Disable the runtime transpiler cache by default inside Docker containers.
# On ephemeral containers, the cache is not useful
ARG BUN_RUNTIME_TRANSPILER_CACHE_PATH=0
ENV BUN_RUNTIME_TRANSPILER_CACHE_PATH=${BUN_RUNTIME_TRANSPILER_CACHE_PATH}
RUN apk add libgcc libstdc++ ripgrep

FROM base AS build-amd64
COPY dist/adacast-linux-x64-baseline-musl/bin/adacast /usr/local/bin/adacast
COPY dist/adacast-linux-x64-baseline-musl/bin/app /usr/local/share/adacast/app
COPY dist/adacast-linux-x64-baseline-musl/bin/python /usr/local/share/adacast/python

FROM base AS build-arm64
COPY dist/adacast-linux-arm64-musl/bin/adacast /usr/local/bin/adacast
COPY dist/adacast-linux-arm64-musl/bin/app /usr/local/share/adacast/app
COPY dist/adacast-linux-arm64-musl/bin/python /usr/local/share/adacast/python

ARG TARGETARCH
FROM build-${TARGETARCH}
RUN adacast --version
ENTRYPOINT ["adacast"]
