# Docker CI Refactor — Design

**Date:** 2026-06-04
**Branch:** `docker-ci-refactor` (off `origin/main` `c8b6697f2`)
**Goal:** Refactor the monolithic `docker-build.yml` into an sglang-style architecture — a thin orchestrator plus reusable sub-workflows — while preserving all current behavior. No new image-build capabilities beyond an `image_repo` staging input.

## Motivation

`docker-build.yml` today is a single file with three jobs (`check-upstream`, `build-and-push`, `build-and-push-dev-glm`). The build steps are duplicated between the two build jobs, the prune logic and image repo are hardcoded, and there is no safe way to test workflow changes without pushing to the production `radixark/miles` repo.

The sglang `release-docker-dev.yml` is "done well" because of its **structure**: a thin trigger workflow + reusable `workflow_call` sub-workflows + a `prepare` job that computes all config as outputs + an `image_repo` input for staging. We adopt that structure.

Out of scope (not selected by the user): reliability changes (retry/cache), merging/removing `release-docker.yaml`, and multi-arch manifest merging. `release-docker.yaml` is left untouched.

## File Structure

```
.github/workflows/
  docker-build.yml              # thin orchestrator: triggers + prepare job, calls reusables
  _docker-build-and-publish.yml # reusable (workflow_call): build + push ONE variant
  _docker-cleanup.yml           # reusable (workflow_call): prune old dev tags
docker/
  build.py                      # + dev-glm variant, + --image-repo override
  glm5/Dockerfile.dev-glm       # FROM ${BASE_IMAGE} (parameterized base)
```

## Components

### 1. `docker-build.yml` — orchestrator

**Triggers (unchanged):**
- `push` to `main` on `docker/Dockerfile`
- `schedule`: `0 0,12 * * *` (every 12h)
- `workflow_dispatch` with inputs: `variant`, `image_tag` (dev/latest/custom), `custom_tag`, `dockerfile`, `simulate_schedule`, **and new `image_repo` (default `radixark/miles`)**

**New:** top-level `concurrency` block (group keyed on workflow + ref, `cancel-in-progress: true`) so overlapping runs don't race.

**`prepare` job** (`runs-on: ubuntu-latest`): merges today's `check-upstream` logic and emits outputs consumed by downstream jobs:
- `should_build` — schedule gate via cached upstream SHAs (sglang `sglang-miles` + NVIDIA `Megatron-LM/main`); for push/dispatch always `true`.
- `variants` — JSON array of variants to build. Schedule/push → `["primary","dev-glm"]` (dev-glm depends on the freshly built `dev`). Dispatch → `[<chosen variant>]`.
- `image_tag`, `custom_tag`, `dockerfile`, `image_repo` — passthrough/derived.
- `retag_latest` — true on schedule/simulate.
- `prune` — true on schedule.

**Build call:** `uses: ./.github/workflows/_docker-build-and-publish.yml` with `strategy.matrix` over `fromJSON(needs.prepare.outputs.variants)`, `secrets: inherit`, gated on `should_build`. `dev-glm` ordering relative to `primary` is handled by either `max-parallel: 1` with ordered matrix, or a dependency note (see Open Questions — resolved: matrix entries run in order is NOT guaranteed, so dev-glm is split into a second matrix-less call that `needs` the primary build on schedule/push).

**Cleanup call:** `uses: ./.github/workflows/_docker-cleanup.yml` when `prune == true`, `secrets: inherit`.

### 2. `_docker-build-and-publish.yml` — reusable build

`on: workflow_call` with inputs: `variant` (required), `image_tag`, `custom_tag`, `dockerfile`, `image_repo` (default `radixark/miles`), `push` (bool), `retag_latest` (bool).

Steps (the previously-duplicated path, now single):
1. `actions/checkout@v4`
2. `docker/setup-buildx-action@v3` (moby/buildkit, network=host)
3. install `python3` + `typer`
4. `docker/login-action@v3` with `secrets.DOCKERHUB_USERNAME` / `secrets.DOCKERHUB_TOKEN`
5. `python docker/build.py --variant <variant> --image-tag <image_tag> --dockerfile <dockerfile> --image-repo <image_repo> [--custom-tag ...] [--push]`
6. if `retag_latest`: `docker buildx imagetools create -t <image_repo>:latest <image_repo>:dev`

`runs-on: self-hosted` (unchanged from today).

### 3. `_docker-cleanup.yml` — reusable prune

`on: workflow_call` with inputs: `image` (default `radixark/miles`), `tag_pattern` (default `^dev-[0-9]{12}$`), `keep` (default `20`). Body = today's Docker Hub JWT-login + paginated tag list + sort + delete-oldest logic, parameterized by the inputs instead of hardcoded constants. Uses `secrets.DOCKERHUB_USERNAME` / `secrets.DOCKERHUB_TOKEN`.

### 4. `docker/build.py` changes

- Add `--image-repo` option (default empty). When set, it overrides each variant's hardcoded `image`, so `--image-repo radixark/miles-staging` pushes the whole tag set to staging.
- Add a `dev-glm` entry to `VARIANTS` and to the `Variant` enum: image `radixark/miles`, dockerfile default `docker/glm5/Dockerfile.dev-glm`, no postfix, and a build arg that points its base at `<image-repo>:dev` (so staging dev-glm layers on staging dev).

### 5. `docker/glm5/Dockerfile.dev-glm` change

```dockerfile
ARG BASE_IMAGE=radixark/miles:dev
FROM ${BASE_IMAGE}
RUN pip install git+https://github.com/huggingface/transformers.git@76732b4e7120808ff989edbd16401f61fa6a0afa
```

`build.py` passes `--build-arg BASE_IMAGE=<image-repo>:dev` for the `dev-glm` variant.

## Behavior Preservation Checklist

| Behavior | Today | After |
|---|---|---|
| push-to-main on Dockerfile builds dev | yes | yes (orchestrator → build reusable) |
| 12h schedule with upstream-SHA gate | yes | yes (prepare job) |
| dispatch: variant/image_tag/custom_tag/dockerfile | yes | yes + `image_repo` |
| dev / latest / custom + timestamped dev tag | yes (build.py) | yes (build.py unchanged for tagging) |
| `latest → dev` retag on schedule | yes | yes (`retag_latest` input) |
| prune old `dev-YYYYMMDDHHMM` tags, keep 20 | yes | yes (cleanup reusable) |
| dev-glm rebuilt after dev | yes (separate job) | yes (dev-glm variant via build.py, runs after primary) |

## Open Questions (resolved)

- **dev-glm via matrix ordering:** matrix entry order is not a hard guarantee, and dev-glm needs `primary`/`dev` to exist first. Resolved: keep `primary` (and any dispatch-chosen variant) in the matrix build job; run `dev-glm` as a separate job that `needs` the build job, only on schedule/push (and dispatch with `image_tag=dev`).
- **Staging secrets:** `radixark/miles-staging` is assumed to live under the same Docker Hub account, so the same `DOCKERHUB_USERNAME`/`DOCKERHUB_TOKEN` secrets work. Verify the staging repo exists before first staging run.

## Testing

- `python docker/build.py --variant dev-glm --image-tag dev --image-repo radixark/miles-staging --dry-run` prints the expected `docker buildx build` command with the right base-image build arg and tags.
- `--dry-run` for `primary` with `--image-repo radixark/miles-staging` shows staging tags.
- A `workflow_dispatch` run with `image_repo=radixark/miles-staging` builds + pushes to staging end-to-end without touching production tags.
