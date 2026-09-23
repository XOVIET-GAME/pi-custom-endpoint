# pi-custom-endpoint

> This project is a fork of the original [ratatulieoi/better-custom](https://github.com/ratatulieoi/better-custom) repository.

A better way to add custom providers for Pi and Oh My Pi (OMP).

## Features

- Add, edit, or delete custom providers from an interactive wizard
- Supports:
  - OpenAI-compatible endpoints
  - Anthropic-compatible endpoints
  - Ollama-compatible endpoints
- Uses the running host's agent directory automatically
  - Pi: `models.json`
  - OMP: `models.yml` / `models.yaml`
- API key modes:
  - API key (stored verbatim in the active models config)
  - none (writes a placeholder so the provider still loads)
  - existing `$ENV` and `!command` keys are still resolved when re-probing
- Auto-probe `/models` for OpenAI-compatible endpoints
- Multi-select model picker for probed models
- Unique provider names — the wizard refuses to overwrite an existing provider
- Image input enabled by default (`input: ["text", "image"]`) so vision-capable
  models receive images instead of having them silently dropped
- Reasoning enabled by default at the `xhigh` ceiling for newly added models (selectable up to `max`)
- Safe delete flow for whole providers or individual models

## Install

### Pi

From npm:
```bash
pi install npm:pi-custom-endpoint
```

From GitHub:
```bash
pi install https://github.com/XOVIET-GAME/pi-custom-endpoint
```

### Oh My Pi (OMP)

> **Note:** OMP uses `bun` internally for all plugin operations (`install`, `uninstall`, `update`). Bun must be available in your system `PATH`.

**Step 1 — Install Bun & add to PATH (one-time setup):**

* **macOS / Linux:**
  ```bash
  command -v bun >/dev/null 2>&1 || curl -fsSL https://bun.sh/install | bash
  # Then restart your terminal (or run: source ~/.bashrc / source ~/.zshrc)
  ```
* **Windows — PowerShell** (run once, then open a new terminal):
  ```powershell
  # Install Bun if missing:
  if (-not (Get-Command bun -ErrorAction SilentlyContinue)) { irm bun.sh/install.ps1 | iex }
  # Add to permanent user PATH:
  [System.Environment]::SetEnvironmentVariable("Path", $env:Path + ";$env:USERPROFILE\.bun\bin", "User")
  ```
  > ⚠️ These commands require **PowerShell** (`pwsh` or `Windows PowerShell`), not `cmd.exe`.  
  > If using **cmd.exe**, use this instead for the current session:
  > ```cmd
  > set PATH=%PATH%;%USERPROFILE%\.bun\bin
  > ```

**Step 2 — Install extension:**

```bash
# From npm:
omp install pi-custom-endpoint

# From GitHub:
omp install https://github.com/XOVIET-GAME/pi-custom-endpoint
```

### Update

**Pi:**
```bash
# Re-install from npm to update:
pi install npm:pi-custom-endpoint@latest

# Or update directly if installed via npm:
pi update pi-custom-endpoint
```

**Oh My Pi (OMP):**
```bash
# Update to latest npm release:
omp install pi-custom-endpoint@latest

# Or force re-install:
omp install pi-custom-endpoint --force
```

### Uninstall / Remove

**Pi:**
```bash
pi remove pi-custom-endpoint
```

**Oh My Pi (OMP):**
```bash
omp plugin uninstall pi-custom-endpoint
```
> **Note (Windows):** If you see `Error: Executable not found in $PATH: "bun"`, add Bun to your PATH first (see install steps above), then open a new terminal and retry.

## Usage

After installing, reload pi if needed, then run:

```text
/endpoint-custom
```

The wizard can:

1. Add a provider
2. Edit a provider
3. Delete a provider

### Add a provider

Guides you through:

- provider style (OpenAI / Anthropic / Ollama)
- endpoint
- provider name (must be unique)
- API key method (API key or none)
- model discovery (auto-probe `/models`) or manual model entry

Newly added models default to `input: ["text", "image"]` and `reasoning: true`
at the `xhigh` ceiling (selectable up to `max`). Tune any of this later via Edit provider.

### Edit a provider

Pick a provider, then choose:

- Re-probe for new models — query `/models` again and add ones not yet configured
- Set context window (all models) — apply one `contextWindow` to every model
- Edit per model — pick a model and edit a single field:
  - Reasoning ceiling (`off` → `max`)
  - Vision (text+image vs text-only)
  - Context window
  - Max output tokens
  - Headers / endpoint override (per-model `baseUrl` and JSON `headers`)
  - Delete this model
- Add models manually
- Rename provider — change the provider name (key) in the active models config

Per-model edits change one field in place, so untouched fields (cost, headers,
overrides) are preserved.

### Delete a provider

Lists configured providers and removes the selected one after confirmation.

## How reasoning maps to Pi and OMP

Pi and Oh My Pi (OMP) support canonical thinking levels from `off` through `max`
(`off, minimal, low, medium, high, xhigh, max`). New providers and added models
continue to default to `xhigh`, with `max` selectable as the maximum ceiling.

When a model has `reasoning: true`, legacy `thinkingLevelMap` behavior remains for
backwards compatibility: `minimal` through `high` are available by default, while
`xhigh` or `max` ceilings and lower-level caps map explicitly.

### Native OMP `max` vs. provider wire aliases

- **Native `max` capability:** OMP represents native reasoning levels using
  `thinking: { mode: "effort" | "anthropic-adaptive", efforts: [...] }`. Selecting the `max` ceiling includes
  `max` in `thinking.efforts` (`[minimal, low, medium, high, xhigh, max]`), unlocking
  native `max` in OMP while maintaining `thinkingLevelMap` for Pi and backwards compatibility.
- **Provider wire alias (`{xhigh: "max"}`):** Existing configs may map `xhigh` to
  `"max"` in `thinkingLevelMap` (e.g. `{ xhigh: "max" }`). This is a provider wire-value
  override for the `xhigh` level, **not** native OMP `max`. A model with `{xhigh: "max"}`
  still operates with an `xhigh` ceiling unless native `max` is selected.

## Configuration

The extension uses the host-provided agent directory instead of hard-coding
`~/.pi/agent`. Existing `models.yml`, `models.yaml`, or `models.json` files are
kept in their current format. A fresh OMP config is created as `models.yml`;
normal Pi continues to use `models.json`.

Saving YAML rewrites its formatting and does not preserve comments.

## Files

- `index.ts` — extension entry point
- `package.json` — pi package manifest

## License

MIT
