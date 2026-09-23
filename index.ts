import * as CodingAgent from "@mariozechner/pi-coding-agent";
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { Key, matchesKey, truncateToWidth } from "@mariozechner/pi-tui";
import { execSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { parse as parseYaml, stringify as stringifyYaml } from "yaml";

type ProviderApi = "openai-completions" | "anthropic-messages";
type ProviderStyle = "openai" | "anthropic" | "ollama";
type ApiKeyMode = "env" | "literal" | "shell" | "none";
// pi / omp reasoning ceilings. "off" means no reasoning; the rest are the levels a
// model is allowed to use. Native max requires thinking.efforts containing max.
type ReasoningCeiling = "off" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max";
const REASONING_LEVELS: ReasoningCeiling[] = ["off", "minimal", "low", "medium", "high", "xhigh", "max"];

// Per-model knobs the wizard can write. apiKey lives at provider scope, not here.
type ModelOptions = {
	reasoning: ReasoningCeiling;
	vision: boolean;
	contextWindow?: number;
};

type ModelsConfig = {
	providers?: Record<string, any>;
};

type ProbeItem = {
	value: string;
	label: string;
	description?: string;
};

type SelectItem = {
	value: string;
	label: string;
	suffix?: string;
	description?: string;
	searchText?: string;
};

type CommandContext = Parameters<Parameters<ExtensionAPI["registerCommand"]>[1]["handler"]>[1];

const AGENT_DIR = CodingAgent.getAgentDir();
const IS_OMP = "logger" in CodingAgent || /(^|[\\/])\.?omp([\\/]|$)/i.test(AGENT_DIR);
// OMP prefers YAML and still accepts legacy JSON; normal Pi uses JSON only.
const MODELS_JSON_PATH = (IS_OMP ? ["models.yml", "models.yaml", "models.json"] : ["models.json"])
	.map((name) => join(AGENT_DIR, name))
	.find(existsSync) ?? join(AGENT_DIR, IS_OMP ? "models.yml" : "models.json");
const IS_YAML_CONFIG = /\.ya?ml$/i.test(MODELS_JSON_PATH);
const BUILTIN_PROVIDER_IDS = new Set([
	"anthropic",
	"openai",
	"azure-openai",
	"google",
	"vertex",
	"bedrock",
	"mistral",
	"groq",
	"cerebras",
	"xai",
	"openrouter",
	"vercel-ai-gateway",
	"zai",
	"huggingface",
	"kimi-for-coding",
	"minimax",
	"ollama",
]);

function ensureConfigDir() {
	mkdirSync(dirname(MODELS_JSON_PATH), { recursive: true });
}

function loadModelsConfig(): ModelsConfig {
	ensureConfigDir();
	if (!existsSync(MODELS_JSON_PATH)) {
		return { providers: {} };
	}

	const raw = readFileSync(MODELS_JSON_PATH, "utf8").trim();
	if (!raw) return { providers: {} };

	const parsed = (IS_YAML_CONFIG ? parseYaml(raw) : JSON.parse(raw)) as ModelsConfig;
	if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
		throw new Error("models config must be an object");
	}
	if (!parsed.providers || typeof parsed.providers !== "object") {
		parsed.providers = {};
	}
	return parsed;
}

function saveModelsConfig(config: ModelsConfig) {
	ensureConfigDir();
	const content = IS_YAML_CONFIG ? stringifyYaml(config, { lineWidth: 0 }) : JSON.stringify(config, null, 2);
	writeFileSync(MODELS_JSON_PATH, `${content.trimEnd()}\n`, "utf8");
}

function hasExplicitScheme(input: string): boolean {
	return /^[a-z]+:\/\//i.test(input.trim());
}

function addDefaultScheme(input: string): string {
	if (hasExplicitScheme(input)) return input;
	const lower = input.toLowerCase();
	const isLocal =
		lower.startsWith("localhost") ||
		lower.startsWith("127.") ||
		lower.startsWith("0.0.0.0") ||
		lower.startsWith("10.") ||
		lower.startsWith("192.168.") ||
		lower.startsWith("172.16.") ||
		lower.startsWith("172.17.") ||
		lower.startsWith("172.18.") ||
		lower.startsWith("172.19.") ||
		lower.startsWith("172.20.") ||
		lower.startsWith("172.21.") ||
		lower.startsWith("172.22.") ||
		lower.startsWith("172.23.") ||
		lower.startsWith("172.24.") ||
		lower.startsWith("172.25.") ||
		lower.startsWith("172.26.") ||
		lower.startsWith("172.27.") ||
		lower.startsWith("172.28.") ||
		lower.startsWith("172.29.") ||
		lower.startsWith("172.30.") ||
		lower.startsWith("172.31.") ||
		lower.startsWith("[");
	return `${isLocal ? "http" : "https"}://${input}`;
}

function stripSuffix(pathname: string, suffix: string): string {
	return pathname.endsWith(suffix) ? pathname.slice(0, -suffix.length) || "/" : pathname;
}

function normalizeEndpoint(input: string, api: ProviderApi): string {
	const url = new URL(addDefaultScheme(input.trim()));
	let pathname = url.pathname.replace(/\/+$/, "") || "/";

	if (api === "openai-completions") {
		pathname = stripSuffix(pathname, "/chat/completions");
		pathname = stripSuffix(pathname, "/responses");
		pathname = stripSuffix(pathname, "/completions");
		pathname = stripSuffix(pathname, "/models");
	} else {
		pathname = stripSuffix(pathname, "/messages");
	}

	pathname = pathname === "/" ? "" : pathname;
	const port = url.port ? `:${url.port}` : "";
	return `${url.protocol}//${url.hostname}${port}${pathname}`;
}

function slugify(value: string): string {
	return value
		.toLowerCase()
		.replace(/[^a-z0-9]+/g, "-")
		.replace(/^-+|-+$/g, "")
		.replace(/--+/g, "-");
}

function suggestProviderId(endpoint: string): string {
	const url = new URL(addDefaultScheme(endpoint));
	const host = url.hostname.replace(/^www\./, "").replace(/^api\./, "");
	const hostSlug = slugify(`${host}${url.port ? `-${url.port}` : ""}`) || "provider";
	return `custom-${hostSlug}`;
}

function dedupe(values: string[]): string[] {
	return Array.from(new Set(values));
}

function buildProbeUrl(baseUrl: string): string {
	const withSlash = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;
	return new URL("models", withSlash).toString();
}

function resolveApiKeyForProbe(mode: ApiKeyMode, storedValue?: string): string | undefined {
	if (!storedValue || mode === "none") return undefined;
	if (mode === "literal") return storedValue;
	if (mode === "env") return process.env[storedValue]?.trim() || undefined;
	if (mode === "shell") {
		try {
			return execSync(storedValue, {
				encoding: "utf8",
				stdio: ["ignore", "pipe", "pipe"],
			}).trim();
		} catch {
			return undefined;
		}
	}
	return undefined;
}

function serializeApiKey(mode: ApiKeyMode, value?: string, style?: ProviderStyle): string | undefined {
	if (mode === "none") return style === "ollama" ? "ollama" : "dummy";
	if (!value) return undefined;
	// pi resolves an apiKey by prefix: "!cmd" runs a shell command, "$VAR" reads an
	// env var, anything else is a literal. See pi-ai resolve-config-value.
	if (mode === "shell") return value.startsWith("!") ? value : `!${value}`;
	if (mode === "env") return value.startsWith("$") ? value : `$${value}`;
	return value;
}

async function probeOpenAIModels(baseUrl: string, apiKeyMode: ApiKeyMode, apiKeyValue?: string): Promise<ProbeItem[]> {
	const headers: Record<string, string> = {
		accept: "application/json",
		"accept-encoding": "identity",
	};
	const resolvedKey = resolveApiKeyForProbe(apiKeyMode, apiKeyValue);
	if (resolvedKey) {
		headers.authorization = `Bearer ${resolvedKey}`;
	}

	const response = await fetch(buildProbeUrl(baseUrl), { headers });
	if (!response.ok) {
		const body = await response.text().catch(() => "");
		throw new Error(`Probe failed (${response.status} ${response.statusText})${body ? `: ${body.slice(0, 200)}` : ""}`);
	}

	const json = (await response.json()) as any;
	const rawModels = Array.isArray(json) ? json : Array.isArray(json?.data) ? json.data : [];
	const ids = dedupe(
		rawModels
			.map((item: any) => (typeof item?.id === "string" ? item.id.trim() : ""))
			.filter(Boolean),
	).sort((a, b) => a.localeCompare(b));

	return ids.map((id) => ({ value: id, label: id }));
}

function normalizeSelectItems(items: Array<string | SelectItem>): SelectItem[] {
	return items.map((item) => (typeof item === "string" ? { value: item, label: item } : item));
}

async function selectOne(
	ctx: CommandContext,
	title: string,
	items: Array<string | SelectItem>,
	options?: { initialIndex?: number },
): Promise<string | null> {
	const normalizedItems = normalizeSelectItems(items);
	if (normalizedItems.length === 0) return null;

	return await ctx.ui.custom<string | null>((tui, theme, _kb, done) => {
		let cursor = Math.max(0, Math.min(options?.initialIndex ?? 0, normalizedItems.length - 1));
		let query = "";
		let cachedLines: string[] | undefined;
		const maxVisible = 12;

		function getVisibleItems() {
			const lowerQuery = query.trim().toLowerCase();
			if (!lowerQuery) return normalizedItems;
			return normalizedItems.filter((item) => {
				const haystack = `${item.label} ${item.suffix ?? ""} ${item.description ?? ""} ${item.searchText ?? ""}`.toLowerCase();
				return haystack.includes(lowerQuery);
			});
		}

		function refresh() {
			const visibleItems = getVisibleItems();
			if (visibleItems.length === 0) cursor = 0;
			else if (cursor >= visibleItems.length) cursor = visibleItems.length - 1;
			cachedLines = undefined;
			tui.requestRender();
		}

		return {
			render(width: number) {
				if (cachedLines) return cachedLines;

				const visibleItems = getVisibleItems();
				const safeWidth = Math.max(10, width);
				const lines: string[] = [];
				const add = (line = "") => lines.push(truncateToWidth(line, safeWidth));
				const border = theme.fg("accent", "─".repeat(safeWidth));

				add(border);
				add(` ${theme.fg("accent", theme.bold(title))}`);
				add(` ${theme.fg("text", `Search: ${query || "-"}`)}`);
				add();

				if (visibleItems.length === 0) {
					add(theme.fg("warning", " No matches."));
				} else {
					const start = Math.max(0, Math.min(cursor - Math.floor(maxVisible / 2), Math.max(0, visibleItems.length - maxVisible)));
					const end = Math.min(visibleItems.length, start + maxVisible);

					for (let i = start; i < end; i++) {
						const item = visibleItems[i];
						const active = i === cursor;
						const prefix = active ? theme.fg("accent", "> ") : "  ";
						const label = active ? theme.fg("accent", item.label) : theme.fg("text", item.label);
						const suffix = item.suffix ? theme.fg("dim", item.suffix) : "";
						add(`${prefix}${label}${suffix}`);
						if (item.description) {
							for (const line of item.description.split("\n")) {
								add(`   ${theme.fg("muted", line)}`);
							}
						}
					}

					if (visibleItems.length > maxVisible) {
						add();
						add(theme.fg("dim", ` ${start + 1}-${end} of ${visibleItems.length}`));
					}
				}

				add();
				add(theme.fg("dim", " Type to search • ↑↓ move (wraps) • enter confirm • backspace delete • esc cancel"));
				add(border);

				cachedLines = lines;
				return lines;
			},
			invalidate() {
				cachedLines = undefined;
			},
			handleInput(data: string) {
				const visibleItems = getVisibleItems();
				if (matchesKey(data, Key.up)) {
					if (visibleItems.length === 0) return;
					cursor = cursor === 0 ? visibleItems.length - 1 : cursor - 1;
					refresh();
					return;
				}
				if (matchesKey(data, Key.down)) {
					if (visibleItems.length === 0) return;
					cursor = cursor === visibleItems.length - 1 ? 0 : cursor + 1;
					refresh();
					return;
				}
				if (matchesKey(data, Key.enter)) {
					const item = visibleItems[cursor];
					done(item?.value ?? null);
					return;
				}
				if (matchesKey(data, Key.escape)) {
					done(null);
					return;
				}
				if (data === "\u007f" || data === "\b") {
					if (query.length > 0) {
						query = query.slice(0, -1);
						refresh();
					}
					return;
				}
				if (data >= " " && data !== "\u001b" && data !== "\r" && data !== "\n") {
					query += data;
					cursor = 0;
					refresh();
				}
			},
		};
	});
}

async function pickMany(
	ctx: CommandContext,
	title: string,
	items: ProbeItem[],
): Promise<string[] | null> {
	return await ctx.ui.custom<string[] | null>((tui, theme, _kb, done) => {
		let cursor = 0;
		let query = "";
		const selected = new Set<string>();
		let cachedLines: string[] | undefined;
		const maxVisible = 12;

		function getVisibleItems() {
			const lowerQuery = query.trim().toLowerCase();
			if (!lowerQuery) return items;
			return items.filter((item) => {
				const haystack = `${item.label} ${item.value} ${item.description ?? ""}`.toLowerCase();
				return haystack.includes(lowerQuery);
			});
		}

		function refresh() {
			const visibleItems = getVisibleItems();
			if (visibleItems.length === 0) cursor = 0;
			else if (cursor >= visibleItems.length) cursor = visibleItems.length - 1;
			cachedLines = undefined;
			tui.requestRender();
		}

		return {
			render(width: number) {
				if (cachedLines) return cachedLines;

				const visibleItems = getVisibleItems();
				const safeWidth = Math.max(10, width);
				const lines: string[] = [];
				const add = (line = "") => lines.push(truncateToWidth(line, safeWidth));
				const border = theme.fg("accent", "─".repeat(safeWidth));

				add(border);
				add(` ${theme.fg("accent", theme.bold(title))}`);
				add(` ${theme.fg("text", `Search: ${query || "-"}`)}`);
				add(` ${theme.fg("muted", `${selected.size} selected • ${visibleItems.length}/${items.length} shown`)}`);
				add();

				if (visibleItems.length === 0) {
					add(theme.fg("warning", " No matching models."));
				} else {
					const start = Math.max(0, Math.min(cursor - Math.floor(maxVisible / 2), Math.max(0, visibleItems.length - maxVisible)));
					const end = Math.min(visibleItems.length, start + maxVisible);

					for (let i = start; i < end; i++) {
						const item = visibleItems[i];
						const active = i === cursor;
						const checked = selected.has(item.value);
						const prefix = active ? theme.fg("accent", "> ") : "  ";
						const box = checked ? theme.fg("success", "[x]") : theme.fg("muted", "[ ]");
						const label = active ? theme.fg("accent", item.label) : theme.fg("text", item.label);
						add(`${prefix}${box} ${label}`);
						if (item.description) {
							add(`     ${theme.fg("muted", item.description)}`);
						}
					}

					if (visibleItems.length > maxVisible) {
						add();
						add(theme.fg("dim", ` ${start + 1}-${end} of ${visibleItems.length}`));
					}
				}

				add();
				add(theme.fg("dim", " Type to search • ↑↓ move (wraps) • space toggle • enter confirm • backspace delete • esc cancel"));
				if (selected.size === 0) {
					add(theme.fg("warning", " Select at least one model before confirming."));
				}
				add(border);

				cachedLines = lines;
				return lines;
			},
			invalidate() {
				cachedLines = undefined;
			},
			handleInput(data: string) {
				const visibleItems = getVisibleItems();
				if (matchesKey(data, Key.up)) {
					if (visibleItems.length === 0) return;
					cursor = cursor === 0 ? visibleItems.length - 1 : cursor - 1;
					refresh();
					return;
				}
				if (matchesKey(data, Key.down)) {
					if (visibleItems.length === 0) return;
					cursor = cursor === visibleItems.length - 1 ? 0 : cursor + 1;
					refresh();
					return;
				}
				if (data === " ") {
					const value = visibleItems[cursor]?.value;
					if (!value) return;
					if (selected.has(value)) selected.delete(value);
					else selected.add(value);
					refresh();
					return;
				}
				if (matchesKey(data, Key.enter)) {
					if (selected.size > 0) done(Array.from(selected));
					return;
				}
				if (matchesKey(data, Key.escape)) {
					done(null);
					return;
				}
				if (data === "\u007f" || data === "\b") {
					if (query.length > 0) {
						query = query.slice(0, -1);
						refresh();
					}
					return;
				}
				if (data >= " " && data !== "\u001b" && data !== "\r" && data !== "\n") {
					query += data;
					cursor = 0;
					refresh();
				}
			},
		};
	});
}

async function promptApiKey(
	ctx: CommandContext,
): Promise<{ mode: ApiKeyMode; value?: string } | null> {
	const choice = await selectOne(ctx, "API key", [
		{ value: "literal", label: "API key", description: "Stored verbatim in the active models config" },
		{ value: "none", label: "None", description: "No key; a placeholder is written so the provider still loads" },
	]);
	if (!choice) return null;
	if (choice === "none") return { mode: "none" };

	const value = await ctx.ui.input("API key", "saved directly in the active models config");
	if (value === undefined) return null;
	const trimmed = value.trim();
	if (!trimmed) return { mode: "none" };
	return { mode: "literal", value: trimmed };
}

function reasoningLabel(level: ReasoningCeiling): string {
	if (level === "off") return "Off - no reasoning";
	if (level === "max") return "max - maximum reasoning (only if the model supports it)";
	if (level === "xhigh") return "xhigh - extra high reasoning";
	return `${level} - cap reasoning at ${level}`;
}

// Prompts for a reasoning ceiling. Returns null if cancelled.
async function promptReasoning(ctx: CommandContext, current?: ReasoningCeiling): Promise<ReasoningCeiling | null> {
	const items: SelectItem[] = REASONING_LEVELS.map((level) => ({
		value: level,
		label: reasoningLabel(level),
	}));
	const initialIndex = current ? REASONING_LEVELS.indexOf(current) : 0;
	const choice = await selectOne(ctx, "Reasoning", items, { initialIndex: Math.max(0, initialIndex) });
	return (choice as ReasoningCeiling | null) ?? null;
}

// When a model is capped at xhigh or max, some providers name that level differently
// (e.g. "max" for xhigh, or "maximum" for max). Offer an optional override for the
// provider-facing wire string.
async function promptReasoningProviderString(
	ctx: CommandContext,
	level: "xhigh" | "max",
	current?: string,
): Promise<string | undefined> {
	const value = await ctx.ui.input(
		`${level} provider value (blank = ${level})`,
		current && current !== level ? `current: ${current}` : `e.g. ${level === "xhigh" ? "max" : "maximum"} (leave blank to send "${level}")`,
	);
	if (value === undefined) return undefined;
	const trimmed = value.trim();
	return trimmed || undefined;
}


async function promptVision(ctx: CommandContext, current?: boolean): Promise<boolean | null> {
	const choice = await selectOne(ctx, "Image input (vision)", [
		{ value: "yes", label: "Yes - send text + images", description: "Sets input: [text, image]" },
		{ value: "no", label: "No - text only", description: "Sets input: [text]" },
	], { initialIndex: current === false ? 1 : 0 });
	if (!choice) return null;
	return choice === "yes";
}

// Prompts for a context window size in tokens. Returns:
//   number  -> set/replace contextWindow
//   0       -> clear contextWindow (user typed 0)
//   null    -> cancelled, leave unchanged
async function promptContextWindow(ctx: CommandContext, current?: number): Promise<number | null> {
	const value = await ctx.ui.input(
		"Context window (tokens)",
		current ? `current: ${current} (blank = keep, 0 = clear)` : "e.g. 128000 (blank = unset)",
	);
	if (value === undefined) return null;
	const trimmed = value.trim();
	if (!trimmed) return null;
	const parsed = Number.parseInt(trimmed.replace(/[_,]/g, ""), 10);
	if (!Number.isFinite(parsed) || parsed < 0) {
		ctx.ui.notify("Enter a whole number of tokens (0 to clear).", "warning");
		return null;
	}
	return parsed;
}

// Prompts for max output tokens. Same return contract as promptContextWindow:
// number to set, 0 to clear, null to leave unchanged.
async function promptMaxTokens(ctx: CommandContext, current?: number): Promise<number | null> {
	const value = await ctx.ui.input(
		"Max output tokens",
		current ? `current: ${current} (blank = keep, 0 = clear)` : "e.g. 8192 (blank = unset)",
	);
	if (value === undefined) return null;
	const trimmed = value.trim();
	if (!trimmed) return null;
	const parsed = Number.parseInt(trimmed.replace(/[_,]/g, ""), 10);
	if (!Number.isFinite(parsed) || parsed < 0) {
		ctx.ui.notify("Enter a whole number of tokens (0 to clear).", "warning");
		return null;
	}
	return parsed;
}

// Read the reasoning ceiling + vision flags already stored on a model entry,
// mirroring pi / omp thinking metadata so edit defaults match reality.
function readModelOptions(model: any): ModelOptions {
	const vision = Array.isArray(model?.input) ? model.input.includes("image") : true;
	const contextWindow = typeof model?.contextWindow === "number" ? model.contextWindow : undefined;
	if (!model || model.reasoning === false || (model.reasoning !== true && !model.thinking)) {
		return { reasoning: "off", vision, contextWindow };
	}

	// 1. Native OMP thinking.efforts metadata: highest recognized effort determines ceiling.
	// Recognizes native max before xhigh when modern metadata has it.
	const efforts = Array.isArray(model?.thinking?.efforts) ? model.thinking.efforts : undefined;
	if (efforts && efforts.length > 0) {
		for (let i = REASONING_LEVELS.length - 1; i >= 1; i--) {
			const level = REASONING_LEVELS[i];
			if (efforts.includes(level)) {
				return { reasoning: level, vision, contextWindow };
			}
		}
	}

	// 2. Fall back to legacy thinkingLevelMap.
	// Existing {xhigh: "max"} must still read as ceiling xhigh (provider alias, not native max).
	const map = model?.thinkingLevelMap;
	let ceiling: ReasoningCeiling = "high";
	if (map && typeof map === "object") {
		if (map.max !== undefined && map.max !== null) {
			ceiling = "max";
		} else if (map.xhigh !== undefined && map.xhigh !== null) {
			ceiling = "xhigh";
		} else {
			for (let i = REASONING_LEVELS.indexOf("high"); i >= 1; i--) {
				const level = REASONING_LEVELS[i];
				if (map[level] === null) continue;
				ceiling = level;
				break;
			}
		}
	}
	return { reasoning: ceiling, vision, contextWindow };
}

function readReasoningProviderString(model: any, level: "xhigh" | "max"): string | undefined {
	const compatVal = model?.compat?.reasoningEffortMap?.[level];
	if (typeof compatVal === "string" && compatVal.trim()) return compatVal.trim();
	const mapVal = model?.thinkingLevelMap?.[level];
	if (typeof mapVal === "string" && mapVal.trim()) return mapVal.trim();
	return undefined;
}

async function promptModelIdsOneByOne(
	ctx: CommandContext,
	style: ProviderStyle,
): Promise<string[] | null> {
	const modelIds: string[] = [];
	const firstPlaceholder =
		style === "anthropic"
			? "e.g. claude-sonnet-4-5 (blank to finish)"
			: style === "ollama"
				? "e.g. llama3.1:8b or qwen2.5-coder:7b (blank to finish)"
				: "e.g. gpt-4o-mini or qwen/qwen3-coder (blank to finish)";
	const nextPlaceholder =
		style === "anthropic"
			? "another Anthropic-style model id (blank to finish)"
			: style === "ollama"
				? "another Ollama model id (blank to finish)"
				: "another OpenAI-style model id (blank to finish)";

	while (true) {
		const value = await ctx.ui.input(modelIds.length === 0 ? "Model id" : "Add another model id", modelIds.length === 0 ? firstPlaceholder : nextPlaceholder);
		if (value === undefined) return null;
		const trimmed = value.trim();
		if (!trimmed) {
			if (modelIds.length === 0) {
				ctx.ui.notify("Add at least one model.", "warning");
				continue;
			}
			return modelIds;
		}
		if (modelIds.includes(trimmed)) {
			ctx.ui.notify(`Model already added: ${trimmed}`, "warning");
			continue;
		}
		modelIds.push(trimmed);
	}
}

// Apply a reasoning ceiling to an entry in place, preserving other fields.
// - Native OMP reasoning requires thinking: { mode, efforts: [...] }.
//   thinking.efforts contains every canonical level up to the ceiling (native max is exposed here).
// - thinkingLevelMap is preserved for pi / backwards compatibility.
//   Legacy {xhigh: "max"} remains ceiling xhigh with wire override "max".
// - compat.reasoningEffortMap is populated only for non-default wire values, preserving unrelated compat fields.
// - off cleans up reasoning, thinking, thinkingLevelMap, and xhigh/max entries in compat.reasoningEffortMap.
function applyReasoning(entry: any, ceiling: ReasoningCeiling, providerStringOverride?: string, api?: ProviderApi) {
	const override = providerStringOverride?.trim();
	const compat = entry.compat && typeof entry.compat === "object" ? entry.compat : undefined;
	const effortMap = compat?.reasoningEffortMap && typeof compat.reasoningEffortMap === "object"
		? compat.reasoningEffortMap
		: undefined;
	const existingThinking = entry.thinking && typeof entry.thinking === "object"
		? entry.thinking
		: undefined;

	if (ceiling === "off") {
		delete entry.reasoning;
		if (existingThinking) {
			delete existingThinking.mode;
			delete existingThinking.efforts;
			delete existingThinking.requiresEffort;
			delete existingThinking.defaultLevel;
			if (Object.keys(existingThinking).length === 0) {
				delete entry.thinking;
			}
		} else {
			delete entry.thinking;
		}
		delete entry.thinkingLevelMap;
		if (effortMap) {
			delete effortMap.xhigh;
			delete effortMap.max;
			if (Object.keys(effortMap).length === 0) {
				delete compat?.reasoningEffortMap;
			}
			if (compat && Object.keys(compat).length === 0) {
				delete entry.compat;
			}
		}
		return;
	}

	entry.reasoning = true;

	// OMP-native metadata: mode depends on provider API (openai-completions => effort, anthropic-messages => anthropic-adaptive).
	// Preserves unrelated thinking metadata (e.g. effortMap, effortBudgets, effortRouting).
	const canonicalEfforts = REASONING_LEVELS.filter((l): l is Exclude<ReasoningCeiling, "off"> => l !== "off");
	const ceilingIndex = canonicalEfforts.indexOf(ceiling as Exclude<ReasoningCeiling, "off">);
	const resolvedApi: ProviderApi = api || (entry?.api as ProviderApi) || "openai-completions";
	const mode = resolvedApi === "anthropic-messages" ? "anthropic-adaptive" : "effort";
	entry.thinking = {
		...(existingThinking ?? {}),
		mode,
		efforts: canonicalEfforts.slice(0, ceilingIndex + 1),
	};
	delete entry.thinking.requiresEffort;

	// Legacy thinkingLevelMap for pi / backcompat
	const map: Record<string, string | null> = {};
	if (ceiling === "max") {
		map.xhigh = "xhigh";
		map.max = override || "max";
	} else if (ceiling === "xhigh") {
		map.xhigh = override || "xhigh";
	} else {
		for (const level of ["minimal", "low", "medium", "high"] as const) {
			if (REASONING_LEVELS.indexOf(level) > REASONING_LEVELS.indexOf(ceiling)) {
				map[level] = null;
			}
		}
	}
	if (Object.keys(map).length > 0) entry.thinkingLevelMap = map;
	else delete entry.thinkingLevelMap;

	// Wire-value override in compat.reasoningEffortMap for non-default wire values
	if (effortMap) {
		delete effortMap.xhigh;
		delete effortMap.max;
	}

	const nonDefaultOverride =
		ceiling === "xhigh" && override && override !== "xhigh"
			? { level: "xhigh", value: override }
			: ceiling === "max" && override && override !== "max"
				? { level: "max", value: override }
				: null;

	if (nonDefaultOverride) {
		const activeCompat = compat ?? (entry.compat = {} as Record<string, unknown>);
		const activeEffortMap =
			(activeCompat.reasoningEffortMap && typeof activeCompat.reasoningEffortMap === "object"
				? activeCompat.reasoningEffortMap
				: (activeCompat.reasoningEffortMap = {})) as Record<string, string>;
		activeEffortMap[nonDefaultOverride.level] = nonDefaultOverride.value;
	} else if (effortMap) {
		if (Object.keys(effortMap).length === 0) {
			delete compat?.reasoningEffortMap;
		}
		if (compat && Object.keys(compat).length === 0) {
			delete entry.compat;
		}
	}
}

function buildModelEntry(id: string, opts: ModelOptions, providerStringOverride?: string, api?: ProviderApi): any {
	const entry: Record<string, unknown> = {
		id,
		// Default to text+image so pi forwards images upstream. Without this,
		// custom models default to text-only and images are silently dropped.
		input: opts.vision ? ["text", "image"] : ["text"],
	};

	if (typeof opts.contextWindow === "number" && opts.contextWindow > 0) {
		entry.contextWindow = opts.contextWindow;
	}

	applyReasoning(entry, opts.reasoning, providerStringOverride, api);
	return entry;
}

function buildProviderConfig(
	style: ProviderStyle,
	api: ProviderApi,
	baseUrl: string,
	apiKey: { mode: ApiKeyMode; value?: string },
	modelIds: string[],
	opts: ModelOptions,
	providerStringOverride?: string,
) {
	const serializedApiKey = serializeApiKey(apiKey.mode, apiKey.value, style);
	const config: any = {
		baseUrl,
		api,
		...(serializedApiKey ? { apiKey: serializedApiKey } : {}),
		models: modelIds.map((id) => buildModelEntry(id, opts, providerStringOverride, api)),
	};

	if (style === "ollama") {
		if (!config.apiKey) config.apiKey = "ollama";
		config.compat = {
			supportsDeveloperRole: false,
			supportsReasoningEffort: false,
		};
	}

	return config;
}

function describeProvider(providerId: string, provider: any): string {
	const modelCount = Array.isArray(provider?.models) ? provider.models.length : 0;
	const endpoint = typeof provider?.baseUrl === "string" ? provider.baseUrl : "(no baseUrl)";
	const api = typeof provider?.api === "string" ? provider.api : "(no api)";
	return `${providerId}\n${api} • ${modelCount} model${modelCount === 1 ? "" : "s"}\n${endpoint}`;
}

function describeProviderInline(providerId: string, provider: any): { label: string; suffix: string; searchText: string } {
	const modelCount = Array.isArray(provider?.models) ? provider.models.length : 0;
	const endpoint = typeof provider?.baseUrl === "string" ? provider.baseUrl : "(no baseUrl)";
	const api = typeof provider?.api === "string" ? provider.api : "(no api)";
	const suffix = ` • ${api} • ${endpoint} • ${modelCount} model${modelCount === 1 ? "" : "s"}`;
	return {
		label: providerId,
		suffix,
		searchText: `${providerId} ${api} ${endpoint} ${modelCount}`,
	};
}

function providerModelItems(provider: any): SelectItem[] {
	const models = Array.isArray(provider?.models) ? provider.models : [];
	return models
		.map((model: any) => {
			const id = typeof model === "string" ? model.trim() : typeof model?.id === "string" ? model.id.trim() : "";
			if (!id) return null;

			const details: string[] = [];
			if (model && typeof model === "object") {
				const opts = readModelOptions(model);
				if (opts.reasoning !== "off") details.push(`reasoning:${opts.reasoning}`);
				if (Array.isArray(model.input) && model.input.includes("image")) details.push("vision");
				if (typeof model.contextWindow === "number") details.push(`context ${model.contextWindow}`);
				if (typeof model.maxTokens === "number") details.push(`max ${model.maxTokens}`);
			}

			return {
				value: id,
				label: id,
				suffix: details.length > 0 ? ` • ${details.join(" • ")}` : "",
				searchText: `${id} ${details.join(" ")}`,
			};
		})
		.filter((item): item is SelectItem => item !== null);
}

function normalizeStoredEndpoint(provider: any): string {
	const endpoint = typeof provider?.baseUrl === "string" ? provider.baseUrl.trim() : "";
	if (!endpoint) return "";
	const api: ProviderApi = provider?.api === "anthropic-messages" ? "anthropic-messages" : "openai-completions";
	try {
		return normalizeEndpoint(endpoint, api);
	} catch {
		return endpoint.replace(/\/+$/, "");
	}
}

function findProvidersByEndpoint(config: ModelsConfig, endpoint: string): string[] {
	return Object.entries(config.providers ?? {})
		.filter(([, provider]) => normalizeStoredEndpoint(provider) === endpoint)
		.map(([providerId]) => providerId)
		.sort((a, b) => a.localeCompare(b));
}

async function editProviderFlow(ctx: CommandContext) {
	let cursor = 0;

	while (true) {
		let config: ModelsConfig;
		try {
			config = loadModelsConfig();
		} catch (error) {
			ctx.ui.notify(`Could not read ${MODELS_JSON_PATH}: ${error instanceof Error ? error.message : String(error)}`, "error");
			return;
		}

		config.providers ||= {};
		const providerIds = Object.keys(config.providers).sort((a, b) => a.localeCompare(b));
		if (providerIds.length === 0) {
			ctx.ui.notify(`No providers found in ${MODELS_JSON_PATH}`, "warning");
			return;
		}

		const choice = await selectOne(
			ctx,
			"Edit provider",
			providerIds.map((providerId) => {
				const inline = describeProviderInline(providerId, config.providers?.[providerId]);
				return {
					value: providerId,
					label: inline.label,
					suffix: inline.suffix,
					searchText: inline.searchText,
				};
			}),
			{ initialIndex: Math.min(cursor, providerIds.length - 1) },
		);
		if (!choice) return;

		cursor = providerIds.indexOf(choice);
		await editSingleProvider(ctx, choice);
	}
}

// Per-provider action menu. Returns when the user backs out to the provider list.
async function editSingleProvider(ctx: CommandContext, providerId: string) {
	while (true) {
		let config: ModelsConfig;
		try {
			config = loadModelsConfig();
		} catch (error) {
			ctx.ui.notify(`Could not read ${MODELS_JSON_PATH}: ${error instanceof Error ? error.message : String(error)}`, "error");
			return;
		}
		const provider = config.providers?.[providerId];
		if (!provider) {
			ctx.ui.notify(`Provider "${providerId}" no longer exists.`, "warning");
			return;
		}

		const modelCount = Array.isArray(provider.models) ? provider.models.length : 0;
		const action = await selectOne(ctx, `Edit ${providerId}`, [
			{ value: "probe", label: "Re-probe for new models", description: "Query /models again and add ones not already configured" },
			{ value: "context", label: "Set context window (all models)", description: `Apply one contextWindow to all ${modelCount} model${modelCount === 1 ? "" : "s"}` },
			{ value: "models", label: "Edit per model", description: `${modelCount} model${modelCount === 1 ? "" : "s"} — reasoning, vision, context, max tokens, headers, delete` },
			{ value: "add", label: "Add models manually", description: "Type model ids to add" },
			{ value: "rename", label: "Rename provider", description: "Change the provider name in the models config" },
			{ value: "back", label: "Back", description: "Return to the provider list" },
		]);
		if (!action || action === "back") return;

		if (action === "models") {
			await editProviderModels(ctx, providerId);
		} else if (action === "probe") {
			await reprobeProvider(ctx, providerId);
		} else if (action === "context") {
			await setProviderContextWindow(ctx, providerId);
		} else if (action === "add") {
			await addModelsToProvider(ctx, providerId);
		} else if (action === "rename") {
			// Reassign so the menu keeps editing the same provider under its new name.
			const renamed = await renameProvider(ctx, providerId);
			if (renamed) providerId = renamed;
		}
	}
}

// Rename a provider's key in the models config, preserving its config and original
// position in the file. Returns the new id on success, or null if cancelled,
// unchanged, or rejected. Only touches the models config — a currently-selected model
// pinned to the old provider id must be reselected via /model afterwards.
async function renameProvider(ctx: CommandContext, providerId: string): Promise<string | null> {
	let config: ModelsConfig;
	try {
		config = loadModelsConfig();
	} catch (error) {
		ctx.ui.notify(`Could not read ${MODELS_JSON_PATH}: ${error instanceof Error ? error.message : String(error)}`, "error");
		return null;
	}
	config.providers ||= {};
	if (!config.providers[providerId]) {
		ctx.ui.notify(`Provider "${providerId}" no longer exists.`, "warning");
		return null;
	}

	const input = await ctx.ui.input("Rename provider", `current: ${providerId}`);
	if (input === undefined) return null;
	// Slugify so names stay consistent with the Add flow.
	const newId = slugify(input.trim());
	if (!newId || newId === providerId) return null;

	if (config.providers[newId]) {
		ctx.ui.notify(`Provider "${newId}" already exists. Choose a different name.`, "warning");
		return null;
	}

	if (BUILTIN_PROVIDER_IDS.has(newId)) {
		const ok = await ctx.ui.confirm(
			"Override built-in provider?",
			`"${newId}" matches a built-in provider id. Saving this will override that provider in the active models config. Continue?`,
		);
		if (!ok) return null;
	}

	// Rebuild key-by-key so the renamed entry keeps its position rather than
	// jumping to the bottom (a naive delete + reassign would reorder it).
	const rebuilt: Record<string, any> = {};
	for (const [key, value] of Object.entries(config.providers)) {
		rebuilt[key === providerId ? newId : key] = value;
	}
	config.providers = rebuilt;

	try {
		saveModelsConfig(config);
	} catch (error) {
		ctx.ui.notify(`Could not write ${MODELS_JSON_PATH}: ${error instanceof Error ? error.message : String(error)}`, "error");
		return null;
	}

	ctx.ui.notify(`Renamed "${providerId}" → "${newId}".`, "info");
	return newId;
}

// Apply a single contextWindow value to every model in the provider, preserving
// each model's reasoning/vision config. A value of 0 clears it from all models.
async function setProviderContextWindow(ctx: CommandContext, providerId: string) {
	let provider: any;
	try {
		provider = loadModelsConfig().providers?.[providerId];
	} catch (error) {
		ctx.ui.notify(`Could not read ${MODELS_JSON_PATH}: ${error instanceof Error ? error.message : String(error)}`, "error");
		return;
	}
	const models = Array.isArray(provider?.models) ? provider.models : [];
	if (models.length === 0) {
		ctx.ui.notify(`Provider "${providerId}" has no models.`, "warning");
		return;
	}

	// Prefill with the shared value if every model already agrees, else blank.
	const windows = models.map((m: any) => (typeof m?.contextWindow === "number" ? m.contextWindow : undefined));
	const shared = windows.every((w: number | undefined) => w === windows[0]) ? windows[0] : undefined;

	const result = await promptContextWindow(ctx, shared);
	if (result === null) return;

	const saved = await mutateProvider(ctx, providerId, (p) => {
		const list = Array.isArray(p.models) ? p.models : [];
		for (const m of list) {
			const opts = readModelOptions(m);
			opts.contextWindow = result === 0 ? undefined : result;
			const wireOverride = (opts.reasoning === "xhigh" || opts.reasoning === "max")
				? readReasoningProviderString(m, opts.reasoning)
				: undefined;
			const modelApi: ProviderApi = (m?.api as ProviderApi) || (p?.api as ProviderApi) || (provider?.api as ProviderApi) || "openai-completions";
			const rebuilt = buildModelEntry(modelIdOf(m), opts, wireOverride, modelApi);
			if (m?.api) rebuilt.api = m.api;
			if (m.thinking && typeof m.thinking === "object") {
				rebuilt.thinking = { ...m.thinking, ...rebuilt.thinking };
			}
			if (m.compat && typeof m.compat === "object") {
				const existingCompat = { ...m.compat };
				let mergedEffortMap: Record<string, string> | undefined;
				if (existingCompat.reasoningEffortMap && typeof existingCompat.reasoningEffortMap === "object") {
					const { xhigh: _oldXhigh, max: _oldMax, ...unrelatedEffortMap } = existingCompat.reasoningEffortMap;
					mergedEffortMap = { ...unrelatedEffortMap, ...(rebuilt.compat?.reasoningEffortMap ?? {}) };
				} else if (rebuilt.compat?.reasoningEffortMap) {
					mergedEffortMap = { ...rebuilt.compat.reasoningEffortMap };
				}

				const mergedCompat: Record<string, unknown> = {
					...existingCompat,
					...(rebuilt.compat ?? {}),
				};
				if (mergedEffortMap && Object.keys(mergedEffortMap).length > 0) {
					mergedCompat.reasoningEffortMap = mergedEffortMap;
				} else {
					delete mergedCompat.reasoningEffortMap;
				}

				if (Object.keys(mergedCompat).length > 0) {
					rebuilt.compat = mergedCompat;
				} else {
					delete rebuilt.compat;
					delete m.compat;
				}
			}
			Object.assign(m, rebuilt);
			if (result === 0) delete m.contextWindow;
		}
		return true;
	});
	if (saved) {
		ctx.ui.notify(
			result === 0
				? `Cleared context window on all ${models.length} model${models.length === 1 ? "" : "s"}.`
				: `Set context window ${result} on all ${models.length} model${models.length === 1 ? "" : "s"}.`,
			"info",
		);
	}
}

// Load config, hand the provider to a mutator, and save if it returns true.
async function mutateProvider(
	ctx: CommandContext,
	providerId: string,
	mutate: (provider: any) => boolean | Promise<boolean>,
): Promise<boolean> {
	let config: ModelsConfig;
	try {
		config = loadModelsConfig();
	} catch (error) {
		ctx.ui.notify(`Could not read ${MODELS_JSON_PATH}: ${error instanceof Error ? error.message : String(error)}`, "error");
		return false;
	}
	const provider = config.providers?.[providerId];
	if (!provider) {
		ctx.ui.notify(`Provider "${providerId}" no longer exists.`, "warning");
		return false;
	}

	const changed = await mutate(provider);
	if (!changed) return false;

	try {
		saveModelsConfig(config);
	} catch (error) {
		ctx.ui.notify(`Could not write ${MODELS_JSON_PATH}: ${error instanceof Error ? error.message : String(error)}`, "error");
		return false;
	}
	return true;
}

// Pick a model, then a field to edit. Each edit mutates one field in place so
// other fields (headers, overrides, cost) are preserved.
async function editProviderModels(ctx: CommandContext, providerId: string) {
	let cursor = 0;
	while (true) {
		let provider: any;
		try {
			provider = loadModelsConfig().providers?.[providerId];
		} catch (error) {
			ctx.ui.notify(`Could not read ${MODELS_JSON_PATH}: ${error instanceof Error ? error.message : String(error)}`, "error");
			return;
		}
		const modelItems = providerModelItems(provider);
		if (modelItems.length === 0) {
			ctx.ui.notify(`Provider "${providerId}" has no models.`, "warning");
			return;
		}

		const choice = await selectOne(ctx, `Edit model in ${providerId}`, modelItems, {
			initialIndex: Math.min(cursor, modelItems.length - 1),
		});
		if (!choice) return;
		cursor = modelItems.findIndex((item) => item.value === choice);

		const deleted = await editSingleModel(ctx, providerId, choice);
		if (deleted) cursor = Math.max(0, cursor - 1);
	}
}

// Field-picker for one model. Returns true if the model was deleted (so the
// caller can adjust its cursor).
async function editSingleModel(ctx: CommandContext, providerId: string, modelId: string): Promise<boolean> {
	while (true) {
		let provider: any;
		let model: any;
		try {
			provider = loadModelsConfig().providers?.[providerId];
			model = findModel(provider, modelId);
		} catch (error) {
			ctx.ui.notify(`Could not read ${MODELS_JSON_PATH}: ${error instanceof Error ? error.message : String(error)}`, "error");
			return false;
		}
		if (!model) {
			ctx.ui.notify(`Model "${modelId}" no longer exists.`, "warning");
			return false;
		}

		const opts = readModelOptions(model);
		const ctxWin = typeof model.contextWindow === "number" ? model.contextWindow : "unset";
		const maxTok = typeof model.maxTokens === "number" ? model.maxTokens : "unset";
		const hasHeaders = model.headers && Object.keys(model.headers).length > 0;
		const override = model.baseUrl || model.api ? "set" : "unset";

		const field = await selectOne(ctx, `Edit ${modelId}`, [
			{ value: "reasoning", label: "Reasoning", suffix: ` • ${opts.reasoning}`, description: "Set the reasoning ceiling (off → max)" },
			{ value: "vision", label: "Vision", suffix: ` • ${opts.vision ? "on" : "off"}`, description: "Toggle image input (text+image vs text-only)" },
			{ value: "context", label: "Context window", suffix: ` • ${ctxWin}`, description: "Max context tokens for this model" },
			{ value: "maxtokens", label: "Max output tokens", suffix: ` • ${maxTok}`, description: "Max tokens this model may generate" },
			{ value: "override", label: "Headers / endpoint override", suffix: ` • ${hasHeaders ? "headers" : override}`, description: "Per-model HTTP headers and api/baseUrl override" },
			{ value: "delete", label: "Delete this model", description: "Remove this model from the provider" },
			{ value: "back", label: "Back", description: "Return to the model list" },
		]);
		if (!field || field === "back") return false;

		if (field === "reasoning") {
			const reasoning = await promptReasoning(ctx, opts.reasoning);
			if (reasoning === null) continue;
			let wireOverride: string | undefined;
			if (reasoning === "xhigh" || reasoning === "max") {
				wireOverride = await promptReasoningProviderString(
					ctx,
					reasoning,
					readReasoningProviderString(model, reasoning),
				);
			}
			await mutateModel(ctx, providerId, modelId, (m, p) => {
				const modelApi: ProviderApi = (m?.api as ProviderApi) || (p?.api as ProviderApi) || (provider?.api as ProviderApi) || "openai-completions";
				applyReasoning(m, reasoning, wireOverride, modelApi);
			});
		} else if (field === "vision") {
			const vision = await promptVision(ctx, opts.vision);
			if (vision === null) continue;
			await mutateModel(ctx, providerId, modelId, (m) => { m.input = vision ? ["text", "image"] : ["text"]; });
		} else if (field === "context") {
			const result = await promptContextWindow(ctx, typeof model.contextWindow === "number" ? model.contextWindow : undefined);
			if (result === null) continue;
			await mutateModel(ctx, providerId, modelId, (m) => { if (result === 0) delete m.contextWindow; else m.contextWindow = result; });
		} else if (field === "maxtokens") {
			const result = await promptMaxTokens(ctx, typeof model.maxTokens === "number" ? model.maxTokens : undefined);
			if (result === null) continue;
			await mutateModel(ctx, providerId, modelId, (m) => { if (result === 0) delete m.maxTokens; else m.maxTokens = result; });
		} else if (field === "override") {
			await editModelOverride(ctx, providerId, modelId);
		} else if (field === "delete") {
			const ok = await ctx.ui.confirm("Delete model?", `Remove "${modelId}" from "${providerId}"?`);
			if (!ok) continue;
			const saved = await mutateProvider(ctx, providerId, (p) => {
				const models = Array.isArray(p.models) ? p.models : [];
				const index = models.findIndex((m: any) => modelIdOf(m) === modelId);
				if (index === -1) return false;
				models.splice(index, 1);
				return true;
			});
			if (saved) ctx.ui.notify(`Deleted "${modelId}".`, "info");
			return true;
		}
	}
}

// Mutate a single model entry in place and save.
async function mutateModel(ctx: CommandContext, providerId: string, modelId: string, mutate: (model: any, provider?: any) => void): Promise<boolean> {
	return mutateProvider(ctx, providerId, (p) => {
		const models = Array.isArray(p.models) ? p.models : [];
		const index = models.findIndex((m: any) => modelIdOf(m) === modelId);
		if (index === -1) return false;
		// Strings become objects so per-field knobs have somewhere to live.
		if (typeof models[index] === "string") models[index] = { id: modelId, input: ["text", "image"] };
		mutate(models[index], p);
		return true;
	}).then((saved) => {
		if (saved) ctx.ui.notify(`Updated "${modelId}".`, "info");
		return saved;
	});
}

// Edit per-model HTTP headers and api/baseUrl endpoint override.
async function editModelOverride(ctx: CommandContext, providerId: string, modelId: string) {
	let model: any;
	try {
		model = findModel(loadModelsConfig().providers?.[providerId], modelId);
	} catch {
		model = undefined;
	}
	const currentBase = typeof model?.baseUrl === "string" ? model.baseUrl : "";
	const currentHeaders = model?.headers && typeof model.headers === "object" ? JSON.stringify(model.headers) : "";

	const base = await ctx.ui.input("baseUrl override (blank = use provider, \"-\" to clear)", currentBase || "e.g. https://api.example.com/v1");
	if (base === undefined) return;
	const headers = await ctx.ui.input("Headers as JSON (blank = keep, \"-\" to clear)", currentHeaders || 'e.g. {"x-api-version":"2024-01"}');
	if (headers === undefined) return;

	let parsedHeaders: Record<string, string> | null | undefined;
	const trimmedHeaders = headers.trim();
	if (trimmedHeaders === "-") parsedHeaders = null;
	else if (trimmedHeaders) {
		try {
			const obj = JSON.parse(trimmedHeaders);
			if (!obj || typeof obj !== "object" || Array.isArray(obj)) throw new Error("not an object");
			parsedHeaders = obj;
		} catch (error) {
			ctx.ui.notify(`Invalid headers JSON: ${error instanceof Error ? error.message : String(error)}`, "error");
			return;
		}
	}

	await mutateModel(ctx, providerId, modelId, (m) => {
		const trimmedBase = base.trim();
		if (trimmedBase === "-") delete m.baseUrl;
		else if (trimmedBase) m.baseUrl = trimmedBase;
		if (parsedHeaders === null) delete m.headers;
		else if (parsedHeaders) m.headers = parsedHeaders;
	});
}

function modelIdOf(model: any): string {
	return typeof model === "string" ? model.trim() : typeof model?.id === "string" ? model.id.trim() : "";
}

function findModel(provider: any, id: string): any {
	const models = Array.isArray(provider?.models) ? provider.models : [];
	return models.find((m: any) => modelIdOf(m) === id);
}

// Resolve a stored provider's apiKey reference back into mode+value so we can
// reuse it for probing. Anything other than $VAR or !cmd is treated as literal.
function apiKeyFromProvider(provider: any): { mode: ApiKeyMode; value?: string } {
	const raw = typeof provider?.apiKey === "string" ? provider.apiKey : "";
	if (!raw || raw === "dummy" || raw === "ollama") return { mode: "none" };
	if (raw.startsWith("!")) return { mode: "shell", value: raw.slice(1) };
	if (raw.startsWith("$")) return { mode: "env", value: raw.slice(1) };
	return { mode: "literal", value: raw };
}

async function addModelEntriesToProvider(ctx: CommandContext, providerId: string, ids: string[]) {
	const existing = new Set<string>();
	try {
		const provider = loadModelsConfig().providers?.[providerId];
		for (const m of Array.isArray(provider?.models) ? provider.models : []) existing.add(modelIdOf(m));
	} catch {
		// fall through; mutateProvider re-reads and reports errors
	}
	const fresh = dedupe(ids).filter((id) => id && !existing.has(id));
	if (fresh.length === 0) {
		ctx.ui.notify("Nothing to add — all selected models already exist.", "info");
		return;
	}

	// Added models default to reasoning on (xhigh ceiling) + text+image. Tune per
	// model later via Edit provider → Edit a model.
	const saved = await mutateProvider(ctx, providerId, (p) => {
		const models = Array.isArray(p.models) ? p.models : [];
		const providerApi: ProviderApi = (p?.api as ProviderApi) || "openai-completions";
		for (const id of fresh) models.push(buildModelEntry(id, { reasoning: "xhigh", vision: true }, undefined, providerApi));
		p.models = models;
		return true;
	});
	if (saved) ctx.ui.notify(`Added ${fresh.length} model${fresh.length === 1 ? "" : "s"} to "${providerId}".`, "info");
}

async function reprobeProvider(ctx: CommandContext, providerId: string) {
	let provider: any;
	try {
		provider = loadModelsConfig().providers?.[providerId];
	} catch (error) {
		ctx.ui.notify(`Could not read ${MODELS_JSON_PATH}: ${error instanceof Error ? error.message : String(error)}`, "error");
		return;
	}
	if (provider?.api === "anthropic-messages") {
		ctx.ui.notify("Anthropic-style endpoints don't expose /models. Use 'Add models manually'.", "warning");
		return;
	}
	const baseUrl = typeof provider?.baseUrl === "string" ? provider.baseUrl : "";
	if (!baseUrl) {
		ctx.ui.notify(`Provider "${providerId}" has no baseUrl to probe.`, "error");
		return;
	}

	const apiKey = apiKeyFromProvider(provider);
	let probed: ProbeItem[];
	try {
		ctx.ui.notify(`Probing ${buildProbeUrl(baseUrl)} ...`, "info");
		probed = await probeOpenAIModels(baseUrl, apiKey.mode, apiKey.value);
	} catch (error) {
		ctx.ui.notify(`Probe failed: ${error instanceof Error ? error.message : String(error)}`, "error");
		return;
	}

	const existing = new Set((Array.isArray(provider.models) ? provider.models : []).map(modelIdOf));
	const novel = probed.filter((item) => !existing.has(item.value));
	if (novel.length === 0) {
		ctx.ui.notify("No new models — everything the endpoint returned is already configured.", "info");
		return;
	}

	const picked = await pickMany(ctx, `New models for ${providerId}`, novel);
	if (!picked || picked.length === 0) return;
	await addModelEntriesToProvider(ctx, providerId, picked);
}

async function addModelsToProvider(ctx: CommandContext, providerId: string) {
	let provider: any;
	try {
		provider = loadModelsConfig().providers?.[providerId];
	} catch (error) {
		ctx.ui.notify(`Could not read ${MODELS_JSON_PATH}: ${error instanceof Error ? error.message : String(error)}`, "error");
		return;
	}
	const style: ProviderStyle =
		provider?.api === "anthropic-messages" ? "anthropic" : provider?.compat ? "ollama" : "openai";
	const ids = await promptModelIdsOneByOne(ctx, style);
	if (!ids || ids.length === 0) return;
	await addModelEntriesToProvider(ctx, providerId, ids);
}

async function deleteProviderFlow(ctx: CommandContext) {
	let cursor = 0;
	let deletedAny = false;

	while (true) {
		let config: ModelsConfig;
		try {
			config = loadModelsConfig();
		} catch (error) {
			ctx.ui.notify(`Could not read ${MODELS_JSON_PATH}: ${error instanceof Error ? error.message : String(error)}`, "error");
			return;
		}

		config.providers ||= {};
		const providerIds = Object.keys(config.providers).sort((a, b) => a.localeCompare(b));
		if (providerIds.length === 0) {
			ctx.ui.notify(
				deletedAny ? `No providers left in ${MODELS_JSON_PATH}` : `No providers found in ${MODELS_JSON_PATH}`,
				deletedAny ? "info" : "warning",
			);
			return;
		}

		const choice = await selectOne(
			ctx,
			"Delete provider",
			providerIds.map((providerId) => {
				const inline = describeProviderInline(providerId, config.providers?.[providerId]);
				return {
					value: providerId,
					label: inline.label,
					suffix: inline.suffix,
					searchText: inline.searchText,
				};
			}),
			{ initialIndex: Math.min(cursor, providerIds.length - 1) },
		);
		if (!choice) return;

		const provider = config.providers[choice];
		const confirmed = await ctx.ui.confirm("Delete provider?", describeProvider(choice, provider));
		const selectedIndex = providerIds.indexOf(choice);
		cursor = selectedIndex;
		if (!confirmed) continue;

		cursor = selectedIndex + 1;
		delete config.providers[choice];

		try {
			saveModelsConfig(config);
		} catch (error) {
			ctx.ui.notify(`Could not write ${MODELS_JSON_PATH}: ${error instanceof Error ? error.message : String(error)}`, "error");
			return;
		}

		deletedAny = true;
		ctx.ui.notify(`Deleted provider \"${choice}\" from ${MODELS_JSON_PATH}`, "info");
	}
}

async function promptProviderStyle(
	ctx: CommandContext,
): Promise<{ style: ProviderStyle; api: ProviderApi } | null> {
	const providerStyleLabel = await selectOne(ctx, "Provider style", [
		"OpenAI-compatible",
		"Anthropic-compatible",
		"Ollama-compatible",
	]);
	if (!providerStyleLabel) return null;

	const style: ProviderStyle =
		providerStyleLabel === "Anthropic-compatible"
			? "anthropic"
			: providerStyleLabel === "Ollama-compatible"
				? "ollama"
				: "openai";
	const api: ProviderApi = style === "anthropic" ? "anthropic-messages" : "openai-completions";
	return { style, api };
}

async function promptEndpoint(
	ctx: CommandContext,
	style: ProviderStyle,
	api: ProviderApi,
): Promise<{ normalized: string; raw: string } | null> {
	const endpointInput = await ctx.ui.input(
		"Endpoint",
		style === "anthropic"
			? "e.g. https://api.anthropic-proxy.com/v1"
			: style === "ollama"
				? "e.g. http://localhost:11434/v1"
				: "e.g. https://api.example.com/v1 or http://localhost:11434/v1",
	);
	if (endpointInput === undefined) return null;
	const raw = endpointInput.trim();
	if (!raw) {
		ctx.ui.notify("Endpoint is required.", "error");
		return null;
	}

	try {
		return { normalized: normalizeEndpoint(raw, api), raw };
	} catch (error) {
		ctx.ui.notify(`Invalid endpoint: ${error instanceof Error ? error.message : String(error)}`, "error");
		return null;
	}
}

async function confirmEndpointReuse(ctx: CommandContext, normalizedEndpoint: string): Promise<boolean> {
	let config: ModelsConfig;
	try {
		config = loadModelsConfig();
	} catch (error) {
		ctx.ui.notify(`Could not read ${MODELS_JSON_PATH}: ${error instanceof Error ? error.message : String(error)}`, "error");
		return false;
	}

	const providersWithSameEndpoint = findProvidersByEndpoint(config, normalizedEndpoint);
	if (providersWithSameEndpoint.length === 0) return true;

	return ctx.ui.confirm(
		"Endpoint already exists",
		`This endpoint is already used by:\n${providersWithSameEndpoint.map((id) => `- ${id}`).join("\n")}\n\nAdd another provider with the same endpoint?`,
	);
}

async function promptProviderId(ctx: CommandContext, normalizedEndpoint: string): Promise<string | null> {
	let existingIds = new Set<string>();
	try {
		existingIds = new Set(Object.keys(loadModelsConfig().providers ?? {}));
	} catch {
		// If config can't be read, persistProvider surfaces the error later.
	}

	const providerIdSuggestion = suggestProviderId(normalizedEndpoint);
	const suggestionTaken = existingIds.has(providerIdSuggestion);

	while (true) {
		const providerNameInput = await ctx.ui.input(
			suggestionTaken ? "Provider name (must be unique)" : `Provider name (blank = ${providerIdSuggestion})`,
			"e.g. custom-example-com",
		);
		if (providerNameInput === undefined) return null;
		const providerId = slugify(providerNameInput.trim() || providerIdSuggestion);
		if (!providerId) {
			ctx.ui.notify("Provider name is required.", "error");
			continue;
		}

		// Provider names must be unique — never silently overwrite an existing one.
		if (existingIds.has(providerId)) {
			ctx.ui.notify(`Provider "${providerId}" already exists. Choose a different name.`, "warning");
			continue;
		}

		if (BUILTIN_PROVIDER_IDS.has(providerId)) {
			const ok = await ctx.ui.confirm(
				"Override built-in provider?",
				`"${providerId}" matches a built-in provider id. Saving this will override that provider in the active models config. Continue?`,
			);
			if (!ok) continue;
		}
		return providerId;
	}
}

async function persistProvider(ctx: CommandContext, providerId: string, providerConfig: any): Promise<boolean> {
	let config: ModelsConfig;
	try {
		config = loadModelsConfig();
	} catch (error) {
		ctx.ui.notify(`Could not read ${MODELS_JSON_PATH}: ${error instanceof Error ? error.message : String(error)}`, "error");
		return false;
	}

	config.providers ||= {};
	if (config.providers[providerId]) {
		// Names are validated unique at prompt time; this only triggers if the
		// config changed underneath us. Refuse rather than overwrite.
		ctx.ui.notify(`Provider "${providerId}" already exists. Not overwriting.`, "error");
		return false;
	}

	config.providers[providerId] = providerConfig;
	try {
		saveModelsConfig(config);
	} catch (error) {
		ctx.ui.notify(`Could not write ${MODELS_JSON_PATH}: ${error instanceof Error ? error.message : String(error)}`, "error");
		return false;
	}
	return true;
}

async function addProviderFlow(ctx: CommandContext) {
	const styleChoice = await promptProviderStyle(ctx);
	if (!styleChoice) return;
	const { style, api } = styleChoice;

	const endpoint = await promptEndpoint(ctx, style, api);
	if (!endpoint) return;
	if (!(await confirmEndpointReuse(ctx, endpoint.normalized))) return;

	const providerId = await promptProviderId(ctx, endpoint.normalized);
	if (!providerId) return;

	const apiKey = await promptApiKey(ctx);
	if (!apiKey) return;
	if (apiKey.mode === "none") {
		ctx.ui.notify(
			style === "ollama"
				? 'No API key selected. Using "ollama" automatically in the models config.'
				: 'No API key selected. Using "dummy" automatically in the models config.',
			"info",
		);
	}

	const modelIds = await collectModelIds(ctx, style, api, apiKey, endpoint.normalized, endpoint.raw);
	if (!modelIds || modelIds.length === 0) return;

	const providerConfig = buildProviderConfig(
		style,
		api,
		endpoint.normalized,
		apiKey,
		dedupe(modelIds),
		// New providers default to text+image, reasoning on (xhigh ceiling). Tune
		// per model later via Edit provider → Edit a model.
		{ reasoning: "xhigh", vision: true },
	);
	if (!(await persistProvider(ctx, providerId, providerConfig))) return;

	ctx.ui.notify(`Saved provider \"${providerId}\" to ${MODELS_JSON_PATH}`, "info");
	ctx.ui.notify("Open /model to use your new provider.", "info");
}

async function collectModelIds(
	ctx: CommandContext,
	style: ProviderStyle,
	api: ProviderApi,
	apiKey: { mode: ApiKeyMode; value?: string },
	normalizedEndpoint: string,
	trimmedEndpointInput: string,
): Promise<string[] | null> {
	if (api !== "openai-completions") {
		return promptModelIdsOneByOne(ctx, style);
	}

	const modelMode = await selectOne(ctx, "Models", ["Auto probe from /models", "Add manually"]);
	if (!modelMode) return null;
	if (modelMode !== "Auto probe from /models") {
		return promptModelIdsOneByOne(ctx, style);
	}

	try {
		ctx.ui.notify(`Probing ${buildProbeUrl(normalizedEndpoint)} ...`, "info");
		const probedModels = await probeOpenAIModels(normalizedEndpoint, apiKey.mode, apiKey.value);
		if (probedModels.length === 0) {
			ctx.ui.notify("Probe succeeded but returned no models. Switching to manual entry.", "warning");
			return promptModelIdsOneByOne(ctx, style);
		}
		return pickMany(ctx, "Select models", probedModels);
	} catch (error) {
		const schemeHint = hasExplicitScheme(trimmedEndpointInput) ? "" : "\n\nNo http:// or https:// was provided.";
		ctx.ui.notify(
			`Auto probe failed: ${error instanceof Error ? error.message : String(error)}.${schemeHint}\n\nSwitching to manual entry.`,
			"warning",
		);
		return promptModelIdsOneByOne(ctx, style);
	}
}

export default function customEndpointWizard(pi: ExtensionAPI) {
	pi.registerCommand("endpoint-custom", {
		description: "Wizard for adding, editing, or deleting custom providers in models configuration",
		handler: async (_args, ctx) => {
			const action = await selectOne(ctx, "Custom Endpoints", ["Add provider", "Edit provider", "Delete provider"]);
			if (!action) return;
			if (action === "Edit provider") {
				await editProviderFlow(ctx);
				return;
			}
			if (action === "Delete provider") {
				await deleteProviderFlow(ctx);
				return;
			}
			await addProviderFlow(ctx);
		},
	});
}
