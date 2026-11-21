// Whether this person may use the panel, asked once with a single `GET v1/schema`: mnemiq resolves what Superset lets
// them read, and answers the opening questions (`starters`) from it, or refuses with its own sentence. A person who
// is not a builder is told the roles building needs, and the panel asks nothing else.
import type { PanelApi } from "./api";

export type Readiness =
  | { kind: "ready"; starters: string[] }
  | { kind: "not-a-builder" | "signed-out" | "unavailable"; sentence: string };

const UNREADABLE = "mnemiq's answer could not be read; try again shortly.";

export async function probe(api: PanelApi, signal?: AbortSignal): Promise<Readiness> {
  const answer = await api.get<{ starters?: unknown }>("/schema", signal);
  if (!answer.ok) {
    const { kind, sentence } = answer.refusal;
    if (kind === "signed-out") return { kind, sentence };
    // Superset refusing this person anything of the panel's says, as the roles sentence does, that they may not
    // build here: each is told in its own words.
    if (kind === "not-a-builder" || kind === "not-permitted" || kind === "not-for-building") {
      return { kind: "not-a-builder", sentence };
    }
    return { kind: "unavailable", sentence };
  }
  const starters = answer.body?.starters;
  if (!Array.isArray(starters) || !starters.every((s) => typeof s === "string")) {
    return { kind: "unavailable", sentence: UNREADABLE };
  }
  return { kind: "ready", starters };
}
