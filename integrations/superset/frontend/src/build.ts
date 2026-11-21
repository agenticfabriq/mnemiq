// Building a dashboard step by step: mnemiq plans a goal into a build, and the panel runs it one step per request
// (`POST v1/superset/builds/{id}/next`) until it is done or stops. The driver lives in module memory, outside the
// panel's component: rc2 unmounts the panel on every close, and closing it must not stop a build while the page
// stays loaded. Leaving or reloading the page stops it after the step in flight; the build stays `building` on the
// server, and the panel offers to finish it.
import type { PanelApi } from "./api";
import { update } from "./store";

export type StepView = {
  key: string;
  kind: string;
  status: string;
  object_id: number | null;
  started_at: string | null;
  lease_until: string | null;
};

export type PlanSummary = {
  charts: { key: string; title: string; family: string; measures: string[] }[];
  measures: { name: string; label: string | null; expression: string | null; kind: string | null; currency: string | null }[];
};

/** A build as mnemiq answers it (`_build` in mnemiq's server/superset_build.py). */
export type BuildView = {
  build_id: string;
  status: string;
  dashboard_id: number | null;
  steps: StepView[];
  checks: Record<string, string>;
  ran: string | null;
  message: string | null;
  plan: PlanSummary | null;
  check_reasons: Record<string, string>;
};

export type LeftOutPart = { part: string; chart_id: number; title: string | null };

/** What starting a goal answers: a build; an offer of a dashboard that already covers it; a plan refused part by
 * part; a goal the dashboard already shows whole; or, the panel's own, a request refused with its sentence. */
export type StartAnswer =
  | { build: BuildView; left_out?: LeftOutPart[] }
  | { offer: { dashboard_id: number; title: string | null; reason: string | null; message: string } }
  | { refused: { reason: string; parts: { part: string; reason: string }[] } }
  | { shown: { dashboard_id: number; left_out: LeftOutPart[]; message: string } }
  | { failed: string };

export type BuildTarget = { dashboardId?: number; fresh?: boolean };

/** The build the panel shows: its goal and target, what starting it answered, the build as last read, and the
 * sentence it stopped with, if it stopped before it was done. */
export type BuildRecord = {
  goal: string;
  target: BuildTarget;
  answer: StartAnswer | null;
  view: BuildView | null;
  stopped: string | null;
};

/** The words for each chart's check status; only a chart whose numbers mnemiq matched is called checked. */
export const CHECK_WORDS = {
  checked: "checked",
  mismatch: "numbers differ from mnemiq's",
  empty: "no rows",
  cut: "result cut short",
  format_wrong: "format does not fit the measure",
  not_checked: "not checked",
} as const;

// On a 409 (another request holds this build's step), wait this long before each of five more tries.
const WAITS_S = [2, 4, 8, 16, 16];

// A build run with no record of its own yet (none shown or started on this page).
const UNNAMED: BuildRecord = { goal: "", target: {}, answer: null, view: null, stopped: null };

export const BUSY_STOPPED = "Another request is running this build's step; finish the build once it is done.";

type Options = { sleep?: (ms: number) => Promise<void> };

export class BuildDriver {
  private api: PanelApi | null = null;
  private current: string | null = null;
  private stopping = false;
  private readonly sleep: (ms: number) => Promise<void>;

  constructor({ sleep }: Options = {}) {
    this.sleep = sleep ?? ((ms) => new Promise((done) => setTimeout(done, ms)));
  }

  /** The client the driver asks mnemiq with (the panel's, for this page). */
  attach(api: PanelApi): void {
    this.api = api;
  }

  /** The build this page is running, if any. */
  running(): string | null {
    return this.current;
  }

  /** Stops after the step in flight. */
  stop(): void {
    this.stopping = true;
  }

  /** Shows `view` (or none yet) as the panel's build, for `goal`: what Finish shows before its first step answers. */
  show(goal: string, target: BuildTarget, view: BuildView | null): void {
    update((state) => ({ ...state, build: { goal, target, answer: null, view, stopped: null } }));
  }

  /** Plans `goal` onto `target` (a dashboard, or a new one), and runs the build it answers. */
  async start(goal: string, target: BuildTarget): Promise<StartAnswer> {
    const api = this.need();
    const body = {
      goal,
      ...(target.dashboardId !== undefined ? { dashboard_id: target.dashboardId } : {}),
      ...(target.fresh ? { fresh: true } : {}),
    };
    update((state) => ({ ...state, build: { goal, target, answer: null, view: null, stopped: null } }));
    const answer = await api.post<StartAnswer>("/superset/builds", body);
    const found: StartAnswer = answer.ok ? answer.body : { failed: answer.refusal.sentence };
    const view = "build" in found ? found.build : null;
    update((state) => ({
      ...state,
      build: { goal, target, answer: found, view, stopped: "failed" in found ? found.failed : null },
    }));
    if (view?.status === "building") void this.run(view.build_id);
    return found;
  }

  /** Runs `buildId` one step per request until it is done or stops. Never two loops, so never two steps at once. */
  async run(buildId: string): Promise<void> {
    if (this.current !== null) return;
    const api = this.need();
    this.current = buildId;
    this.stopping = false;
    update((state) => ({ ...state, runningBuild: buildId,
      build: state.build ? { ...state.build, stopped: null } : state.build }));
    let waited = 0;
    try {
      while (!this.stopping) {
        const answer = await api.post<{ build: BuildView }>(`/superset/builds/${encodeURIComponent(buildId)}/next`, {});
        if (answer.ok) {
          waited = 0;
          const view = answer.body.build;
          update((state) => ({ ...state, build: { ...(state.build ?? UNNAMED), view } }));
          if (view.status !== "building") return;
          continue;
        }
        if (answer.refusal.kind === "busy" && waited < WAITS_S.length) {
          await this.sleep(WAITS_S[waited++] * 1000);
          continue;
        }
        // Any other refusal (a timeout, mnemiq unreachable, signed out) stops the driver; the build stays
        // `building` on the server, and the panel offers to finish it.
        const sentence = answer.refusal.kind === "busy" ? BUSY_STOPPED : answer.refusal.sentence;
        update((state) => ({ ...state, build: { ...(state.build ?? UNNAMED), stopped: sentence } }));
        return;
      }
    } finally {
      this.current = null;
      update((state) => ({ ...state, runningBuild: null }));
    }
  }

  /** For tests: no build running, and the next `run` free to start. */
  reset(): void {
    this.stopping = true;
    this.current = null;
  }

  private need(): PanelApi {
    if (this.api === null) throw new Error("the build driver has no client: attach the panel's first");
    return this.api;
  }
}

/** The page's driver. */
export const driver = new BuildDriver();

/** For tests: the page's driver idle. */
export function resetBuilds(): void {
  driver.reset();
}
