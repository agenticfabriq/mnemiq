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

/** Where a goal goes, worked out when it is sent: a target, or a sentence saying why nothing can be built. */
export type TargetResolver = () => Promise<BuildTarget | { refusal: string }>;

/** The build the panel shows: its goal and target, its id once mnemiq has made it, what starting it answered, the
 * build as last read, and the sentence it stopped with, if it stopped before it was done. */
export type BuildRecord = {
  goal: string;
  target: BuildTarget;
  buildId: string | null;
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

// On a 409 (another request holds this build's step), or an answer in which no step moved (mnemiq left one
// uncertain: Superset busy, silent or unreachable), wait this long before each of five more tries.
const WAITS_S = [2, 4, 8, 16, 16];

// A build run with no record of its own yet (none shown or started on this page).
const UNNAMED: BuildRecord = { goal: "", target: {}, buildId: null, answer: null, view: null, stopped: null };

export const BUSY_STOPPED = "Another request is running this build's step; finish the build once it is done.";
export const ALREADY_BUILDING = "A build is already being planned or run on this page; wait for it to finish.";
const NO_PROGRESS = "The build's step did not move after several tries; finish the build later.";
const UNREADABLE = "mnemiq's answer for this build's step could not be read; finish the build later.";

type Options = { sleep?: (ms: number) => Promise<void> };

type Phase = { kind: "idle" } | { kind: "planning" } | { kind: "running"; buildId: string };

const doneSteps = (view: BuildView | null) => view?.steps.filter((step) => step.status === "done").length ?? -1;

/** Whether `value` reads as a build mnemiq answered: an id, a status and a list of steps. */
function isBuildView(value: unknown): value is BuildView {
  const view = value as Partial<BuildView> | null;
  return typeof view?.build_id === "string" && typeof view.status === "string" && Array.isArray(view.steps);
}

export class BuildDriver {
  private api: PanelApi | null = null;
  // One phase for the page: idle, planning a goal, or running one build. A start, a Finish and a run are each refused
  // unless it is idle, so the page never plans or runs two builds, and never runs two steps at once.
  private phase: Phase = { kind: "idle" };
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
    return this.phase.kind === "running" ? this.phase.buildId : null;
  }

  /** Whether a goal is being planned or a build run on this page. */
  busy(): boolean {
    return this.phase.kind !== "idle";
  }

  /** Stops after the step in flight. */
  stop(): void {
    this.stopping = true;
  }

  /** Shows `view` (or none yet) as the panel's build `buildId`, for `goal`, when nothing is being planned or run. */
  show(goal: string, target: BuildTarget, view: BuildView | null, buildId: string | null = view?.build_id ?? null): boolean {
    if (this.busy()) return false;
    update((state) => ({ ...state, build: { goal, target, buildId, answer: null, view, stopped: null } }));
    return true;
  }

  /** Runs on the stopped build `buildId` for `goal`, when nothing else is being planned or run: what Finish does. */
  resume(goal: string, target: BuildTarget, buildId: string): boolean {
    if (!this.show(goal, target, null, buildId)) return false;
    void this.run(buildId);
    return true;
  }

  /** Plans `goal` onto `target` (a dashboard, or a new one), or onto the target `target` works out when called, and
   * runs the build it answers. Refused, with nothing changed, while another goal is planned or a build runs. */
  async start(goal: string, target: BuildTarget | TargetResolver): Promise<StartAnswer> {
    if (this.busy()) return { failed: ALREADY_BUILDING };
    const api = this.need();
    this.stopping = false;
    this.enter({ kind: "planning" });
    update((state) => ({ ...state, build: { goal, target: {}, buildId: null, answer: null, view: null, stopped: null } }));
    let handed = false;
    try {
      const resolved = typeof target === "function" ? await target() : target;
      if ("refusal" in resolved) return this.failed(goal, resolved.refusal);
      const body = {
        goal,
        ...(resolved.dashboardId !== undefined ? { dashboard_id: resolved.dashboardId } : {}),
        ...(resolved.fresh ? { fresh: true } : {}),
      };
      const answer = await api.post<StartAnswer>("/superset/builds", body);
      const found: StartAnswer = answer.ok ? answer.body : { failed: answer.refusal.sentence };
      const view = "build" in found && isBuildView(found.build) ? found.build : null;
      update((state) => ({
        ...state,
        build: { goal, target: resolved, buildId: view?.build_id ?? null, answer: found, view,
          stopped: "failed" in found ? found.failed : null },
      }));
      if (view?.status === "building") {
        handed = true;
        this.enter({ kind: "running", buildId: view.build_id });
        void this.loop(view.build_id, view);
      }
      return found;
    } catch (error) {
      if ((error as { name?: string })?.name === "AbortError") throw error;
      return this.failed(goal, "mnemiq could not be asked to plan this goal; try again shortly.");
    } finally {
      if (!handed) this.enter({ kind: "idle" });
    }
  }

  /** Runs `buildId` one step per request until it is done or stops, unless something is being planned or run. */
  async run(buildId: string): Promise<void> {
    if (this.busy()) return;
    this.need();
    this.stopping = false;
    this.enter({ kind: "running", buildId });
    await this.loop(buildId, null);
  }

  /** For tests: nothing planned or running, and the next start free to begin. */
  reset(): void {
    this.stopping = true;
    this.phase = { kind: "idle" };
  }

  private async loop(buildId: string, last: BuildView | null): Promise<void> {
    const api = this.need();
    let waited = 0;
    try {
      while (!this.stopping) {
        const answer = await api.post<{ build?: unknown }>(`/superset/builds/${encodeURIComponent(buildId)}/next`, {});
        if (answer.ok) {
          const view = answer.body?.build;
          if (!isBuildView(view) || view.build_id !== buildId) return this.stopped(buildId, UNREADABLE);
          this.write(buildId, (record) => ({ ...record, view }));
          if (view.status !== "building") return;
          const moved = doneSteps(view) > doneSteps(last);
          last = view;
          if (moved) {
            waited = 0;
            continue;
          }
          if (waited < WAITS_S.length) {
            await this.sleep(WAITS_S[waited++] * 1000);
            continue;
          }
          return this.stopped(buildId, view.message?.trim() || NO_PROGRESS);
        }
        if (answer.refusal.kind === "busy" && waited < WAITS_S.length) {
          await this.sleep(WAITS_S[waited++] * 1000);
          continue;
        }
        // Any other refusal (a timeout, mnemiq unreachable, signed out) stops the driver; the build stays
        // `building` on the server, and the panel offers to finish it.
        return this.stopped(buildId, answer.refusal.kind === "busy" ? BUSY_STOPPED : answer.refusal.sentence);
      }
    } catch (error) {
      if ((error as { name?: string })?.name !== "AbortError") this.stopped(buildId, UNREADABLE);
    } finally {
      this.enter({ kind: "idle" });
    }
  }

  /** Writes `buildId`'s record, and only its own: a record of another build is left as it is. */
  private write(buildId: string, fn: (record: BuildRecord) => BuildRecord): void {
    update((state) => {
      const record = state.build ?? { ...UNNAMED, buildId };
      return record.buildId === buildId ? { ...state, build: fn(record) } : state;
    });
  }

  private stopped(buildId: string, sentence: string): void {
    this.write(buildId, (record) => ({ ...record, stopped: sentence }));
  }

  private failed(goal: string, sentence: string): StartAnswer {
    const found = { failed: sentence };
    update((state) => ({ ...state, build: { goal, target: {}, buildId: null, answer: found, view: null, stopped: sentence } }));
    return found;
  }

  private enter(phase: Phase): void {
    this.phase = phase;
    update((state) => ({ ...state, planning: phase.kind === "planning",
      runningBuild: phase.kind === "running" ? phase.buildId : null }));
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
