import React, { useSyncExternalStore, type CSSProperties } from "react";
import { Alert, Button, Spin, Typography, theme } from "antd";
import { CHECK_WORDS, driver, type BuildRecord, type BuildView, type StepView } from "../build";
import { pageAppRoot } from "../page";
import { getSnapshot, subscribe } from "../store";
import { reloadKeeping } from "../reload";

const STEP_WORDS: Record<string, string> = { pending: "to do", done: "done", failed: "stopped",
  uncertain: "not sure it was made" };

/** A step's name for the person: a chart by its planned title, the others by what they make. */
function stepName(step: StepView, view: BuildView): string {
  if (step.kind === "chart") return view.plan?.charts?.find((chart) => chart.key === step.key)?.title ?? step.key;
  if (step.kind === "dataset") return `Dataset ${step.key}`;
  if (step.kind === "dashboard") return "Dashboard";
  if (step.kind === "layout") return "Layout";
  if (step.kind === "check") return "Checks";
  return step.key;
}

const dashboardUrl = (id: number) => `${pageAppRoot()}/dashboard/${id}/`;

/** The build the panel shows: the goal; the plan; the steps; each chart's check; mnemiq's message; and, once it is
 * done, where to see it. A start that offered, refused or found the goal shown says so, with its answers. */
export function BuildCard({ onScreenId = null }: { onScreenId?: number | null }) {
  const { build, planning } = useSyncExternalStore(subscribe, getSnapshot);
  const { token } = theme.useToken();
  if (build === null) return null;
  const muted: CSSProperties = { color: token.colorTextSecondary };
  return (
    <section aria-label="Build" style={{ display: "flex", flexDirection: "column", gap: token.paddingXS,
      padding: token.paddingSM, border: `1px solid ${token.colorBorderSecondary}`, borderRadius: token.borderRadius }}>
      <Typography.Text strong>{build.goal}</Typography.Text>
      {planning && build.answer === null ? (
        <div role="status" aria-label="Planning" style={muted}><Spin size="small" /> Planning the dashboard</div>
      ) : null}
      <Answer build={build} muted={muted} />
      {build.view ? <Progress view={build.view} muted={muted} onScreenId={onScreenId} /> : null}
      {build.stopped && !("failed" in (build.answer ?? {})) ? <Alert type="warning" showIcon message={build.stopped} /> : null}
    </section>
  );
}

function Answer({ build, muted }: { build: BuildRecord; muted: CSSProperties }) {
  const answer = build.answer;
  if (answer === null || "build" in answer) {
    return answer && "build" in answer && answer.left_out?.length ? (
      <div style={muted}>
        Already on the dashboard:{" "}
        {answer.left_out.map((part) => `${part.part} (${part.title ?? `chart ${part.chart_id}`})`).join("; ")}
      </div>
    ) : null;
  }
  if ("failed" in answer) return <Alert type="error" showIcon message={answer.failed} />;
  if ("offer" in answer) {
    const { dashboard_id, title, reason, message } = answer.offer;
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <span>{message}</span>
        {reason ? <span style={muted}>{reason}</span> : null}
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          <a href={dashboardUrl(dashboard_id)}>Open it</a>
          <Button size="small" onClick={() => void driver.start(build.goal, { dashboardId: dashboard_id })}>
            Add these charts to it
          </Button>
          <Button size="small" onClick={() => void driver.start(build.goal, { fresh: true })}>Build a new one</Button>
        </div>
        {title ? <span style={muted}>{title}</span> : null}
      </div>
    );
  }
  if ("refused" in answer) {
    return (
      <div>
        <span>{answer.refused.reason}</span>
        <ul style={{ margin: "4px 0 0", paddingLeft: 18 }}>
          {answer.refused.parts.map((part) => <li key={part.part}>{part.part}: {part.reason}</li>)}
        </ul>
      </div>
    );
  }
  if (!("shown" in answer) || typeof answer.shown?.message !== "string") {
    return <Alert type="warning" showIcon message="mnemiq answered this goal in a way this panel does not read; update the panel or mnemiq." />;
  }
  return (
    <div>
      <span>{answer.shown.message}</span>
      <div style={{ marginTop: 4 }}>
        <a href={dashboardUrl(answer.shown.dashboard_id)}>Open it</a>
      </div>
    </div>
  );
}

function Progress({ view, muted, onScreenId }: { view: BuildView; muted: CSSProperties; onScreenId: number | null }) {
  const { token } = theme.useToken();
  // A mnemiq older than the panel may leave out a field: each is read as absent, never as an error.
  const checks = view.checks ?? {};
  const reasons = view.check_reasons ?? {};
  const charts = view.steps.filter((step) => step.kind === "chart" && step.object_id !== null
    && checks[String(step.object_id)] !== undefined);
  return (
    <>
      {view.plan?.charts ? (
        <section aria-label="Plan" style={muted}>
          {view.plan.charts.map((chart) => (
            <div key={chart.key}>
              {chart.title} <span>({chart.family}{chart.measures?.length ? `: ${chart.measures.join(", ")}` : ""})</span>
            </div>
          ))}
        </section>
      ) : null}
      <ol aria-label="Steps" style={{ margin: 0, paddingLeft: 18 }}>
        {view.steps.map((step) => (
          <li key={step.key}>{stepName(step, view)}: {STEP_WORDS[step.status] ?? step.status}</li>
        ))}
      </ol>
      {charts.length > 0 ? (
        <section aria-label="Checks" style={{ display: "flex", flexDirection: "column", gap: 2 }}>
          {charts.map((step) => {
            const id = String(step.object_id);
            const status = checks[id] as keyof typeof CHECK_WORDS;
            const reason = reasons[id];
            return (
              <div key={id}>
                {stepName(step, view)}:{" "}
                <span style={{ color: status === "checked" ? token.colorSuccess : token.colorWarning }}>
                  {CHECK_WORDS[status] ?? checks[id]}
                </span>
                {reason ? <span style={muted}> ({reason})</span> : null}
              </div>
            );
          })}
        </section>
      ) : null}
      {view.message ? <span>{view.message}</span> : null}
      {view.status === "done" && view.dashboard_id !== null ? (
        view.dashboard_id === onScreenId ? (
          <Button size="small" onClick={() => reloadKeeping(getSnapshot().conversationId)}>Reload to see the changes</Button>
        ) : (
          <a href={dashboardUrl(view.dashboard_id)}>Open the dashboard</a>
        )
      ) : null}
    </>
  );
}
