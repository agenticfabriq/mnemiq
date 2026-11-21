import React, { useEffect, useState, useSyncExternalStore } from "react";
import { Button, theme } from "antd";
import type { PanelApi } from "../api";
import { driver } from "../build";
import { getSnapshot, subscribe } from "../store";

type Unfinished = {
  build_id: string;
  goal: string;
  dashboard_id: number | null;
  status: string;
  steps_done: number;
  steps_total: number;
};

// The builds the person said "Not now" to, for the life of the page.
const dismissed = new Set<string>();

/** For tests: nothing dismissed. */
export function resetDismissed(): void {
  dismissed.clear();
}

/** The person's builds that stopped before they were done (of the dashboard on screen, on one), each with Finish,
 * which runs it on from where it stopped, and Not now. Read again whenever this page stops planning or running a
 * build (a start that failed may still have made one). Finish waits while another build is planned or runs. */
export function UnfinishedBuilds({ api, dashboardId }: { api: PanelApi; dashboardId: number | null }) {
  const { runningBuild, planning } = useSyncExternalStore(subscribe, getSnapshot);
  const busy = planning || runningBuild !== null;
  const [builds, setBuilds] = useState<Unfinished[]>([]);
  const [, setShown] = useState(0);
  const { token } = theme.useToken();
  useEffect(() => {
    if (busy) return;
    const asking = new AbortController();
    const query = dashboardId !== null ? `?dashboard_id=${dashboardId}` : "";
    api.get<{ builds?: Unfinished[] }>(`/superset/builds${query}`, asking.signal)
      .then((answer) => setBuilds(answer.ok && Array.isArray(answer.body?.builds) ? answer.body.builds : []))
      .catch(() => {});
    return () => asking.abort();
  }, [api, dashboardId, busy]);
  const offered = builds.filter((build) => !dismissed.has(build.build_id) && build.build_id !== runningBuild);
  return (
    <>
      {offered.map((build) => (
        <div key={build.build_id} role="status" aria-label="Unfinished build"
          style={{ display: "flex", flexDirection: "column", gap: 4, padding: token.paddingXS,
            borderRadius: token.borderRadius, background: token.colorWarningBg }}>
          <span>'{build.goal}' stopped after {build.steps_done} of {build.steps_total} steps</span>
          <div style={{ display: "flex", gap: 8 }}>
            <Button size="small" type="primary" disabled={busy} onClick={() => {
              driver.resume(build.goal, build.dashboard_id !== null ? { dashboardId: build.dashboard_id } : {},
                build.build_id);
            }}>Finish</Button>
            <Button size="small" onClick={() => {
              dismissed.add(build.build_id);
              setShown((n) => n + 1);
            }}>Not now</Button>
          </div>
        </div>
      ))}
    </>
  );
}
