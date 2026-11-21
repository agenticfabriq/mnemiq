import React, { useSyncExternalStore } from "react";
import { Badge, Button } from "antd";
import { chat } from "@apache-superset/core";
import { getSnapshot, subscribe } from "./store";

/** The collapsed entry point Superset mounts: opens and closes the panel, and shows a dot while a build runs. */
export function Trigger() {
  const { runningBuild } = useSyncExternalStore(subscribe, getSnapshot);
  const running = runningBuild !== null;
  return (
    <Badge dot={running} data-testid={running ? "mnemiq-running" : undefined}>
      <Button
        shape="round"
        aria-label={running ? "mnemiq, a build is running" : "mnemiq"}
        onClick={() => (chat.isOpen() ? chat.close() : chat.open())}
      >
        mnemiq
      </Button>
    </Badge>
  );
}
