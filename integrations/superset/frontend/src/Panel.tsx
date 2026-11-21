import React, { useEffect, useState, type CSSProperties } from "react";
import { Button, theme } from "antd";
import { chat } from "@apache-superset/core";

// 640 px tall, at most 80% of the window: `min(640px, 80vh)`, spelt so jsdom, whose CSS parser drops `min()`, keeps it.
const FLOATING: CSSProperties = { width: 420, height: 640, maxHeight: "80vh" };
const DOCKED: CSSProperties = { width: "100%", height: "100%" };

/** The panel Superset mounts, floating over the page or docked beside it. Its styles are inline, under one root
 * element, with antd's colours, so it brings no stylesheet of its own. */
export function Panel() {
  const [mode, setMode] = useState(chat.getDisplayMode());
  useEffect(() => {
    const changed = chat.onDidChangeDisplayMode((next) => setMode(next));
    setMode(chat.getDisplayMode());
    return () => changed.dispose();
  }, []);
  const { token } = theme.useToken();
  const root: CSSProperties = {
    ...(mode === "floating" ? FLOATING : DOCKED),
    display: "flex",
    flexDirection: "column",
    boxSizing: "border-box",
    background: token.colorBgContainer,
    color: token.colorText,
    border: mode === "floating" ? `1px solid ${token.colorBorderSecondary}` : undefined,
    borderRadius: mode === "floating" ? token.borderRadiusLG : undefined,
    fontFamily: token.fontFamily,
    fontSize: token.fontSize,
    overflow: "hidden",
  };
  const header: CSSProperties = {
    display: "flex",
    alignItems: "center",
    gap: token.paddingXS,
    padding: `${token.paddingXS}px ${token.padding}px`,
    borderBottom: `1px solid ${token.colorBorderSecondary}`,
  };
  return (
    <section data-mnemiq-panel={mode} style={root} aria-label="mnemiq">
      <header style={header}>
        <strong style={{ flex: 1 }}>mnemiq</strong>
        {mode === "floating" ? (
          <Button size="small" type="text" onClick={() => chat.setDisplayMode("panel")}>Dock</Button>
        ) : (
          <Button size="small" type="text" onClick={() => chat.setDisplayMode("floating")}>Float</Button>
        )}
        <Button size="small" type="text" onClick={() => chat.close()}>Close</Button>
      </header>
      <div style={{ flex: 1, overflow: "auto", padding: token.padding }} />
    </section>
  );
}
