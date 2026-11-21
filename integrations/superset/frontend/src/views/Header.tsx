import React, { type CSSProperties } from "react";
import { Button, theme } from "antd";
import { chat } from "@apache-superset/core";

type Mode = ReturnType<typeof chat.getDisplayMode>;

/** The panel's header: its name, and Dock or Float, and Close, each through the host. */
export function Header({ mode }: { mode: Mode }) {
  const { token } = theme.useToken();
  const style: CSSProperties = {
    display: "flex",
    alignItems: "center",
    gap: token.paddingXS,
    padding: `${token.paddingXS}px ${token.padding}px`,
    borderBottom: `1px solid ${token.colorBorderSecondary}`,
  };
  return (
    <header style={style}>
      <strong style={{ flex: 1 }}>mnemiq</strong>
      {mode === "floating" ? (
        <Button size="small" type="text" onClick={() => chat.setDisplayMode("panel")}>Dock</Button>
      ) : (
        <Button size="small" type="text" onClick={() => chat.setDisplayMode("floating")}>Float</Button>
      )}
      <Button size="small" type="text" onClick={() => chat.close()}>Close</Button>
    </header>
  );
}
