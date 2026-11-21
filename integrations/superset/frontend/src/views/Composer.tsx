import React, { useState } from "react";
import { Button, Checkbox, Input, Segmented, theme } from "antd";

export type ComposerMode = "ask" | "build";

type Props = {
  busy: { ask: boolean; build: boolean };
  /** The dashboard on screen, when the person may read it: a Build goes onto it unless "New dashboard" is ticked. */
  onDashboard: string | null;
  onAsk: (text: string) => boolean;
  onBuild: (goal: string, fresh: boolean) => boolean;
};

/** Where the person writes: what they are doing (Ask or Build; Change joins them when it arrives), the text, and
 * Send. Enter sends, Shift+Enter starts a new line. */
export function Composer({ busy, onDashboard, onAsk, onBuild }: Props) {
  const { token } = theme.useToken();
  const [mode, setMode] = useState<ComposerMode>("ask");
  const [text, setText] = useState("");
  const [fresh, setFresh] = useState(false);
  const blocked = busy[mode];
  const send = () => {
    const said = text.trim();
    if (!said || blocked) return;
    if (mode === "ask" ? onAsk(said) : onBuild(said, fresh || onDashboard === null)) setText("");
  };
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: token.paddingXS }}>
      <Segmented<ComposerMode> size="small" value={mode} onChange={setMode}
        options={[{ label: "Ask", value: "ask" }, { label: "Build", value: "build" }]} />
      <Input.TextArea
        aria-label={mode === "ask" ? "Ask mnemiq" : "Describe the dashboard"}
        placeholder={mode === "ask" ? "Ask about your data" : "What should the dashboard show?"}
        autoSize={{ minRows: 2, maxRows: 6 }}
        value={text}
        onChange={(event) => setText(event.target.value)}
        onPressEnter={(event) => {
          if (event.shiftKey) return;
          event.preventDefault();
          send();
        }}
      />
      <div style={{ display: "flex", alignItems: "center", gap: token.paddingXS }}>
        {mode === "build" && onDashboard !== null ? (
          <Checkbox checked={fresh} onChange={(event) => setFresh(event.target.checked)}>New dashboard</Checkbox>
        ) : null}
        {mode === "build" && onDashboard !== null && !fresh ? (
          <span style={{ color: token.colorTextSecondary }}>Onto {onDashboard}</span>
        ) : null}
        <Button type="primary" disabled={blocked || !text.trim()} onClick={send} style={{ marginLeft: "auto" }}>
          {mode === "ask" ? "Send" : "Build"}
        </Button>
      </div>
    </div>
  );
}
