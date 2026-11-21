import React, { useEffect, useState } from "react";
import { Button, Checkbox, Input, Segmented, theme } from "antd";

export type ComposerMode = "ask" | "build" | "change";

type Props = {
  busy: { ask: boolean; build: boolean; change: boolean };
  /** The dashboard on screen, when the person may read it: a Build goes onto it unless "New dashboard" is ticked.
   * `onBuild`'s second argument is that tick. */
  onDashboard: string | null;
  onAsk: (text: string) => boolean;
  onBuild: (goal: string, fresh: boolean) => boolean;
  /** Changes the dashboard on screen; offered only on a dashboard the person may read. */
  onChange: (request: string) => boolean;
};

const LABELS: Record<ComposerMode, { box: string; placeholder: string; send: string }> = {
  ask: { box: "Ask mnemiq", placeholder: "Ask about your data", send: "Send" },
  build: { box: "Describe the dashboard", placeholder: "What should the dashboard show?", send: "Build" },
  change: { box: "Describe the change", placeholder: "What should change on this dashboard?", send: "Change" },
};

/** Where the person writes: what they are doing (Ask, Build, and on a dashboard they may read, Change), the text, and
 * Send. Enter sends, Shift+Enter starts a new line. */
export function Composer({ busy, onDashboard, onAsk, onBuild, onChange }: Props) {
  const { token } = theme.useToken();
  const [mode, setMode] = useState<ComposerMode>("ask");
  const [text, setText] = useState("");
  const [fresh, setFresh] = useState(false);
  const blocked = busy[mode];
  // Change goes away off a dashboard the person may read; so does the mode.
  useEffect(() => {
    if (onDashboard === null && mode === "change") setMode("ask");
  }, [onDashboard, mode]);
  const send = () => {
    const said = text.trim();
    if (!said || blocked) return;
    // "New dashboard" is offered, and so counts, only on a dashboard; off one the goal is sent with neither.
    const sent = mode === "ask" ? onAsk(said)
      : mode === "build" ? onBuild(said, fresh && onDashboard !== null) : onChange(said);
    if (sent) setText("");
  };
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: token.paddingXS }}>
      <Segmented<ComposerMode> size="small" value={mode} onChange={setMode}
        options={[{ label: "Ask", value: "ask" }, { label: "Build", value: "build" },
          ...(onDashboard !== null ? [{ label: "Change", value: "change" as const }] : [])]} />
      <Input.TextArea
        aria-label={LABELS[mode].box}
        placeholder={LABELS[mode].placeholder}
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
          {LABELS[mode].send}
        </Button>
      </div>
    </div>
  );
}
