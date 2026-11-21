import React, { useState } from "react";
import { Button, Input, Segmented, theme } from "antd";

export type ComposerMode = "ask";

/** Where the person writes: what they are doing (Ask; Build and Change join it as they arrive), the text, and Send.
 * Enter sends, Shift+Enter starts a new line. */
export function Composer({ busy, onAsk }: { busy: boolean; onAsk: (text: string) => boolean }) {
  const { token } = theme.useToken();
  const [mode, setMode] = useState<ComposerMode>("ask");
  const [text, setText] = useState("");
  const send = () => {
    const question = text.trim();
    if (!question || busy) return;
    if (onAsk(question)) setText("");
  };
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: token.paddingXS }}>
      <Segmented<ComposerMode> size="small" value={mode} onChange={setMode} options={[{ label: "Ask", value: "ask" }]} />
      <Input.TextArea
        aria-label="Ask mnemiq"
        placeholder="Ask about your data"
        autoSize={{ minRows: 2, maxRows: 6 }}
        value={text}
        onChange={(event) => setText(event.target.value)}
        onPressEnter={(event) => {
          if (event.shiftKey) return;
          event.preventDefault();
          send();
        }}
      />
      <Button type="primary" disabled={busy || !text.trim()} onClick={send} style={{ alignSelf: "flex-end" }}>
        Send
      </Button>
    </div>
  );
}
