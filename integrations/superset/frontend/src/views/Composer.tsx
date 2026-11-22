import React, { useState } from "react";
import { Button, Input, theme } from "antd";

type Props = {
  busy: boolean;
  /** Where the person is: on a dashboard with no charts yet, the box invites a description of what it should show. */
  placeholder: string;
  onSend: (text: string) => boolean;
};

/** Where the person writes: one box for a question, a dashboard to build, or a change to the one on screen; mnemiq
 * works out which, and asks when it cannot tell. Enter sends, Shift+Enter starts a new line. */
export function Composer({ busy, placeholder, onSend }: Props) {
  const { token } = theme.useToken();
  const [text, setText] = useState("");
  const send = () => {
    const said = text.trim();
    if (!said || busy) return;
    if (onSend(said)) setText("");
  };
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: token.paddingXS }}>
      <Input.TextArea
        aria-label="Message mnemiq"
        placeholder={placeholder}
        autoSize={{ minRows: 2, maxRows: 6 }}
        value={text}
        onChange={(event) => setText(event.target.value)}
        onPressEnter={(event) => {
          if (event.shiftKey || event.nativeEvent.isComposing) return;
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
