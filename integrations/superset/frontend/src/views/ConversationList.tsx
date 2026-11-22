import React, { useState } from "react";
import { Alert, Button, theme } from "antd";
import type { PanelApi } from "../api";
import { listConversations, type ConversationSummary } from "../conversation";
import { newConversation, openConversation } from "../talk";

/** The person's conversations, newest first, a page at a time, each opened onto the screen; and New, which clears the
 * screen for a conversation mnemiq makes with its first message. */
export function ConversationList({ api, current, busy }: { api: PanelApi; current: string | null; busy: boolean }) {
  const { token } = theme.useToken();
  const [open, setOpen] = useState(false);
  const [listed, setListed] = useState<ConversationSummary[]>([]);
  const [next, setNext] = useState<string | null>(null);
  const [refused, setRefused] = useState<string | null>(null);
  const page = async (before: string | null) => {
    const found = await listConversations(api, before);
    if (!found.ok) {
      setRefused(found.refusal.sentence);
      return;
    }
    setRefused(null);
    setListed((now) => (before === null ? found.body.conversations : [...now, ...found.body.conversations]));
    setNext(found.body.next);
  };
  const toggle = () => {
    if (!open) void page(null);
    setOpen(!open);
  };
  return (
    <section aria-label="Conversations" style={{ display: "flex", flexDirection: "column", gap: token.paddingXS }}>
      <div style={{ display: "flex", gap: token.paddingXS }}>
        <Button size="small" aria-expanded={open} onClick={toggle}>Conversations</Button>
        <Button size="small" disabled={busy} onClick={() => { newConversation(); setOpen(false); }}>New</Button>
      </div>
      {open ? (
        <div role="list" style={{ display: "flex", flexDirection: "column", gap: 2 }}>
          {refused ? <Alert type="warning" showIcon message={refused} /> : null}
          {listed.length === 0 && !refused ? <span style={{ color: token.colorTextSecondary }}>No conversations yet.</span> : null}
          {listed.map((item) => (
            <Button key={item.id} role="listitem" size="small" type={item.id === current ? "primary" : "text"}
              disabled={busy} style={{ justifyContent: "space-between", display: "flex", textAlign: "left" }}
              onClick={() => { void openConversation(api, item.id); setOpen(false); }}>
              <span style={{ overflow: "hidden", textOverflow: "ellipsis" }}>{item.title || "Untitled"}</span>
              <span style={{ color: token.colorTextTertiary, marginLeft: 8 }}>{item.updated_at.slice(0, 10)}</span>
            </Button>
          ))}
          {next ? <Button size="small" type="link" onClick={() => void page(next)}>More</Button> : null}
        </div>
      ) : null}
    </section>
  );
}
