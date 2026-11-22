import React, { useState } from "react";
import { Alert, Button, theme } from "antd";
import type { PanelApi } from "../api";
import { deleteConversation, listConversations, type ConversationSummary } from "../conversation";
import { newConversation, openConversation } from "../talk";

/** The person's conversations, newest first, a page at a time: each opened onto the screen, or deleted after a second
 * click says so; and New, which clears the screen for a conversation mnemiq makes with its first message. Nothing
 * here moves while a message, confirm or decline is in flight. */
export function ConversationList({ api, current, busy }: { api: PanelApi; current: string | null; busy: boolean }) {
  const { token } = theme.useToken();
  const [open, setOpen] = useState(false);
  const [listed, setListed] = useState<ConversationSummary[]>([]);
  const [next, setNext] = useState<string | null>(null);
  const [refused, setRefused] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);
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
    setDeleting(null);
    setOpen(!open);
  };
  const remove = async (item: ConversationSummary) => {
    const done = await deleteConversation(api, item.id);
    setDeleting(null);
    if (!done.ok) {
      setRefused(done.refusal.sentence);
      return;
    }
    setListed((now) => now.filter((each) => each.id !== item.id));
    if (item.id === current) newConversation();
  };
  const name = (item: ConversationSummary) => item.title || "Untitled";
  return (
    <section aria-label="Conversations" style={{ display: "flex", flexDirection: "column", gap: token.paddingXS }}>
      <div style={{ display: "flex", gap: token.paddingXS }}>
        <Button size="small" aria-expanded={open} disabled={busy} onClick={toggle}>Conversations</Button>
        <Button size="small" disabled={busy} onClick={() => { newConversation(); setOpen(false); }}>New</Button>
      </div>
      {open ? (
        <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
          {refused ? <Alert type="warning" showIcon message={refused} /> : null}
          {listed.length === 0 && !refused ? <span style={{ color: token.colorTextSecondary }}>No conversations yet.</span> : null}
          <ul style={{ listStyle: "none", margin: 0, padding: 0, display: "flex", flexDirection: "column", gap: 2 }}>
            {listed.map((item) => (
              <li key={item.id} style={{ display: "flex", alignItems: "center", gap: 4 }}>
                <Button size="small" type={item.id === current ? "primary" : "text"} disabled={busy}
                  aria-current={item.id === current ? "true" : undefined}
                  style={{ flex: 1, justifyContent: "space-between", display: "flex", textAlign: "left", minWidth: 0 }}
                  onClick={() => { void openConversation(api, item.id); setOpen(false); }}>
                  <span style={{ overflow: "hidden", textOverflow: "ellipsis" }}>{name(item)}</span>
                  <span style={{ color: token.colorTextTertiary, marginLeft: 8 }}>{item.updated_at.slice(0, 10)}</span>
                </Button>
                {deleting === item.id ? (
                  <>
                    <Button size="small" danger disabled={busy} onClick={() => void remove(item)}>Delete it</Button>
                    <Button size="small" onClick={() => setDeleting(null)}>Keep it</Button>
                  </>
                ) : (
                  <Button size="small" type="text" aria-label={`Delete ${name(item)}`} disabled={busy}
                    onClick={() => setDeleting(item.id)}>Delete</Button>
                )}
              </li>
            ))}
          </ul>
          {next ? <Button size="small" type="link" onClick={() => void page(next)}>More</Button> : null}
        </div>
      ) : null}
    </section>
  );
}
