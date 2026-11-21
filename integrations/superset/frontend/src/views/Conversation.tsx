import React from "react";
import { Spin, theme } from "antd";
import type { Turn } from "../shared";
import { AnswerCard } from "./AnswerCard";

/** The conversation for the life of the page: each question, and each answer as it streams in. */
export function Conversation({ turns }: { turns: Turn[] }) {
  const { token } = theme.useToken();
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: token.padding }}>
      {turns.map((turn) =>
        turn.role === "user" ? (
          <p key={turn.id} style={{ margin: 0, alignSelf: "flex-end", maxWidth: "85%", padding: token.paddingXS,
            borderRadius: token.borderRadius, background: token.colorFillTertiary }}>{turn.text}</p>
        ) : turn.status === "running" && !turn.text ? (
          <div key={turn.id} role="status" aria-label="Working" style={{ color: token.colorTextSecondary }}>
            <Spin size="small" /> {turn.steps?.at(-1)?.name ?? "Working"}
          </div>
        ) : (
          <AnswerCard key={turn.id} turn={turn} />
        ),
      )}
    </div>
  );
}
