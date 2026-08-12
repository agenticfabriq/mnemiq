/**
 * The transcript. assistant-ui owns the viewport, auto-scroll and composer
 * mechanics; every pixel of the rendering is ours.
 *
 * Uses the children render form of ThreadPrimitive.Messages and AuiIf -- the
 * `components={{...}}` and `*Primitive.If` forms are deprecated at 0.15.4.
 */

import {
  AuiIf,
  ComposerPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
  useAuiState,
} from "@assistant-ui/react";

import { TURN_PART } from "../lib/runtime";
import type { Turn } from "../lib/store";
import { AnswerView } from "./AnswerView";
import { EmptyState } from "./EmptyState";
import { TurnArtifacts } from "./TurnArtifacts";
import { VerdictStripe } from "./VerdictStripe";

const PARTS = { Text: AnswerView, data: { by_name: { [TURN_PART]: TurnArtifacts } } };

function UserTurn() {
  return (
    <MessagePrimitive.Root className="flex justify-end">
      <div className="max-w-[46rem] border border-rule bg-surface px-3 py-2 text-[13px]">
        <MessagePrimitive.Parts />
      </div>
    </MessagePrimitive.Root>
  );
}

function AssistantTurn() {
  // The stripe leads the turn, so it is rendered here rather than as a part -- parts
  // render in content order and the payload part necessarily arrives after the prose.
  const parts = useAuiState((s) => s.message.parts);
  let turn: Turn | undefined;
  for (const part of parts) {
    if (part.type === "data" && part.name === TURN_PART) turn = part.data as Turn;
  }

  return (
    <MessagePrimitive.Root className="flex flex-col gap-3.5 border-t border-rule pt-3">
      {turn && <VerdictStripe turn={turn} />}
      <MessagePrimitive.Parts components={PARTS} />
    </MessagePrimitive.Root>
  );
}

export function Thread({
  starters,
  onPick,
}: {
  starters: string[];
  onPick: (question: string) => void;
}) {
  return (
    <ThreadPrimitive.Root className="flex h-full flex-col">
      <ThreadPrimitive.Viewport className="flex-1 overflow-y-auto">
        <div className="mx-auto flex w-full max-w-3xl flex-col gap-5 px-5 py-6">
          <AuiIf condition={(s) => s.thread.isEmpty}>
            <EmptyState starters={starters} onPick={onPick} />
          </AuiIf>
          <ThreadPrimitive.Messages>
            {({ message }) => (message.role === "user" ? <UserTurn /> : <AssistantTurn />)}
          </ThreadPrimitive.Messages>
        </div>
      </ThreadPrimitive.Viewport>

      <ThreadPrimitive.ViewportFooter className="border-t border-rule bg-surface">
        <ComposerPrimitive.Root className="mx-auto flex w-full max-w-3xl items-end gap-2 px-5 py-3">
          <ComposerPrimitive.Input
            rows={1}
            placeholder="Ask about your data"
            className="max-h-40 flex-1 resize-none bg-transparent py-1.5 text-[13px] placeholder:text-graphite focus:outline-none"
          />
          <ComposerPrimitive.Send className="label border border-rule px-2.5 py-1.5 hover:border-brass hover:text-brass disabled:opacity-40">
            Ask
          </ComposerPrimitive.Send>
        </ComposerPrimitive.Root>
      </ThreadPrimitive.ViewportFooter>
    </ThreadPrimitive.Root>
  );
}
