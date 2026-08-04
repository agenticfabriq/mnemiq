/** Answer prose -- the only proportional type in the interface. */

import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { TextMessagePartProps } from "@assistant-ui/react";

export function AnswerView({ text }: TextMessagePartProps) {
  if (!text) return null;
  return (
    <div className="prose-answer">
      <Markdown remarkPlugins={[remarkGfm]}>{text}</Markdown>
    </div>
  );
}
