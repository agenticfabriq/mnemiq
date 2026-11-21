// The workbench's own modules the panel shares, and the only file that reaches into the workbench: how an answer's
// stream is read into turns, how a run is ended, what history is sent back, and what a turn's outcome is called. They
// import nothing but each other (src/shared.test.ts), so they run on the host's React 18 as on the workbench's 19.
export {
  closeRun,
  CONNECTION_ENDED,
  CONNECTION_ENDED_CODE,
  isRunning,
  reduce,
  settle,
  userTurn,
  type Turn,
  type TurnStatus,
} from "../../../../workbench/src/lib/store";
export { SseDecoder } from "../../../../workbench/src/lib/sse";
export { historyFrom } from "../../../../workbench/src/lib/history";
export * from "../../../../workbench/src/lib/verdict";
export type * from "../../../../workbench/src/lib/types";
