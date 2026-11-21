// The part of Superset 7.0.0rc2's `@apache-superset/core` the panel calls, declared by hand: the 0.2.0 that rc2 ships
// is not on npm (npm's latest is 0.1.0). Each signature is rc2's own, from
// superset-frontend/packages/superset-core/src/{chat,navigation,authentication,common}/index.ts. At run time the
// module is the host's, given to the container through its share scope (`ExtensionsLoader.loadModule`).
declare module "@apache-superset/core" {
  import type { ComponentType } from "react";

  export interface Disposable {
    dispose(): void;
  }
  export interface Event<T> {
    (listener: (e: T) => unknown, thisArgs?: unknown): Disposable;
  }

  export namespace chat {
    interface Chat {
      id: string;
      name: string;
      description?: string;
    }
    type DisplayMode = "floating" | "panel";
    function registerChat(chat: Chat, trigger: ComponentType, panel: ComponentType): Disposable;
    function open(): void;
    function close(): void;
    function isOpen(): boolean;
    const onDidOpen: Event<void>;
    const onDidClose: Event<void>;
    function getDisplayMode(): DisplayMode;
    function setDisplayMode(displayMode: DisplayMode): void;
    const onDidChangeDisplayMode: Event<DisplayMode>;
  }

  export namespace navigation {
    type Page =
      | "dashboard"
      | "dashboard_list"
      | "explore"
      | "chart_list"
      | "sqllab"
      | "query_history"
      | "saved_queries"
      | "dataset"
      | "dataset_list"
      | "home";
    function getPage(): Page;
    const onDidChangePage: Event<Page>;
  }

  export namespace authentication {
    function getCSRFToken(): Promise<string | undefined>;
  }
}
