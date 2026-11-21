// The panel's entry, which Superset's loader runs once when it loads the container (`ExtensionsLoader.loadModule`:
// `container.get("./index")`, then the factory): it registers mnemiq as the page's chat. rc2 mounts the trigger only
// for a signed-in user (`ExtensionsStartup`, `AppContent`).
import { chat } from "@apache-superset/core";
import { Panel } from "./Panel";
import { Trigger } from "./Trigger";

chat.registerChat(
  { id: "agenticfabriq.mnemiq", name: "mnemiq", description: "Ask about your data and build dashboards" },
  Trigger,
  Panel,
);
