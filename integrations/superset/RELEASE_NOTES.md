The mnemiq extension for Apache Superset, `agenticfabriq.mnemiq`: mnemiq's panel beside every dashboard, and the
routes and MCP tools that let mnemiq answer, build and change dashboards as the person asking.

**It needs a mnemiq server in Superset mode.** The extension is its panel and its doors into Superset; the answers come
from mnemiq. See [mnemiq with Superset](https://github.com/agenticfabriq/mnemiq/blob/main/docs/superset.md) for the
server, the service key and the identity provider's token.

**Superset 7.0.0rc2.** Built and tested against Superset 7.0.0rc2, whose chat contribution the panel uses.

**Install:**

1. Download the `.supx` below. Check it against its `.sha256`: `sha256sum -c mnemiq-<version>.supx.sha256`.
2. Put it in a folder of its own, and in `superset_config.py` set `EXTENSIONS_PATH` to that folder and turn on
   `FEATURE_FLAGS = {"ENABLE_EXTENSIONS": True}`.
3. Configure the extension as
   [its README](https://github.com/agenticfabriq/mnemiq/blob/main/integrations/superset/README.md#configuration) says,
   then restart Superset's web server and its MCP service: extensions load at startup. After the first load, run
   `superset init`, so roles receive the extension's route permission.
