# mnemiq with Open WebUI, Keycloak and oauth2-proxy

A local stack that runs mnemiq behind Open WebUI and its own workbench, with every request answered as its own
signed-in user. [docs/open-webui.md](../../docs/open-webui.md) explains each setting and what it costs to change it.

From the repository root:

```
export MNEMIQ_LLM_BASE_URL=...  MNEMIQ_LLM_API_KEY=...  MNEMIQ_LLM_MODEL=...   # your model, as in the quickstart
python examples/open-webui/make-secrets.py secrets
docker compose -f examples/open-webui/compose.yaml up -d --build     # add --profile option-b for option B
python examples/open-webui/make-secrets.py users
python examples/open-webui/make-secrets.py check
```

Run `users` straight after `up`. The first account to sign up to an Open WebUI instance becomes its admin, and `users`
creates that account. Until it runs, anything that reaches the port could take it, which here means this machine only.

`docker compose -f examples/open-webui/compose.yaml down` keeps every account, Keycloak's and Open WebUI's alike.
`down -v` removes them all together, with the demo data, and the steps above start over. Two things follow from
Keycloak keeping its database:

- **A changed `realm.json` takes effect only after `down -v`,** or the same change made in Keycloak's admin console:
  Keycloak imports the realm only when it does not exist yet.
- **Delete `.secrets/` only together with `down -v`.** Keycloak keeps the admin password it first started with, so
  `secrets` would write a new one, and `users` stops with `Keycloak refused the admin sign-in (401)`.

Then open Open WebUI at <http://localhost:13000> and the workbench at <http://localhost:14180>, and sign in with a
user from `examples/open-webui/.secrets/users.txt`. Everything listens on 127.0.0.1 only. This is a local demo: a
deployment serves Keycloak over https, keeps its secrets in its own store, and fronts mnemiq with its own network
policy.
