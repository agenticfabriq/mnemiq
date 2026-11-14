"""Set up the Superset 7.0.0rc2 that mnemiq's live contract suite answers through. Run with Superset's own interpreter,
with `SUPERSET_CONFIG_PATH` naming `superset_config.py` beside this file and its environment set (that file says what
it reads):

    python setup_superset.py keys          before Superset first starts: the test identity provider's key pair, its
                                           key set, its issuer, audience and tenant, and Superset's secret key, each
                                           written once and kept after that
    python setup_superset.py extension     before Superset starts: mnemiq's Superset extension, built from this
                                           checkout into `MNEMIQ_TEST_SUPERSET_EXTENSION_DIR`, which every process
                                           loads; run `superset init` once it is in place, so roles hold its route
    python setup_superset.py               once `superset db upgrade` and `superset init` have run: the source
                                           database, the users, roles and row rules, the users' passwords, and the
                                           service user's API key
    python setup_superset.py narrow west   narrows west's row rule to west's active policies
    python setup_superset.py widen west    puts it back to all of west's policies
    python setup_superset.py filter-and-report DASHBOARD USER
                                           as USER, a native filter on the dashboard and an active report that names it

Everything is made through Superset's commands and its security manager, never by writing a model, wherever Superset
has one: a dataset built from the model directly carries no catalog, and the access check then matches it to nothing.
Running it again changes nothing that is already in place.

What it makes, against the database `MNEMIQ_TEST_SUPERSET_PG_DSN` names (seeded from `seed.sql`):

- the database `mnemiq_live`, and a dataset for `policy` and for nothing else;
- roles `west_rls` and `east_rls`, each the subject of a Regular row rule on `policy` for its own region, and
  `policy_reader`, which may read the `policy` dataset;
- users `west` and `east` (Alpha, sql_lab and their rule's role), `analyst` (Alpha and sql_lab, whom no rule narrows),
  `gamma_only` (Gamma), `policy_only` (Gamma, sql_lab and `policy_reader`: a builder granted one dataset and not the
  database), and `svc` (Alpha and sql_lab), the service user enrichment reads as;
- an API key for `svc`, written to `svc.key` in the keys directory, readable by its owner alone, and never printed;
- a password for each of those users, random per install, kept in `passwords.json` in the keys directory, readable by
  its owner alone and never printed: the suite signs in to the web server with it to call the extension's token route.
  Every other call reaches Superset with a token or an API key.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import os
import secrets
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

DATABASE_NAME = "mnemiq_live"
SCHEMA = "public"
REGIONS = ("west", "east")
# The issuer and audience the test identity provider's tokens carry. Read back from `issuer.json` by Superset's config
# and by the suite, so this is the one place they are written.
ISSUER = "https://idp.mnemiq-live.invalid/realms/live"
AUDIENCE = "mnemiq-live"
KEY_ID = "live-1"
# The tenant mnemiq's deployment in the suite serves, which a token's `tenant` claim must name.
TENANT = "live"
# Everyone `provision` makes, who signs in with a password from `passwords.json`.
USERS = ("west", "east", "analyst", "gamma_only", "policy_only", "svc")


def _keys_dir() -> Path:
    return Path(os.environ["MNEMIQ_TEST_SUPERSET_KEYS_DIR"])


def _write_private(path: Path, content: bytes) -> None:
    """Write `content` to a new file only its owner may read, created that way rather than narrowed after."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)


def keys() -> None:
    """The test identity provider's RSA key pair, its key set (what mnemiq's door reads), its issuer and audience,
    and Superset's secret key. A file already there is kept: a running Superset holds the old key."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    folder = _keys_dir()
    folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    private_path = folder / "private.pem"
    if not private_path.exists():
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        _write_private(private_path, key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                       serialization.NoEncryption()))
    key = serialization.load_pem_private_key(private_path.read_bytes(), password=None)
    public = key.public_key()
    (folder / "public.pem").write_bytes(public.public_bytes(serialization.Encoding.PEM,
                                                            serialization.PublicFormat.SubjectPublicKeyInfo))
    numbers = public.public_numbers()
    jwk = {"kty": "RSA", "use": "sig", "alg": "RS256", "kid": KEY_ID, "n": _b64_uint(numbers.n),
           "e": _b64_uint(numbers.e)}
    (folder / "jwks.json").write_text(json.dumps({"keys": [jwk]}, indent=2))
    (folder / "issuer.json").write_text(json.dumps({"issuer": ISSUER, "audience": AUDIENCE, "kid": KEY_ID,
                                                    "tenant": TENANT}, indent=2))
    if not (folder / "secret_key").exists():
        _write_private(folder / "secret_key", secrets.token_urlsafe(48).encode())
    print(f"keys in place: {sorted(path.name for path in folder.iterdir())}")


def extension() -> None:
    """mnemiq's Superset extension, built from this checkout by the packer it ships with
    (`scripts/build_superset_extension.py`) into the directory `MNEMIQ_TEST_SUPERSET_EXTENSION_DIR` names, which
    `superset_config.py` loads. Running processes keep the extension they started with."""
    script = REPO / "scripts" / "build_superset_extension.py"
    spec = importlib.util.spec_from_file_location("build_superset_extension", script)
    packer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(packer)
    bundle = packer.build(Path(os.environ["MNEMIQ_TEST_SUPERSET_EXTENSION_DIR"]))
    print(f"extension built: dist/ and {bundle.name}")


def _passwords() -> dict[str, str]:
    """Each suite user's password, made once per install and kept in `passwords.json`, readable by its owner alone."""
    path = _keys_dir() / "passwords.json"
    found = json.loads(path.read_text()) if path.exists() else {}
    if all(username in found for username in USERS):
        return found
    found = {**{username: secrets.token_urlsafe(24) for username in USERS}, **found}
    staged = path.with_name("passwords.json.new")
    staged.unlink(missing_ok=True)
    _write_private(staged, json.dumps(found, indent=2).encode())
    staged.replace(path)
    return found


def _b64_uint(value: int) -> str:
    raw = value.to_bytes((value.bit_length() + 7) // 8, "big")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def provision() -> None:
    from superset import db, security_manager
    from superset.app import create_app

    app = create_app()
    with app.app_context(), app.test_request_context():
        from superset.utils.core import override_user

        admin = security_manager.find_user(username="admin")
        if admin is None:
            sys.exit("no admin user: run `superset fab create-admin` first")
        with override_user(admin):
            database = _database()
            policy = _policy_dataset(database)
            rule_roles = {region: _rule_role(region, policy) for region in REGIONS}
            reader = _reader_role(policy)
        alpha, gamma, sql_lab = (security_manager.find_role(name) for name in ("Alpha", "Gamma", "sql_lab"))
        passwords = _passwords()
        for region in REGIONS:
            _user(region, [alpha, sql_lab, rule_roles[region]], passwords[region])
        _user("analyst", [alpha, sql_lab], passwords["analyst"])
        _user("gamma_only", [gamma], passwords["gamma_only"])
        _user("policy_only", [gamma, sql_lab, reader], passwords["policy_only"])
        svc = _user("svc", [alpha, sql_lab], passwords["svc"])
        _service_key(svc)
        db.session.commit()
        _report(database, policy)


def _database():
    from superset import db
    from superset.commands.database.create import CreateDatabaseCommand
    from superset.models.core import Database

    found = db.session.query(Database).filter_by(database_name=DATABASE_NAME).one_or_none()
    if found is not None:
        return found
    return CreateDatabaseCommand({"database_name": DATABASE_NAME,
                                  "sqlalchemy_uri": os.environ["MNEMIQ_TEST_SUPERSET_PG_DSN"],
                                  "expose_in_sqllab": True}).run()


def _policy_dataset(database):
    from superset import db
    from superset.commands.dataset.create import CreateDatasetCommand
    from superset.connectors.sqla.models import SqlaTable

    found = (db.session.query(SqlaTable)
             .filter_by(table_name="policy", schema=SCHEMA, database_id=database.id).one_or_none())
    if found is not None:
        return found
    return CreateDatasetCommand({"database": database.id, "schema": SCHEMA, "table_name": "policy"}).run()


def _rule_role(region: str, policy):
    """The role whose members west's (or east's) Regular rule on `policy` filters, with that rule."""
    from superset import db, security_manager
    from superset.commands.security.create import CreateRLSRuleCommand
    from superset.connectors.sqla.models import RowLevelSecurityFilter
    from superset.subjects.utils import get_or_create_role_subject

    role = security_manager.find_role(f"{region}_rls") or security_manager.add_role(f"{region}_rls")
    name = _rule_name(region)
    if db.session.query(RowLevelSecurityFilter).filter_by(name=name).one_or_none() is None:
        subject = get_or_create_role_subject(role.id)
        db.session.commit()
        CreateRLSRuleCommand({"name": name, "filter_type": "Regular", "clause": _clause(region), "tables": [policy.id],
                              "subjects": [subject.id], "description": "mnemiq live contract suite"}).run()
    return role


def _reader_role(policy):
    """The role that may read the `policy` dataset and nothing else."""
    from superset import security_manager

    role = security_manager.find_role("policy_reader") or security_manager.add_role("policy_reader")
    access = security_manager.find_permission_view_menu("datasource_access", policy.perm)
    if access is None:
        sys.exit(f"no datasource_access permission for {policy.perm}")
    security_manager.add_permission_role(role, access)
    return role


def _user(username: str, roles: list, password: str):
    from superset import security_manager
    from werkzeug.security import check_password_hash

    if any(role is None for role in roles):
        sys.exit(f"a role for {username} is missing: run `superset init` first")
    user = security_manager.find_user(username=username)
    if user is None:
        user = security_manager.add_user(username, username, "Live", f"{username}@mnemiq-live.invalid", roles,
                                         password=password)
        if not user:
            sys.exit(f"Superset did not create the user {username}")
        return user
    if sorted(role.name for role in user.roles) != sorted(role.name for role in roles):
        user.roles = roles
        security_manager.update_user(user)
    if not user.password or not check_password_hash(user.password, password):
        security_manager.reset_password(user.id, password)
    return user


def _service_key(svc) -> None:
    """An API key for `svc` in `svc.key`, made once: a key file already there that Superset still accepts is kept."""
    from superset import security_manager

    path = _keys_dir() / "svc.key"
    if path.exists():
        if security_manager.validate_api_key(path.read_text().strip()) is not None:
            return
        path.unlink()
    made = security_manager.create_api_key(svc, "mnemiq-live-enrichment")
    if not made:
        sys.exit("Superset did not create the service user's API key")
    _write_private(path, made["key"].encode())


def _rule_name(region: str) -> str:
    return f"{region} only"


def _clause(region: str, *, narrowed: bool = False) -> str:
    return f"region = '{region}' AND status = 'active'" if narrowed else f"region = '{region}'"


def set_rule(region: str, *, narrowed: bool) -> None:
    """West's (or east's) row rule narrowed to its region's active policies, or widened back to all of them, through
    Superset's own command for updating a rule."""
    from superset import db, security_manager
    from superset.app import create_app

    if region not in REGIONS:
        sys.exit(f"no rule for {region!r}: one of {REGIONS}")
    app = create_app()
    with app.app_context(), app.test_request_context():
        from superset.commands.security.update import UpdateRLSRuleCommand
        from superset.connectors.sqla.models import RowLevelSecurityFilter
        from superset.utils.core import override_user

        rule = db.session.query(RowLevelSecurityFilter).filter_by(name=_rule_name(region)).one()
        with override_user(security_manager.find_user(username="admin")):
            UpdateRLSRuleCommand(rule.id, {"clause": _clause(region, narrowed=narrowed)}).run()
        db.session.expire_all()
        rule = db.session.query(RowLevelSecurityFilter).filter_by(name=_rule_name(region)).one()
        print(f"rule {rule.name!r}: {rule.clause}")


def filter_and_report(dashboard_id: int, username: str) -> None:
    """As `username`, a native filter on the dashboard (a region picker on `policy`) and an active report on the
    dashboard that names the filter, each made through Superset's own command; prints the filter's id and the report's
    as JSON. A layout save that dropped the filter would deactivate the report
    (`UpdateDashboardCommand.process_native_filter_diff`)."""
    from superset import db, security_manager
    from superset.app import create_app

    app = create_app()
    with app.app_context(), app.test_request_context():
        from superset.commands.dashboard.update import UpdateDashboardNativeFiltersCommand
        from superset.commands.report.create import CreateReportScheduleCommand
        from superset.connectors.sqla.models import SqlaTable
        from superset.utils.core import override_user

        policy = db.session.query(SqlaTable).filter_by(table_name="policy", schema=SCHEMA).one()
        filter_id = f"NATIVE_FILTER-{secrets.token_hex(6)}"
        native_filter = {
            "id": filter_id, "name": "Region", "type": "NATIVE_FILTER", "filterType": "filter_select",
            "targets": [{"datasetId": policy.id, "column": {"name": "region"}}],
            "controlValues": {"multiSelect": True, "enableEmptyFilter": False}, "defaultDataMask": {},
            "cascadeParentIds": [], "scope": {"rootPath": ["ROOT_ID"], "excluded": []}, "description": ""}
        with override_user(security_manager.find_user(username=username)):
            UpdateDashboardNativeFiltersCommand(dashboard_id, {"modified": [native_filter], "deleted": [],
                                                               "reordered": []}).run()
            report = CreateReportScheduleCommand({
                "type": "Report", "name": f"mnemiq live report {secrets.token_hex(4)}", "crontab": "0 9 * * *",
                "dashboard": dashboard_id, "creation_method": "alerts_reports", "active": True,
                "report_format": "PNG", "timezone": "UTC",
                "recipients": [{"type": "Email", "recipient_config_json": {"target": "live@mnemiq-live.invalid"}}],
                "extra": {"dashboard": {"nativeFilters": [{"nativeFilterId": filter_id, "filterType": "filter_select",
                                                           "columnName": "region", "filterValues": ["west"]}]}},
            }).run()
        print(json.dumps({"filter_id": filter_id, "report_id": report.id}))


def _report(database, policy) -> None:
    from superset import db, security_manager
    from superset.connectors.sqla.models import RowLevelSecurityFilter

    print(f"database {database.database_name} id {database.id}; dataset policy id {policy.id} catalog {policy.catalog}")
    for rule in db.session.query(RowLevelSecurityFilter).all():
        print(f"rule {rule.name!r}: {rule.filter_type} {rule.clause} on {[table.table_name for table in rule.tables]}")
    for username in (*REGIONS, "analyst", "gamma_only", "policy_only", "svc"):
        user = security_manager.find_user(username=username)
        print(f"user {username}: {sorted(role.name for role in user.roles)}")
    print("service key in place:", (_keys_dir() / "svc.key").exists())
    print("passwords in place:", (_keys_dir() / "passwords.json").exists())


def main(argv: list[str]) -> None:
    if argv == ["keys"]:
        keys()
    elif argv == ["extension"]:
        extension()
    elif len(argv) == 3 and argv[0] == "filter-and-report":
        filter_and_report(int(argv[1]), argv[2])
    elif not argv:
        provision()
    elif len(argv) == 2 and argv[0] in ("narrow", "widen"):
        set_rule(argv[1], narrowed=argv[0] == "narrow")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
