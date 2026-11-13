-- The source database behind the live contract suite's Superset: synthetic, every row generated here.
--
-- policy           300 policies, 100 in each of three regions; in each region 60 active, 20 expired, 20 cancelled.
--                  The one table Superset has a dataset for, and the one its row rules filter.
-- party            two parties per policy, with no dataset: what a builder granted only datasets cannot read. Its
--                  `role` holds two values, so profiling records a partition for each.
-- policy_by_region a view over policy.
-- other.median     a function in a schema off the search path Superset sets, so an unqualified call cannot reach it.
-- "Policy Lines"   a table whose name needs quoting.
-- coverage         a partitioned table that a foreign key references and that declares one of its own, so Postgres
--                  keeps a constraint row for each of its partitions beside the declared one.

CREATE TABLE policy (
    id integer PRIMARY KEY,
    region text NOT NULL,
    status text NOT NULL,
    premium numeric(10, 2) NOT NULL,
    issued date NOT NULL,
    doc json
);

INSERT INTO policy (id, region, status, premium, issued, doc)
SELECT g,
       (ARRAY['west', 'east', 'central'])[1 + (g - 1) % 3],
       (ARRAY['active', 'active', 'expired', 'cancelled', 'active'])[1 + ((g - 1) / 3) % 5],
       round(100 + (g * 37) % 900 + ((g * 13) % 100) / 100.0, 2),
       DATE '2024-01-01' + (g * 7) % 365,
       CASE WHEN g % 25 = 0 THEN NULL
            ELSE json_build_object('channel', (ARRAY['web', 'agent', 'phone'])[1 + (g * 5) % 3], 'riders', g % 4)
       END
FROM generate_series(1, 300) AS g;

CREATE TABLE party (
    id integer PRIMARY KEY,
    policy_id integer NOT NULL REFERENCES policy (id),
    role text NOT NULL
);

INSERT INTO party (id, policy_id, role)
SELECT g, 1 + (g - 1) / 2, CASE WHEN g % 2 = 1 THEN 'insured' ELSE 'nominee' END
FROM generate_series(1, 600) AS g;

CREATE VIEW policy_by_region AS
SELECT region, count(*) AS policies
FROM policy
GROUP BY region;

CREATE SCHEMA other;

CREATE FUNCTION other.median(value integer) RETURNS integer
LANGUAGE sql IMMUTABLE
AS $$ SELECT value $$;

CREATE TABLE "Policy Lines" (
    line_id integer PRIMARY KEY,
    policy_id integer NOT NULL REFERENCES policy (id),
    amount numeric(10, 2) NOT NULL
);

INSERT INTO "Policy Lines" (line_id, policy_id, amount)
SELECT g, 1 + (g - 1) % 300, round(10 + (g * 17) % 190 + ((g * 7) % 100) / 100.0, 2)
FROM generate_series(1, 450) AS g;

CREATE TABLE coverage (
    id integer PRIMARY KEY,
    policy_id integer NOT NULL REFERENCES policy (id),
    kind text NOT NULL
) PARTITION BY RANGE (id);

CREATE TABLE coverage_low PARTITION OF coverage FOR VALUES FROM (1) TO (301);
CREATE TABLE coverage_high PARTITION OF coverage FOR VALUES FROM (301) TO (601);

INSERT INTO coverage (id, policy_id, kind)
SELECT g, 1 + (g - 1) % 300, (ARRAY['base', 'rider', 'waiver'])[1 + g % 3]
FROM generate_series(1, 600) AS g;

CREATE TABLE coverage_claim (
    id integer PRIMARY KEY,
    coverage_id integer NOT NULL REFERENCES coverage (id),
    amount numeric(10, 2) NOT NULL
);

INSERT INTO coverage_claim (id, coverage_id, amount)
SELECT g, 1 + (g * 7) % 600, round(50 + (g * 29) % 950 + ((g * 3) % 100) / 100.0, 2)
FROM generate_series(1, 120) AS g;

ANALYZE;
