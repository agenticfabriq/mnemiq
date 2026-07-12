from scripts.acme_ddl_translate import translate

TSQL = """
CREATE TABLE Claim
(
    Claim_Identifier     int  NOT NULL ,
    Claims_Made_Date     datetime  NULL ,
    Claim_Status_Code    varchar(5)  NULL ,
     PRIMARY KEY (Claim_Identifier ASC)
)

CREATE TABLE Claim_Amount
(
    Claim_Amount_Identifier bigint  NOT NULL ,
    Claim_Amount         decimal(15,2)  NULL ,
     PRIMARY KEY (Claim_Amount_Identifier ASC),
     FOREIGN KEY (Claim_Amount_Identifier) REFERENCES Claim(Claim_Identifier)
)
"""


def test_translate_splits_two_statements():
    out = translate(TSQL)
    assert len(out) == 2


def test_translate_maps_datetime_and_strips_asc():
    out = "\n".join(translate(TSQL))
    assert "datetime" not in out.lower()
    assert "timestamp" in out.lower()
    assert "ASC" not in out
    assert out.count(";") == 2  # each statement terminated


def test_translate_drops_keys_without_dangling_comma():
    out = "\n".join(translate(TSQL))
    assert "foreign key" not in out.lower()
    assert "primary key" not in out.lower()
    import re

    assert not re.search(r",\s*\)", out)  # no dangling comma before a closing paren
