/**
 * Real payloads captured from a live ACME store via POST /v1/ask, not hand-written.
 * If the wire contract drifts, these stop type-checking against AnswerPayload.
 */

import type { AnswerPayload } from "../lib/types";

/** A scalar answer: one row, one column. */
export const SCALAR_ANSWER: AnswerPayload = {
  "answer": "There are 2 claims.",
  "deferred": false,
  "failed": false,
  "reason_code": null,
  "mode": "thinking",
  "cached": false,
  "agreement": null,
  "judge_engaged": null,
  "judge_fell_back": null,
  "judge_override": null,
  "candidates_executed": null,
  "attempts": 1,
  "corrected": false,
  "sql": "SELECT COUNT(claim_identifier) AS claim_count FROM claim LIMIT 1000",
  "tables_used": [
    "claim"
  ],
  "enrichment_version": "e91e3340c22b",
  "timing": {
    "execute_ms": 13.723208976443857,
    "total_ms": 6734.467541973572
  },
  "preview": {
    "columns": [
      "claim_count"
    ],
    "rows": [
      [
        2
      ]
    ],
    "row_count": 1,
    "truncated": false
  }
};

/** Eight rows, two columns. Postgres numerics arrive as strings. */
export const TABLE_ANSWER: AnswerPayload = {
  "answer": "Fire claims listed are: identifier 1 with amounts 1000.00, 1100.00, 1200.00, and 1300.00; identifier 2 with amounts 2100.00, 2200.00, 2300.00, and 2400.00.",
  "deferred": false,
  "failed": false,
  "reason_code": null,
  "mode": "thinking",
  "cached": false,
  "agreement": null,
  "judge_engaged": null,
  "judge_fell_back": null,
  "judge_override": null,
  "candidates_executed": null,
  "attempts": 1,
  "corrected": false,
  "sql": "SELECT c.claim_identifier, ca.claim_amount FROM claim AS c LEFT JOIN claim_amount AS ca ON ca.claim_identifier = c.claim_identifier WHERE c.catastrophe_identifier = 4 AND NOT c.claim_identifier IS NULL ORDER BY c.claim_identifier, ca.claim_amount_identifier LIMIT 1000",
  "tables_used": [
    "claim",
    "claim_amount"
  ],
  "enrichment_version": "e91e3340c22b",
  "timing": {
    "execute_ms": 19.902583968359977,
    "total_ms": 17688.249040977098
  },
  "preview": {
    "columns": [
      "claim_identifier",
      "claim_amount"
    ],
    "rows": [
      [
        1,
        "1000.00"
      ],
      [
        1,
        "1100.00"
      ],
      [
        1,
        "1200.00"
      ],
      [
        1,
        "1300.00"
      ],
      [
        2,
        "2100.00"
      ],
      [
        2,
        "2200.00"
      ],
      [
        2,
        "2300.00"
      ],
      [
        2,
        "2400.00"
      ]
    ],
    "row_count": 8,
    "truncated": false
  }
};

/** A real refusal: the only claim status column is entirely NULL. */
export const DEFERRAL: AnswerPayload = {
  "answer": "Cannot answer because the only claim status column, claim.claim_status_code, is entirely NULL and must not be used for grouping or aggregation.",
  "deferred": true,
  "failed": false,
  "reason_code": "unanswerable",
  "mode": "thinking",
  "cached": false,
  "agreement": null,
  "judge_engaged": null,
  "judge_fell_back": null,
  "judge_override": null,
  "candidates_executed": null,
  "attempts": null,
  "corrected": null,
  "sql": null,
  "tables_used": null,
  "enrichment_version": null,
  "timing": null,
  "preview": null
};
