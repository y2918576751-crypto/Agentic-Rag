---
name: multi-clue-search
description: Identify a target entity when a question combines multiple factual clues, dates, places or relationships that require evidence from different documents.
---

Build a short constraint ledger in state_update.subqueries and state_update.unresolved.
Start with the most discriminating clue, such as a named institution, an unusual award or an exact date.
Use retrieve to establish candidate entities; keep candidates distinct rather than merging similarly named people.
Search the remaining constraints using the candidate's name. Use open_document or find_in_document to check the source wording.
Associate each supported constraint with an evidence ID. A passage proving one condition does not prove all conditions.
If two sources disagree, check dates and entity identity before accepting either.
Finish only when the candidate satisfies the question's required constraints; provide the minimal requested name, date or value in answer.
If evidence cannot resolve a constraint within the search budget, state the uncertainty rather than inventing a fact.
