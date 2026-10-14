---
name: search-recovery
description: Recover when retrieval is empty, unrelated, ambiguous or repeatedly returns the same documents without resolving the current question.
---

Diagnose the failure before spending another retrieval call: overly long query, spelling or alias mismatch, vague terms, or a missing connecting entity.
Keep one discriminating concept while simplifying the rest of the query; remove speculative facts not yet supported by documents.
Try a documented alias, alternate spelling or a shorter phrase. Do not manufacture synonyms for a named entity.
Select another available method within retrieve if exact keyword matching or semantic matching is the likely bottleneck.
When a useful document is already known, find_in_document or open_document may be more effective than searching again.
Record the changed hypothesis in state_update.unresolved or state_update.subqueries. Avoid repeating an identical retrieval request.
After two unsuccessful reformulations, switch the clue being pursued or report the remaining evidence gap.
