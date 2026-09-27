"""Action policy — a pure function of the INTERNAL risk read, never of output tokens.

Given the loop's outcome ``(r_star, safe)`` it selects one of four actions. This
is the whole point of ACSL: the gate is an internal signal at a mid-network
layer, so surface-form jailbreaks that change the wording but not the internal
risk representation do not move the decision.
"""

from __future__ import annotations

# Risk bands in head log-odds units (see head.risk_logit; 0.0 == p(harm)=0.5).
ANSWER_BELOW: float = -1.0   # comfortably safe -> answer
REFUSE_ABOVE: float = 2.0    # clearly harmful -> hard refuse/route


def action_policy(r_star, safe: bool) -> str:
    """Return one of ``'answer' | 'steer' | 'refuse_or_route' | 'flag'``.

    safe (loop converged at/below threshold):
        * r_star <= ANSWER_BELOW  -> 'answer'   (comfortably safe)
        * otherwise               -> 'steer'    (resolved; proceed on steered state)
    not safe (fail closed):
        * r_star >= REFUSE_ABOVE  -> 'refuse_or_route'  (clearly harmful)
        * otherwise               -> 'flag'             (unresolved; route for review)
    """
    r = float(r_star)
    if safe:
        return "answer" if r <= ANSWER_BELOW else "steer"
    # fail closed
    return "refuse_or_route" if r >= REFUSE_ABOVE else "flag"
