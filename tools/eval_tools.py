"""
Eval tool implementations for PinchBench Maxing agent.

Tools: get_state, request_approval
"""

from agents.base import log_print


# ── get_state ────────────────────────────────────────────────────────────────

def get_state(args: dict, cfg, state) -> dict:
    """Return the full current agent state dict."""
    try:
        return {
            "status": "success",
            "result": state.to_dict(),
            "cost_usd": 0.0,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ── request_approval ─────────────────────────────────────────────────────────

def request_approval(args: dict, cfg, state) -> dict:
    """Pause execution and request human approval via stdin."""
    try:
        reason = args.get("reason", "No reason provided")

        log_print("")
        log_print("=" * 62)
        log_print("  APPROVAL REQUIRED")
        log_print("=" * 62)
        log_print(f"  Reason: {reason}")
        log_print("")
        log_print("  Type 'yes' to approve, anything else to deny:")

        try:
            response = input("  > ").strip().lower()
        except EOFError:
            response = "no"

        approved = response in ("yes", "y")

        if approved:
            log_print("  APPROVED")
        else:
            log_print(f"  DENIED (response: {response!r})")

        log_print("=" * 62)

        return {
            "status": "success",
            "result": {
                "approved": approved,
            },
            "cost_usd": 0.0,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
