"""
Eval tool implementations for PinchBench Maxing agent.

Tools: get_state, request_approval, write_note
"""

from datetime import datetime
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


# ── write_note ────────────────────────────────────────────────────────────────

def write_note(args: dict, cfg, state) -> dict:
    """Write a note to the scratchpad — persists across turns."""
    try:
        note = args.get("note", "").strip()
        if not note:
            return {"status": "error", "error": "note cannot be empty"}

        entry = {
            "timestamp": datetime.utcnow().strftime("%H:%M:%S"),
            "note": note,
        }
        state.scratchpad.append(entry)

        # Keep last 20 notes to avoid unbounded growth
        if len(state.scratchpad) > 20:
            state.scratchpad = state.scratchpad[-20:]

        log_print(f"[NOTE] {note}")
        return {
            "status": "success",
            "result": {"saved": True, "total_notes": len(state.scratchpad)},
            "cost_usd": 0.0,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
