"""
CuratorAgent — multi-stage quality gate between data generation and training.

Pipeline:
  Gate 1: Score all examples via LLM judge
  Gate 2: Repair borderline examples (score 2-3) — fix instead of discard
  Gate 3: Filter below min_judge_score
  Gate 4: Deduplicate semantically similar examples
  Gate 5: Verify train.jsonl is non-empty
  Gate 6: Snapshot dataset to HuggingFace (versioned, public)

Only high-quality, diverse examples make it through to fine-tuning.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

from .base import Agent, AgentState

_PROJECT_ROOT = Path(__file__).parent.parent


class CuratorAgent(Agent):
    name = "curator"

    def run(self, state: AgentState, cfg) -> AgentState:
        min_score = cfg.data.min_judge_score
        scores_file = cfg.data_dir / "scores.json"

        # ── Gate 1: Score all examples ─────────────────────────────────────
        judge = str(_PROJECT_ROOT / "llm_judge.py")
        self.log("Running LLM judge on all examples (this may take a while)...")
        self.run_cmd([sys.executable, judge, "run"])

        if not scores_file.exists():
            raise RuntimeError(
                "llm_judge.py run completed but scores.json was not created. "
                "Check ANTHROPIC_API_KEY and llm_judge.py output."
            )
        self.log(f"  Scores written to {scores_file}")

        # ── Gate 2: Repair borderline examples ─────────────────────────────
        repair_script = _PROJECT_ROOT / "example_repair.py"
        if repair_script.exists():
            self.log("Repairing borderline examples (score 2-3)...")
            rc = self.run_cmd(
                [sys.executable, str(repair_script), "run",
                 "--min-score", "2", "--max-score", str(min_score)],
                check=False,
            )
            if rc != 0:
                self.log(f"  WARNING: repair exited {rc} (non-fatal, continuing)")
            else:
                report_file = cfg.data_dir / "repair_report.json"
                if report_file.exists():
                    report = json.loads(report_file.read_text())
                    self.log(f"  Repair: {report.get('improved', 0)} improved, "
                             f"{report.get('failed', 0)} failed, "
                             f"{report.get('success_rate', 0)}% success rate")

        # ── Gate 3: Filter by min score ────────────────────────────────────
        self.log(f"Filtering examples with min score {min_score}/5...")
        self.run_cmd([sys.executable, judge, "filter", "--min", str(min_score)])

        # ── Gate 4: Deduplicate ────────────────────────────────────────────
        dedup_script = _PROJECT_ROOT / "dedup.py"
        if dedup_script.exists():
            self.log("Deduplicating semantically similar examples...")
            rc = self.run_cmd(
                [sys.executable, str(dedup_script), "run"],
                check=False,
            )
            if rc != 0:
                self.log(f"  WARNING: dedup exited {rc} (non-fatal, continuing)")
            else:
                report_file = cfg.data_dir / "dedup_report.json"
                if report_file.exists():
                    report = json.loads(report_file.read_text())
                    removed = report.get("removed", 0)
                    pct = report.get("percent_removed", 0)
                    self.log(f"  Dedup: removed {removed} examples ({pct}%)")

                    if pct > 30:
                        self.log(f"  WARNING: {pct}% removed — generation may lack diversity")

        # ── Gate 5: Verify output is non-empty ─────────────────────────────
        train_file = cfg.train_file
        if not train_file.exists() or train_file.stat().st_size == 0:
            raise RuntimeError(
                f"train.jsonl is empty after curation (score >= {min_score} + dedup). "
                "Lower min_judge_score, regenerate data, or check dedup threshold."
            )
        n = sum(1 for line in train_file.read_text().splitlines() if line.strip())
        self.log(f"  {n} examples passed full curation pipeline (score >= {min_score}/5)")

        # ── Gate 6: Snapshot to HuggingFace ────────────────────────────────
        self._push_to_huggingface(cfg, state, n)

        return state

    def _push_to_huggingface(self, cfg, state: AgentState, n_examples: int) -> None:
        """Push curated dataset to HuggingFace for versioning and public access."""
        try:
            repo_id = cfg.huggingface.dataset_repo
        except AttributeError:
            self.log("  No huggingface.dataset_repo in config — skipping HF push")
            return

        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            self.log("  HF_TOKEN not set — skipping HuggingFace push")
            self.log("  To enable: export HF_TOKEN=hf_... or huggingface-cli login")
            return

        try:
            from huggingface_hub import HfApi

            api = HfApi(token=hf_token)
            api.create_repo(repo_id, repo_type="dataset", exist_ok=True)

            version = f"v{state.model_version}"
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            commit_msg = (f"Iteration {state.iteration} ({version}): "
                          f"{n_examples} examples after curation")

            files_to_upload = []
            for fname in ["train.jsonl", "val.jsonl", "scores.json"]:
                path = cfg.data_dir / fname
                if path.exists():
                    files_to_upload.append((str(path), fname))

            # Also upload loop state and dataset card
            state_file = cfg.data_dir / "loop_state.json"
            if state_file.exists():
                files_to_upload.append((str(state_file), "loop_state.json"))

            dataset_card = Path(__file__).parent.parent / "dataset_card.md"
            if dataset_card.exists():
                files_to_upload.append((str(dataset_card), "README.md"))

            for local, remote in files_to_upload:
                api.upload_file(
                    path_or_fileobj=local,
                    path_in_repo=remote,
                    repo_id=repo_id,
                    repo_type="dataset",
                    commit_message=commit_msg,
                )

            self.log(f"  Pushed {len(files_to_upload)} files to "
                     f"huggingface.co/datasets/{repo_id} ({commit_msg})")

        except ImportError:
            self.log("  WARNING: huggingface_hub not installed — skipping HF push")
        except Exception as e:
            self.log(f"  WARNING: HuggingFace push failed: {e} (non-fatal)")


# ── Standalone entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    from utils.config import load_config
    cfg   = load_config()
    state = AgentState()
    agent = CuratorAgent()
    agent.run(state, cfg)
