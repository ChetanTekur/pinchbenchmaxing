"""
CuratorAgent — multi-stage quality gate between data generation and training.

Pipeline:
  1. Score all examples via LLM judge
  2. Repair borderline examples (score 2-3)
  3. Filter below min_judge_score
  4. Deduplicate semantically similar examples
  5. Verify train.jsonl is non-empty
  6. Snapshot to HuggingFace
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

        # ── 1. Score ───────────────────────────────────────────────────────
        self.log("Scoring all examples...")
        judge = str(_PROJECT_ROOT / "datagen" / "llm_judge.py")
        self.run_cmd([sys.executable, judge, "run"])

        if not scores_file.exists():
            raise RuntimeError(
                "llm_judge.py completed but scores.json not created. "
                "Check ANTHROPIC_API_KEY."
            )
        self.log(f"  Scores saved to {scores_file}")

        # ── 2. Repair ─────────────────────────────────────────────────────
        repair_script = _PROJECT_ROOT / "datagen" / "example_repair.py"
        if repair_script.exists():
            self.log("Repairing borderline examples (score 2-3)...")
            rc = self.run_cmd(
                [sys.executable, str(repair_script), "run",
                 "--min-score", "2", "--max-score", str(min_score)],
                check=False,
            )
            if rc != 0:
                self.log(f"  WARNING: repair exited {rc} (continuing)")
            else:
                report_file = cfg.data_dir / "repair_report.json"
                if report_file.exists():
                    r = json.loads(report_file.read_text())
                    self.log(f"  Done: {r.get('improved', 0)} improved, "
                             f"{r.get('failed', 0)} failed "
                             f"({r.get('success_rate', 0)}% success)")

        # ── 3. Filter ─────────────────────────────────────────────────────
        self.log(f"Filtering below score {min_score}/5...")
        self.run_cmd([sys.executable, judge, "filter", "--min", str(min_score)])

        # ── 4. Dedup ──────────────────────────────────────────────────────
        dedup_script = _PROJECT_ROOT / "datagen" / "dedup.py"
        if dedup_script.exists():
            self.log("Deduplicating similar examples...")
            rc = self.run_cmd(
                [sys.executable, str(dedup_script), "run"],
                check=False,
            )
            if rc != 0:
                self.log(f"  WARNING: dedup exited {rc} (continuing)")
            else:
                report_file = cfg.data_dir / "dedup_report.json"
                if report_file.exists():
                    r = json.loads(report_file.read_text())
                    self.log(f"  Done: removed {r.get('removed', 0)} "
                             f"({r.get('percent_removed', 0)}%)")
                    if r.get("percent_removed", 0) > 30:
                        self.log(f"  WARNING: {r['percent_removed']}% removed "
                                 "— generation may lack diversity")

        # ── 5. Verify ─────────────────────────────────────────────────────
        train_file = cfg.train_file
        if not train_file.exists() or train_file.stat().st_size == 0:
            raise RuntimeError(
                f"train.jsonl empty after curation (score >= {min_score} + dedup). "
                "Lower min_judge_score or regenerate data."
            )
        n = sum(1 for line in train_file.read_text().splitlines() if line.strip())
        self.log(f"Curation complete: {n} examples ready for training")

        # ── 6. HuggingFace snapshot ───────────────────────────────────────
        self._push_to_huggingface(cfg, state, n)

        return state

    def _push_to_huggingface(self, cfg, state: AgentState, n_examples: int) -> None:
        try:
            repo_id = cfg.huggingface.dataset_repo
        except AttributeError:
            return

        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            self.log("HF_TOKEN not set — skipping HuggingFace push")
            return

        try:
            from huggingface_hub import HfApi

            api = HfApi(token=hf_token)
            api.create_repo(repo_id, repo_type="dataset", exist_ok=True)

            version = f"v{state.model_version}"
            commit_msg = (f"Iteration {state.iteration} ({version}): "
                          f"{n_examples} examples")

            files_to_upload = []
            for fname in ["train.jsonl", "val.jsonl", "scores.json", "loop_state.json"]:
                path = cfg.data_dir / fname
                if path.exists():
                    files_to_upload.append((str(path), fname))

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

            self.log(f"Pushed to huggingface.co/datasets/{repo_id}")

        except ImportError:
            self.log("WARNING: huggingface_hub not installed — skipping push")
        except Exception as e:
            self.log(f"WARNING: HuggingFace push failed: {e}")


if __name__ == "__main__":
    from utils.config import load_config
    cfg   = load_config()
    state = AgentState()
    CuratorAgent().run(state, cfg)
