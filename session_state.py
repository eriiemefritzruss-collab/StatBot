import json
import os
from datetime import datetime


class SessionState:
    def __init__(self, session_cache_path):
        self.session_cache_path = session_cache_path
        self.state_path = os.path.join(session_cache_path, "session_state.json")
        self.state = {
            "created_at": datetime.utcnow().isoformat(),
            "uploaded_files": [],
            "dataset": {},
            "artifacts": [],
            "recent_requests": [],
            "request_summaries": [],
            "current_stage": "",
            "last_plan": "",
            "last_successful_code": "",
            "last_result_summary": "",
            "last_error": "",
            "tool_history": [],
            "execution_history": [],
        }
        self._load()

    def _load(self):
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    self.state.update(loaded)
            except Exception as e:
                print(f"Failed to load session state: {e}")

    def save(self):
        try:
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2, ensure_ascii=True)
        except Exception as e:
            print(f"Failed to save session state: {e}")

    def register_upload(self, filename, dataset_info=None):
        files = self.state["uploaded_files"]
        if filename not in files:
            files.append(filename)
        if dataset_info:
            self.state["dataset"] = {
                "filename": filename,
                "num_rows": dataset_info.get("num_rows"),
                "num_features": dataset_info.get("num_features"),
                "features": [str(v) for v in dataset_info.get("features", [])],
                "missing_val": {
                    str(k): int(v) for k, v in dataset_info.get("missing_val", {}).items()
                } if hasattr(dataset_info.get("missing_val", {}), "items") else {},
            }
        self.save()

    def register_request(self, user_message):
        self.state["recent_requests"].append(user_message)
        self.state["recent_requests"] = self.state["recent_requests"][-8:]
        self.save()

    def register_summary(self, summary):
        self.state["request_summaries"].append(summary)
        self.state["request_summaries"] = self.state["request_summaries"][-8:]
        self.save()

    def register_plan(self, plan):
        self.state["last_plan"] = plan
        self.save()

    def register_stage(self, stage, detail=""):
        self.state["current_stage"] = stage if not detail else f"{stage}: {detail}"
        self.save()

    def register_tool(self, tool_name, rationale, category="", assumptions=""):
        self.state["tool_history"].append({
            "tool": tool_name,
            "category": category,
            "rationale": rationale,
            "assumptions": assumptions,
            "at": datetime.utcnow().isoformat(),
        })
        self.state["tool_history"] = self.state["tool_history"][-12:]
        self.save()

    def register_execution(self, code, result_summary):
        self.state["last_successful_code"] = code
        self.state["last_result_summary"] = result_summary
        self.state["last_error"] = ""
        self.state["execution_history"].append({
            "summary": result_summary,
            "code_preview": code[:240],
            "at": datetime.utcnow().isoformat(),
        })
        self.state["execution_history"] = self.state["execution_history"][-8:]
        self.save()

    def register_error(self, error_message):
        self.state["last_error"] = error_message
        self.state["current_stage"] = "error"
        self.save()

    def register_artifact(self, path, kind):
        artifact = {
            "path": str(path),
            "kind": kind,
            "at": datetime.utcnow().isoformat(),
        }
        self.state["artifacts"].append(artifact)
        self.state["artifacts"] = self.state["artifacts"][-20:]
        self.save()

    def clear(self):
        self.state.update({
            "uploaded_files": [],
            "dataset": {},
            "artifacts": [],
            "recent_requests": [],
            "request_summaries": [],
            "current_stage": "",
            "last_plan": "",
            "last_successful_code": "",
            "last_result_summary": "",
            "last_error": "",
            "tool_history": [],
            "execution_history": [],
        })
        self.save()

    def build_memory_context(self):
        dataset = self.state.get("dataset", {})
        artifact_list = self.state.get("artifacts", [])[-5:]
        tool_list = self.state.get("tool_history", [])[-5:]
        execution_history = self.state.get("execution_history", [])[-3:]

        lines = ["Session memory summary:"]
        if self.state.get("current_stage"):
            lines.append(f"- Current stage: {self.state['current_stage']}")
        if dataset:
            lines.append(
                f"- Active dataset: {dataset.get('filename')} with "
                f"{dataset.get('num_rows')} rows and {dataset.get('num_features')} columns."
            )
            features = dataset.get("features", [])
            if features:
                lines.append(f"- Dataset columns: {', '.join(features[:12])}")
        recent_requests = self.state.get("recent_requests", [])
        if recent_requests:
            lines.append("- Recent user requests:")
            for item in recent_requests[-3:]:
                lines.append(f"  * {item}")
        request_summaries = self.state.get("request_summaries", [])
        if request_summaries:
            lines.append("- Recent completed work:")
            for item in request_summaries[-3:]:
                lines.append(f"  * {item}")
        if tool_list:
            lines.append("- Recently selected built-in tools:")
            for item in tool_list:
                detail = item["tool"]
                if item.get("category"):
                    detail += f" [{item['category']}]"
                lines.append(f"  * {detail}: {item['rationale']}")
                if item.get("assumptions"):
                    lines.append(f"    assumptions: {item['assumptions']}")
        if artifact_list:
            lines.append("- Recent artifacts:")
            for item in artifact_list:
                lines.append(f"  * {item['kind']}: {item['path']}")
        if self.state.get("last_result_summary"):
            lines.append(f"- Last result summary: {self.state['last_result_summary']}")
        if execution_history:
            lines.append("- Recent execution summaries:")
            for item in execution_history:
                lines.append(f"  * {item['summary']}")
        if self.state.get("last_error"):
            lines.append(f"- Last error: {self.state['last_error']}")
        return "\n".join(lines)
