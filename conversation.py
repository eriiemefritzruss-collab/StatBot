import os
import openai
import json
from programmer import Programmer
from inspector import Inspector
from builtin_skills import route_builtin_skill, embed_skill_code, summarize_request
from cache.cache import *
from prompt_engineering.prompts import *
import warnings
import traceback
import zipfile
from kernel import *
from display import *
from pathlib import Path
from session_state import SessionState
from utils.utils import *
# warnings.filterwarnings("ignore")


class Conversation:

    def __init__(self, config) -> None:
        self.config = config    
        self.api_timeout = float(config.get("api_timeout", 20))
        self.client = openai.OpenAI(
            api_key=config['api_key'],
            base_url=config['base_url_conv_model'],
            timeout=self.api_timeout
        )
        self.model = config['conv_model']
        self.programmer = Programmer(
            api_key=config['api_key'],
            model=config['programmer_model'],
            base_url=config['base_url_programmer'],
            timeout=self.api_timeout
        )
        self.inspector = Inspector(
            api_key=config['api_key'],
            model=config['inspector_model'],
            base_url=config['base_url_inspector'],
            timeout=self.api_timeout
        )
        self.session_cache_path = config["session_cache_path"]
        self.chat_history_display = config["chat_history_display"] if "chat_history_display" in config else []
        self.retrieval = self.config['retrieval']
        self.analysis_profile = config.get("analysis_profile", "general")
        self.statistical_planning = config.get("statistical_planning", False)
        self.kernel = CodeKernel(session_cache_path=self.session_cache_path, max_exe_time=config['max_exe_time'])
        self.max_attempts = config['max_attempts']
        self.error_count = 0
        self.repair_count = 0
        self.file_list = []
        self.figure_list = config["figure_list"] if "figure_list" in config else []
        self.function_repository = {}
        self.my_data_cache = None
        self.session_state = SessionState(self.session_cache_path)
        self._active_skill_name = None
        self.run_code(IMPORT)


    def add_functions(self, function_lib: dict) -> None:
        self.function_repository = function_lib

    def add_data(self, data_path) -> None:
        self.my_data_cache = data_cache(data_path)
        suffix = Path(data_path).suffix.lower()
        reader = "pd.read_csv" if suffix == ".csv" else "pd.read_excel"
        load_code = (
            f"data = {reader}({data_path!r})\n"
            "print(f'Dataset loaded into variable data with shape: {data.shape}')\n"
            "data.head()"
        )
        self.run_code(load_code)

    def check_folder(self):
        current_files = os.listdir(self.session_cache_path)
        new_files = set(current_files) - set(self.file_list)
        self.file_list = current_files
        display = False
        display_link = ''
        if new_files:
            display = True
            for file in new_files:
                file_link = os.path.join(self.session_cache_path,file)
                if os.path.splitext(file)[1] in ['.png','.jpg','.jpeg']:
                    display_link += display_image(file_link)
                    absolute_path = Path(file_link).resolve()
                    print("absolute_path",absolute_path)
                    self.figure_list.append(absolute_path)
                    self.session_state.register_artifact(absolute_path, "image")
                else:
                    display_link += display_download_file(file_link, file)
                    self.session_state.register_artifact(file_link, "file")
        print("display_link:", display_link)
        return display, display_link


    def save_conv(self):
        with open(os.path.join(self.session_cache_path, 'programmer_msg.json'), 'w') as f:
            json.dump(self.programmer.messages, f, indent=4)
            f.close()
        print(f"Conversation saved in {os.path.join(self.session_cache_path, 'programmer_msg.json')}")
        with open(os.path.join(self.session_cache_path, 'inspector_msg.json'), 'w') as f:
            json.dump(self.inspector.messages, f, indent=4)
            f.close()
        print(f"Conversation saved in {os.path.join(self.session_cache_path, 'inspector_msg.json')}")
        with open(os.path.join(self.session_cache_path, 'config.json'), 'w') as f:
            config = {
            "config": self.config,
            "model": self.model,
            "session_cache_path": self.session_cache_path,
            "chat_history_display": self.chat_history_display,
            "retrieval": self.retrieval,
            "max_attempts": self.max_attempts,
            "error_count": self.error_count,
            "repair_count": self.repair_count,
            "file_list": [str(p) for p in self.file_list],
            "figure_list": [str(p) for p in self.figure_list],
            "function_repository": self.function_repository,
            "session_state": self.session_state.state,
        }
            json.dump(config, f, indent=4)
        print(f"Config saved in {os.path.join(self.session_cache_path, 'config.json')}")

    def add_programmer_msg(self, message: dict):
        self.programmer.messages.append(message)

    def add_programmer_repair_msg(self, bug_code: str, error_msg: str, fix_method: str, role="user"):
        message = {"role": role,
                   "content": CODE_FIX.format(bug_code=bug_code, error_message=error_msg, fix_method=fix_method)}
        self.programmer.messages.append(message)

    def add_inspector_msg(self, bug_code: str, error_msg: str, role="user"):
        message = {"role": role, "content": CODE_INSPECT.format(bug_code=bug_code, error_message=error_msg)}
        self.inspector.messages.append(message)

    def run_code(self, code):
        try:
            sign, msg_llm, exe_res = execute(code, self.kernel)
        except Exception as e:  # this error is due to the outer programme, not the error in the kernel
            print(f'Error in executing code (outer): {e}')
            sign, msg_llm, exe_res = 'text', f'{e}\nThis error is due to the outer programme, not the error in the kernel, you should tell the user to check the system code.', str(e) # tell the user, the code have problems.

        return sign, msg_llm, exe_res

    def rendering_code(self):
        for i in range(len(self.programmer.messages) - 1, 0, -1):
            if self.programmer.messages[i]["role"] == "assistant":
                is_python, code = extract_code(self.programmer.messages[i]["content"])
                if is_python:
                    return code
        return None

    def show_data(self) -> pd.DataFrame:
        if self.my_data_cache is None:
            print("User do not upload a data.")
            return pd.DataFrame()
        return self.my_data_cache.data

    def document_generation(self, chat_history):
        print("Report generating...")
        self.session_state.register_stage("reporting", "Generating markdown report from the session history.")
        formatted_chat = []
        for item in chat_history:
            formatted_chat.append({"role": "user", "content": item[0]})
            formatted_chat.append({"role": "assistant", "content": item[1]})
        report_prompt = STAT_REPORT_PROMPT if self.analysis_profile == "statistics" else Basic_Report
        report_context = self.session_state.build_memory_context()
        self.messages = [{"role": "system", "content": report_prompt}] + formatted_chat + [{"role": "user", "content": f"Now, you should generate a report according to the above chat history (Do not give further suggestions at the end of report).\nNote: Here is figure list with links in the chat history: {self.figure_list}\n\nAdditional session context:\n{report_context}"}]
        try:
            report = self.call_chat_model().choices[0].message.content
            self.messages.append({"role": "assistant", "content": report})
        except Exception as e:
            print(f"Falling back to local report generation: {e}")
            self.session_state.register_error(f"Report model timeout: {e}")
            report = self._build_local_report(chat_history)
        mkd_path = os.path.join(self.session_cache_path, 'report.md')
        with open(mkd_path, "w") as f:
            f.write(report)
            f.close()
        self.session_state.register_artifact(mkd_path, "report")
        return mkd_path

    def _build_local_report(self, chat_history):
        dataset = self.session_state.state.get("dataset", {})
        artifacts = self.session_state.state.get("artifacts", [])[-8:]
        tool_history = self.session_state.state.get("tool_history", [])[-8:]
        summaries = self.session_state.state.get("request_summaries", [])[-8:]
        execution_history = self.session_state.state.get("execution_history", [])[-5:]
        recent_requests = self.session_state.state.get("recent_requests", [])[-5:]
        lines = [
            "# StatBot Technical Analysis Report",
            "",
            "## 1. Title",
            "",
            "StatBot: AI-Assisted Statistical Analysis Session Report",
            "",
            "## 2. Abstract",
            "",
            "This report was generated using StatBot's local fallback pipeline because the external report model was temporarily unavailable. The report summarizes the uploaded dataset, the agent's statistical workflow, the tools selected during execution, and the observable outputs recorded in the current session.",
            "",
            "## 3. Introduction",
            "",
            "The goal of this session was to support data analysis through an AI+statistics agent that combines planning, tool selection, code execution, and artifact tracking. When the large language model endpoint timed out, StatBot preserved reproducibility by generating a deterministic report from the recorded session state.",
            "",
            "## 4. Data and Variables",
            "",
        ]
        if dataset:
            lines.extend([
                f"- Active dataset: `{dataset.get('filename', 'unknown')}`",
                f"- Number of rows: {dataset.get('num_rows', 'unknown')}",
                f"- Number of columns: {dataset.get('num_features', 'unknown')}",
            ])
            features = dataset.get("features", [])
            if features:
                lines.append(f"- Variables: {', '.join(features)}")
            missing = dataset.get("missing_val", {})
            if missing:
                lines.append("- Missing-value summary:")
                for key, value in missing.items():
                    lines.append(f"  - {key}: {value}")
        else:
            lines.append("- No structured dataset metadata were available in the current session state.")
        lines.extend([
            "",
            "## 5. Methodology",
            "",
            "### 5.1 Agent workflow",
            "",
            "StatBot executes a multi-stage pipeline consisting of request intake, statistical planning, built-in tool routing or dynamic code generation, notebook execution, artifact collection, and result interpretation.",
            "",
            "### 5.2 Built-in tools and statistical methods",
            "",
        ])
        if tool_history:
            for item in tool_history:
                tool_line = f"- `{item.get('tool', 'unknown')}`"
                if item.get("category"):
                    tool_line += f" ({item['category']})"
                tool_line += f": {item.get('rationale', '')}"
                lines.append(tool_line)
                if item.get("assumptions"):
                    lines.append(f"  - Assumptions: {item['assumptions']}")
        else:
            lines.append("- No built-in statistical tool was recorded in this session.")
        lines.extend([
            "",
            "### 5.3 Recent user requests",
            "",
        ])
        if recent_requests:
            for request in recent_requests:
                lines.append(f"- {request}")
        else:
            lines.append("- No user requests were recorded.")
        lines.extend([
            "",
            "## 6. Results",
            "",
        ])
        if execution_history:
            for idx, item in enumerate(execution_history, start=1):
                lines.append(f"### 6.{idx} Execution Summary")
                lines.append("")
                lines.append(item.get("summary", "No summary available."))
                lines.append("")
        else:
            lines.append("No successful execution summary was available.")
            lines.append("")
        if summaries:
            lines.append("### 6.X Session-level request summaries")
            lines.append("")
            for item in summaries:
                lines.append(f"- {item}")
            lines.append("")
        if artifacts:
            lines.append("### 6.Y Generated artifacts")
            lines.append("")
            for item in artifacts:
                lines.append(f"- {item.get('kind', 'artifact')}: `{item.get('path', '')}`")
            lines.append("")
        lines.extend([
            "## 7. Interpretation",
            "",
            "The available evidence shows that the agent successfully preserved dataset metadata, execution traces, and generated artifacts. Even when the external model timed out, the underlying analysis environment remained functional, enabling the session to retain reproducible intermediate results.",
            "",
            "## 8. Limitations and Robustness",
            "",
            "- External LLM connectivity timed out during at least part of the session.",
            "- The report is generated from local session memory and execution traces, so it is less narrative than a fully model-written report.",
            "- Additional validation should be performed once the model endpoint is reachable again.",
            "",
            "## 9. Conclusion",
            "",
            "This fallback report demonstrates that StatBot can degrade gracefully under network instability while preserving core agent capabilities: tool use, code execution, memory tracking, and artifact generation.",
        ])
        if self.figure_list:
            lines.extend(["", "## 10. Figures", ""])
            for figure in self.figure_list:
                lines.append(f"![{Path(figure).name}](file={figure})")
        return "\n".join(lines)

    def export_code(self):
        print("Exporting notebook...")
        notebook_path = os.path.join(self.session_cache_path, 'notebook.ipynb')
        try:
            self.kernel.write_to_notebook(notebook_path)
        except Exception as e:
            print(f"An error occurred when exporting notebook: {e}")
        return notebook_path

    def call_chat_model(self, functions=None, include_functions=False):
        params = {
            "model": self.model,
            "messages": self.messages,
        }

        if include_functions:
            params["functions"] = functions
            params["function_call"] = "auto"

        return self.client.chat.completions.create(**params)

    def _dataset_context_for_planning(self):
        if self.my_data_cache is None:
            return "No structured dataset metadata is available yet."

        try:
            info = self.my_data_cache.general_info or self.my_data_cache.get_description()
            return str(info)
        except Exception as e:
            return f"Dataset metadata could not be summarized: {e}"

    def _build_statistical_plan(self, user_message):
        if self.analysis_profile != "statistics" or not self.statistical_planning:
            return None

        planning_messages = [
            {"role": "system", "content": STAT_PLANNER_PROMPT},
            {
                "role": "user",
                "content": (
                    f"User request:\n{user_message}\n\n"
                    f"Uploaded files: {self.file_list}\n\n"
                    f"Dataset metadata:\n{self._dataset_context_for_planning()}"
                )
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=planning_messages
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Failed to build statistical plan: {e}")
            return (
                "1. Data Overview Checks: inspect rows, columns, types, missing values, and summary statistics.\n"
                "2. Research Question / Task Type: infer the task from the user request and available variables.\n"
                "3. Statistical Method Selection: choose tests or models based on variable types and assumptions.\n"
                "4. Assumption and Diagnostic Checks: check normality, variance, outliers, multicollinearity, and residuals when relevant.\n"
                "5. Inference / Modeling Outputs: report statistics, p-values, confidence intervals, effect sizes, coefficients, or validation metrics.\n"
                "6. Interpretation and Next Steps: explain the result, limitations, and robustness checks."
            )


    def _build_contextual_request(self, user_message, statistical_plan=None):
        memory_context = self.session_state.build_memory_context()
        request = user_message
        if statistical_plan:
            request = STAT_PLAN_INJECTION.format(user_message=user_message, plan=statistical_plan)
        return f"{request}\n\n{memory_context}"

    def _fallback_result_response(self, exe_res, error):
        preview_lines = []
        for line in exe_res.splitlines():
            stripped = line.strip()
            if stripped:
                preview_lines.append(stripped)
            if len(preview_lines) >= 6:
                break
        preview_text = " | ".join(preview_lines) if preview_lines else "The execution finished successfully, and detailed outputs are shown in the execution panel above."
        lines = [
            "### Local fallback summary",
            "The analysis code executed successfully, but the model-based explanation timed out.",
            f"Observed output preview: {preview_text}",
            f"Technical note: {error}",
            "",
            "Next, you can:",
            "[1]Review the execution results above and inspect the generated artifacts or tables.",
            "[2]Retry the interpretation step after restoring model connectivity.",
            "[3]Use Edit Code to refine the analysis or request a more targeted statistical method.",
        ]
        return "\n".join(lines)


    def clear(self):
        # os.removedirs(self.session_cache_path)
        clear_working_path(self.session_cache_path)
        self.messages = []
        self.programmer.clear()
        self.inspector.clear()
        self.kernel.shutdown()
        del self.kernel
        self.kernel = CodeKernel(session_cache_path=self.session_cache_path, max_exe_time=self.config['max_exe_time'])
        self.my_data_cache = None
        self.session_state.clear()
        self.session_state.register_stage("idle", "Waiting for the next request.")


    def stream_workflow(self, chat_history_display, code=None) -> object:
        try:
            chat_history_display[-1][1] = ""
            self.session_state.register_stage("received", "User request accepted.")
            yield chat_history_display
            if code is not None:
                self.session_state.register_stage("human_edit", "User provided code manually.")
                prog_response = HUMAN_LOOP.format(code=code)
                self.add_programmer_msg({"role": "user", "content": prog_response})
            else:
                self._active_skill_name = None
                user_message = self.programmer.messages[-1]["content"]
                self.session_state.register_request(user_message)
                statistical_plan = self._build_statistical_plan(user_message)
                if statistical_plan:
                    self.session_state.register_stage("planning", "Built a statistical analysis plan.")
                    chat_history_display[-1][1] += f"### Statistical analysis plan\n{statistical_plan}\n\n"
                    yield chat_history_display
                    self.session_state.register_plan(statistical_plan)

                self.programmer.messages[-1]["content"] = self._build_contextual_request(user_message, statistical_plan)
                builtin_skill = route_builtin_skill(
                    user_message,
                    self.my_data_cache.data if self.my_data_cache is not None else None
                )
                if builtin_skill:
                    self._active_skill_name = builtin_skill["name"]
                    self.session_state.register_tool(
                        builtin_skill["name"],
                        builtin_skill["rationale"],
                        builtin_skill.get("category", ""),
                        builtin_skill.get("assumptions", ""),
                    )
                    self.session_state.register_stage("tool_selection", f"Selected built-in tool {builtin_skill['name']}.")
                    chat_history_display[-1][1] += (
                        f"### statbot built-in tool selected\n"
                        f"- Tool: `{builtin_skill['name']}`\n"
                        f"- Category: `{builtin_skill.get('category', 'general')}`\n"
                        f"- What it does: {builtin_skill.get('description', 'Run a structured statistical analysis.')}\n"
                        f"- Why: {builtin_skill['rationale']}\n\n"
                    )
                    if builtin_skill.get("assumptions"):
                        chat_history_display[-1][1] += f"- Assumptions: {builtin_skill['assumptions']}\n\n"
                    yield chat_history_display
                    prog_response = embed_skill_code(builtin_skill)
                else:
                    self.session_state.register_stage("code_generation", "Generating analysis code dynamically.")
                    prog_response = ''
                    for message in self.programmer._call_chat_model_streaming(retrieval=self.retrieval, kernel=self.kernel):
                        chat_history_display[-1][1] += message
                        yield chat_history_display
                        prog_response += message
                self.add_programmer_msg({"role": "assistant", "content": prog_response})

            is_python, code = extract_code(prog_response)
            print("is_python:", is_python)

            if is_python:
                self.session_state.register_stage("execution", "Executing Python analysis code.")
                chat_history_display[-1][1] += '\n🖥️ Execute code...'
                yield chat_history_display
                sign, msg_llm, exe_res = self.run_code(code)
                print("Executing result:", exe_res)
                if sign and 'error' not in sign:
                    yield from self._handle_execution_result(exe_res, msg_llm, chat_history_display)
                else:
                    self.error_count += 1
                    round = 0
                    while 'error' in sign and round < self.max_attempts:
                        self.session_state.register_stage("repair", f"Repair attempt {round + 1}.")
                        chat_history_display[-1][1] = f'⭕ Execution error, try to repair the code, attempts: {round + 1}....\n'
                        yield chat_history_display
                        self.add_inspector_msg(code, msg_llm)
                        if round == 3:
                            insp_response = "Try other packages or methods."
                        else:
                            insp_response = self.inspector._call_chat_model().choices[0].message.content
                        self.inspector.messages.append({"role": "assistant", "content": insp_response})

                        self.add_programmer_repair_msg(code, msg_llm, insp_response)
                        prog_response = ''
                        for message in self.programmer._call_chat_model_streaming():
                            chat_history_display[-1][1] += message
                            prog_response += message
                            yield chat_history_display
                        chat_history_display[-1][1] += '\n🖥️ Execute code...\n'
                        yield chat_history_display
                        self.add_programmer_msg({"role": "assistant", "content": prog_response})
                        is_python, code = extract_code(prog_response)
                        if is_python:
                            sign, msg_llm, exe_res = self.run_code(code)
                            if sign and 'error' not in sign:
                                self.repair_count += 1
                                break
                        round += 1

                    if round == self.max_attempts:
                        return prog_response + f"\nSorry, I can't fix the code with {self.max_attempts} attempts, can you help me to modified it or give some suggestions?"

                    yield from self._handle_execution_result(exe_res, msg_llm, chat_history_display)

        except Exception as e:
            if chat_history_display and chat_history_display[-1][1] is None:
                chat_history_display[-1][1] = ""
            chat_history_display[-1][1] += f"\nSorry, there is an error in the program, please try again.\n\nError detail: {e}"
            yield chat_history_display
            print(f"An error occurred: {e}")
            traceback.print_exc()
            self.session_state.register_error(str(e))
            if self.programmer.messages[-1]["role"] == "user":
                self.programmer.messages.append({"role": "assistant", "content": f"An error occurred in program: {e}"})

    def _handle_execution_result(self, exe_res, msg_llm, chat_history_display):
        chat_history_display[-1][1] += display_exe_results(exe_res)
        yield chat_history_display

        summary = msg_llm.replace("\n", " ").strip()[:500]
        self.session_state.register_execution(self.rendering_code() or "", summary)
        self.session_state.register_summary(
            summarize_request(
                self.session_state.state.get("recent_requests", [""])[-1],
                self._active_skill_name
            )
        )
        self.session_state.register_stage("result_interpretation", "Explaining execution results and collecting artifacts.")

        display, link_info = self.check_folder()
        chat_history_display[-1][1] += f"{link_info}" if display else ''
        yield chat_history_display

        result_prompt = STAT_RESULT_PROMPT if self.analysis_profile == "statistics" else RESULT_PROMPT
        self.add_programmer_msg({"role": "user", "content": result_prompt.format(msg_llm)})
        prog_response = ''
        try:
            for message in self.programmer._call_chat_model_streaming():
                chat_history_display[-1][1] += message
                yield chat_history_display
                prog_response += message
        except Exception as e:
            print(f"Falling back to local execution summary: {e}")
            self.session_state.register_error(f"Result interpretation timeout: {e}")
            prog_response = self._fallback_result_response(exe_res, e)
            chat_history_display[-1][1] += f"\n{prog_response}"
            yield chat_history_display

        self.add_programmer_msg({"role": "assistant", "content": prog_response})
        if prog_response.startswith("### Local fallback summary"):
            formatted_response = chat_history_display[-1][1]
        else:
            formatted_response = display_suggestions(prog_response, chat_history_display[-1][1])
            if not formatted_response or formatted_response.strip() == "#":
                formatted_response = chat_history_display[-1][1]
        chat_history_display[-1][1] = formatted_response
        self.session_state.register_stage("completed", "Finished the current request.")
        yield chat_history_display

    
    def update_config(self, conv_model, programmer_model, inspector_model, api_key,
                      base_url_conv_model, base_url_programmer, base_url_inspector,
                      max_attempts, max_exe_time):

        client_config_changed = any([
            self.config['api_key'] != api_key,
            self.config['base_url_conv_model'] != base_url_conv_model,
            self.config['base_url_programmer'] != base_url_programmer,
            self.config['base_url_inspector'] != base_url_inspector,
        ])

        if client_config_changed:
            self.config['api_key'] = api_key
            self.config['base_url_conv_model'] = base_url_conv_model
            self.config['base_url_programmer'] = base_url_programmer
            self.config['base_url_inspector'] = base_url_inspector
            self.client = openai.OpenAI(api_key=api_key, base_url=base_url_conv_model, timeout=self.api_timeout)
            self.programmer.client = openai.OpenAI(api_key=api_key, base_url=base_url_programmer, timeout=self.api_timeout)
            self.inspector.client = openai.OpenAI(api_key=api_key, base_url=base_url_inspector, timeout=self.api_timeout)

        if self.model != conv_model:
            self.model = conv_model
            self.config['conv_model'] = conv_model

        if self.kernel.max_exe_time != max_exe_time:
            self.kernel.max_exe_time = max_exe_time
            self.config['max_exe_time'] = max_exe_time

        if self.programmer.model != programmer_model:
            self.programmer.model = programmer_model
            self.config['programmer_model'] = programmer_model

        if self.inspector.model != inspector_model:
            self.inspector.model = inspector_model
            self.config['inspector_model'] = inspector_model

        if self.max_attempts != max_attempts:
            self.config['max_attempts'] = max_attempts
