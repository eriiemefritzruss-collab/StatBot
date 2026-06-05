# StatBot

`StatBot` is a statistics-first data analysis agent built for the course project of *Introduction to Artificial Intelligence*. It is developed on top of the original `LAMBDA` project, a large-model-based data agent designed for natural-language-driven data analysis and code execution.

## Demo Video

The project demo video is embedded below and can be played directly on the GitHub repository page:

docs/statbot_demo.mp4

## Background

The original `LAMBDA` project explores how large language models can assist users in data analysis by turning natural language requests into executable code, running that code in a Jupyter kernel, and returning results through an interactive interface. Its core idea is to combine conversational intelligence with a reproducible execution environment, so that users can interact with data without manually writing every line of code.

Based on that foundation, this project focuses on the **AI + statistics** direction. Our goal is to make the system better suited for statistical analysis tasks such as descriptive analysis, hypothesis testing, regression modeling, and statistical reporting.

## What We Changed

Compared with the original `LAMBDA` workflow, this project adds several practical improvements that make the agent more suitable for a course project and for real statistical analysis use:

### 1. Statistics-oriented built-in tools

We added a structured built-in tool registry for common statistical analysis tasks, so the agent can directly route requests to more stable analysis templates instead of relying only on free-form code generation.

Supported statistical capabilities now include:

- data overview and descriptive statistics
- missing-value analysis
- outlier screening
- normality checking
- t-tests
- Wilcoxon signed-rank test
- Mann-Whitney U test
- ANOVA
- Kruskal-Wallis test
- chi-square test
- Fisher exact test
- two-proportion test
- correlation analysis
- OLS regression
- logistic regression
- trend analysis and visualization

### 2. Better context and memory management

We added session-level state tracking so the agent can remember:

- uploaded files and dataset summaries
- recent requests
- selected statistical tools
- execution history
- generated artifacts such as notebooks, reports, and figures

This makes the system more coherent across multi-step analysis sessions.

### 3. Session-safe frontend behavior

We fixed the frontend/backend session isolation problem. Each browser session now gets its own `StatBot` instance, which prevents features like `Edit Code` from accidentally showing code from another session.

### 4. Graceful fallback under model timeout

When the external LLM service times out, the system no longer fails completely after code execution. Instead:

- successful analysis results are still shown
- a local fallback summary is generated for the user
- report generation falls back to a local markdown report

This makes the system much more robust for demo and classroom use.

### 5. Improved frontend interaction flow

We improved the UI-side behavior for:

- upload status feedback
- clearer error surfacing
- notebook export
- dialogue saving
- report generation feedback

## Installation

First, clone the repository and enter the project directory:

```bash
git clone <your-repository-url>
cd statbot
```

We recommend using Python `3.10` in a clean environment.

If you use Conda:

```bash
conda create -n statbot python=3.10
conda activate statbot
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

Install a local Jupyter kernel for code execution:

```bash
ipython kernel install --name python3 --user
```

## Configuration

Edit [config.yaml](config.yaml) and configure the model endpoints you want to use.

The current default configuration uses an OpenAI-compatible API format:

```yaml
conv_model : "qwen-max"
programmer_model : "qwen-max"
inspector_model : "qwen-max"
api_key : "${OPENAI_API_KEY}"
base_url_conv_model : "https://dashscope.aliyuncs.com/compatible-mode/v1"
base_url_programmer : "https://dashscope.aliyuncs.com/compatible-mode/v1"
base_url_inspector : "https://dashscope.aliyuncs.com/compatible-mode/v1"
```

You can either:

1. set the API key in your shell environment, or
2. replace the placeholder with your own key locally

Recommended environment-variable usage:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

## Run

Start the web app with:

```bash
IPYKERNEL=python3 python statbot_app.py
```

Then open the local Gradio URL shown in the terminal, usually:

```text
http://127.0.0.1:8000
```

If port `8000` is occupied, the app will automatically switch to another local port such as `8001`.

## Example Data

A demo dataset is included for testing:

- [demo_data/ecommerce_demo.csv](demo_data/ecommerce_demo.csv)

You can upload it and try prompts such as:

- `Please read this dataset and tell me how many rows and columns it has.`
- `Analyze sales revenue and profit by region.`
- `Run a correlation analysis among numeric variables.`
- `Fit an OLS regression model for profit.`

## Technical Report

The project technical document is available at:

- [docs/project_technical_report.md](docs/project_technical_report.md)

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
