# OAGI Lux Samples

This repository demonstrates how to leverage **Lux** in various use cases. Lux is a Computer Use foundation model from OpenAGI that enables you to build AI agents capable of automating desktop and web tasks through visual understanding and action execution.

## What is Lux?

Lux provides a `TaskerAgent` that can:
- Execute multi-step workflows defined as "todos"
- Take screenshots and understand UI elements visually
- Perform actions like clicking, typing, and scrolling via PyAutoGUI
- Track execution history and export detailed reports

## Examples

### Web Automation

| Example | Description |
|---------|-------------|
| `amazon_scraping.py` | Searches Amazon for a product, sorts by best sellers, and uses a VLM to extract product details from the results page |
| `cvs_tasker.py` | Navigates CVS.com to schedule a flu shot appointment by filling out forms and selecting options |

### Software QA

| Example | Description |
|---------|-------------|
| `software_qa.py` | Automates UI testing of the Nuclear Player app by clicking through all sidebar buttons and verifying each page loads correctly |

## Getting Started

1. Install the `oagi` package
    ```bash
    pip install -r tasker_examples/requirements.txt
    ```
2. Set your API key:
    ```bash
    export OAGI_API_KEY="your-api-key"
    ```
3. Run an example:
    ```bash
    python tasker_examples/amazon_scraping.py --product_name "headphones"
    ```

## Key Components

- **`TaskerAgent`** - The core agent that executes todo-based workflows
- **`AsyncScreenshotMaker`** - Captures screenshots for visual analysis
- **`AsyncPyautoguiActionHandler`** - Executes mouse/keyboard actions
- **`AsyncAgentObserver`** - Records execution history for debugging