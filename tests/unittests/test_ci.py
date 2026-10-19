from pathlib import Path


def test_documentation_changes_start_ci():
    workflow = Path(".github/workflows/ci.yaml").read_text()
    pull_request = workflow.partition("  pull_request:\n")[2].partition(
        "\n\n"
    )[0]
    documentation_paths = {
        line.strip()
        for line in pull_request.splitlines()
        if ".md" in line or ".gitignore" in line
    }
    assert documentation_paths == {'- ".gitignore"', '- "**/*.md"'}
