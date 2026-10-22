"""Refresh the offline HTML's embedded metrics engine; no NPU or dependencies."""

from pathlib import Path
import re


def main():
    directory = Path(__file__).resolve().parent
    html = directory / "urma_trace_dashboard.html"
    source = (directory / "urma_dashboard_metrics.js").read_text(encoding="utf-8")
    if "</script" in source.lower():
        raise ValueError("Metrics engine contains an unsafe inline script terminator")
    bundle = "<!-- ANALYZER_BEGIN -->\n<script>\n" + source + "\n</script>\n<!-- ANALYZER_END -->"
    content = html.read_text(encoding="utf-8")
    if "<!-- ANALYZER_SCRIPT -->" in content:
        content = content.replace("<!-- ANALYZER_SCRIPT -->", bundle)
    else:
        content, count = re.subn(
            r"<!-- ANALYZER_BEGIN -->.*?<!-- ANALYZER_END -->", lambda _: bundle, content, flags=re.DOTALL
        )
        if count != 1:
            raise ValueError("Expected exactly one metrics engine placeholder")
    html.write_text(content, encoding="utf-8")
    print(html)


if __name__ == "__main__":
    main()
