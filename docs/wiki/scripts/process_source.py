#!/usr/bin/env python3
"""
Process a new file from Raw Resources/ into wiki pages.

Usage:
    python process_source.py <path_to_file>

Reads the file, extracts key concepts using the Claude API, and writes
wiki pages to BTRL/Thesis/wiki/ with wikilinks and cross-references.
"""

import sys
import os
import json
import re
import subprocess
from pathlib import Path
from datetime import datetime

WIKI_DIR = Path(__file__).parent.parent
RAW_RESOURCES_DIR = WIKI_DIR.parent / "Raw Resources"
PROCESSED_LOG = WIKI_DIR / "scripts" / ".processed_files.json"


def load_processed_log():
    if PROCESSED_LOG.exists():
        with open(PROCESSED_LOG) as f:
            return json.load(f)
    return {}


def save_processed_log(log):
    with open(PROCESSED_LOG, "w") as f:
        json.dump(log, f, indent=2)


def get_new_files():
    """Return files in Raw Resources that haven't been processed yet."""
    log = load_processed_log()
    new_files = []
    for path in RAW_RESOURCES_DIR.iterdir():
        if path.is_file() and path.suffix in {".pdf", ".txt", ".md", ".docx"}:
            key = path.name
            mtime = path.stat().st_mtime
            if key not in log or log[key]["mtime"] != mtime:
                new_files.append(path)
    return new_files


def read_file_content(path: Path) -> str:
    """Read file content — handles text files; PDFs need extraction."""
    if path.suffix == ".txt" or path.suffix == ".md":
        return path.read_text(encoding="utf-8", errors="replace")
    elif path.suffix == ".pdf":
        # Try pdftotext if available
        try:
            result = subprocess.run(
                ["pdftotext", str(path), "-"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                return result.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return f"[PDF file: {path.name} — install pdftotext for automatic extraction]"
    return f"[Unsupported file type: {path.suffix}]"


def slugify(name: str) -> str:
    """Convert a name to a wiki page slug."""
    name = name.lower()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"\s+", "-", name.strip())
    return name


def call_claude(prompt: str) -> str:
    """Call Claude API to process content."""
    try:
        import anthropic
        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except ImportError:
        return "[anthropic package not installed — run: pip install anthropic]"
    except Exception as e:
        return f"[Claude API error: {e}]"


def build_prompt(filename: str, content: str) -> str:
    """Build the prompt for Claude to process a source document."""
    existing_pages = [p.stem for p in WIKI_DIR.glob("*.md") if p.stem != "home"]
    existing_str = ", ".join(f"[[{p}]]" for p in existing_pages)

    # Truncate content to avoid token limits
    content_preview = content[:6000] + ("..." if len(content) > 6000 else "")

    return f"""You are building a research wiki for a thesis project on BTRL (Bayesian Transformer Reinforcement Learning).

Existing wiki pages: {existing_str}

A new source document has been added to Raw Resources/:
Filename: {filename}
Content:
---
{content_preview}
---

Please produce:
1. A wiki page in Markdown format for this document. The page should:
   - Have a clear title (H1)
   - Summarise the key concepts and findings from the document
   - Use [[wikilink]] syntax to link to related existing wiki pages where relevant
   - Be structured with clear H2 sections
   - End with a ## Links section listing related pages

2. A brief list (3-5 bullet points) of concepts that might warrant their own new wiki pages.

3. A brief update (2-3 sentences) to add to the learnings.md file under a new session entry for today ({datetime.now().strftime('%Y-%m-%d')}).

Format your response as JSON:
{{
  "wiki_page": {{
    "filename": "suggested-slug.md",
    "content": "# Title\\n\\n..."
  }},
  "new_pages_needed": ["concept1", "concept2"],
  "learnings_update": "Session entry text..."
}}
"""


def update_learnings(update_text: str):
    """Prepend a new session entry to learnings.md."""
    learnings_path = WIKI_DIR / "learnings.md"
    if not learnings_path.exists():
        return

    existing = learnings_path.read_text()
    date_str = datetime.now().strftime("%Y-%m-%d")

    new_entry = f"""
## Session: {date_str} — New Source Processed

{update_text}

---
"""
    # Insert after the first H1 line and any introductory text before the first H2
    first_h2 = existing.find("\n## ")
    if first_h2 > -1:
        updated = existing[:first_h2] + "\n" + new_entry + existing[first_h2:]
    else:
        updated = existing + "\n" + new_entry

    learnings_path.write_text(updated)


def update_home(new_page_slug: str, new_page_title: str, source_file: str):
    """Add the new source and wiki page to home.md."""
    home_path = WIKI_DIR / "home.md"
    if not home_path.exists():
        return

    existing = home_path.read_text()
    table_marker = "| `" + source_file + "`"
    if table_marker not in existing:
        new_row = f"| `{source_file}` | [[{new_page_slug}]] |"
        # Find the last row in the source documents table
        last_row_end = existing.rfind("\n", 0, existing.rfind(" |"))
        if last_row_end > -1:
            updated = existing[:last_row_end + 1] + new_row + "\n" + existing[last_row_end + 1:]
            home_path.write_text(updated)


def process_file(path: Path):
    """Process a single file from Raw Resources/ into a wiki page."""
    print(f"Processing: {path.name}")

    content = read_file_content(path)
    prompt = build_prompt(path.name, content)

    print("  Calling Claude API...")
    response_text = call_claude(prompt)

    # Parse JSON response
    try:
        # Extract JSON from response (Claude may add extra text)
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            data = json.loads(json_match.group())
        else:
            raise ValueError("No JSON found in response")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  Warning: Could not parse JSON response: {e}")
        # Fallback: write raw response as a wiki page
        slug = slugify(path.stem)
        wiki_path = WIKI_DIR / f"{slug}.md"
        wiki_path.write_text(f"# {path.stem}\n\nSource: `{path.name}`\n\n{response_text}")
        print(f"  Written (raw): {wiki_path}")
        return

    # Write wiki page
    page_info = data.get("wiki_page", {})
    page_slug = page_info.get("filename", f"{slugify(path.stem)}.md")
    if not page_slug.endswith(".md"):
        page_slug += ".md"
    page_content = page_info.get("content", "")

    wiki_path = WIKI_DIR / page_slug
    wiki_path.write_text(page_content)
    print(f"  Written: {wiki_path}")

    # Update learnings.md
    learnings_update = data.get("learnings_update", "")
    if learnings_update:
        update_learnings(learnings_update)
        print("  Updated learnings.md")

    # Update home.md
    page_stem = page_slug.replace(".md", "")
    update_home(page_stem, path.stem, path.name)
    print("  Updated home.md")

    # Log new page suggestions
    new_pages = data.get("new_pages_needed", [])
    if new_pages:
        print(f"  Suggested new wiki pages: {', '.join(new_pages)}")

    # Mark as processed
    log = load_processed_log()
    log[path.name] = {
        "mtime": path.stat().st_mtime,
        "processed_at": datetime.now().isoformat(),
        "wiki_page": page_slug,
    }
    save_processed_log(log)


def main():
    if len(sys.argv) > 1:
        # Process specific file
        path = Path(sys.argv[1])
        if not path.exists():
            print(f"File not found: {path}")
            sys.exit(1)
        process_file(path)
    else:
        # Process all new/changed files
        new_files = get_new_files()
        if not new_files:
            print("No new files to process.")
            return
        print(f"Found {len(new_files)} new file(s) to process.")
        for path in new_files:
            process_file(path)


if __name__ == "__main__":
    main()
