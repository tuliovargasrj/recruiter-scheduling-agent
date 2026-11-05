---
name: publish-check
description: Audit this repository before it goes public. Use when asked to check, prepare, or publish the repo, or before the first push.
---

Report each item as pass or fail. Do not publish or push anything, and ask before any command that calls the paid API.

1. Scan tracked files and full git history for secrets, tokens, and `.env` files.
2. Confirm no private, client, employer, or personal data is present.
3. Run every command in the README from a clean copy and confirm it works as written.
4. Confirm the example conversation in the README matches real output.
5. Confirm `uv run pytest` passes with no API key set.
6. Confirm all code, comments, and documentation are in English.
