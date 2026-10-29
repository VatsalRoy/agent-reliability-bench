"""What the agent is shown: tool names and the format of tool results.

A Variant changes only the *presentation*. The real tools, database, policy,
tasks, and checker stay the same. This is how the TRACE-style probes work:

  none        the normal setup
  rename      meaning-preserving synonyms (get_order -> lookup_order_record)
  reformat    tool results as pretty-printed, key-sorted JSON
  misleading  POSITIVE CONTROL: names shuffled so each name describes a
              different tool (descriptions stay truthful). This SHOULD hurt;
              if it doesn't, the probe setup can't detect real effects.
"""

import copy
import json
import re
from dataclasses import dataclass, field

from env.tools import TOOL_SCHEMAS

REAL_NAMES = [s["function"]["name"] for s in TOOL_SCHEMAS]


@dataclass(frozen=True)
class Variant:
    name: str
    aliases: dict = field(default_factory=dict)  # real name -> name shown to the agent
    pretty: bool = False

    def __post_init__(self):
        shown = [self.aliases.get(n, n) for n in REAL_NAMES]
        if len(set(shown)) != len(shown):
            raise ValueError(f"Variant {self.name}: two tools would share a name")

    # ----- names -----
    def shown_name(self, real: str) -> str:
        return self.aliases.get(real, real)

    def real_name(self, shown: str) -> str | None:
        """The real tool behind a shown name, or None if no tool has that name."""
        for real in REAL_NAMES:
            if self.shown_name(real) == shown:
                return real
        return None

    def rename_text(self, text: str) -> str:
        """Replace every real tool name in text with its shown name (in one pass,
        so shuffled names can't be replaced twice)."""
        if not self.aliases:
            return text
        pattern = re.compile(r"\b(" + "|".join(map(re.escape, self.aliases)) + r")\b")
        return pattern.sub(lambda m: self.aliases[m.group(1)], text)

    # ----- what the agent sees -----
    def schemas(self) -> list[dict]:
        out = copy.deepcopy(TOOL_SCHEMAS)
        for s in out:
            s["function"]["name"] = self.shown_name(s["function"]["name"])
        return out

    def format_result(self, result: dict) -> str:
        if self.pretty:
            text = json.dumps(result, default=str, indent=2, sort_keys=True)
        else:
            text = json.dumps(result, default=str)
        return self.rename_text(text)  # error messages can mention tool names

    def fingerprint(self) -> str:
        """Extra text for the setup hash. Empty for the normal setup, so its
        hash matches runs made before variants existed."""
        if self.name == "none":
            return ""
        return json.dumps({"variant": self.name, "aliases": self.aliases, "pretty": self.pretty},
                          sort_keys=True)


RENAME = {
    "get_customer": "lookup_customer_profile",
    "get_order": "lookup_order_record",
    "search_policy": "query_policy_docs",
    "create_ticket": "open_support_case",
    "update_ticket_status": "set_case_status",
    "issue_refund": "process_refund_payment",
    "cancel_order": "void_order",
    "send_reply": "message_customer",
}

# Each name now belongs to a different tool. Descriptions are unchanged.
MISLEADING = {
    "get_order": "issue_refund",
    "issue_refund": "get_order",
    "get_customer": "search_policy",
    "search_policy": "get_customer",
    "cancel_order": "send_reply",
    "send_reply": "cancel_order",
    "create_ticket": "update_ticket_status",
    "update_ticket_status": "create_ticket",
}

VARIANTS = {
    "none": Variant("none"),
    "rename": Variant("rename", RENAME),
    "reformat": Variant("reformat", pretty=True),
    "misleading": Variant("misleading", MISLEADING),
}
BASE = VARIANTS["none"]