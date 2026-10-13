import importlib
import os
import unittest

os.environ.setdefault("OPENAI_API_KEY", "test")

gtm_agent = importlib.import_module("gtm_agent.gtm_agent")


class EmailAttachmentBehaviorTests(unittest.TestCase):
    def test_email_tool_documents_plain_text_only(self):
        self.assertIn(
            "Sends plain-text email only; attachments are not supported.",
            gtm_agent.send_prospect_email.description,
        )

    def test_prompt_discloses_unsupported_documents_without_changing_plain_text(self):
        self.assertIn(
            "plain-text only and cannot attach files", gtm_agent.SYSTEM_PROMPT
        )
        self.assertIn(
            "Never claim in an email body or final response that a file is attached or enclosed.",
            gtm_agent.SYSTEM_PROMPT,
        )
        self.assertIn(
            "tell the rep it could not be included because attachments are unsupported",
            gtm_agent.SYSTEM_PROMPT,
        )
        self.assertIn(
            "handle ordinary plain-text requests normally", gtm_agent.SYSTEM_PROMPT
        )


if __name__ == "__main__":
    unittest.main()
