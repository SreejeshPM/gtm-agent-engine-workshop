import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test")

from gtm_agent import data_service
from gtm_agent.gtm_agent import SYSTEM_PROMPT, get_prospect, send_prospect_email


class SendProspectEmailTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(config={"metadata": {"user_id": ""}})

    def send(self, prospect):
        return send_prospect_email.func(prospect, "Subject", "Body", self.runtime)

    def test_blocks_disqualified_prospect_without_sending(self):
        prospect = {**get_prospect.func("LEAD-50001")["prospect"], "disqualified": False}

        result = self.send(prospect)

        self.assertEqual(result["status"], "blocked")
        self.assertIn("LEAD-50001", result["error"])
        self.assertIn("disqualified", result["error"])
        self.assertIn("not sent", result["error"])
        self.assertNotIn("message_id", result)

    def test_sends_non_disqualified_prospect(self):
        prospect = {**get_prospect.func("LEAD-12853")["prospect"], "disqualified": True}

        result = self.send(prospect)

        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["to"], "omar.okafor@lakesideanalytics.com")
        self.assertEqual(result["subject"], "Subject")

    def test_missing_prospect_id_fails_without_lookup_or_send(self):
        with patch.object(data_service, "get_prospect_record") as get_record:
            result = self.send({"email": "prospect@example.com"})

        self.assertEqual(result["status"], "failed")
        self.assertIn("prospect ID", result["error"])
        get_record.assert_not_called()

    def test_unknown_prospect_id_fails_without_sending(self):
        result = self.send({"prospect_id": "LEAD-UNKNOWN", "email": "prospect@example.com"})

        self.assertEqual(result["status"], "failed")
        self.assertIn("LEAD-UNKNOWN", result["error"])
        self.assertIn("could not be verified", result["error"])
        self.assertNotIn("message_id", result)

    def test_prompt_explains_disqualification_even_if_rep_insists(self):
        self.assertIn("first use get_prospect to check", SYSTEM_PROMPT)
        self.assertIn("Never draft or send an email to a disqualified prospect", SYSTEM_PROMPT)
        self.assertIn("no email was sent even if they insist", SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
