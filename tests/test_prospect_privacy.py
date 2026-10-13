import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test")

from gtm_agent import data_service
from gtm_agent.gtm_agent import build_prospect_profile, get_prospect, score_prospect


SENSITIVE_FIELDS = ("tax_id", "date_of_birth", "card_on_file", "credit_check_ref")


def nested_keys(value):
    if isinstance(value, dict):
        return set(value).union(*(nested_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(nested_keys(item) for item in value))
    return set()


class ProspectPrivacyTests(unittest.TestCase):
    def assert_no_sensitive_fields(self, value):
        self.assertTrue(set(SENSITIVE_FIELDS).isdisjoint(nested_keys(value)))

    def test_get_prospect_returns_only_contact_fields(self):
        result = get_prospect.func("LEAD-50001")

        self.assertEqual(result["found"], True)
        self.assertEqual(
            result["prospect"],
            {
                "prospect_id": "LEAD-50001",
                "name": "Priya Nair",
                "email": "priya.nair@brightwaveapps.com",
                "disqualified": True,
            },
        )
        self.assert_no_sensitive_fields(result)

    def test_new_profile_excludes_billing_qualification(self):
        with patch.object(data_service, "get_profile_from_db", return_value={"prospect_profile": None}), \
             patch.object(data_service, "save_profile_to_db") as save_profile:
            result = build_prospect_profile.func("LEAD-71001")

        profile = result["prospect_profile"]
        self.assertTrue(result["found"])
        self.assertNotIn("billing_qualification", profile)
        self.assertEqual(profile["annual_revenue"], 60000000)
        self.assertIn("tech_stack", profile)
        self.assert_no_sensitive_fields(result)
        self.assert_no_sensitive_fields(save_profile.call_args.args[1])

    def test_cached_profile_is_cleaned_before_return(self):
        existing = {
            "prospect_id": "LEAD-71001",
            "name": "Noah Williams",
            "billing_qualification": {field: "sensitive" for field in SENSITIVE_FIELDS},
            "tech_stack": ["Snowflake", "dbt"],
        }
        with patch.object(data_service, "get_profile_from_db", return_value={"prospect_profile": existing}):
            result = build_prospect_profile.func("LEAD-71001")

        self.assertTrue(result["found"])
        self.assertNotIn("billing_qualification", result["prospect_profile"])
        self.assertEqual(result["prospect_profile"]["tech_stack"], ["Snowflake", "dbt"])
        self.assertIn("billing_qualification", existing)
        self.assert_no_sensitive_fields(result)

    def test_scoring_prompt_uses_allowlisted_profile_and_keeps_score_output(self):
        profile = {
            "prospect_id": "LEAD-71001",
            "name": "Noah Williams",
            "annual_revenue": 60000000,
            "tech_stack": ["Snowflake", "dbt"],
            "account_details": [{"industry": "Software"}],
            "engagement_history": [{"channel": "Webinar"}],
            "billing_qualification": {field: "sensitive" for field in SENSITIVE_FIELDS},
            "unapproved_field": "omit",
        }
        score = {"score": 88, "max_score": 100, "justification": "Strong fit", "rubric_breakdown": {}}
        response = SimpleNamespace(model_dump=lambda: score)
        llm = SimpleNamespace(invoke=unittest.mock.Mock(return_value=response))
        with patch("gtm_agent.gtm_agent._scoring_llm", llm):
            result = score_prospect.func(profile, data_service.get_offering("OFFER-10005"))

        user_message = llm.invoke.call_args.args[0][1]["content"]
        prompt_profile = json.loads(user_message.split("\n\nProspect profile:\n", 1)[1])
        self.assertEqual(
            set(prompt_profile),
            {
                "prospect_id", "name", "annual_revenue", "tech_stack",
                "account_details", "engagement_history",
            },
        )
        self.assertEqual(prompt_profile["tech_stack"], data_service.fetch_tech_stack("LEAD-71001"))
        self.assertEqual(result["score"], 88)
        self.assertEqual(result["justification"], "Strong fit")
        self.assert_no_sensitive_fields(user_message)
        self.assert_no_sensitive_fields(result)


if __name__ == "__main__":
    unittest.main()
