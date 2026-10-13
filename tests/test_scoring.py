import os
import unittest
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test")

from gtm_agent import data_service
from gtm_agent.gtm_agent import ScoringJustification, score_prospect


class ScoreProspectTests(unittest.TestCase):
    def score(self, profile, offering):
        with patch("gtm_agent.gtm_agent._scoring_llm") as scoring_llm:
            scoring_llm.invoke.return_value = ScoringJustification(justification="Fit summary.")
            return score_prospect.invoke({"prospect_profile": profile, "offering": offering})

    def test_identical_lead_and_offering_have_deterministic_numeric_fields(self):
        profile = {
            "prospect_id": "LEAD-90001",
            **data_service.get_prospect_record("LEAD-90001"),
        }
        offering = data_service.get_offering("OFFER-10007")

        first = self.score(profile, offering)
        second = self.score(profile, offering)

        numeric_fields = ("score", "max_score", "rubric_breakdown")
        self.assertEqual(
            {field: first[field] for field in numeric_fields},
            {field: second[field] for field in numeric_fields},
        )
        self.assertEqual(first["rubric_breakdown"]["tech_stack_match"], 75.0)
        self.assertEqual(first["score"], 91.7)

    def test_empty_required_technology_list_scores_full_match(self):
        offering = {
            "required_tech_stack": [],
            "min_annual_revenue": 1,
            "description": "Offering.",
        }

        result = self.score({"annual_revenue": 1}, offering)

        self.assertEqual(result["rubric_breakdown"]["tech_stack_match"], 100.0)

    def test_segment_fit_is_case_insensitive_and_mismatch_scores_zero(self):
        offering = {
            "required_tech_stack": [],
            "min_annual_revenue": 1,
            "target_segment": "Enterprise",
            "description": "Offering.",
        }

        matching = self.score({"annual_revenue": 1, "segment": "eNtErPrIsE"}, offering)
        mismatching = self.score({"annual_revenue": 1, "segment": "Mid-Market"}, offering)

        self.assertEqual(matching["rubric_breakdown"]["segment_fit"], 100.0)
        self.assertEqual(mismatching["rubric_breakdown"]["segment_fit"], 0.0)

    def test_revenue_fit_at_above_and_below_minimum(self):
        offering = {
            "required_tech_stack": [],
            "min_annual_revenue": 100,
            "description": "Offering.",
        }

        at_minimum = self.score({"annual_revenue": 100}, offering)
        above_minimum = self.score({"annual_revenue": 150}, offering)
        below_minimum = self.score({"annual_revenue": 25}, offering)

        self.assertEqual(at_minimum["rubric_breakdown"]["revenue_fit"], 100.0)
        self.assertEqual(above_minimum["rubric_breakdown"]["revenue_fit"], 100.0)
        self.assertEqual(below_minimum["rubric_breakdown"]["revenue_fit"], 25.0)

    def test_missing_revenue_counts_as_zero(self):
        offering = {
            "required_tech_stack": [],
            "min_annual_revenue": 100,
            "description": "Offering.",
        }

        result = self.score({}, offering)

        self.assertEqual(result["rubric_breakdown"]["revenue_fit"], 0.0)

    def test_missing_minimum_returns_existing_error_result(self):
        offering = {"required_tech_stack": [], "description": "Offering."}

        result = score_prospect.invoke({"prospect_profile": {}, "offering": offering})

        self.assertEqual(result, {"score": None, "error": "Cannot score without a valid offering."})

    def test_zero_minimum_scores_nonnegative_revenue_as_full_fit(self):
        offering = {
            "required_tech_stack": [],
            "min_annual_revenue": 0,
            "description": "Offering.",
        }

        result = self.score({"annual_revenue": 25}, offering)

        self.assertEqual(result["rubric_breakdown"]["revenue_fit"], 100.0)


if __name__ == "__main__":
    unittest.main()
