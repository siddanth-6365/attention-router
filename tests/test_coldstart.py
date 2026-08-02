"""Priors for senders with no history with this receiver."""

from tests.conftest import event, message

from attention_router import coldstart


class TestColdDetection:
    def test_known_sender_is_not_cold(self, tiny_dataset):
        assert not coldstart.is_cold(tiny_dataset, message(
            user_id="u_engaged", sender_user_id="u_seller"))

    def test_unknown_sender_is_cold(self, tiny_dataset):
        assert coldstart.is_cold(tiny_dataset, message(
            user_id="u_engaged", sender_user_id="u_stranger"))

    def test_no_counterpart_at_all_is_cold(self, tiny_dataset):
        assert coldstart.is_cold(tiny_dataset, message(sender_user_id="", business_id=""))


class TestPriorNeverCompetesWithHistory:
    def test_direct_history_suppresses_the_prior(self, tiny_dataset):
        prior = coldstart.compute(tiny_dataset, message(
            user_id="u_engaged", sender_user_id="u_seller"), has_direct_history=True)
        assert not prior.available

    def test_absent_prior_still_gives_usable_guidance(self, tiny_dataset):
        prior = coldstart.compute(tiny_dataset, message(
            user_id="u_engaged", sender_user_id="u_nobody"))
        assert not prior.available
        assert "no prior information" in prior.describe().lower()


class TestSenderReputation:
    """Only abuse crosses the recipient boundary.

    Whether you want a given person's messages is a fact about your
    relationship with them, not about them. Measured on the corpus, barely a
    fifth of personal senders provoke the same reaction from everyone, and an
    earlier version that borrowed engagement preferences across recipients
    measurably reduced accuracy. Fraud is different: a phishing script targets
    everyone, so a reported sender stays reported.
    """

    def test_engagement_preference_does_not_transfer(self, tiny_dataset):
        """u_seller is read by one user and muted by another - unusable as a prior."""
        prior = coldstart.compute(tiny_dataset, message(
            user_id="u_new", sender_user_id="u_seller"))
        assert not prior.available

    def test_the_receivers_own_rows_are_excluded(self, tiny_dataset):
        """A prior must describe other people, or it is just history again."""
        prior = coldstart.compute(tiny_dataset, message(
            user_id="u_rejecting", sender_user_id="u_seller"))
        assert prior.observations < coldstart.MIN_OBSERVATIONS or not prior.available

    def test_reported_senders_read_as_risky(self, tiny_dataset):
        for index in range(4):
            hid = f"r_{index}"
            tiny_dataset.history[hid] = {
                "message_id": hid, "user_id": f"u_victim{index}",
                "conversation_type": "personal", "group_id": "", "business_id": "",
                "sender_user_id": "u_fraud", "created_at": "2026-06-01 10:00",
                "message_text": "share your OTP", "media_type": "", "media_id": "",
                "forwarded_count": "0",
            }
            tiny_dataset.events[(f"u_victim{index}", hid)] = event(
                hid, f"u_victim{index}", "reports")
        prior = coldstart.compute(tiny_dataset, message(
            user_id="u_engaged", sender_user_id="u_fraud"))
        assert prior.leaning == "risky"


class TestThreshold:
    def test_too_few_observations_yields_nothing(self, tiny_dataset):
        """One data point is noise, not a prior."""
        hid = "solo"
        tiny_dataset.history[hid] = {
            "message_id": hid, "user_id": "u_other", "conversation_type": "personal",
            "group_id": "", "business_id": "", "sender_user_id": "u_rare",
            "created_at": "2026-06-01 10:00", "message_text": "hi",
            "media_type": "", "media_id": "", "forwarded_count": "0",
        }
        tiny_dataset.events[("u_other", hid)] = event(hid, "u_other", "acts_fast")
        assert not coldstart.compute(tiny_dataset, message(
            user_id="u_engaged", sender_user_id="u_rare")).available


class TestDescription:
    def test_a_prior_labels_itself_as_weaker_than_history(self, tiny_dataset):
        for index in range(4):
            hid = f"d_{index}"
            tiny_dataset.history[hid] = {
                "message_id": hid, "user_id": f"u_x{index}",
                "conversation_type": "personal", "group_id": "", "business_id": "",
                "sender_user_id": "u_abuser", "created_at": "2026-06-01 10:00",
                "message_text": "send the OTP", "media_type": "", "media_id": "",
                "forwarded_count": "0",
            }
            tiny_dataset.events[(f"u_x{index}", hid)] = event(hid, f"u_x{index}", "reports")

        prior = coldstart.compute(tiny_dataset, message(
            user_id="u_engaged", sender_user_id="u_abuser"))
        assert prior.available
        text = prior.describe().lower()
        assert "prior" in text and "no direct history" in text
