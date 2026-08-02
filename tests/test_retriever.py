"""Evidence retrieval — the mechanism the whole system rests on."""

from tests.conftest import message

from attention_router.retriever import EvidenceRetriever, tokenize

LISTING = "Selling a denim jacket, size M. Pickup near Gate 2 this weekend."


class TestTokenizer:
    def test_drops_stopwords_and_short_tokens(self):
        tokens = tokenize("The bus is at the gate by 7")
        assert "the" not in tokens and "is" not in tokens
        assert "bus" in tokens and "gate" in tokens

    def test_case_and_punctuation_insensitive(self):
        assert tokenize("Gate-2, PICKUP!") == tokenize("gate 2 pickup")


class TestPersonalisation:
    """Identical text, identical sender, opposite outcomes."""

    def test_same_message_yields_opposite_verdicts(self, tiny_dataset):
        retriever = EvidenceRetriever(tiny_dataset)
        engaged = retriever.retrieve(message(
            user_id="u_engaged", sender_user_id="u_seller", message_text=LISTING))
        rejecting = retriever.retrieve(message(
            user_id="u_rejecting", sender_user_id="u_seller", message_text=LISTING))

        assert engaged.tier == rejecting.tier == "counterpart"
        assert set(engaged.message_ids).isdisjoint(rejecting.message_ids)
        assert engaged.signal()["verdict"] != rejecting.signal()["verdict"]
        assert rejecting.signal()["verdict"] == "actively_rejected"

    def test_rejection_is_strong_evidence(self, tiny_dataset):
        retriever = EvidenceRetriever(tiny_dataset)
        result = retriever.retrieve(message(
            user_id="u_rejecting", sender_user_id="u_seller", message_text=LISTING))
        assert result.strength() == "strong"


class TestCitationCount:
    def test_repeated_rejection_cites_a_pattern(self, tiny_dataset):
        """Two rows that agree on rejection demonstrate a pattern, so cite both."""
        retriever = EvidenceRetriever(tiny_dataset)
        result = retriever.retrieve(message(
            user_id="u_rejecting", sender_user_id="u_seller", message_text=LISTING))
        assert len(result.message_ids) == 2

    def test_positive_engagement_cites_one(self, tiny_dataset):
        """Engagement is not a 'pattern of rejection', so one citation suffices."""
        retriever = EvidenceRetriever(tiny_dataset)
        result = retriever.retrieve(message(
            user_id="u_engaged", sender_user_id="u_seller", message_text=LISTING))
        assert len(result.message_ids) == 1


class TestTiersAndGaps:
    def test_unknown_sender_finds_nothing(self, tiny_dataset):
        retriever = EvidenceRetriever(tiny_dataset)
        result = retriever.retrieve(message(
            user_id="u_engaged", sender_user_id="u_stranger", message_text="hello"))
        assert result.tier != "counterpart"

    def test_no_evidence_means_no_signal(self, tiny_dataset):
        retriever = EvidenceRetriever(tiny_dataset)
        result = retriever.retrieve(message(
            user_id="u_nobody", sender_user_id="u_stranger", message_text="hello"))
        assert result.evidence == []
        assert result.signal()["verdict"] == "no_signal"
        assert result.strength() == "none"

    def test_a_message_never_cites_itself(self, tiny_dataset):
        retriever = EvidenceRetriever(tiny_dataset)
        existing = next(iter(tiny_dataset.history.values()))
        result = retriever.retrieve(existing)
        assert existing["message_id"] not in result.message_ids


class TestDeterminism:
    def test_repeated_retrieval_is_identical(self, tiny_dataset):
        retriever = EvidenceRetriever(tiny_dataset)
        query = message(user_id="u_rejecting", sender_user_id="u_seller",
                        message_text=LISTING)
        runs = [retriever.retrieve(query).message_ids for _ in range(5)]
        assert all(run == runs[0] for run in runs)
