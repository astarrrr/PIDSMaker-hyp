from pidsmaker.detection.evaluation_methods.evaluation_utils import normalize_tw_malicious_nodes


def test_normalize_tw_malicious_nodes_accepts_internal_node_ids():
    tw_to_malicious_nodes = {
        0: ["123", "123", "456"],
    }

    normalized = normalize_tw_malicious_nodes(
        tw_to_malicious_nodes=tw_to_malicious_nodes,
        uuid_to_node_id={},
    )

    assert normalized == {0: {"123": 2, "456": 1}}


def test_normalize_tw_malicious_nodes_merges_uuid_and_internal_node_id_counts():
    tw_to_malicious_nodes = {
        1: ["UUID-A", "321", "UUID-A", "UNKNOWN-UUID"],
    }

    normalized = normalize_tw_malicious_nodes(
        tw_to_malicious_nodes=tw_to_malicious_nodes,
        uuid_to_node_id={"UUID-A": "321"},
    )

    assert normalized == {1: {"321": 3}}
