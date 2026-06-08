from inference.common.redis_list_transport import normalize_incoming_message


def test_normalize_task_already_wrapped():
    raw = {"type": "task", "task_id": "t1", "task_data": {"type": "image", "id": "t1"}}
    msg = normalize_incoming_message(raw)
    assert msg["type"] == "task"
    assert msg["task_id"] == "t1"
    assert isinstance(msg["task_data"], dict)


def test_normalize_cancel():
    raw = {"type": "cancel", "task_id": "t2"}
    msg = normalize_incoming_message(raw)
    assert msg["type"] == "cancel"
    assert msg["task_id"] == "t2"
    assert "timestamp" in msg


def test_normalize_task_data_without_type():
    raw = {"type": "image", "id": "t3", "params": {"job_type": "MK"}}
    msg = normalize_incoming_message(raw)
    assert msg["type"] == "task"
    assert msg["task_id"] == "t3"
    assert msg["task_data"]["id"] == "t3"

