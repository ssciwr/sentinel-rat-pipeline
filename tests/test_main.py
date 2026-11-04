"""Tests for the batch callback in main."""

from unittest.mock import MagicMock

import pytest

from sentinel_rat_pipeline import main
from sentinel_rat_pipeline.config import Settings


@pytest.fixture
def mocks(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, MagicMock]:
    monkeypatch.setattr(main, "settings", Settings(watch_folder="/data/images"))
    run_analysis = MagicMock()
    persist = MagicMock()
    monkeypatch.setattr(main, "run_analysis_remote", run_analysis)
    monkeypatch.setattr(main, "persist_results", persist)
    return run_analysis, persist


def test_on_new_images_analyzes_batch_and_persists_each(mocks) -> None:
    run_analysis, persist = mocks
    run_analysis.return_value = [
        {"image_path": "cam01/b.jpg", "detections": []},
        {"image_path": "a.jpg", "detections": []},
    ]

    main.on_new_images(["/data/images/a.jpg", "/data/images/cam01/b.jpg"])

    run_analysis.assert_called_once_with(["a.jpg", "cam01/b.jpg"])
    # host-relative paths are stored, the files are read at their container path
    assert [(c.args, c.kwargs) for c in persist.call_args_list] == [
        (
            ("a.jpg", {"image_path": "a.jpg", "detections": []}),
            {"file_path": "/data/images/a.jpg"},
        ),
        (
            ("cam01/b.jpg", {"image_path": "cam01/b.jpg", "detections": []}),
            {"file_path": "/data/images/cam01/b.jpg"},
        ),
    ]


def test_on_new_images_skips_paths_outside_watch_folder(mocks) -> None:
    run_analysis, persist = mocks
    run_analysis.return_value = [{"image_path": "a.jpg", "detections": []}]

    main.on_new_images(["/elsewhere/b.jpg", "/data/images/a.jpg"])

    run_analysis.assert_called_once_with(["a.jpg"])
    assert [c.args[0] for c in persist.call_args_list] == ["a.jpg"]


def test_on_new_images_analysis_failure_persists_nothing(mocks) -> None:
    run_analysis, persist = mocks
    run_analysis.side_effect = RuntimeError("ML service down")

    main.on_new_images(["/data/images/a.jpg"])

    persist.assert_not_called()


def test_on_new_images_continues_after_missing_or_failing_result(mocks) -> None:
    run_analysis, persist = mocks
    run_analysis.return_value = [
        {"image_path": "a.jpg", "detections": []},
        {"image_path": "c.jpg", "detections": []},
    ]
    persist.side_effect = [LookupError("unknown camera"), 1]

    main.on_new_images(
        ["/data/images/a.jpg", "/data/images/b.jpg", "/data/images/c.jpg"]
    )

    # b.jpg has no result, a.jpg fails to persist, c.jpg is still persisted
    assert [c.args[0] for c in persist.call_args_list] == ["a.jpg", "c.jpg"]


def test_on_new_images_splits_batch_larger_than_max_size(
    mocks, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        main, "settings", Settings(watch_folder="/data/images", max_batch_size=2)
    )
    run_analysis, persist = mocks
    run_analysis.side_effect = lambda paths: [
        {"image_path": p, "detections": []} for p in paths
    ]

    main.on_new_images([f"/data/images/{name}.jpg" for name in "abcde"])

    assert [c.args[0] for c in run_analysis.call_args_list] == [
        ["a.jpg", "b.jpg"],
        ["c.jpg", "d.jpg"],
        ["e.jpg"],
    ]
    assert persist.call_count == 5


def test_on_new_images_failed_chunk_does_not_stop_others(
    mocks, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        main, "settings", Settings(watch_folder="/data/images", max_batch_size=1)
    )
    run_analysis, persist = mocks
    run_analysis.side_effect = [
        RuntimeError("ML service down"),
        [{"image_path": "b.jpg", "detections": []}],
    ]

    main.on_new_images(["/data/images/a.jpg", "/data/images/b.jpg"])

    assert [c.args[0] for c in persist.call_args_list] == ["b.jpg"]


@pytest.mark.parametrize("process_backlog", [True, False])
def test_main_passes_backlog_flag_to_watcher(
    tmp_path, monkeypatch: pytest.MonkeyPatch, process_backlog: bool
) -> None:
    monkeypatch.setattr(
        main,
        "settings",
        Settings(
            watch_folder=str(tmp_path),
            database_url="postgresql://fake:5432/db",
            process_backlog=process_backlog,
        ),
    )
    start_watch = MagicMock()
    monkeypatch.setattr(main, "start_watch", start_watch)

    main.main()

    start_watch.assert_called_once_with(
        main.on_new_images,
        on_deleted=main.on_image_deleted,
        process_backlog=process_backlog,
        backlog_filter=main._unprocessed_images,
    )


def test_unprocessed_images_drops_stored_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main, "settings", Settings(watch_folder="/data/images"))
    existing = MagicMock(return_value={"a.jpg"})
    monkeypatch.setattr(main, "existing_image_paths", existing)
    restore_missing = MagicMock()
    monkeypatch.setattr(main, "restore_missing_images", restore_missing)

    remaining = main._unprocessed_images(["/data/images/a.jpg", "/data/images/b.jpg"])

    existing.assert_called_once_with(["a.jpg", "b.jpg"])
    assert remaining == ["/data/images/b.jpg"]
    # a stored image found at startup is in the folder, so it is not missing
    restore_missing.assert_called_once_with(["a.jpg"])


def test_unprocessed_images_drops_images_outside_watch_folder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main, "settings", Settings(watch_folder="/data/images"))
    existing = MagicMock(return_value=set())
    monkeypatch.setattr(main, "existing_image_paths", existing)
    restore_missing = MagicMock()
    monkeypatch.setattr(main, "restore_missing_images", restore_missing)

    remaining = main._unprocessed_images(["/elsewhere/a.jpg", "/data/images/b.jpg"])

    existing.assert_called_once_with(["b.jpg"])
    assert remaining == ["/data/images/b.jpg"]
    restore_missing.assert_not_called()


def test_to_host_relative_accepts_resolved_watch_folder(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "real").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "real")
    monkeypatch.setattr(
        main, "settings", Settings(watch_folder=str(tmp_path / "link"))
    )

    assert main._to_host_relative(str(tmp_path / "link" / "a.jpg")) == "a.jpg"
    assert main._to_host_relative(str(tmp_path / "real" / "a.jpg")) == "a.jpg"
