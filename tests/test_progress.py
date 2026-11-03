from dactilograph.progress import Progress


def test_missing_file_gives_empty_progress(tmp_path):
    progress = Progress.load(tmp_path / "no-existe.json")
    assert progress.learned == set()
    assert progress.high_score == 0


def test_corrupt_file_gives_empty_progress(tmp_path):
    path = tmp_path / "progreso.json"
    path.write_text("{no es json", encoding="utf-8")
    assert Progress.load(path) == Progress()


def test_progress_survives_a_save_and_load(tmp_path):
    path = tmp_path / "carpeta" / "progreso.json"
    progress = Progress()
    progress.learn("Ñ")
    progress.record_score(12)
    progress.save(path)
    assert Progress.load(path) == Progress({"Ñ"}, 12)


def test_learning_a_letter_twice_is_not_new():
    progress = Progress()
    assert progress.learn("A")
    assert not progress.learn("A")


def test_only_a_higher_score_is_a_record():
    progress = Progress(high_score=10)
    assert not progress.record_score(10)
    assert progress.record_score(11)
    assert progress.high_score == 11
