from fogbench import splits


def test_satu_fold_per_pasien(recordings):
    folds = splits.loso_folds({r.subject for r in recordings})
    assert len(folds) == 6
    assert [f.test_subject for f in folds] == [1, 2, 3, 4, 5, 6]


def test_fold_selalu_sama(recordings):
    subs = {r.subject for r in recordings}
    a = splits.loso_folds(subs)
    b = splits.loso_folds(list(subs)[::-1])
    assert a == b, "urutan fold harus sama walau urutan masukan berbeda"


def test_pasien_uji_tidak_ada_di_latih(recordings):
    for fold in splits.loso_folds({r.subject for r in recordings}):
        train, test = splits.split_recordings(recordings, fold)
        assert {r.subject for r in train}.isdisjoint({r.subject for r in test})
