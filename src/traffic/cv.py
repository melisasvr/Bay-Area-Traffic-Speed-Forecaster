"""Rolling-origin (expanding-window) CV. Test windows tile the end of the series,
each preceded by a validation block (early stopping). Targets never cross a split
boundary (see FeatureBuilder.sample_idx), which acts as the purge/gap."""


def expanding_window_folds(n, n_folds, val_len, test_len, min_train):
    for i in range(n_folds):
        test_end = n - (n_folds - 1 - i) * test_len
        val_end = test_end - test_len
        train_end = val_end - val_len
        if train_end < min_train:
            raise ValueError("Not enough data: reduce folds / val_len / test_len")
        yield train_end, val_end, test_end
