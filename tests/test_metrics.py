import numpy as np

from util import compute_miou, post_process_binary


def test_compute_miou_is_one_for_perfect_prediction():
    ground_truth = np.array([[0, 0], [1, 1]], dtype=np.uint8)
    result = compute_miou(ground_truth.copy(), ground_truth)
    assert np.isclose(result["miou"], 1.0)


def test_compute_miou_ignores_unlabelled_pixels():
    prediction = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    ground_truth = np.array([[0, 255], [1, 255]], dtype=np.uint8)
    result = compute_miou(prediction, ground_truth)
    assert np.isclose(result["miou"], 1.0)


def test_post_processing_returns_binary_mask():
    mask = np.array([[0, 2], [-1, 5]], dtype=np.int32)
    processed = post_process_binary(mask, open_ks=0, close_ks=0)
    assert processed.dtype == np.uint8
    assert set(np.unique(processed)) <= {0, 1}
