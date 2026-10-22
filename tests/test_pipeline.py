import numpy as np

from scanlag.pipeline.flags import shape_flags, track_flags
from scanlag.pipeline.plates import Plate, circle_mask, crop
from scanlag.pipeline.scans import SCAN_NAME


def test_scan_name_parsing():
    m = SCAN_NAME.match("epec-amp7_1-5.1u_20260224_2025.tiff")
    assert m["prefix"] == "epec-amp7" and m["scanner"] == "1-5.1u"
    m = SCAN_NAME.match("vapc_scan_1-1.2_20250807_1511.tif")
    assert m["prefix"] == "vapc_scan" and m["scanner"] == "1-1.2"
    assert SCAN_NAME.match("vapc_scan_1-3_20250806_1726_motions.mat") is None


def test_track_flags():
    smooth = np.array([0, 0, 10, 14, 20, 28, 40, 55, 70])
    assert track_flags(smooth, np.zeros(9, bool)) == []
    gap = smooth.copy()
    gap[5] = 0
    assert any(f.startswith("gap") for f in track_flags(gap, np.zeros(9, bool)))
    shrink = smooth.copy()
    shrink[6] = 15
    assert any(f.startswith("shrinks") for f in track_flags(shrink, np.zeros(9, bool)))
    jump = smooth.copy()
    jump[7:] = [200, 210]
    assert any(f.startswith("jump") for f in track_flags(jump, np.zeros(9, bool)))
    merged = np.zeros(9, bool)
    merged[7:] = True
    assert track_flags(jump, merged) == []  # the jump is a merge that lasts, already marked as such
    flicker = np.zeros(9, bool)
    flicker[6] = True
    assert any(f.startswith("transient merge") for f in track_flags(smooth, flicker))


def test_shape_flags():
    from skimage.measure import regionprops

    disk = np.zeros((40, 40), int)
    yy, xx = np.mgrid[:40, :40]
    disk[(yy - 20) ** 2 + (xx - 20) ** 2 <= 64] = 1
    assert shape_flags(regionprops(disk)[0]) == []
    line = np.zeros((40, 40), int)
    line[18:22, 2:38] = 1
    assert any(f.startswith("elongated") for f in shape_flags(regionprops(line)[0]))


def test_crop_and_mask():
    img = np.arange(100).reshape(10, 10)
    c = crop(img, -2, 8, 4)
    assert c.shape == (4, 4) and c[0, 2] == img[8, 0] and c[0, 0] == 0
    m = circle_mask(100, Plate(1, 50.0, 50.0, 50.0), 0.8)
    assert m[50, 50] and not m[50, 5] and abs(m.sum() - np.pi * 40**2) < 200


def test_find_scanners_groups_and_names(tmp_path):
    from scanlag.pipeline.scans import find_scanners

    for d, scanner, stamps in (("T2", "1-3", ["20250806_1726", "20250806_1926"]), ("T24", "1-1.2", ["20250807_1511"])):
        (tmp_path / d / "ROI" / "sep_1").mkdir(parents=True)
        for s in stamps:
            (tmp_path / d / f"vapc_scan_{scanner}_{s}.tif").write_bytes(b"")
        (tmp_path / d / "ROI" / "sep_1" / f"vapc_scan_{scanner}_{stamps[0]}_1.tif").write_bytes(b"")  # MATLAB crop
    (tmp_path / "T2" / "run_vapc_scan20250806_1726.log").write_text("2025/08/06 17:26:30 - Starting.\n")
    scanners = {s.name: s for s in find_scanners(tmp_path)}
    assert set(scanners) == {"1-3", "1-1.2"}
    assert len(scanners["1-3"].scans) == 2 and scanners["1-3"].times_min == [0.0, 120.0]
