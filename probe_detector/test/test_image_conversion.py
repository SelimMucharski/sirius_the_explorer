from probe_detector.infrastructure.probe_detector_node import image_to_bgr
from sensor_msgs.msg import Image


def test_image_conversion_is_writable_bgr():
    for enc, px in (('rgb8', [1, 2, 3]), ('bgr8', [3, 2, 1]), ('rgba8', [1, 2, 3, 9]),
                    ('mono8', [7])):
        n = len(px)
        msg = Image(height=1, width=2, encoding=enc, step=2 * n)
        msg.data = bytes(px * 2)
        img = image_to_bgr(msg)
        assert img.shape == (1, 2, 3) and img.flags.writeable
        assert list(img[0, 0]) == ([7, 7, 7] if enc == 'mono8' else [3, 2, 1])
