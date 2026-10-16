"""Window handling shared by the capture and check views."""

import cv2

# Added in OpenCV 4.5.2; older builds simply do not get raised.
TOPMOST = getattr(cv2, "WND_PROP_TOPMOST", None)


def open_window(name, size=None):
    """A resizable window, sized if a size is given."""
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    if size:
        cv2.resizeWindow(name, *size)


def bring_to_front(name):
    """Raise the window above whatever is covering it.

    Turning the always-on-top hint on and straight back off raises the window and then
    lets it behave like any other -- it does not stay pinned over your other work. Call
    it after the first frame is shown: an empty window has nothing to raise yet. The Qt
    backend has no getter for the property, so there is nothing to read back.
    """
    if TOPMOST is None:
        return False
    try:
        cv2.setWindowProperty(name, TOPMOST, 1)
        cv2.waitKey(1)
        cv2.setWindowProperty(name, TOPMOST, 0)
    except cv2.error:
        return False
    return True
