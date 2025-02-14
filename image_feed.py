import cv2
import os

class ImageFeed:
    def __init__(self, image_folder, loop=False):
        self.image_folder = image_folder
        self.valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
        self.images = sorted([img for img in os.listdir(image_folder) if img.lower().endswith(self.valid_extensions)])
        self.index = 0
        self.loop = loop
        self.opened = len(self.images) > 0  # Check if images exist

        if self.opened:
            first_image = cv2.imread(os.path.join(image_folder, self.images[0]))
            if first_image is not None:
                self.height, self.width, self.layers = first_image.shape
            else:
                self.opened = False  # If first image is None, mark as closed

    def isOpened(self):
        """Simulates cap.isOpened() - returns True if there are images to read"""
        return self.opened

    def read(self):
        """Simulates cap.read() - returns (True, image) if available, else (False, None)"""
        if self.index >= len(self.images):
            if self.loop:
                self.index = 0  # Restart if looping is enabled
            else:
                return False, None  # End of sequence

        img_path = os.path.join(self.image_folder, self.images[self.index])
        frame = cv2.imread(img_path)
        self.index += 1
        return True, frame

    def set(self, prop_id, value):
        """Simulates cap.set() - only supports resetting frame index"""
        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            self.index = int(value)

    def release(self):
        """Simulates cap.release()"""
        self.opened = False  # Mark as closed