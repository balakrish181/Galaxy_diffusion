import numpy as np
from PIL import Image
from torchvision import transforms

class ImageFFTComparer:
    def __init__(self, image_size=(256, 256)):
        self.img_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

    def read_image(self, image_path):
        img = Image.open(image_path).convert('L')
        img = self.img_transform(img)
        img = img.numpy()
        img = img.squeeze()
        return img

    def calculate_fft(self, image):
        fft_image = np.fft.fft2(image)
        fft_shifted = np.fft.fftshift(fft_image)
        magnitude_spectrum = np.abs(fft_shifted)
        return magnitude_spectrum

    def mean_squared_error(self, img1, img2):
        return np.mean((img1 - img2) ** 2)

    def mse_fft(self, image_path1, image_path2):
        img1 = self.read_image(image_path1)
        img2 = self.read_image(image_path2)
        fft_img1 = self.calculate_fft(img1)
        fft_img2 = self.calculate_fft(img2)
        mse = self.mean_squared_error(fft_img1, fft_img2)
        return mse

if __name__ == "__main__":
    pass

