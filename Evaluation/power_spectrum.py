import numpy as np
import matplotlib.pyplot as plt
import cv2
import Pk_library as PKL
from pathlib import Path
import argparse

class PowerSpectraAnalyzer:
    def __init__(self, image_paths1, image_paths2, box_size, MAS):
        self.image_paths1 = image_paths1
        self.image_paths2 = image_paths2
        self.box_size = box_size
        self.MAS = MAS

    def get_power_spectra(self, delta, box_size, MAS, array=False):
        BoxSize = box_size
        MAS     = MAS
        threads = 1

        # Compute the Pk of the image
        Pk2D = PKL.Pk_plane(delta, BoxSize, MAS, threads)

        # Get the attributes of the routine
        k      = Pk2D.k      # k in h/Mpc
        Pk     = Pk2D.Pk     # Pk in (Mpc/h)^2
        Nmodes = Pk2D.Nmodes # Number of modes in the different k bins
        
        return k, Pk, Nmodes

    def average_power_spectra(self, image_paths):
        k_list = []
        Pk_list = []
        
        for path in image_paths:
            # Read the images in grayscale (0 flag) and convert to float32
            delta = cv2.imread(str(path), 0).astype('float32')
            
            k, Pk, _ = self.get_power_spectra(delta, self.box_size, self.MAS, array=True)
            k_list.append(k)
            Pk_list.append(Pk)
        
        # Convert lists to numpy arrays for easier processing
        k_list = np.array(k_list)
        Pk_list = np.array(Pk_list)
        
        # Compute the average power spectrum
        average_Pk = np.mean(Pk_list, axis=0)
        
        # Assuming that k is the same for all images
        k = k_list[0] if len(k_list) > 0 else None

        return k, average_Pk

    def average_compare_power_spectra(self, k1, Pk1, k2, Pk2,output_path="power_spectra_comparison.png"):
        plt.figure()
        plt.loglog(k1, Pk1, label='Original')
        plt.loglog(k2, Pk2, label='Generated', linestyle='--')
        plt.xlabel('k (h/Mpc)')
        plt.ylabel('log Power Spectrum P(k) [(Mpc/h)^2]')
        plt.legend()
        plt.savefig(output_path)
        plt.close()

    def mean_absolute_fractional_difference(self, k1, average_Pk1, k2, average_Pk2):
        if not np.array_equal(k1, k2):
            raise ValueError("The wavenumbers k1 and k2 must be identical.")
        
        fractional_diff = np.abs(average_Pk2 - average_Pk1) / average_Pk1
        
        mean_abs_frac_diff = np.mean(fractional_diff)
        
        return mean_abs_frac_diff

    def run_analysis(self):
        # Compute the average power spectrum for both sets of images
        k1, average_Pk1 = self.average_power_spectra(self.image_paths1)
        k2, average_Pk2 = self.average_power_spectra(self.image_paths2)
        
        # Compare and plot the power spectra
        self.average_compare_power_spectra(k1, average_Pk1, k2, average_Pk2)
        self.save_power_spectrum_ratio(k1,average_Pk1,k2,average_Pk2)
        
        # Calculate the mean absolute fractional difference
        mean_abs_frac_diff = self.mean_absolute_fractional_difference(k1, average_Pk1, k2, average_Pk2)
        print(f"Mean Absolute Fractional Difference: {mean_abs_frac_diff}")

    def save_power_spectrum_ratio(self, k1, Pk1, k2, Pk2, output_path="power_spectrum_ratio.png"):
        if not np.array_equal(k1, k2):
            raise ValueError("The wavenumbers k1 and k2 must be identical.")

        # Calculate the ratio of the original power spectrum to the generated power spectrum
        ratio = Pk1 / Pk2

        # Plot the ratio
        plt.figure()
        plt.xscale('log')
        plt.plot(k1, ratio, label='Original/Generated Ratio')
        #plt.hlines(y=np.ones(k1.size))
        plt.xlabel('k (h/Mpc)')
        plt.ylabel('Power Spectrum Ratio P_original(k) / P_generated(k)')
        plt.legend()

        # Save the plot to a file
        plt.savefig(output_path)
        plt.close()  # Close the figure to free memory

        print(f"Power spectrum ratio plot saved to {output_path}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Analyze power spectra of images.")
    parser.add_argument("image_dir1", type=str, help="Directory path for the first set of images (original images)")
    parser.add_argument("image_dir2", type=str, help="Directory path for the second set of images (Generated images)")
    parser.add_argument("--box_size", type=float, default=1000, help="Box size for the power spectra analysis")
    parser.add_argument("--MAS", type=str, default='CIC', help="Mass Assignment Scheme (e.g., 'CIC')")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    image_paths1 = list(Path(args.image_dir1).glob('*.png'))
    image_paths2 = list(Path(args.image_dir2).glob('*.png'))
    
    analyzer = PowerSpectraAnalyzer(image_paths1, image_paths2, args.box_size, args.MAS)
    analyzer.run_analysis()
