import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import argparse
import sys
import pickle  # Import the pickle module

from similarity_metrics.fsim_quality import FSIMsimilarity
from similarity_metrics.issm_quality import ISSMsimilarity
from CopulaSimilarity.CSM import CopulaBasedSimilarity as CSMSimilarity
from similarity_metrics.ssim_rmse_psnr_metrics import ssim

class SimilarityMetrics:
    def __init__(self, patch_size=8):
        self.fsim_similarity = FSIMsimilarity()
        self.issm_similarity = ISSMsimilarity()
        self.copula_similarity = CSMSimilarity(patch_size=patch_size)

    def compute_ssim(self, reference_frame, current_frame):
        return ssim(reference_frame, current_frame)

    def compute_fsim(self, reference_frame, current_frame):
        return self.fsim_similarity.fsim(reference_frame, current_frame)

    def compute_issm(self, reference_frame, current_frame):
        return self.issm_similarity.issm(reference_frame, current_frame)

    def compute_csm(self, reference_frame, current_frame):
        csm = self.copula_similarity.compute_local_similarity(reference_frame, current_frame)
        return csm, np.mean(csm)

def update_progress_bar(total_frames, processed_frames, bar_length=50):
    percent_complete = (processed_frames / total_frames) * 100
    num_hashes = int((percent_complete / 100) * bar_length)
    bar = '#' * num_hashes + '-' * (bar_length - num_hashes)
    sys.stdout.write(f'\rProgress: |{bar}| {percent_complete:.2f}%')
    sys.stdout.flush()

def process_video(video_path, resolution_factor, output_video, ssim, fsim, issm, save_final_frame, show_live_window, patch_size):
    metrics = SimilarityMetrics(patch_size=patch_size)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Total frames: {total_frames}')

    ret, reference_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        return

    ref_resized = cv2.resize(
        reference_frame, 
        (reference_frame.shape[1] // resolution_factor, 
         reference_frame.shape[0] // resolution_factor)
    )
    ref_rgb = cv2.cvtColor(ref_resized, cv2.COLOR_BGR2RGB)  

    ssim_results = []
    fsim_results = []
    issm_results = []
    csm_results = []
    csm_maps = []  # To store CSM maps for each frame
    frame_indices = []

    fig = plt.figure(figsize=(20, 12)) 
    
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.5])  

    # Axes for the first row
    video_ax = fig.add_subplot(gs[0, 0])
    diff_ax = fig.add_subplot(gs[0, 1])
    csm_ax = fig.add_subplot(gs[0, 2])
    
    plot_ax = fig.add_subplot(gs[1, :])
    
    font_title = 24 
    font_labels = 20  
    font_ticks = 18  

    # Initialize the plots
    video_ax.set_title('(a)', fontsize=font_title)
    diff_ax.set_title('(b)', fontsize=font_title)
    csm_ax.set_title('(c)', fontsize=font_title)
    plot_ax.set_title('(d)', fontsize=font_title)
    
    plot_ax.plot([], [], 'g-', label='SSIM')
    plot_ax.plot([], [], 'b-', label='FSIM')
    plot_ax.plot([], [], 'm-', label='ISSM')
    plot_ax.plot([], [], 'r-', label='CSM')
    plot_ax.set_xlim(0, 100)
    plot_ax.set_ylim(0, 1)
    plot_ax.set_xlabel('Frame Index', fontsize=font_labels)
    plot_ax.set_ylabel('Metric Value', fontsize=font_labels)
    plot_ax.legend(loc='lower left', fontsize=font_labels)
    
    for ax in [video_ax, diff_ax, csm_ax, plot_ax]:
        ax.tick_params(axis='both', which='major', labelsize=font_ticks)

    writer = FFMpegWriter(fps=30, codec='libx264')

    with writer.saving(fig, output_video, dpi=100):
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_resized = cv2.resize(
                frame, 
                (ref_resized.shape[1], ref_resized.shape[0]) 
            )
            current_rgb = cv2.cvtColor(current_resized, cv2.COLOR_BGR2RGB)
            ssim_value = metrics.compute_ssim(ref_rgb, current_rgb)
            fsim_value = metrics.compute_fsim(ref_rgb, current_rgb)
            issm_value = metrics.compute_issm(ref_rgb, current_rgb)
            csm_map, csm_mean = metrics.compute_csm(ref_rgb, current_rgb)
            diff_frame = cv2.absdiff(current_rgb, ref_rgb)

            frame_indices.append(frame_count)
            if ssim:
                ssim_results.append(ssim_value)
            if fsim:
                fsim_results.append(fsim_value)
            if issm:
                issm_results.append(issm_value)
            csm_results.append(csm_mean)
            csm_maps.append(csm_map)  # Save the CSM map for later use

            video_ax.imshow(current_rgb)
            video_ax.axis('off')

            diff_rgb = cv2.cvtColor(diff_frame, cv2.COLOR_BGR2RGB)
            diff_ax.imshow(diff_rgb)
            diff_ax.axis('off')

            csm_map_resized = cv2.resize(csm_map, (current_resized.shape[1], current_resized.shape[0]))  
            csm_ax.imshow(csm_map_resized, cmap='hot', interpolation='nearest')
            csm_ax.axis('off')

            plot_ax.clear()
            if ssim:
                plot_ax.plot(frame_indices, ssim_results, 'g-', label='SSIM')
            if fsim:
                plot_ax.plot(frame_indices, fsim_results, 'b-', label='FSIM')
            if issm:
                plot_ax.plot(frame_indices, issm_results, 'm-', label='ISSM')
            plot_ax.plot(frame_indices, csm_results, 'r-', label='CSM')
            plot_ax.set_xlim(0, len(frame_indices))
            plot_ax.set_ylim(0, 1)
            plot_ax.set_xlabel('Frame Index', fontsize=font_labels)
            plot_ax.set_ylabel('Metric Value', fontsize=font_labels)
            plot_ax.set_title('(d)', fontsize=font_title)
            plot_ax.legend(loc='lower left', fontsize=font_labels)

            plt.draw()
            writer.grab_frame()

            if show_live_window:
                plt.pause(0.001)

            update_progress_bar(total_frames, frame_count)

            frame_count += 1

    if save_final_frame:
        fig.savefig('final_frame.pdf', bbox_inches='tight')

    cap.release()
    plt.close(fig)

    ''' #pickle data saving data (optional)
    data = {
        'frame_indices': frame_indices,
        'ssim_results': ssim_results,
        'fsim_results': fsim_results,
        'issm_results': issm_results,
        'csm_results': csm_results,
        'csm_maps': csm_maps 
    }

    with open('similarity_metrics_data.pkl', 'wb') as f:
        pickle.dump(data, f)

    print("Data has been saved to similarity_metrics_data.pkl")
    '''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Video analysis')
    parser.add_argument('--path_to_video', help='Path to video file')
    parser.add_argument('--output_video_path', help='Path to output video file')
    parser.add_argument('--ssim', action='store_true', help='Compute SSIM')
    parser.add_argument('--fsim', action='store_true', help='Compute FSIM')
    parser.add_argument('--issm', action='store_true', help='Compute ISSM')
    parser.add_argument('--save_final_frame', action='store_true', help='Save final frame')
    parser.add_argument('--show_live_window', action='store_true', help='Show live window')
    parser.add_argument('--resolution_factor', type=int, default=8, help='Resolution factor')
    parser.add_argument('--patch_size', type=int, default=8, help='Patch size for CopulaBasedSimilarity') 
    args = parser.parse_args()

    process_video(args.path_to_video, args.resolution_factor, args.output_video_path, args.ssim, args.fsim, args.issm, args.save_final_frame, args.show_live_window, args.patch_size)
