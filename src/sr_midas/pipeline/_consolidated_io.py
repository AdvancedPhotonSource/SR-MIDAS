"""
Binary I/O for MIDAS consolidated peak files.

Replicates the binary format defined in PeaksFittingConsolidatedIO.h
so that downstream MIDAS stages (MergeOverlappingPeaksAllZarr, etc.)
can read SR-MIDAS output directly.
"""

import struct
import numpy as np

N_PEAK_COLS = 29


def write_allpeaks_ps_bin(filepath, nr_frames, frame_peak_data):
    """Write AllPeaks_PS.bin in MIDAS consolidated format.

    Args:
        filepath: Output file path.
        nr_frames: Total number of frames in the dataset.
        frame_peak_data: List of length nr_frames.  Each element is either
            a np.ndarray of shape (n_peaks, 29) with dtype float64, or
            None for frames with 0 peaks.

    Binary layout (matches PeaksFittingConsolidatedIO.h):
        Header:
            int32   nFrames
            int32   nPeaks[nFrames]
            int64   offsets[nFrames]
        Data:
            For each frame with peaks: nPeaks * 29 float64 values (row-major).
    """
    n_peaks_arr = np.zeros(nr_frames, dtype=np.int32)
    for f in range(nr_frames):
        if frame_peak_data[f] is not None:
            n_peaks_arr[f] = frame_peak_data[f].shape[0]

    # header_size = sizeof(int32) + nFrames*sizeof(int32) + nFrames*sizeof(int64)
    header_size = 4 + nr_frames * 4 + nr_frames * 8

    offsets = np.zeros(nr_frames, dtype=np.int64)
    data_off = header_size
    for f in range(nr_frames):
        offsets[f] = data_off
        data_off += int(n_peaks_arr[f]) * N_PEAK_COLS * 8  # 8 bytes per float64

    with open(filepath, 'wb') as fh:
        fh.write(struct.pack('<i', nr_frames))
        fh.write(n_peaks_arr.tobytes())
        fh.write(offsets.tobytes())
        for f in range(nr_frames):
            if frame_peak_data[f] is not None and n_peaks_arr[f] > 0:
                data = np.ascontiguousarray(frame_peak_data[f], dtype=np.float64)
                fh.write(data.tobytes())


def write_allpeaks_px_bin(filepath, nr_frames, nr_pixels, frame_pixel_data):
    """Write AllPeaks_PX.bin in MIDAS consolidated format.

    Args:
        filepath: Output file path.
        nr_frames: Total number of frames.
        nr_pixels: Detector dimension (max of height, width).
        frame_pixel_data: List of length nr_frames.  Each element is either
            a list of (pixel_y, pixel_z) tuples (one per peak), or None.
            pixel_y and pixel_z are 1-D int16 arrays of detector-row and
            detector-column indices for non-zero pixels in the peak region.

    Binary layout (matches PeaksFittingConsolidatedIO.h):
        Header:
            int32   nFrames
            int32   NrPixels
            int32   nPeaks[nFrames]
            int64   offsets[nFrames]
        Data:
            For each frame, for each peak:
                int32   nPixels
                int16   y,z pairs × nPixels  (interleaved: y0,z0,y1,z1,...)
    """
    n_peaks_arr = np.zeros(nr_frames, dtype=np.int32)
    for f in range(nr_frames):
        if frame_pixel_data[f] is not None:
            n_peaks_arr[f] = len(frame_pixel_data[f])

    # header_size = 2*sizeof(int32) + nFrames*sizeof(int32) + nFrames*sizeof(int64)
    header_size = 4 + 4 + nr_frames * 4 + nr_frames * 8

    offsets = np.zeros(nr_frames, dtype=np.int64)
    data_off = header_size
    for f in range(nr_frames):
        offsets[f] = data_off
        if frame_pixel_data[f] is not None:
            for (py, pz) in frame_pixel_data[f]:
                n_px = len(py)
                # int32 nPixels + nPx * 2 * sizeof(int16)
                data_off += 4 + n_px * 2 * 2

    with open(filepath, 'wb') as fh:
        fh.write(struct.pack('<i', nr_frames))
        fh.write(struct.pack('<i', nr_pixels))
        fh.write(n_peaks_arr.tobytes())
        fh.write(offsets.tobytes())
        for f in range(nr_frames):
            if frame_pixel_data[f] is not None:
                for (py, pz) in frame_pixel_data[f]:
                    n_px = len(py)
                    fh.write(struct.pack('<i', n_px))
                    interleaved = np.empty(n_px * 2, dtype=np.int16)
                    interleaved[0::2] = np.asarray(py, dtype=np.int16)
                    interleaved[1::2] = np.asarray(pz, dtype=np.int16)
                    fh.write(interleaved.tobytes())
