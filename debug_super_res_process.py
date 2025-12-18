import numpy as np #type:ignore
import pandas as pd #type: ignore
import zarr #type: ignore
from scipy.ndimage import label, find_objects, gaussian_filter, median_filter #type:ignore
from scipy.optimize import curve_fit #type:ignore
from skimage.feature import peak_local_max #type:ignore
from skimage.segmentation import watershed #type:ignore
from skimage.measure import regionprops #type:ignore

import torch #type:ignore
import torch.nn as nn #type:ignore
from torch.amp import autocast #type:ignore

import os, time, datetime, logging, argparse, json, math
from copy import deepcopy

SEP = os.sep


# Setup logging
def setup_logging(resultDir):
    """Setup logging configuration with timestamped log file
    Args:
        midas_layer_dir (str): Path to the MIDAS layer directory
    Returns:
        logger (logging.Logger): Logger object for logging messages
    """

    log_dir = f"{resultDir}SRlogs{SEP}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filepath = f"{log_dir}peakfit_execution_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized for super-resolution workflow. Log filepath: {log_filepath}")
    return logger



# ----------------------------
def compute_diffraction_rings(spacegroup, a, wavelength, Lsd, max_index=5):
    """The code currently supports only cubic systems. Needs improvement.
    Calculates diffraction rings for a given cubic space group and lattice parameter.
    Args:
        spacegroup: int, space group number
        a: float, lattice parameter
        wavelength: float, wavelength of the x-ray
        Lsd: float, sample-to-detector distance
        max_index: int, maximum index for h, k, l
    Returns:
        rings_R: list of float, diffraction ring positions (same units as Lsd)
    """

    space_groups_lookup = {
    '225': 'F',
    '229': 'B'
    }


    def is_reflection_allowed(h, k, l, spacegroup):
        """    
        bravais can be 'P', 'I', or 'F' for cubic systems.
        Limited to just cubic systems for now
        """
        # Simple cubic (P): all integer (h,k,l) are allowed in principle.
        if space_groups_lookup[str(spacegroup)].upper() == 'P':
            return True
        
        # Body-centered (I): h+k+l must be even
        elif space_groups_lookup[str(spacegroup)].upper() == 'I':
            return ((h + k + l) % 2) == 0
        
        # Face-centered (F): h, k, l must be all even or all odd
        elif space_groups_lookup[str(spacegroup)].upper() == 'F':
            # (h, k, l) all even -> (h%2, k%2, l%2) = (0,0,0)
            # (h, k, l) all odd  -> (h%2, k%2, l%2) = (1,1,1)
            # So sum them up and see if sum==0 or sum==3
            modsum = (h % 2) + (k % 2) + (l % 2)
            return (modsum == 0 or modsum == 3)
        
        else:
            # If the input bravais is unrecognized, default to no reflection
            return False


    def generate_hkl_cubic(max_index=5):
        """
        Generate a list of (h, k, l) triplets from 0 to max_index.
        You can change max_index based on how many reflections you want to consider.
        """
        hkls = []
        for h in range(0, max_index + 1):
            for k in range(0, max_index + 1):
                for l in range(0, max_index + 1):
                    if not (h == 0 and k == 0 and l == 0):
                        hkls.append((h, k, l))
        return hkls

    hkls = generate_hkl_cubic(max_index=max_index)
    rings_two_theta = []
    
    for (h, k, l) in hkls:
        # Check reflection conditions (centering)
        if not is_reflection_allowed(h, k, l, spacegroup):
            continue
        
        # Calculate d-spacing and two-theta angles
        d = a / math.sqrt(h**2 + k**2 + l**2)
        sin_theta = wavelength / (2.0 * d)
        if sin_theta <= 1.0:
            two_theta = 2.0 * math.degrees(math.asin(sin_theta))
            rings_two_theta.append(round(two_theta, 3))
        else:
            continue
        
    # Sort and remove duplicates
    rings_two_theta = np.sort(np.unique(np.array(rings_two_theta)))
    
    # Calculate ring positions using experiment geometry
    rings_R = Lsd * np.tan(np.deg2rad(rings_two_theta)) # same units as Lsd

    return rings_R
    

def read_hkls_csv(hkls_csv_path):
    """Read the hkls.csv file and return it as a pandas dataframe"""
    df_hkls = pd.read_csv(hkls_csv_path, sep=' ')
    return df_hkls


# ----------------------------
class CNNSR(torch.nn.Module):

    def __init__(self, l_ch_nrs, l_ker_size, l_act_func, X_channels):
        """Initializaion function of the class; runs automatically when class is created
        Args:
            l_ch_nrs (list of int): list with number of channels in each successive CNN layer
            l_ker_size (list of odd int): list with kernel size for each successive CNN layer
            l_act_func (list of str): list with activation function ('r', 's', 'lr', 't') each successive CNN layer
            X_channels (int): no. of channels in input X data
        """

        super(CNNSR, self).__init__()

        self.cnn_ops = []
        cnn_out_chs = tuple(l_ch_nrs)
        cnn_in_chs = (X_channels, ) + cnn_out_chs[:-1]

        for (in_ch, out_ch, ker_sz, act_func) in zip(
            cnn_in_chs, cnn_out_chs, l_ker_size, l_act_func):

            # Add convolution layer
            pad_sz = int(ker_sz / 2)
            self.cnn_ops += [
                torch.nn.Conv2d(in_channels=in_ch, out_channels=out_ch,\
                               kernel_size=ker_sz, padding=pad_sz)
            ]

            # Add activation layer
            if act_func == "r": self.cnn_ops += [torch.nn.ReLU()]
            if act_func == "s": self.cnn_ops += [torch.nn.Sigmoid()]
            if act_func == "lr": self.cnn_ops += [torch.nn.LeakyReLU()]
            if act_func == "t": self.cnn_ops += [torch.nn.Tanh()]

        self.cnn_layers = torch.nn.Sequential(*self.cnn_ops)

    def forward(self, x):

        _out = x
        for layer in self.cnn_layers:
            _out = layer(_out)

        return (_out)
    

# ----------------------------
def load_trained_CNNSR(midasVerDir, mod_dir, mod_itr, torch_devs):
    """ loads trained CNNSR model based on input directory and iteration.\
    NOTE: The model directory should contain the following files:
        - mod-it{mod_itr}.pth: model weights
        - _train_args.json: model training arguments
    Args:
        mod_dir (str): directory containing the trained model
        mod_itr (int): model iteration
        torch_devs (torch.device): device to be used for loading model
    Returns:
        mod (torch model): loaded model instance with weights
        mod_args (dict): arguments used for model training
        X_channels (list): channels used in input data for model (0-Intensity, 1-Radius, 2-Eta)
    """
    
    mod_args_filepath = f"{midasVerDir}{SEP}{mod_dir}{SEP}_train_args.json"
    with open(mod_args_filepath, 'r') as f:
        mod_args = json.load(f)

    X_channels = [0]
    if mod_args["useRch"] in ["True", "true", True, "Yes", "yes", "Y", "y"]:
        X_channels.append(1)
    if mod_args["useEtach"] in ["True", "true", True, "Yes", "yes", "Y", "y"]:
        X_channels.append(2)

    in_ch = len(X_channels)
    
    mod_itr_path = f"{midasVerDir}{SEP}{mod_dir}{SEP}mod-it{mod_itr}.pth"

    l_layer_params = mod_args["arch"].split("_")
    l_ch_nrs = [int(i.split("-")[0]) for i in l_layer_params]
    l_ker_size = [int(i.split("-")[1]) for i in l_layer_params]
    l_act_func = [str(i.split("-")[2]) for i in l_layer_params]

    mod = CNNSR(l_ch_nrs, l_ker_size, l_act_func, in_ch)
    mod = nn.DataParallel(mod)
    mod.load_state_dict(torch.load(mod_itr_path, map_location=torch_devs, weights_only=True))
    mod = mod.to(torch_devs)

    return (mod, mod_args, X_channels)


# ----------------------------
def patches_from_detector_frame(frame_exp, patch_size=int(20), threshold=0, connectivity_dim=8):
    """ extracts patches from input detector frame using connected component algorithm. First a labelled array is created where different spots\
    are marked with different label. Each spot is then isolated and pasted onto a patch of size=patch_size such that the max intensity pixel is\
    at the center of the patch.
    Args:
        frame_exp (arr): experimental detector frame
        patch_size (int; default=20): size of patches that are extracted
        threshold (float; default=0): pixels below threshold are set to zero
        connectivity_dim (int; default=8): if set to 8: it accepts continuity with diagonal cells; otherwise it uses 4-connectivity only
    Returns:
        patches (arr): extracted patch array (N,patch_size,patch_size)
        patches_Z00 (list of int): Z coordinates of (0,0) pixels in patches
        patches_Y00 (list of int): Y coordinates of (0,0) pixels in patches
        nr_pixels_in_patch (list of int): no. of non-zero pixels in the patch
    """

    # Create a binary mask for regions above the threshold
    binary_mask = frame_exp > threshold

    # Label connected regions in the binary mask
    if connectivity_dim in [8, 8.0, "8"]:
        # Define an 8-connectivity structure that includes diagonals
        structure = np.array([[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1]])

        labeled_array, num_features = label(binary_mask, structure)
        
    else:
        labeled_array, num_features = label(binary_mask) # uses 4-connectivity

    # Find bounding boxes for each labeled region
    bounding_boxes = find_objects(labeled_array)
    
    # Extract and store only the region corresponding to each spot
    spots = []
    patches = []
    patches_Y00, patches_Z00 = [], []
    nr_pixels_in_patch = []

    for i, bbox in enumerate(bounding_boxes[:-1]):
        if bbox is not None:  # Ensure there is a valid region
            # Extract the sub-array corresponding to the current spot
            spot = frame_exp[bbox]

            # Mask out values not belonging to the current spot
            spot = np.where(labeled_array[bbox] == (i + 1), spot, 0)

            # Find the row, col position of the maximum intensity pixel within the spot
            r_max_spot, c_max_spot = np.unravel_index(np.argmax(spot), spot.shape)  # (row, col)
            Zpx_max_spot = bbox[0].start + r_max_spot
            Ypx_max_spot = bbox[1].start + c_max_spot

            Z00_patch = Zpx_max_spot - int(patch_size/2)
            Y00_patch = Ypx_max_spot - int(patch_size/2)

            # If Z00 and Y00 are such that cropped region will be within frame boundaries
            if (Z00_patch >= 0) and \
                (Z00_patch < frame_exp.shape[0] - patch_size) and \
                (Y00_patch >= 0) and \
                (Y00_patch < frame_exp.shape[1] - patch_size):


                # Create patch by padding the spot on all sides to ensure it can be cropped properly
                patch = np.pad(spot, pad_width=patch_size, mode='constant', constant_values=0)
                
                # Find the row, col position of the maximum intensity pixel within the patch
                r_max_patch, c_max_patch = np.unravel_index(np.argmax(patch), patch.shape)  # (row, col)

                # Crop the patch such that max int pixel will be at center of patch
                rs = int(r_max_patch - int(patch_size/2))
                cs = int(c_max_patch - int(patch_size/2))
                rf, cf = rs + patch_size, cs + patch_size
                patch = patch[rs:rf, cs:cf]
                patches.append(patch)

                patches_Z00.append(Z00_patch)
                patches_Y00.append(Y00_patch)

                spots.append(spot)
                nr_pixels_in_patch.append(np.count_nonzero(spot))


    patches = np.array(patches)

    return (patches, patches_Z00, patches_Y00, nr_pixels_in_patch)


# ----------------------------
def weighted_center_of_mass(patch, threshold=0.5):
    """ Computes the weighted center of mass (centroid) of a 2D patch.
    Args:
        patch (ndarray): 2D NumPy array representing the patch.
    Returns:
        tuple: (Z_center, Y_center) coordinates of the centroid.
            Z represents row, Y represents col
    """

    # Ensure the patch is a NumPy array
    patch = np.array(patch)
    patch[patch < threshold*np.max(patch)] = 0

    # Get the dimensions of the patch
    height, width = patch.shape

    # Generate coordinate grids
    Z_indices, Y_indices = np.indices((height, width))

    # Compute the total mass (sum of all pixel intensities)
    total_mass = np.sum(patch)

    # Avoid division by zero
    if total_mass == 0:
        Y_center, Z_center = width / 2, height / 2

    else:
        # Compute the weighted sums
        Y_center = np.sum(Y_indices * patch) / total_mass
        Z_center = np.sum(Z_indices * patch) / total_mass

    com_coords = (Z_center, Y_center)
    
    return com_coords



# ----------------------------
def com_peak_coords(patch, locmax_peak_coords, threshold=0.5, peak_crop_size=20):
    """ Calculate center of mass peak coordinates in a patch.
    Args:
        patch (arr): input patch (with single or multiple peaks)
        locmax_peak_coords (arr): peak coords obtained from local maxima
        threshold (float; default=0.5): a fraction value to threshold cropped peak patches
        peak_crop_size (int): size of peak patches that are cropped around each local max peak position.
            center of mass calculation is done in the cropped patch
    Returns:
        com_peak_coord (arr): array with com peak coordinates; column0 has Z (row) coordinates, column1 has Y (column) coordinates
    """
    
    n_locmax_peaks = len(locmax_peak_coords)
    com_peakY, com_peakZ = [], []

    for i in range(0, n_locmax_peaks):

        # Y00 & Z00 positions below are (0,0) px position of cropped peak patch from the full patch
        # These are NOT the Y00 & Z00 positions of the patch wrt detector coordinate frame
        peak_patch_Y00 = locmax_peak_coords[i][1] - int(peak_crop_size/2)
        peak_patch_Z00 = locmax_peak_coords[i][0] - int(peak_crop_size/2)
        
        row_T, row_B = max(0,peak_patch_Z00), min(peak_patch_Z00 + peak_crop_size, patch.shape[0])
        col_L, col_R = max(0,peak_patch_Y00), min(peak_patch_Y00 + peak_crop_size, patch.shape[1])
        
        peak_patch = patch[row_T:row_B, col_L:col_R]
        Z_center, Y_center = weighted_center_of_mass(peak_patch, threshold)

        com_peakY.append(peak_patch_Y00+Y_center)
        com_peakZ.append(peak_patch_Z00+Z_center)

    com_peak_coords = np.zeros(shape=(n_locmax_peaks,2))
    com_peak_coords[:,0] = np.array(com_peakZ)
    com_peak_coords[:,1] = np.array(com_peakY)

    return (com_peak_coords)


# ----------------------------
def watershed_peaks(patch, peak_coords, mask_thresh):
    """Apply watershed algorithm to allocate regions 
    Args:
        patch (arr): input patch
        peak_coords (arr): coordinates of peaks in patch
        mask_thresh (float): fraction of max intensity used as threshold for mask
    Returns:
        labelled_patch (arr): labelled patch
        peak_label_values (arr): label values of peaks
        peaks_Isum (arr): sum of intensities of peaks
    """

    n_peaks_in_patch = len(peak_coords)
    # Create markers for watershed algorithm
    markers = np.zeros_like(patch, dtype=np.int32)
    for i in range(n_peaks_in_patch):
        markers[peak_coords[i][0], peak_coords[i][1]] = i + 1

    # Apply watershed with a mask
    mask_thresh = mask_thresh * np.max(patch)
    labelled_patch = watershed(-patch, markers, mask=patch>mask_thresh)

    # Get properties of labelled regions
    label_props = regionprops(labelled_patch, intensity_image=patch)
    peak_label_values = [prop.label for prop in label_props]
    peaks_Isum = [prop.mean_intensity * prop.area for prop in label_props]
    peaks_NrPixels = [prop.num_pixels for prop in label_props]


    return (labelled_patch, peak_label_values, peaks_Isum, peaks_NrPixels)



# ----------------------------
def multi_pv_fit(patch, Y00, Z00, Ypx_BC, Zpx_BC, lrsz, srfac, min_distance, threshold_rel, gauss_filter_sigma=0, median_filter_size=1, lr_int_thresh=10.0):
    """ performs multiple pseduo-voigt fit on input patch. No. of peaks and their approx location is identified using "loc_max_coords"
    Args:
        patch (arr): input patch (with single or multiple peaks)
        Y00 (int): Y coord of (0,0) px of patch in detector frame in native x1 resolution
        Z00 (int): Z coord of (0,0) px of patch in detector frame in native x1 resolution
        Ypx_BC (float): Beam center Y coord in detector frame
        Zpx_BC (float): Beam center Z coord in detector frame
        lrsz (int): low-res native patch size
        srfac (int): super-res factor of patch being fitted
        min_distance (float): minimum distanc (in pixels) between two peaks
        threshold_rel (float): threshold value to count a pixel as a peak
        gauss_filter_sigma (float; default=1): standard deviation for Gaussian kernel
        median_filter_size (float; default=1): size that is taken from the input array\
            at every element position, to define input to the filter function
        lr_int_thresh (float; default=10.0): threshold used to filter out pixels
    Returns:
        fit_peaks (dict): dictionary with properties of fitted peaks; the peak positions here are in detector frame coordinates
        fit_peak_coords (arr): array with pv-fit peak coordinates; column0 has Z (row) coordinates, column1 has Y (column) coordinates;\
            the peak positions here are in patch coordinates
        fit_patch (arr): fitted patch
    """

    # Threshold int for SRx patch
    SRx_thresholdInt = lr_int_thresh / (srfac * srfac)

    def multiPV2d_diffLGwidth(RE_coords, *params):
        RR, EE = RE_coords
        n_paramsPerPeak = 8
        n_peaks = len(params) // n_paramsPerPeak
        patchI = np.zeros_like(RR, dtype=np.float32)
        
        for i in range(n_peaks):
            Rpx, EtaDeg, sigGR, sigGEta, sigLR, sigLEta, LGmix, IMax = params[i * n_paramsPerPeak : (i + 1) * n_paramsPerPeak]
            G = IMax * np.exp(-0.5 * ((RR - Rpx) / sigGR)**2 - 0.5 * ((EE - EtaDeg) / sigGEta)**2)
            L = IMax * 1 / ((1 + ((RR - Rpx) / sigLR)**2) * (1 + ((EE - EtaDeg) / sigLEta)**2))
            patchI += (LGmix * L) + ((1 - LGmix) * G)
        
        patchI[patchI<SRx_thresholdInt] = 0

        return (patchI)

    def multiPV2d_diffLGwidth_forFit(RE_coords, *params):
        patchI = multiPV2d_diffLGwidth(RE_coords, *params)
        return (patchI.ravel())


    # Set parameter bounds to constrain the fit
    dR, dEta, dIMax= 2, 0.1, 400/srfac # R and Eta change allowed from initial estimate

    # Create RR and EE spread over the patch
    dpx_sr = 1 / srfac # pixel step size in super-res patch

    # Y-Z coordinate values
    Ypx = np.arange(Y00, Y00 + lrsz, dpx_sr)
    Zpx = np.arange(Z00, Z00 + lrsz, dpx_sr)

    # Create meshgrids of Radius and Eta values
    grid_YY, grid_ZZ = np.meshgrid(Ypx, Zpx)
    grid_RR = ((Ypx_BC - grid_YY)**2 + (Zpx_BC - grid_ZZ)**2)**0.5
    grid_EE = np.rad2deg(np.arccos((grid_ZZ - Zpx_BC) / grid_RR))
    # Convert Eta_inDeg to -ve where Ypx < Ypx_BC
    grid_EE = grid_EE * ((grid_YY - Ypx_BC) / np.abs(grid_YY - Ypx_BC))

    # Apply Gaussian and median filters to the patch for peak finding
    patch_for_locmax = gaussian_filter(patch, sigma=gauss_filter_sigma)
    patch_for_locmax = median_filter(patch_for_locmax, size=median_filter_size)

    # Find local maxima coords in the filtered patch
    locmax_coords = peak_local_max(patch_for_locmax, min_distance=min_distance, threshold_rel=threshold_rel)

    nPeaks_locmax = len(locmax_coords)
    locmax_Y, locmax_Z = locmax_coords[:,1], locmax_coords[:,0]

    fit_params_names = ["R(px)", "Eta(deg)", "SigGR", "SigGEta", "SigLR", "SigLEta", "LGmix", "IMax"]

    init_guess = [0, 0, 0.3, 0.3, 0.3, 0.3, 0.5, 0] * nPeaks_locmax
    lower_bounds = [0, 0, 0.1, 0.05, 0.1, 0.05, 0, 0] * nPeaks_locmax
    upper_bounds = [0, 0, 3.0, 3.0, 3.0, 3.0, 1, 0] * nPeaks_locmax

    n_paramsPerPeak = len(fit_params_names)

    # Iterate over each locmax peak identified
    for i in range(nPeaks_locmax):
        Rpx_guess, Eta_guess = grid_RR[locmax_Z[i], locmax_Y[i]], grid_EE[locmax_Z[i], locmax_Y[i]]
        IMax_guess = patch[locmax_Z[i], locmax_Y[i]]

        # Update initial guess and bounds of Radius
        init_guess[i * n_paramsPerPeak] = Rpx_guess
        lower_bounds[i * n_paramsPerPeak] = Rpx_guess - dR
        upper_bounds[i * n_paramsPerPeak] = Rpx_guess + dR

        # Update initial guess and bounds of Eta
        init_guess[i * n_paramsPerPeak + 1] = Eta_guess
        lower_bounds[i * n_paramsPerPeak + 1] = Eta_guess - dEta
        upper_bounds[i * n_paramsPerPeak + 1] = Eta_guess + dEta

        # Update initial guess and bounds of IMax
        init_guess[i * n_paramsPerPeak + 7] = IMax_guess
        upper_bounds[i * n_paramsPerPeak + 7] = IMax_guess + dIMax


    popt, pcov = curve_fit(multiPV2d_diffLGwidth_forFit, (grid_RR, grid_EE), patch.ravel(), p0=init_guess, bounds=(lower_bounds, upper_bounds),
            maxfev=1000,  # Increase the number of function evaluations if needed
            method="trf")
    
    fit_patch = multiPV2d_diffLGwidth((grid_RR, grid_EE), *popt).reshape(grid_RR.shape)

    fit_peaks = [] # list to store dictionaries with peak properties
    for i in range(nPeaks_locmax):
        params = popt[i*n_paramsPerPeak:(i+1)*n_paramsPerPeak]
        fit_peaks.append({fit_params_names[j]:params[j] for j in range(0, n_paramsPerPeak)})

        # Get Y-Z coordinates from R-Eta
        Y0_fit = Ypx_BC + (fit_peaks[i]['R(px)'] * np.sin(np.deg2rad(fit_peaks[i]['Eta(deg)'])))
        Z0_fit = Zpx_BC + (fit_peaks[i]['R(px)'] * np.cos(np.deg2rad(fit_peaks[i]['Eta(deg)'])))

        fit_peaks[i]['Y(px)'] = Y0_fit
        fit_peaks[i]['Z(px)'] = Z0_fit

    fit_peak_coords = np.zeros(shape=(len(fit_peaks),2))
    fit_peak_coords[:,0] = np.array([fit_peaks[i]['Z(px)'] for i in range(0, len(fit_peak_coords))])
    fit_peak_coords[:,1] = np.array([fit_peaks[i]['Y(px)'] for i in range(0, len(fit_peak_coords))])

    fit_peak_coords[:,0] = (fit_peak_coords[:,0] - Z00) * srfac
    fit_peak_coords[:,1] = (fit_peak_coords[:,1] - Y00) * srfac


    return fit_peaks, fit_peak_coords, fit_patch



# ----------------------------
def pseudoVoigt2d_diffLGwidth(yy, zz, y0, z0, ySigG, zSigG, ySigL, zSigL, LGmix, IMax, *args):
    """create 2D pseudo-voigt spread over yy and zz coordinates using different gaussian and lorentzian widths
    Args:
        yy (arr): Y coord (pixel) grid array that maps Y distribution over patch
        zz (arr): Z coord (pixel) grid array that maps Y distribution over patch
        y0 (float): Peak position Y coord (pixel)
        z0 (float): Peak position Z coord (pixel)
        ySigG (float): Gaussian peak width in Y-axis
        zSigG (float): Gaussian peak width in Z-axis
        ySigL (float): Lorentzian peak width in Y-axis
        zSigL (float): Lorentzian peak width in Z-axis
        LGmix (float): LGmix factor
        IMax (float): Max intensity
        *args (list): [rot_deg (rotation angle in degrees)]
    Returns:
        PV (arr): pseudo-Voigt peak over Y-Z grid
    """


    # ----------------------------
    def gaussian2d(yy, zz, y0, z0, ySig, zSig, IMax, *args):
        """create 2D Gaussian spread over yy and zz coordinates
        """

        if len(args) > 0: rot_deg = args[0]
        else: rot_deg = 0

        rot_rad = np.deg2rad(rot_deg) # degree to radian

        # Rotate coordinates
        yyRot = (yy - y0) * np.cos(rot_rad) - (zz - z0) * np.sin(rot_rad)
        zzRot = (yy - y0) * np.sin(rot_rad) + (zz - z0) * np.cos(rot_rad)

        # Create gaussian distribution
        G = IMax * np.exp(-0.5 * (yyRot / ySig)**2 - 0.5 * (zzRot / zSig)**2)

        return (G)


    # ----------------------------
    def lorentzian2d(yy, zz, y0, z0, ySig, zSig, IMax, *args):
        """create 2D Lorentzian spread over yy and zz coordinates
        """

        if len(args) > 0: rot_deg = args[0]
        else: rot_deg = 0

        rot_rad = np.deg2rad(rot_deg) # degree to radian

        # Rotate coordinates
        yyRot = (yy - y0) * np.cos(rot_rad) - (zz - z0) * np.sin(rot_rad)
        zzRot = (yy - y0) * np.sin(rot_rad) + (zz - z0) * np.cos(rot_rad)

        # Create lorentzian distribution
        L = IMax * 1 / ((1 + (yyRot / ySig)**2) * (1 + (zzRot / zSig)**2))

        return (L)


    if len(args) > 0: rot_deg = args[0]
    else: rot_deg = 0

    # Create gaussian and lorentzian distributions
    G = gaussian2d(yy, zz, y0, z0, ySigG, zSigG, IMax, *[rot_deg])
    L = lorentzian2d(yy, zz, y0, z0, ySigL, zSigL, IMax, *[rot_deg])
    
    # Create pseudo-voigt distribution
    PV = (LGmix * L) + ((1 - LGmix) * G) # pseudo-voigt 2d

    return (PV)


# ----------------------------
def parse_sr_config_txt(SRconfig_path):
    """Parse the SR config file when it is a txt file
    Args:
        SRconfig_path (str): Path to the SR config text file
    Returns:
        sr_config (dict): Dictionary containing the SR config
    """

    cfg = {}
    with open(SRconfig_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            key = parts[0]
            value = " ".join(parts[1:])  # keep as a single string
            cfg[key] = value

    sr_config = {}
    try: sr_config["minEta"] = float(cfg["minEta"])
    except: pass
    
    try: sr_config["minPxCount"] = int(cfg["minPxCount"])
    except: pass
    
    try: sr_config["skipFitIfExists"] = str(cfg["skipFitIfExists"])
    except: pass
    
    try: sr_config["fitPeakShapePV"] = str(cfg["fitPeakShapePV"])
    except: pass
    
    try: sr_config["R_deviation"] = float(cfg["R_deviation"])
    except: pass
    
    try: sr_config["lrsz"] = int(cfg["lrsz"])
    except: pass
    
    try: sr_config["edge_bound_cutoff_fac"] = float(cfg["edge_bound_cutoff_fac"])
    except: pass
    
    try: sr_config["batch_size"] = int(cfg["batch_size"])
    except: pass

    sr_config["mods_to_use"] = {}
    for i in [2,4,8]:
        sr_config["mods_to_use"][f"SRx{i}"] = {}
        try: sr_config["mods_to_use"][f"SRx{i}"]["mod_dir"] = str(cfg[f"SRx{i}_mod_dir"])
        except: pass

        try: sr_config["mods_to_use"][f"SRx{i}"]["mod_itr"] = int(cfg[f"SRx{i}_mod_itr"])
        except: pass

    sr_config["spot_find_args"] = {}
    try: sr_config["spot_find_args"]["threshold"] = float(cfg["spot_threshold"])
    except: pass

    try: sr_config["spot_find_args"]["patch_size"] = int(cfg["spot_patch_size"])
    except: pass

    sr_config["peak_find_args"] = {}
    try: sr_config["peak_find_args"]["min_d"] = {f"SRx{i}": int(cfg[f"SRx{i}_min_d"]) for i in [1,2,4,8]}
    except: pass

    try: sr_config["peak_find_args"]["thresh_rel"] = {f"SRx{i}": float(cfg[f"SRx{i}_thresh_rel"]) for i in [1,2,4,8]}
    except: pass

    try: sr_config["peak_find_args"]["gauss_filter_sigma"] = {f"SRx{i}": float(cfg[f"SRx{i}_gauss_filter_sigma"]) for i in [1,2,4,8]}
    except: pass

    try: sr_config["peak_find_args"]["median_filter_size"] = {f"SRx{i}": int(cfg[f"SRx{i}_median_filter_size"]) for i in [1,2,4,8]}
    except: pass

    try: sr_config["peak_find_args"]["peak_crop_size"] = {f"SRx{i}": int(cfg[f"SRx{i}_peak_crop_size"]) for i in [1,2,4,8]}
    except: pass

    try: sr_config["peak_find_args"]["pvfit_int_thresh"] = {f"SRx{i}": float(cfg[f"SRx{i}_pvfit_int_thresh"]) for i in [1,2,4,8]}
    except: pass
    sr_config["shift_YZ_pos"] = {}
    for i in [2,4,8]:
        try:
            sr_config["shift_YZ_pos"][f"SRx{i}"] = {
                "shiftYpx": float(cfg[f"SRx{i}_shift_Ypx"]),
                "shiftZpx": float(cfg[f"SRx{i}_shift_Zpx"])
            }
        except: pass

    return sr_config


def update_nested_dict(dict_a, dict_b):
    """Update a nested dictionary with values from another nested dictionary
    Args:
        dict_a (dict): Dictionary to be updated
        dict_b (dict): Dictionary with values to be updated
    """
    for key, value in dict_b.items():
        if key in dict_a:
            # If both values are dictionaries, recurse
            if isinstance(dict_a[key], dict) and isinstance(value, dict):
                update_nested_dict(dict_a[key], value)
            else:
                # Update the value
                dict_a[key] = value
        else:
            # Key doesn't exist in A, add it
            dict_a[key] = value


parser = argparse.ArgumentParser(description="Routine to create SR patches & perform pseudoVoigt fit for a given MIDAS layer.\
                    The analysis take midas ZArr file as input from the directory provided")

parser.add_argument("-midasZarrDir",type=str,  help="Directory containing the Zarr file")
parser.add_argument("-srfac",       type=int,   default=8,  help="(1/2/4/8; def=8) Super-resolution factor")
parser.add_argument("-SRconfig_path", type=str,   default="sr_config.json",  help="SR config file path. If not provided, the default configration file will be used")
parser.add_argument("-midasVerDir", type=str,   default="",  help="MIDAS version directory used to create paths to trained models")
parser.add_argument("-ps_fn", type=str,   default="",  help="Parameter file name")

args, unparsed = parser.parse_known_args()

# If the last character of the midasZarrDir is not the separator, add one
if args.midasZarrDir[-1] != SEP:
    args.midasZarrDir = args.midasZarrDir + SEP


t0 = time.time()
t_run = 0 
# Setup logging
ts = time.time()
SRlogger = setup_logging(args.midasZarrDir)
tf = time.time()
t_run += tf - ts
SRlogger.info(f"{'-'*5} Time to setup logging: {tf - ts:.5f} s | SR_run_time: {t_run:.5f} s")


SRlogger.info(f"Loading super-res configuration file: {args.SRconfig_path}")
ts = time.time()

with open(args.SRconfig_path, "r") as f:
    sr_config = json.load(f)

"""Below code is used to update the SR config with the parameters from the parameter text file
# Currently disabled since the ps_fn obtained from ff_MIDAS_sr.py is not getting read correctly.
sr_config_in_ps = parse_sr_config_txt(args.ps_fn)
update_nested_dict(sr_config, sr_config_in_ps)
"""

tf = time.time()    
t_run += tf - ts
SRlogger.info(f"{'-'*5} Time to load SR config: {tf - ts:.5f} s | SR_run_time: {t_run:.5f} s")


SRlogger.info(f"Searching for MIDAS zip file in dir: {args.midasZarrDir}")
ts = time.time()
midas_zip_filename = [i for i in os.listdir(args.midasZarrDir) if ".MIDAS.zip" in i][0]
midas_zip_filepath = f"{args.midasZarrDir}{midas_zip_filename}"
tf = time.time()    
t_run += tf - ts
SRlogger.info(f"\t|Found MIDAS zip file: {midas_zip_filename}")
SRlogger.info(f"{'-'*5} Time to find MIDAS zip file: {tf - ts:.5f} s | SR_run_time: {t_run:.5f} s")


SRlogger.info(f"Loading Zarr data & analysis parameters: {midas_zip_filename}")
ts = time.time()
zf= zarr.open(midas_zip_filepath,"r")
exData = np.array(zf["exchange"]["data"])
params = zf["analysis"]["process"]["analysis_parameters"]
del zf
tf = time.time()
t_run += tf - ts
SRlogger.info(f"\t|Nr of frames={exData.shape[0]}")
SRlogger.info(f"{'-'*5} Time to load midas zip file: {tf - ts:.5f} s | SR_run_time: {t_run:.5f} s")


SRlogger.info(f"Extracting parameters needed for super-res workflow")
ts = time.time()
sr_params = {}
sr_params["Ypx_BC"] = float(params["YCen"][0])
sr_params["Zpx_BC"] = float(params["ZCen"][0])
sr_params["ringsToUse"] = list(params["RingThresh"][:][:,0].astype(int))
sr_params["spacegroup"] = int(params["SpaceGroup"][0])
sr_params["lattParam"] = list(params["LatticeParameter"][:].astype(float))
sr_params["xrayLambda"] = float(params["Wavelength"][0]) # in Angstrom
sr_params["Lsd"] = float(params["Lsd"][0])
sr_params["pxSize"] = float(params["PixelSize"][0])
sr_params["SkipFrame"] = list(params["SkipFrame"][:].astype(int))
tf = time.time()
t_run += tf - ts
SRlogger.info(f"{'-'*5} Time to extract parameters: {tf - ts:.5f} s | SR_run_time: {t_run:.5f} s")

SRlogger.info(f"Calculating diffraction ring radii for {sr_params['ringsToUse']} rings")
ts = time.time()

""" Instead of compute_diffraction_rings, read the hkls.csv file and get the ring radii
# The 'compute_diffraction_rings' function is limited only to cubic systems.

rings_Rpx = compute_diffraction_rings(sr_params["spacegroup"], sr_params["lattParam"][0],
                                      sr_params["xrayLambda"], sr_params["Lsd"],
                                      max_index=5) / sr_params["pxSize"]
"""

df_hkls = read_hkls_csv(f"{args.midasZarrDir}hkls.csv")
rings_Rpx = (df_hkls["Radius"].unique() / sr_params["pxSize"]).tolist()

ring_R_values = [rings_Rpx[int(i)-1] for i in (sr_params['ringsToUse'])]
tf = time.time()
t_run += tf - ts
SRlogger.info(f"Ring radii (px) {ring_R_values}")
SRlogger.info(f"{'-'*5} Time to calculate ring radii: {tf - ts:.5f} s | SR_run_time: {t_run:.5f} s")


SRlogger.info(f"Creating list of frame indices to be used for analysis")
ts = time.time()
framesIdx = [int(i) for i in np.arange(0, exData.shape[0], 1)
             if (i+1) not in sr_params["SkipFrame"]]

tf = time.time()
t_run += tf - ts
SRlogger.info(f"{'-'*5} Time to compile frame indices: {tf - ts:.5f} s | SR_run_time: {t_run:.5f} s")

ts = time.time()
peaks_csv_dir = f"{args.midasZarrDir}Temp{SEP}"
if not os.path.isdir(peaks_csv_dir):
    os.mkdir(peaks_csv_dir) # create new output directory
    SRlogger.info(f"Created 'Temp' folder in MIDAS directory (peaks csv files will be saved here)'")
else:
    SRlogger.info(f"Temp folder already exists in MIDAS directory")
tf = time.time()
t_run += tf - ts
SRlogger.info(f"{'-'*5} 'Temp' directory setup completed in {tf - ts:.5f}s | SR_run_time: {t_run:.5f}s")

# Selecting the device for model training
SRlogger.info(f"Selecting the device for running ML models")
ts = time.time()

if torch.cuda.is_available():
    torch_devs = torch.device("cuda")
elif torch.backends.mps.is_available():
    torch_devs = torch.device("mps")
else:
    torch_devs = torch.device("cpu")

tf = time.time()
t_run += tf - ts
SRlogger.info(f"\t|Device selected: {torch_devs}")
SRlogger.info(f"{'-'*5} Time to select the device: {tf - ts:.5f} s | SR_run_time: {t_run:.5f} s")

SRlogger.info(f"Loading the trained models")
ts = time.time()
with torch.no_grad():
    x2mod, x2mod_args, x2mod_ch = load_trained_CNNSR(midasVerDir=args.midasVerDir, mod_dir=sr_config["mods_to_use"]["SRx2"]["mod_dir"], mod_itr=sr_config["mods_to_use"]["SRx2"]["mod_itr"], torch_devs=torch_devs)
    x4mod, x4mod_args,x4mod_ch = load_trained_CNNSR(midasVerDir=args.midasVerDir, mod_dir=sr_config["mods_to_use"]["SRx4"]["mod_dir"], mod_itr=sr_config["mods_to_use"]["SRx4"]["mod_itr"], torch_devs=torch_devs)
    x8mod, x8mod_args, x8mod_ch = load_trained_CNNSR(midasVerDir=args.midasVerDir, mod_dir=sr_config["mods_to_use"]["SRx8"]["mod_dir"], mod_itr=sr_config["mods_to_use"]["SRx8"]["mod_itr"], torch_devs=torch_devs)
    x2mod.eval()
    x4mod.eval()
    x8mod.eval()

tf = time.time()
t_run += tf - ts
SRlogger.info(f"\t|Models loaded in evaluation mode.")
SRlogger.info(f"{'-'*5} Time to load the models: {tf - ts:.5f} s | SR_run_time: {t_run:.5f} s")

SRlogger.info(f"Defining column names for the output csv files.")
# Column names for the output csv file
col_names = ["SpotID", "IntegratedIntensity", "Omega(degrees)", "YCen(px)", "ZCen(px)", "IMax",
             "Radius(px)", "Eta(degrees)", "SigmaR", "SigmaEta", "NrPixels",
             "TotalNrPixelsInPeakRegion","nPeaks", "maxY", "maxZ", "diffY", "diffZ"]

SRlogger.info(f"Calculating omega stepsize based on the number of frames in the data.")
nr_frames = exData.shape[0]
omega_stepsize = 360 / (nr_frames - 1)
SRlogger.info(f"\t|Nr. frames={nr_frames} | Omega stepsize={omega_stepsize}")

SRlogger.info(f"Getting Y-Z peak position pixel shift values to be used due to super-resolution factor")
shiftYpx = sr_config["shift_YZ_pos"][f"SRx{args.srfac}"]["shiftYpx"]
shiftZpx = sr_config["shift_YZ_pos"][f"SRx{args.srfac}"]["shiftZpx"]
SRlogger.info(f"\t|shiftYpx: {shiftYpx}, shiftZpx: {shiftZpx}")


for frame_i in framesIdx:
    t0_frame = time.time()
    SRlogger.info(f"{'='*60}")
    SRlogger.info(f"{'*'*5} Processing frame {frame_i} {'*'*5} ")

    frame_i_csv_savename = f"{midas_zip_filename}_{str(frame_i).zfill(6)}_PS.csv"
    frame_i_csv_savepath = f"{peaks_csv_dir}{frame_i_csv_savename}"

    # Skip if csv file already exists provided skipIfExists is set to 'yes'
    if sr_config["skipFitIfExists"].lower() in ["yes", "y", "true", "t", "1"] and os.path.isfile(frame_i_csv_savepath):
        SRlogger.info(f"{'*'*5}| Frame {frame_i} skipped as csv file already exists.")
        continue

    SRlogger.info(f"\t| Extracting patches from frame {frame_i}")
    
    ts = time.time()

    patches_exp, patches_Z00, patches_Y00, nr_pixels_in_patch = \
            patches_from_detector_frame(exData[frame_i],
                                        patch_size=sr_config["spot_find_args"]["patch_size"],
                                        threshold=sr_config["spot_find_args"]["threshold"],
                                        connectivity_dim=8)
    
    patches_exp = np.expand_dims(patches_exp, axis=1) # expand dimension to make this ready as input for SR model
    patches_Isum = np.sum(patches_exp, axis=(1,2,3)).tolist()
    n_patches = len(patches_exp) # np. of patches

    tf = time.time()
    t_run += tf - ts
    SRlogger.info(f"\t| Nr. of patches in frame {frame_i}: {patches_exp.shape[0]}")
    SRlogger.info(f"{'-'*5}Time to extract patches: {tf - ts:.5f} s | frame_time: {tf - t0_frame:.5f} s | SR_run_time: {t_run:.5f} s")


    SRlogger.info(f"\t| Upscaling experimental patches to create inputfor SRx2")
    ts = time.time()
    upscale_fac = 2
    patches_subset = patches_exp[:, x2mod_ch, :, :] # Extract only the channels we need
    upscaled_patches = np.repeat(np.repeat(patches_subset, upscale_fac, axis=2), upscale_fac, axis=3)
    upscaled_patches = upscaled_patches / (upscale_fac * upscale_fac) # Scale by upscale factor squared
    tf = time.time()
    t_run += tf - ts
    SRlogger.info(f"{'-'*5}Time to create upscaled patches for SRx2 input: {tf - ts:.5f} s | frame_time: {tf - t0_frame:.5f} s | SR_run_time: {t_run:.5f} s")

    SRlogger.info(f"\t| Normalizing upscaled input SRx2 patches")
    ts = time.time()
    max_vals = np.max(upscaled_patches, axis=(2, 3), keepdims=True)
    max_vals = np.where(max_vals == 0, 1, max_vals)  # Avoid division by zero
    Xin_x2 = upscaled_patches / max_vals
    tf = time.time()
    t_run += tf - ts
    SRlogger.info(f"{'-'*5}Time to normalize SRx2 input: {tf - ts:.5f} s | frame_time: {tf - t0_frame:.5f} s | SR_run_time: {t_run:.5f} s")


    SRlogger.info(f"\t| Running inference: SRx2 model")
    ts = time.time()
    n_batches = n_patches // sr_config["batch_size"]

    # Running SRx2 predictions in batches
    with torch.no_grad():
        for i in range(0, n_batches+1):
            i_s, i_f = i * sr_config["batch_size"], (min((i+1)*sr_config["batch_size"], n_patches))
            if i_s < n_patches:
                X_batch = Xin_x2[i_s:i_f]
                X_batch = torch.from_numpy(X_batch.astype(np.float32))
                X_batch = X_batch.to(torch_devs)
                SRx2_pred_batch = x2mod.forward(X_batch).detach().cpu().numpy()

                if i == 0:SRx2_pred = deepcopy(SRx2_pred_batch)
                else: SRx2_pred = np.append(SRx2_pred, SRx2_pred_batch, axis=0)

    tf = time.time()
    t_run += tf - ts
    SRlogger.info(f"{'-'*5} Time to predict SRx2 patches: {tf - ts:.5f} s | frame_time: {tf - t0_frame:.5f} s | SR_run_time: {t_run:.5f} s")


    if args.srfac > 2.5:
        SRlogger.info(f"\t| Upscaling SRx2 predicted patches to create input for SRx4 model")
        ts = time.time()
        upscale_fac = 2
        patches_subset = SRx2_pred[:, x4mod_ch, :, :] # Extract only the channels we need
        upscaled_patches = np.repeat(np.repeat(patches_subset, upscale_fac, axis=2), upscale_fac, axis=3)
        upscaled_patches = upscaled_patches / (upscale_fac * upscale_fac) # Scale by upscale factor squared
        tf = time.time()
        t_run += tf - ts
        SRlogger.info(f"{'-'*5} Time to create upscaled patches for SRx4 input: {tf - ts:.5f} s | frame_time: {tf - t0_frame:.5f} s | SR_run_time: {t_run:.5f} s")
        
        SRlogger.info(f"\t| Normalizing upscaled input SRx4 patches")
        ts = time.time()
        max_vals = np.max(upscaled_patches, axis=(2, 3), keepdims=True)
        max_vals = np.where(max_vals == 0, 1, max_vals)  # Avoid division by zero
        Xin_x4 = upscaled_patches / max_vals
        tf = time.time()
        t_run += tf - ts
        SRlogger.info(f"{'-'*5}Time to normalize SRx4 input: {tf - ts:.5f} s | frame_time: {tf - t0_frame:.5f} s | SR_run_time: {t_run:.5f} s")

        SRlogger.info(f"\t| Running inference: SRx4 model")
        ts = time.time()

        # Running SRx4 predictions in batches
        with torch.no_grad():
            for i in range(0, n_batches+1):
                i_s, i_f = i * sr_config["batch_size"], (min((i+1)*sr_config["batch_size"], n_patches))
                if i_s < n_patches:
                    X_batch = Xin_x4[i_s:i_f]
                    X_batch = torch.from_numpy(X_batch.astype(np.float32))
                    X_batch = X_batch.to(torch_devs)
                    SRx4_pred_batch = x4mod.forward(X_batch).detach().cpu().numpy()

                    if i == 0:
                        SRx4_pred = deepcopy(SRx4_pred_batch)
                    else:
                        SRx4_pred = np.append(SRx4_pred, SRx4_pred_batch, axis=0)

        tf = time.time()
        t_run += tf - ts
        SRlogger.info(f"{'-'*5} Time to predict SRx4 patches: {tf - ts:.5f} s | frame_time: {tf - t0_frame:.5f} s | SR_run_time: {t_run:.5f} s")


    if args.srfac > 4.5:
        SRlogger.info(f"\t| Upscaling SRx4 predicted patches to create input for SRx8 model")
        ts = time.time()
        upscale_fac = 2
        patches_subset = SRx4_pred[:, x8mod_ch, :, :] # Extract only the channels we need
        upscaled_patches = np.repeat(np.repeat(patches_subset, upscale_fac, axis=2), upscale_fac, axis=3)
        upscaled_patches = upscaled_patches / (upscale_fac * upscale_fac) # Scale by upscale factor squared
        tf = time.time()
        t_run += tf - ts
        SRlogger.info(f"{'-'*5} Time to create upscaled patches for SRx8 input: {tf - ts:.5f} s | frame_time: {tf - t0_frame:.5f} s | SR_run_time: {t_run:.5f} s")
        
        SRlogger.info(f"\t| Normalizing upscaled input SRx8 patches")
        ts = time.time()
        max_vals = np.max(upscaled_patches, axis=(2, 3), keepdims=True)
        max_vals = np.where(max_vals == 0, 1, max_vals)  # Avoid division by zero
        Xin_x8 = upscaled_patches / max_vals
        tf = time.time()
        t_run += tf - ts
        SRlogger.info(f"{'-'*5} Time to normalize SRx8 input: {tf - ts:.5f} s | frame_time: {tf - t0_frame:.5f} s | SR_run_time: {t_run:.5f} s")


        SRlogger.info(f"\t| Running inference: SRx8 model")
        ts = time.time()

        # Running SRx8 predictions
        # Use autocast if cuda-GPU is available
        if torch_devs.type == "cuda":
            with torch.no_grad(), autocast(torch_devs.type):
                for i in range(0, n_batches+1):
                    i_s, i_f = i * sr_config["batch_size"], (min((i+1)*sr_config["batch_size"], n_patches))
                    if i_s < n_patches:
                        X_batch = Xin_x8[i_s:i_f]
                        X_batch = torch.from_numpy(X_batch.astype(np.float32))
                        X_batch = X_batch.to(torch_devs)
                        #SRx8_pred_batch = x8mod.forward(X_batch).detach().cpu().numpy()
                        SRx8_pred_batch = x8mod.forward(X_batch)
                        current_sums = torch.sum(SRx8_pred_batch, dim=(1, 2, 3))  # Shape: [248]
                        target_sums = torch.tensor(patches_Isum, device=SRx8_pred_batch.device)
                        scaling_factors = target_sums / current_sums
                        scaling_factors = scaling_factors.view(-1, 1, 1, 1)
                        SRx8_pred_batch = SRx8_pred_batch * scaling_factors
                        SRx8_pred_batch = SRx8_pred_batch.detach().cpu().numpy()

                        if i == 0:
                            SRx8_pred = deepcopy(SRx8_pred_batch)
                        else:
                            SRx8_pred = np.append(SRx8_pred, SRx8_pred_batch, axis=0)

        # Run SRx8 predictions without autocast if cuda is not available
        if torch_devs.type != "cuda":
            with torch.no_grad():
                for i in range(0, n_batches+1):
                    i_s, i_f = i * sr_config["batch_size"], (min((i+1)*sr_config["batch_size"], n_patches))
                    if i_s < n_patches:
                        X_batch = Xin_x8[i_s:i_f]
                        X_batch = torch.from_numpy(X_batch.astype(np.float32))
                        X_batch = X_batch.to(torch_devs)
                        #SRx8_pred_batch = x8mod.forward(X_batch).detach().cpu().numpy()
                        SRx8_pred_batch = x8mod.forward(X_batch)
                        current_sums = torch.sum(SRx8_pred_batch, dim=(1, 2, 3))  # Shape: [248]
                        target_sums = torch.tensor(patches_Isum, device=SRx8_pred_batch.device)
                        scaling_factors = target_sums / current_sums
                        scaling_factors = scaling_factors.view(-1, 1, 1, 1)
                        SRx8_pred_batch = SRx8_pred_batch * scaling_factors
                        SRx8_pred_batch = SRx8_pred_batch.detach().cpu().numpy()

                        if i == 0:
                            SRx8_pred = deepcopy(SRx8_pred_batch)
                        else:
                            SRx8_pred = np.append(SRx8_pred, SRx8_pred_batch, axis=0)


        tf = time.time()
        t_run += tf - ts
        SRlogger.info(f"{'-'*5} Time to predict SRx8 patches: {tf - ts:.5f} s | frame_time: {tf - t0_frame:.5f} s | SR_run_time: {t_run:.5f} s")

    ts = time.time()
    # Based on srfac value, decide which patches will be used for peak analysis
    if args.srfac==2:
        patches_to_fit = SRx2_pred
        SRlogger.info(f"\t| Using SRx2 predicted patches for peak analysis")
    if args.srfac==4:
        patches_to_fit = SRx4_pred
        SRlogger.info(f"\t| Using SRx4 predicted patches for peak analysis")
    if args.srfac==8:
        patches_to_fit = SRx8_pred
        SRlogger.info(f"\t| Using SRx8 predicted patches for peak analysis")
    tf = time.time()
    t_run += tf - ts
    SRlogger.info(f"{'-'*5} Time to decide which patches will be used for peak analysis: {tf - ts:.5f} s | frame_time: {tf - t0_frame:.5f} s | SR_run_time: {t_run:.5f} s")


    ts = time.time()
    spotID = int(0)
    omega = 180 - (omega_stepsize) * (frame_i - 1)
    df_peaks_frame_i = pd.DataFrame(columns=col_names)

    SRlogger.info(f"\t| Initiating peak fitting process for frame {frame_i}")
    if str(sr_config["fitPeakShapePV"]).lower() in ["no", "n", "false", "f", "0"]:
        SRlogger.info(f"\t    |Pseudo-voigt fitting DISABLED. Peak positions will be fitted with center of mass algorithm + watershed")
    if str(sr_config["fitPeakShapePV"]).lower() in ["yes", "y", "true", "t", "1"]:
        SRlogger.info(f"\t    |Pseudo-voigt fitting ENABLED.")
    if str(sr_config["fitPeakShapePV"]).lower() in ["auto", "partial", "mix"]:
        SRlogger.info(f"\t    |If multiple peaks in patch: Use pseudo-Voigt fit. If single peak: Use center of mass algorithm + watershed")
    tf = time.time()
    t_run += tf - ts
    SRlogger.info(f"{'-'*5} Time to initiate peak fitting process for frame {frame_i}: {tf - ts:.5f} s | frame_time: {tf - t0_frame:.5f} s | SR_run_time: {t_run:.5f} s")

    ts = time.time()
    # Get Radius values for all patches in the frame
    R_patches = np.sqrt((sr_params["Ypx_BC"] - (np.array(patches_Y00) + sr_config["spot_find_args"]["patch_size"]/2))**2 +
                        (sr_params["Zpx_BC"] - (np.array(patches_Z00) + sr_config["spot_find_args"]["patch_size"]/2))**2)

    # Find indices of patches that belong to a possible ring
    patches_Idx_for_fitting = [i for (i, R_patch) in enumerate(R_patches) if \
                                min(np.abs(np.array(ring_R_values) - R_patch)) < sr_config["R_deviation"]]

    for patch_i in patches_Idx_for_fitting:
        patch = patches_to_fit[patch_i,0]
        Z00, Y00 = patches_Z00[patch_i], patches_Y00[patch_i]

        if (str(sr_config["fitPeakShapePV"]).lower() not in ["yes", "y", "true", "t", "1"]):
            patch_for_locmax = gaussian_filter(patch, sigma=sr_config["peak_find_args"]["gauss_filter_sigma"][f"SRx{args.srfac}"])
            patch_for_locmax = median_filter(patch_for_locmax, size=sr_config["peak_find_args"]["median_filter_size"][f"SRx{args.srfac}"])

            locmax_peak_coords = peak_local_max(patch_for_locmax,
                                                min_distance=sr_config["peak_find_args"]["min_d"][f"SRx{args.srfac}"],
                                                threshold_rel=sr_config["peak_find_args"]["thresh_rel"][f"SRx{args.srfac}"])
            
            n_peaks_in_patch = len(locmax_peak_coords)

        if (str(sr_config["fitPeakShapePV"]).lower() in ["no", "n", "false", "f", "0"]) or\
            (str(sr_config["fitPeakShapePV"]).lower() in ["auto", "partial", "mix"] and n_peaks_in_patch == 1):

            # Cutting off peaks that are too close to the edges of the patch
            bnd_peak_cutoff = sr_config["edge_bound_cutoff_fac"] * args.srfac
            locmax_peak_coords = locmax_peak_coords[~(locmax_peak_coords[:,:]<bnd_peak_cutoff).any(1)]
            locmax_peak_coords = locmax_peak_coords[~(locmax_peak_coords[:,:]>((sr_config["lrsz"] * args.srfac)-bnd_peak_cutoff)).any(1)]
            
            com_coords = com_peak_coords(patch, locmax_peak_coords,
                                            threshold=0.7,
                                            peak_crop_size=sr_config["peak_find_args"]["peak_crop_size"][f"SRx{args.srfac}"])

            # Apply watershed algorithm to allocate regions
            # Thresh for mask should be same as the threshold used to find local maxima
            # otherwise, watershed may end up ignoring a peak
            labelled_patch, peak_label_values, peaks_Isum, peaks_NrPixels = \
                watershed_peaks(patch, locmax_peak_coords,
                                mask_thresh=sr_config["peak_find_args"]["thresh_rel"][f"SRx{args.srfac}"])

            n_peaks_in_patch = len(com_coords)
            for (peak_i, peak_label) in zip(range(n_peaks_in_patch), peak_label_values):
                spotID += 1

                # Segment out the current peak from the patch
                peak_i_patch = patch * (labelled_patch == peak_label)

                #---------------------------
                # Downscale the peak patch to SRx1
                downscale_fac = int(args.srfac / 1) # downscale factor: number of pixels (in each direction) in input patch per pixel in downscaled image
                h, w = peak_i_patch.shape
                new_h, new_w = h // downscale_fac, w // downscale_fac # Reshape input to group pixels for pooling

                # Reshape and sum using vectorized operations
                reshaped = peak_i_patch[:new_h * downscale_fac, :new_w * downscale_fac].reshape(
                    new_h, downscale_fac, new_w, downscale_fac)
                
                peak_i_patch_SRx1 = np.sum(reshaped, axis=(1, 3))
                #-------

                IntegratedIntensity = np.sum(peak_i_patch_SRx1)
                NrPixels = np.count_nonzero(peak_i_patch_SRx1 > 1E-2)
                IMax = np.max(peak_i_patch_SRx1)
                TotalNrPixelsInPeakRegion = nr_pixels_in_patch[patch_i]

                # Get peak position in detector frame pixels
                YCen_px = Y00 + (com_coords[peak_i][1] * (1/args.srfac)) + float(shiftYpx)
                ZCen_px = Z00 + (com_coords[peak_i][0] * (1/args.srfac)) + float(shiftZpx)
                
                # Get peak R & Eta values from Y & Z positions
                R = math.sqrt((sr_params["Ypx_BC"] - YCen_px)**2 + (sr_params["Zpx_BC"] - ZCen_px)**2)
                Eta = math.degrees(math.acos((ZCen_px - sr_params["Zpx_BC"]) / R))
                
                # Convert Eta_inDeg to -ve where Ypx < Ypx_BC
                Eta = Eta * ((YCen_px - sr_params["Ypx_BC"]) / np.abs(YCen_px - sr_params["Ypx_BC"]))
            
                SigmaR, SigmaEta = 1, 1 # default sigma values

                # Get max Y & Z positions in SRx1 detector frame pixels
                r_max_SRx1, c_max_SRx1 = np.unravel_index(np.argmax(peak_i_patch_SRx1), peak_i_patch_SRx1.shape)  # (row, col)
                maxY = Y00 + c_max_SRx1
                maxZ = Z00 + r_max_SRx1
                # Get difference values between peak positions in SRx1 and MIDAS detector frames
                diffY = maxY - YCen_px
                diffZ = maxZ - ZCen_px

                peak_data = [spotID, IntegratedIntensity, omega, YCen_px, ZCen_px, IMax, R, Eta, SigmaR, SigmaEta, NrPixels, TotalNrPixelsInPeakRegion, n_peaks_in_patch, maxY, maxZ, diffY, diffZ]

                # Add to peaks dataframe only if Eta is within the range of minEta and if the number of pixels is greated than the minimum needed for a peak
                if sr_config["minEta"] == 0:
                    # No need to check filtering criteria for Eta; just add the peak if it has enough pixels
                    if TotalNrPixelsInPeakRegion >= sr_config["minPxCount"]:
                        df_peaks_frame_i.loc[spotID-1] = peak_data
                else:
                    # Check filtering criteria
                    if (sr_config["minEta"] <= abs(Eta) <= (180 - sr_config["minEta"])) and (TotalNrPixelsInPeakRegion >= sr_config["minPxCount"]):
                        df_peaks_frame_i.loc[spotID-1] = peak_data


        if (str(sr_config["fitPeakShapePV"]).lower() in ["yes", "y", "true", "t", "1"]) or\
            (str(sr_config["fitPeakShapePV"]).lower() in ["auto", "partial", "mix"] and n_peaks_in_patch >= 2):

            try:
                # Perform peak fitting on the patch to get peak parameters
                pvfit_peaks, _, _ = multi_pv_fit(patch, patches_Y00[patch_i], patches_Z00[patch_i],
                                    sr_params["Ypx_BC"], sr_params["Zpx_BC"], sr_config["lrsz"], args.srfac,
                                    min_distance=sr_config["peak_find_args"]["min_d"][f"SRx{args.srfac}"],
                                    threshold_rel=sr_config["peak_find_args"]["thresh_rel"][f"SRx{args.srfac}"],
                                    gauss_filter_sigma=sr_config["peak_find_args"]["gauss_filter_sigma"][f"SRx{args.srfac}"],
                                    median_filter_size=sr_config["peak_find_args"]["median_filter_size"][f"SRx{args.srfac}"],
                                    lr_int_thresh=sr_config["peak_find_args"]["pvfit_int_thresh"][f"SRx{args.srfac}"])
                
                n_peaks_in_patch = len(pvfit_peaks)
                for peak_i in range(n_peaks_in_patch):
                    
                    spotID += 1
                    peak_prop = pvfit_peaks[peak_i]
                    R, Eta = peak_prop["R(px)"], peak_prop["Eta(deg)"]
                    
                    # Create RR and EE spread over the patch
                    dpx_sr = 1 / args.srfac # pixel step size in super-res patch

                    # Y-Z coordinate values
                    Ypx = np.arange(Y00, Y00 + sr_config["spot_find_args"]["patch_size"], dpx_sr)
                    Zpx = np.arange(Z00, Z00 + sr_config["spot_find_args"]["patch_size"], dpx_sr)

                    # Create meshgrids of Radius and Eta values
                    grid_YY, grid_ZZ = np.meshgrid(Ypx, Zpx)
                    grid_RR = ((sr_params["Ypx_BC"] - grid_YY)**2 + (sr_params["Zpx_BC"] - grid_ZZ)**2)**0.5
                    grid_EE = np.rad2deg(np.arccos((grid_ZZ - sr_params["Zpx_BC"]) / grid_RR))
                    # Convert Eta_inDeg to -ve where Ypx < Ypx_BC
                    grid_EE = grid_EE * ((grid_YY - sr_params["Ypx_BC"]) / np.abs(grid_YY - sr_params["Ypx_BC"]))

                    peak_fit_patch = pseudoVoigt2d_diffLGwidth(grid_RR, grid_EE,
                                            y0=R, z0=Eta,
                                            ySigG=peak_prop["SigGR"], zSigG=peak_prop["SigGEta"],
                                            ySigL=peak_prop["SigLR"], zSigL=peak_prop["SigLEta"],
                                            LGmix=peak_prop["LGmix"], IMax=peak_prop["IMax"])
                    
                    #---------------------------
                    # Downscale the peak patch to SRx1
                    downscale_fac = int(args.srfac / 1) # downscale factor: number of pixels (in each direction) in input patch per pixel in downscaled image
                    h, w = peak_fit_patch.shape
                    new_h, new_w = h // downscale_fac, w // downscale_fac # Reshape input to group pixels for pooling

                    # Reshape and sum using vectorized operations
                    reshaped = peak_fit_patch[:new_h * downscale_fac, :new_w * downscale_fac].reshape(
                        new_h, downscale_fac, new_w, downscale_fac)
                    
                    peak_fit_patch_SRx1 = np.sum(reshaped, axis=(1, 3))
                    #-------

                    r_max_SRx1, c_max_SRx1 = np.unravel_index(np.argmax(peak_fit_patch_SRx1), peak_fit_patch_SRx1.shape)  # (row, col)
                    maxY = Y00 + c_max_SRx1
                    maxZ = Z00 + r_max_SRx1

                    IntegratedIntensity = np.sum(peak_fit_patch)
                    YCen_px = peak_prop["Y(px)"] + shiftYpx
                    ZCen_px = peak_prop["Z(px)"] + shiftZpx
                    IMax = np.max(peak_fit_patch_SRx1)
                    SigmaR = max(peak_prop["SigGR"], peak_prop["SigLR"])
                    SigmaEta = max(peak_prop["SigGEta"], peak_prop["SigLEta"])
                    NrPixels = np.count_nonzero(peak_fit_patch_SRx1 * patches_exp[patch_i])
                    TotalNrPixelsInPeakRegion = nr_pixels_in_patch[patch_i]

                    diffY = maxY - YCen_px
                    diffZ = maxZ - ZCen_px
                    peak_data = [spotID, IntegratedIntensity, omega, YCen_px, ZCen_px, IMax, R, Eta, SigmaR, SigmaEta, NrPixels, TotalNrPixelsInPeakRegion, n_peaks_in_patch, maxY, maxZ, diffY, diffZ]

                    # Add to peaks dataframe only if Eta is within the range of minEta and if the number of pixels is greated than the minimum needed for a peak
                    if sr_config["minEta"] == 0:
                        # No need to check filtering criteria for Eta; just add the peak if it has enough pixels
                        if TotalNrPixelsInPeakRegion >= sr_config["minPxCount"]:
                            df_peaks_frame_i.loc[spotID-1] = peak_data
                    else:
                        # Check filtering criteria
                        if (sr_config["minEta"] <= abs(Eta) <= (180 - sr_config["minEta"])) and (TotalNrPixelsInPeakRegion >= sr_config["minPxCount"]):
                            df_peaks_frame_i.loc[spotID-1] = peak_data

            except:
                SRlogger.info(f"\tFrame {frame_i}: Patch {patch_i} skipped as pseudoVoigt fit did not converge.")

    tf = time.time()
    t_run += tf - ts
    SRlogger.info(f"{'-'*5} Time to fit peaks (frame {frame_i}): {tf - ts:.5f} s | frame_time: {tf - t0_frame:.5f} s | SR_run_time: {t_run:.5f} s")


    df_peaks_frame_i.to_csv(frame_i_csv_savepath, sep="\t", index=False)

