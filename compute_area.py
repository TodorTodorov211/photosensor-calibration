import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import read_data as reader
from scipy.stats import crystalball
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import sys
#change this folder so the import is correct
sys.path.append("/home/todor/University/MPhys project/MPhys_project/utils/")
from plotting_utils import plot1d
from plotting_utils import get_bin_centres
from plotting_utils import get_bin_index
from read_data import make_heatmap
from scipy.stats import norm
from scipy.stats import moyal
from landaupy import landau
from landaupy import langauss
from copy import deepcopy
from iminuit import Minuit
from iminuit import cost
import iminuit

#change these folders depending on where you want to save data and plots
DATA_FOLDER = "/home/todor/University/MPhys project/phoyosensor-calibration/data/"
PLOTS_FOLDER = "/home/todor/University/MPhys project/phoyosensor-calibration/plots/"
RESULTS_FOLDER = "/home/todor/University/MPhys project/phoyosensor-calibration/results/"

LOC_DATA_SIPM = "/home/todor/University/MPhys project/Data_SiPM/"
LOC_DATA_PMT = "/home/todor/University/MPhys project/Data_PMT/"



def mirror_crystalball(x, beta, m, loc, scale, norm, offset):
    result = crystalball.pdf(-x, beta, m, -loc, scale) * norm + offset
    return result


def linear(x, a, b):
    return a * x + b


def norm_gaus(x, loc, sigma, normalisation):
    return norm.pdf(x, loc, sigma) * normalisation





def chi2(data, model, unc, no_params=21):
    """A function to calculate the chi^2.  Can be replaced with Minuit.cost.LeastSquares

    Parameters
    ----------
    data : list
        experimental data
    model : list
        theoretical values
    unc : list
        errors in the theoretical values
    no_params : int, optional
        number of parameters for reduced chi^2 calculation, by default 21

    Returns
    -------
    float
        chi^2
    """
    chi2 = np.sum((data - model)**2 / unc**2)
    chi2_per_DoF = chi2 / (len(data) - no_params)
    print("Chi2 : {0} and chi2 per DoF : {1}".format(chi2, chi2_per_DoF))
    return chi2




def determine_roi(all_waveforms, plot=False):
    """Determine the Region of Interest (ROI).  Locate the common peak in all waveforms 
    and the beginning of integration region.  An end value for the ROI is provided but is later 
    discarded.

    Parameters
    ----------
    all_waveforms : list of 2d np arrays
        list with each entry being a 2d np array describing a waveform with columns (time, amplitude)
    plot : bool, optional
        plot the determined roi or not, by default False

    Returns
    -------
    int, int, int
        indices of begin of ROI, end of ROI, location of peak
    """
    time = []
    amplitude = []
    bins = len(all_waveforms[0])
    
    print("Determining ROI...")
    for index, waveform in tqdm(enumerate(all_waveforms)):
        for index_inner, single_point in enumerate(waveform):
            time.append(single_point[0])
            amplitude.append(single_point[1])
    
    smallest = np.min(amplitude)
    amplitude = amplitude + np.abs(smallest)

    time = np.array(time)
    time *= 10**9

    hist, edges = np.histogram(time, bins, weights=amplitude)
    plot1d(hist, edges, alpha= 0.2, label='raw signal')
    hist_filtered = savgol_filter(hist, 60, 9)
    hist_deriv = savgol_filter(hist, 60, 9, 1, edges[1] - edges[0])
    visual_extrema = np.where(np.abs(hist_deriv) <= np.max(hist_deriv)/30, hist_filtered, np.full(np.shape(hist), np.nan))
    plot1d(hist_filtered, edges, alpha = 0.2, color='r', label='smoothed signal')
    #plt.scatter(get_bin_centres(edges), visual_extrema,  color='r')
    
    numerical_extrema = np.where(np.isnan(visual_extrema) , np.zeros(np.shape(visual_extrema)), visual_extrema)
    
    max_loc = np.argmax(numerical_extrema)
    plt.scatter(get_bin_centres(edges)[max_loc], visual_extrema[max_loc], color='r')
    #print(visual_extrema[max_loc])

    #define some dummy variables to find the peak before 
    iterator = max_loc
    activate = False
    roi_begin = 0

    while iterator >= 0:

        if(numerical_extrema[iterator] == 0):
            activate = True
        
        else:
            if activate == True:
                roi_begin = iterator
                break


        iterator -= 1


    plt.scatter(get_bin_centres(edges)[roi_begin], visual_extrema[roi_begin], color='r')

    #For the upper ROI limit just take 3 times the 
    diff = max_loc - roi_begin
    roi_end = max_loc + 1 * diff
    

    #plt.scatter(get_bin_centres(edges)[roi_end], hist_filtered[roi_end], color='r')
    plt.xlabel("time[ns]", fontsize = 18)
    plt.ylabel("Summed signal[V]", fontsize = 18)
    #plt.title("ROI determination")
    plt.legend(fontsize=18)
    plt.tick_params(axis='both', labelsize=16)
    plt.tight_layout()
    

    if plot==True:
        #plt.savefig("PLOTS_FOLDER + ROI_example_sipm.png", dpi=600)
        plt.show()
    plt.cla()
    print("ROI determined to be [{0}:{1}]".format(roi_begin, roi_end))

    return roi_begin, roi_end, max_loc


def filter_outliers(all_waveforms, max_loc, roi=[], plot=False):
    """Filters waveforms that peak away from the predetermined peak

    Parameters
    ----------
    all_waveforms : list of ndarray
        a list containing all waveforms, each waveform is an ndarray with first column time[s]
        and second column amplitude[V]
    max_loc : int
        index of the peak in every waveform
    roi : list, optional
        begin and end indices of the roi, by default []
    plot : bool, optional
        plot the waveforms or not, by default False

    Returns
    -------
    list of ndarray
        list of filtered waveforms
    """    
    filtered_waveforms = []
    double_filtered_waveforms = []
    trash = []  #use for testing
    max_diff = 0
    all_baseline_vars = []
    all_diff = []
    trashed_var = 0
    print("Calculating average values...")
    for index, waveform in tqdm(enumerate(all_waveforms)):
        maximum_value_loc = np.argmax(waveform[:, 1])

        
        max_value = np.max(waveform[:, 1])
        min_value = np.min(waveform[:, 1])
        diff = max_value - min_value
        all_diff.append(diff)
        if diff > max_diff:
            max_diff = diff
        
        """
        if diff < max_diff / 10:
            filtered_waveforms.append(waveform)
            continue
        """
        baseline_var = np.var(waveform[:roi[0], 1])
        all_baseline_vars.append(baseline_var)


        """
        if(maximum_value_loc < roi[1] and maximum_value_loc > roi[0]):


            if np.abs(maximum_value_loc - max_loc) > (max_loc - roi[0])*0.7:
                trash.append(waveform)
                continue
            filtered_waveforms.append(waveform)
            continue
        trash.append(waveform)
        """

    average_diff = np.average(all_diff)
    average_baseline_var = np.average(all_baseline_vars)
    baseline_var_cut = np.quantile(all_baseline_vars, 1.00)
    #print(baseline_var_cut)
    
    print("Eliminating outliers...")
    for index1, waveform1 in tqdm(enumerate(all_waveforms)):
        maximum_value_loc = np.argmax(waveform1[:, 1])
        max_value = np.max(waveform1[:, 1])
        min_value = np.min(waveform1[:, 1])
        diff = max_value - min_value
        if diff < average_diff / 2:
            filtered_waveforms.append(waveform1)
            continue
        
        baseline_var = np.var(waveform1[:roi[0], 1])
        if baseline_var > baseline_var_cut:
            trash.append(waveform1)
            trashed_var += 1
            continue
        
        if(maximum_value_loc < roi[1] and maximum_value_loc > roi[0]):


            if np.abs(maximum_value_loc - max_loc) > (max_loc - roi[0])*0.7:
                trash.append(waveform1)
                continue
            filtered_waveforms.append(waveform1)
            continue
        trash.append(waveform1)

        



    if plot == True:
        #make_heatmap(filtered_waveforms)

        #make_heatmap(trash[10:12], True, "example_filtered.png")
        plot_indices = [10, 11]
        lim_max = np.max(np.vstack(filtered_waveforms)[:, 1])
        lim_min = np.min(np.vstack(filtered_waveforms)[:, 1])
        for index in plot_indices:
            plt.plot(trash[index][:, 0], trash[index][:, 1])
        plt.ylim(bottom=lim_min, top=lim_max)
        plt.show()
        #np.savetxt("/home/todor/University/MPhys project/MPhys_project/analyze-lecroy/data/rejected_waveforms_example.csv", np.vstack(trash), delimiter=',')
    
    print("Total entries left after filtering: {}".format(len(filtered_waveforms)))
    
    return filtered_waveforms


def find_area(selected_waveforms, roi=[], no_bins=150, plot=False, save=False, save_loc=DATA_FOLDER, savename="areas_sipm-1_56V_sl.csv"):
    """
    Filters waveforms that have high slope in the baseline region (top and bottom 10%) 
    Calculates integration limits for all waveforms (non-flat ones)
    Shifts all waveforms to 0
    Integrates in the determined regions all waveforms (or up until the average of other regions)
    Saves all areas to a file

    Parameters
    ----------
    selected_waveforms : list of ndarray
        a list containing all waveforms after first filter, each waveform is an ndarray with first column time[s]
        and second column amplitude[V]
    roi : list, optional
        begin and end indices of roi(the end is discarded, only used as an initial estimation), by default []
    no_bins : int, optional
        number of bins in histogram, by default 150
    plot : bool, optional
        show plots of waveforms after filtering and shifting and areas histogram, by default False
    save : bool, optional
        save the areas to csv file or not, by default False
    save_loc : str, optional
        location of the saved csv, by default DATA_FOLDER
    savename : str, optional
        filename of the areas csv, by default "areas_sipm-1_56V_sl.csv"

    Returns
    -------
    list<float>, ndarray, ndarray
        list of all areas, histogrammed areas, bins of the histogram
    """    
    areas = []
    #all_amplitudes = np.array([])
    shifted_waveforms = []
    negative_waveforms = []
    negative_var = []
    slopes = []
    print("Calculating areas...")
    roi_upper_all = [roi[1]]

    for index_1, waveform_1 in enumerate(selected_waveforms):
        #looking for minimum
        baseline = np.average(waveform_1[:roi[0], 1])
        waveform_1[:, 1] -= baseline
        #if(- np.min(waveform_1[:, 1]) < np.max(waveform_1[:, 1] / 0.75)):
            #shifted_waveforms.append(waveform_1)
        
        popt, pcov = curve_fit(linear, waveform_1[:roi[0], 0], waveform_1[:roi[0], 1])
        slopes.append(popt[0])
        """
        below_zero = np.flatnonzero(waveform_1[:, 1] < 0)
        if len(below_zero) < len(waveform_1[:, 1]) /2:

            shifted_waveforms.append(waveform_1)
        else:
            negative_waveforms.append(waveform_1)
            negative_var.append(np.var(waveform_1[:, 1]))
        """
        roi_effective = list(roi)
        peak_loc = np.argmax(waveform_1[:, 1])
        
        
        tail_indices = np.flatnonzero(waveform_1[peak_loc:, 1] <= np.max(waveform_1[:, 1]) * 0.1)
        if(tail_indices.size == 0):
            tail_indices = [0]
        tail_indices += peak_loc
        if tail_indices[0] > roi_effective[1]:
            
            roi_effective[1] = deepcopy( tail_indices[0])
            roi_upper_all.append(roi_effective[1])

        #all_amplitudes = np.append(all_amplitudes, waveform_1[:, 1])
    #var_cut = np.quantile(negative_var, 0.65)
    slope_cut_max = np.quantile(slopes, 0.9)
    slope_cut_min = np.quantile(slopes, 0.1)
    for index_2, waveform_2 in enumerate(selected_waveforms):
        if(slopes[index_2] <= slope_cut_max and slopes[index_2] >= slope_cut_min):
            shifted_waveforms.append(waveform_2)
        else:
            negative_waveforms.append(waveform_2)

    if plot == True:
        make_heatmap(shifted_waveforms, True, "shifted_waveforms_sipm-411_56V.png", False, "Waveforms after filtering for a single PMT at 850V")
        
    #shift = - np.min(all_amplitudes)
    
    #negative_waveforms = []
    
    for index, waveform in tqdm(enumerate(shifted_waveforms)):
        #waveform[:, 1] = waveform[:, 1] + shift #shift to above 0 before integration

        roi_effective = list(roi)
        peak_loc = np.argmax(waveform[:, 1])
        tail_indices = np.flatnonzero(waveform[peak_loc:, 1] <= np.max(waveform[:, 1]) * 0.1)
        if(tail_indices.size == 0):
            tail_indices = [0]
        tail_indices += peak_loc
        #print(tail_indices[0])
        if tail_indices[0] > roi_effective[1]:
            
            roi_effective[1] = deepcopy( tail_indices[0])
            
            
            #print(roi_effective)
        else:
            roi_effective[1] = int(np.average(roi_upper_all))
        
        integral = np.trapz(waveform[roi_effective[0]:roi_effective[1], 1], waveform[roi_effective[0]:roi_effective[1], 0], waveform[1, 0] - waveform[0, 0])
        areas.append(integral)
        if integral < 7e-11 and integral > 3e-11:
            #negative_waveforms.append(waveform)
            continue

    print("New ROI upper limit changed from {0} to {1}".format(roi[1], np.average(roi_upper_all)))  #check which roi limit is better
    

    hist, bins = np.histogram(areas, no_bins)
    if plot == True:
        plot1d(hist, bins)
        plt.xlabel("Charge[nV.s]")
        plt.ylabel("Entries[0.3 counts/(nV.ms)]")
        plt.title("Charge for a single PMT")
        plt.savefig(PLOTS_FOLDER + "areas_example_sipm.png", dpi=600)
        plt.show()
        #make_heatmap(negative_waveforms, True, "cancelled_waveforms.png")
        #np.savetxt("/home/todor/University/MPhys project/MPhys_project/analyze-lecroy/data/cancelled_waveforms_example_2.csv", np.vstack(negative_waveforms), delimiter=',')
    
    if save == True:
        fname = save_loc + savename
        np.savetxt(fname, areas, delimiter=',')
        print("Areas saved to file at: " + fname)

    return areas, hist, bins




    

def procedure_areas_save(sipm_no=411, voltage="56V", Ch=None, sipm_str='1'):
    """Procedure to call the functions to read data, find ROI, filter waveforms,
    integrate and save all areas to file

    Parameters
    ----------
    sipm_no : int, optional
        The number of the sipm as written in the data file names, by default 411
    voltage : str, optional
        voltage as written in the data file names, by default "56V"
    Ch : _type_, optional
        channel prefix added by scope, by default None
    sipm_str : str, optional
        in the case 2 sipms were recorded at once this string is how the filename was recorded, by default '1'
    """
    fname = voltage + "_sipm-" + sipm_str + "_1000--"
    if Ch != None:
        fname = Ch + "--" + fname
    location = LOC_DATA_SIPM+ str(sipm_no) + "/" + voltage + "/"
    all_waveforms = reader.iterate_large_files(0, 25, fname, loc=location)
    roi_begin, roi_end, peak_loc =  determine_roi(all_waveforms, plot=False)
    roi = [roi_begin, roi_end]
    
    filtered_waveforms = filter_outliers(all_waveforms, peak_loc, roi, plot=False)
    areas, hist, bins = find_area(filtered_waveforms, roi, no_bins=300, save=True, plot=True, savename="areas_sipm-" + str(sipm_no) + "_" + voltage + ".csv")


def procedure_bkg():
    """Procedure to analyze and find the histogram of the background (datasets with no LED power)
    """
    all_waveforms = reader.iterate_large_files(0, 23, "C2--57V_sipm-412413_bkg_1000--", loc=LOC_DATA_SIPM + "413/bkg/")
    roi_begin, roi_end, peak_loc =  38, 381, 110
    roi = [roi_begin, roi_end]
    filtered_waveforms = filter_outliers(all_waveforms, peak_loc, roi, plot=False)
    areas, hist, bins = find_area(filtered_waveforms, roi, no_bins=600, save=True, plot=True, savename="areas_sipm-413_bkg_57V.csv")
    plot1d(hist, bins)
    plt.xlabel("Charge[V.s]")
    plt.ylabel("Entries[2.7 counts/(nV.ms)]")
    plt.title("Data readout with no LED")
    plt.show()



        
def find_all_areas(sipm_no=411, voltage=[54, 59], Ch=None, sipm_str='1'):
    """calls procedure_areas_save for all voltages of a single SiPM

    Parameters
    ----------
    sipm_no : int, optional
        SiPM number as written in all data files, by default 411
    voltage : list, optional
        lower and upper limit of voltages to iterate over, by default [54, 59]
    Ch : str, optional
        channel prefix sometimes added by scope in the data file name, by default None
    sipm_str : str, optional
        if 2 sipms are read at once they are saved under the same name.  THis replaces sipm_no, i.e. '412413', by default '1'
    """
    for V in range(voltage[0], voltage[1] + 1):
        
        V_string = str(V) + "V"
        print("saving data for " + V_string)
        procedure_areas_save(sipm_no=sipm_no, voltage=V_string, Ch=Ch, sipm_str=sipm_str)

def make_pretty_plots():
    """Make pretty plots for presentation.  No use for the actual analysis
    """
    all_waveforms = reader.iterate_large_files(0, 25, "56V_sipm-1_1000--", loc=LOC_DATA_SIPM + "411/56V/")
    roi1, roi2, max_loc = determine_roi(all_waveforms, True)
    
    filtered_waveforms = filter_outliers(all_waveforms, max_loc, [roi1, roi2], plot=True)
    
    find_area(filtered_waveforms, [roi1, roi2], 600, True)
    
    rejected_1 = reader.read_large_file("rejected_waveforms_example.csv")
    rejected_2 = reader.read_large_file("cancelled_waveforms_example_2.csv")
    # shift waveforms
    for index, entry in enumerate(rejected_1):
        displacement = np.average(entry[:roi1, 1])
        entry[:, 1] -= displacement
        rejected_1[index] = entry
    indices_1 = [51, 23]
    indices_2 = [123, 222]
    for index_1 in indices_1:
        plt.plot(rejected_1[index_1][:, 0], rejected_1[index_1][:, 1], color='r', label="first filter")
    for index_2 in indices_2:
        plt.plot(rejected_2[index_2][:, 0], rejected_2[index_2][:, 1], color='b', label="second filter")

    plt.xlabel("Time [s]")
    plt.ylabel("Amplified signal [V]")
    plt.title("Rejected waveform examples")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.tight_layout()
    plt.savefig(PLOTS_FOLDER + "rejected_examples.png")
    plt.show()


    


if __name__ == "__main__":
    
    make_pretty_plots()




    
    



