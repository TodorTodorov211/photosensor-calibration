import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import read_data as reader
from scipy.stats import crystalball
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import sys
sys.path.append("utils/")
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



def mirror_crystalball(x, beta, m, loc, scale, norm, offset):
    result = crystalball.pdf(-x, beta, m, -loc, scale) * norm + offset
    return result


def linear(x, a, b):
    return a * x + b


def norm_gaus(x, loc, sigma, normalisation):
    return norm.pdf(x, loc, sigma) * normalisation


def model(x, mu1, mu2, mu3, mu4, mu5, mu7, err1, err2, err3, err4, err5, err7, n1, n2, n3, n4, n5, n7):

    return ( (norm.pdf(x, mu1, err1) * n1) + (norm.pdf(x, mu2, err2) * n2) + (norm.pdf(x, mu3, err3) * n3) + (norm.pdf(x, mu4, err4) * n4) + (norm.pdf(x, mu5, err5) * n5)  + (landau.pdf(x, mu7, err7) * n7) )


def model_gauss(x, mu1, mu2, mu3, mu4, mu5, mu6, mu7, err1, err2, err3, err4, err5, err6, err7, n1, n2, n3, n4, n5, n6, n7, err_gauss):
    return ( (norm.pdf(x, mu1, err1) * n1) + (norm.pdf(x, mu2, err2) * n2) + (norm.pdf(x, mu3, err3) * n3) + (norm.pdf(x, mu4, err4) * n4) + (norm.pdf(x, mu5, err5) * n5) + (norm.pdf(x, mu6, err6) * n6)  + (langauss.pdf(x, mu7, err7, err_gauss) * n7) )


def dep_model(x, mu0, err0, G, sigma, mu7, err7, n1, n2, n3, n4, n5, n7):
    mu=[mu0]
    err=[err0]
    for i in range(1, 5):
        mu.append(mu0 + i * G)
        err.append(np.sqrt(err0**2 + i * sigma**2) )
    
    return model(x, mu[0], mu[1], mu[2], mu[3], mu[4], mu7, err[0], err[1], err[2], err[3], err[4], err7, n1, n2, n3, n4, n5, n7)


def model_cdf(x, mu1, mu2, mu3, mu4, mu5, mu6, mu7, err1, err2, err3, err4, err5, err6, err7, n1, n2, n3, n4, n5, n6, n7):
    return norm.cdf(x, mu1, err1) * n1 + norm.cdf(x, mu2, err2) * n2 + norm.cdf(x, mu3, err3) * n3 + norm.cdf(x, mu4, err4) * n4 + norm.cdf(x, mu5, err5) * n5 + norm.cdf(x, mu6, err6) * n6  + landau.cdf(x, mu7, err7) * n7


def model_cdf_binned(bin_edges, mu1, mu2, mu3, mu4, mu5, mu6, mu7, err1, err2, err3, err4, err5, err6, err7, n1, n2, n3, n4, n5, n6, n7):
    x = get_bin_centres(bin_edges, dtype='ndarray')
    return model_cdf(x, mu1, mu2, mu3, mu4, mu5, mu6, mu7, err1, err2, err3, err4, err5, err6, err7, n1, n2, n3, n4, n5, n6, n7)


def simple_model(x, mu1, delta_mu, err1, delta_err, n1, n2, n3, n4, n5, n6):
    mu = [mu1]
    err = [err1]

    for i in range(1, 6):
        mu.append(mu1 + i * delta_mu)
        err.append(np.sqrt(err1**2 + i * delta_err**2))

    return model(x, mu[0], mu[1], mu[2], mu[3], mu[4], mu[5], err[0], err[1], err[2], err[3], err[4], err[5], n1, n2, n3, n4, n5, n6)


def chi2(data, model, unc, no_params=21):
    chi2 = np.sum((data - model)**2 / unc**2)
    chi2_per_DoF = chi2 / (len(data) - no_params)
    print("Chi2 : {0} and chi2 per DoF : {1}".format(chi2, chi2_per_DoF))
    return chi2


def plot_residuals(x, data, model, unc,fit_region , axes=None, x_scale=1):
    if axes == None:
        axes = plt.gca()
    residuals = data - model
    axes.errorbar(x*x_scale, residuals, unc, fmt='.', linestyle='None')
    axes.plot(x*x_scale, np.zeros(np.shape(x)), ls='--', color='r')
    return axes


def plot_residuals_norm(x, data, model, unc,fit_region , axes=None, x_scale=1):
    if axes == None:
        axes = plt.gca()
    unc_corr = np.where(unc == 0, np.ones(np.shape(unc)), unc)
    residuals = (data - model) / unc_corr
    axes.scatter(x*x_scale, residuals, linestyle='None')
    axes.plot(x*x_scale, np.zeros(np.shape(x)), ls='--', color='r')
    return axes


def determine_roi(all_waveforms, plot=False):
    """Determine the Region of Interest (ROI).  Locate the common peak in all waveforms 
    and the beginning of integration region.

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
    plt.xlabel("time[s]")
    plt.ylabel("Summed signal[V]")
    plt.title("ROI determination")
    plt.legend()
    

    if plot==True:
        plt.savefig("/home/todor/University/MPhys project/MPhys_project/analyze-lecroy/plots/ROI_example_pmt.png", dpi=600)
        plt.show()
    plt.cla()
    print("ROI determined to be [{0}:{1}]".format(roi_begin, roi_end))

    return roi_begin, roi_end, max_loc


def filter_outliers(all_waveforms, max_loc, roi=[], plot=False):
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


def find_area(selected_waveforms, roi=[], no_bins=150, plot=False, save=False, save_loc="/home/todor/University/MPhys project/MPhys_project/analyze-lecroy/data/", savename="areas_sipm-1_56V_sl.csv"):
    
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
        make_heatmap(shifted_waveforms, True, "shifted_waveforms_pmt-0047_850V.png", True, "Waveforms after filtering for a single PMT at 850V")
        
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
        plt.savefig("/home/todor/University/MPhys project/MPhys_project/analyze-lecroy/plots/areas_example_pmt.png", dpi=600)
        plt.show()
        #make_heatmap(negative_waveforms, True, "cancelled_waveforms.png")
        #np.savetxt("/home/todor/University/MPhys project/MPhys_project/analyze-lecroy/data/cancelled_waveforms_example_2.csv", np.vstack(negative_waveforms), delimiter=',')
    
    if save == True:
        fname = save_loc + savename
        np.savetxt(fname, areas, delimiter=',')
        print("Areas saved to file at: " + fname)

    return areas, hist, bins



def indep_gaus_fit(histogram_areas, bins, fit_region=[-0.15e-9, 0.8e-9], method='LeastSquares', plot=False, saveplot=False, fname="", bkg="landau", plot_title=""):
    x_full = get_bin_centres(bins)
    unc = np.sqrt(histogram_areas)
    x = []
    y = []
    err = []
    bins_used = [bins[0]]
    for iterator in range(0, len(x_full)):
        if x_full[iterator] >= fit_region[0] and x_full[iterator] <= fit_region[1]:
            x.append(x_full[iterator])
            y.append(histogram_areas[iterator])
            err.append(unc[iterator])
            bins_used.append(bins[iterator + 1])
    
    err = np.array(err)
    err_corrected = np.where(err <= 1, np.full(np.shape(err), 10), err)
    bins_used = np.array(bins_used)
    y = np.array(y)
    

    #initial guesses:
    error = 1e-11
    min_error = 1e-15
    max_error = 2e-10

    mu = [0, 1.4e-10, 2.8e-10, 4.2e-10, 5.6e-10, 3e-10]
    n = 1.8e-8 / 2
    n_max = 8e-8 / 2



    x = np.array(x)
    cost_f = cost.LeastSquares(x, y, err_corrected, model)
    


    

    if(method == "likelihood"):
        cost_f = cost.ExtendedBinnedNLL(y, bins_used, model_cdf)
        
    
    p0 = [mu[0], mu[1], mu[2], mu[3], mu[4], mu[5], error, error*2.4, error*2.5, error*3, error*3, error*15, n/2, n*1.4, n*1, n*0.45, n/4, n*1.5]
    bounds_lower = [-1e-10, 0, 1e-10, 3e-10, 4e-10, 1e-10, min_error, min_error, min_error, min_error, min_error, min_error, 0, 0, 0, 0, 0, 0]
    bounds_high = [1e-10, 2.4e-10, 3.6e-10, 5e-10, 8e-10, 5e-10, max_error, max_error, max_error, max_error, max_error, max_error*10, n_max, n_max, n_max, n_max, n_max, n_max]
    fitter = Minuit(cost_f, mu1=p0[0], mu2=p0[1], mu3=p0[2], mu4=p0[3], mu5=p0[4], mu7=p0[5], err1=p0[6], err2=p0[7], err3=p0[8], err4=p0[9], err5=p0[10], err7=p0[11], n1=p0[12], n2=p0[13], n3=p0[14], n4=p0[15], n5=p0[16], n7=p0[17])
    if bkg == "langauss":
        cost_f = cost.LeastSquares(x, y, err_corrected, model_gauss)
        p0 = [mu[0], mu[1], mu[2], mu[3], mu[4], mu[5], mu[6], error, error*2.4, error*2.5, error*3, error*3, error*4, error*15, n/2, n*1.4, n*1, n*0.45, n/4, n/6, n*1.5, error*15]
        bounds_lower = [-1e-10, 0, 1e-10, 3e-10, 4e-10, 5e-10, 1e-10, min_error, min_error, min_error, min_error, min_error, min_error, min_error, 0, 0, 0, 0, 0, 0, 0, min_error]
        bounds_high = [1e-10, 2.4e-10, 3.6e-10, 5e-10, 7e-10, 8e-10, 5e-10, max_error, max_error, max_error, max_error, max_error, max_error, max_error*10, n_max, n_max, n_max, n_max, n_max, n_max, n_max, max_error*15]
        fitter = Minuit(cost_f, mu1=p0[0], mu2=p0[1], mu3=p0[2], mu4=p0[3], mu5=p0[4], mu7=p0[5], err1=p0[6], err2=p0[7], err3=p0[8], err4=p0[9], err5=p0[10], err7=p0[11], n1=p0[12], n2=p0[13], n3=p0[14], n4=p0[15], n5=p0[16], n7=p0[17], err_gauss=p0[18])
    #popt, pcov = curve_fit(model, x, histogram_areas, p0, bounds=(bounds_lower, bounds_high), maxfev=50000)
    
    

    limits_minuit = []
    for i in range(0, len(bounds_lower)):
        limits_minuit.append( (bounds_lower[i], bounds_high[i]) )
    
    fitter.limits = limits_minuit
    fitter.migrad()
    fitter.hesse()

    popt = fitter.values
    pcov = fitter.covariance
    perr = np.sqrt(np.diag(pcov))

    print(popt)
    #print(np.flatnonzero(  (err==0) ))
    #print(np.flatnonzero(  (err_corrected==0) ))
    #print(np.flatnonzero(np.isnan(histogram_areas)))
    #print(np.flatnonzero(np.isinf(histogram_areas)))
    #print(np.flatnonzero(np.isnan(x)))
    #print(np.flatnonzero(np.isinf(x)))

    if plot == True:
        plt.errorbar(x_full, histogram_areas, unc, linestyle='None', fmt='.')
        if bkg == "langauss":
            chi2(y, model_gauss(x, *popt), err_corrected)
            plt.plot(x, model_gauss(x, *popt), color='r', label='fit')
            plt.plot(x, model_gauss(x, *p0), color='g', label="Init guess")
            plt.plot(x, langauss.pdf(x, popt[6], popt[13], popt[21])*popt[20], color='r', ls='--')
            plt.plot(x, langauss.pdf(x, p0[6], p0[13], popt[21])*p0[20], color='g', ls='--')
            plt.legend()
            plt.show()
            plot_residuals(x, y, model_gauss(x, *popt), err_corrected, fit_region)
        else:
        
            chi2(y, model(x, *popt), err_corrected)
        
            plt.plot(x, model(x, *popt), color='r', label='fit')
            plt.plot(x, model(x, *p0), color='g', label="Init guess")
            plt.plot(x, landau.pdf(x, popt[5], popt[11])*popt[17], color='r', ls='--')
            plt.plot(x, landau.pdf(x, p0[5], p0[11])*p0[17], color='g', ls='--')
            plt.legend()
            plt.show()            
            plot_residuals(x, y, model(x, *popt), err_corrected, fit_region)

        #fig, axes = plt.subplots(1, 1)
        #plot_residuals(x, y, model(x, *p0), err_corrected, axes[0])
        
        plt.show()

    if saveplot == True:
        
        fig, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        #fig.tight_layout(pad=0.8)
        fig.subplots_adjust(left=0.08, bottom=0.08, right=0.92, top=0.92, wspace=0.005, hspace=0.1)

        fig.set_size_inches(16, 9)
        axes[0].errorbar(np.array(x_full) * 10**9, histogram_areas, unc, linestyle='None', fmt='.')
        if bkg == "langauss":
            chi2(y, model_gauss(x, *popt), err_corrected)
            axes[0].plot(x * 10**9, model_gauss(x, *popt), color='r', label='model')
            axes[0].plot(x * 10**9, langauss.pdf(x, popt[5], popt[11], popt[21])*popt[20], color='r', ls='--', label="background")
            plot_residuals(np.array(x_full), histogram_areas, model_gauss(np.array(x_full), *popt), unc, fit_region, axes[1], 10**9)
        else:
            chi2(y, model(x, *popt), err_corrected)
            axes[0].plot(x * 10**9, model(x, *popt), color='r', label='model')
            axes[0].plot(x * 10**9, landau.pdf(x, popt[5], popt[11])*popt[17], color='r', ls='--', label="background")
            plot_residuals_norm(np.array(x_full), histogram_areas, model(np.array(x_full), *popt), unc, fit_region, axes[1], 10**9)
        axes[1].set_xlabel("Charge [nV.s]", fontsize=22)
        x_span = x_full[-1] - x_full[0]

        axes[0].set_ylabel("Entries [{0:3.1f} counts / (nV.ms)]".format(x_span / len(x_full) * 10**12), fontsize=22)
        axes[1].set_ylabel("Residuals", fontsize=22)
        axes[0].legend(fontsize="20")
        axes[0].tick_params(axis='y', labelsize=15)
        axes[1].tick_params(axis='both', labelsize=15)
        if plot_title != "":
            axes[0].set_title(plot_title, fontsize=23)
        loc = "/home/todor/University/MPhys project/MPhys_project/analyze-lecroy/plots/"
        fig.savefig(loc+fname)
        
        plt.show()




    means = [popt[0], popt[1], popt[2], popt[3], popt[4] ]
    err_means = [perr[0], perr[1], perr[2], perr[3], perr[4] ]

    return means, err_means


def dep_gaus_fit(histogram_areas, bins, fit_region=[-0.15e-9, 0.8e-9], method='LeastSquares', plot=False, saveplot=False, fname="", p0=None, plot_title=""):
    #format the input data
    x_full = get_bin_centres(bins)
    unc = np.sqrt(histogram_areas)
    x = []
    y = []
    err = []
    bins_used = [bins[0]]
    for iterator in range(0, len(x_full)):
        if x_full[iterator] >= fit_region[0] and x_full[iterator] <= fit_region[1]:
            x.append(x_full[iterator])
            y.append(histogram_areas[iterator])
            err.append(unc[iterator])
            bins_used.append(bins[iterator + 1])
    
    err = np.array(err)
    err_corrected = np.where(err <= 1, np.full(np.shape(err), 10), err)
    bins_used = np.array(bins_used)
    y = np.array(y)
    x = np.array(x)
    
    if p0 == None:
        #define the indep params with initial guesses
        #Those are initial guesses for 56V
        mu1 = 0
        err1 = 1.15e-11
        G = 1.39e-10
        erri = 1.27e-11

        mu_bkg = 1e-10
        err_bkg = 1.1e-10

        n = 1.8e-8 / 2
        n_max = 8e-8 / 2

        p0 = [mu1, err1, G, erri, mu_bkg, err_bkg, n/2, n, n/3, n/4, n/5, n]

    #initialize Minuit object
    cost_f = cost.LeastSquares(x, y, err_corrected, dep_model)

    fitter = Minuit(cost_f, *p0)

    fitter.migrad()
    fitter.hesse()
    #fitter.minos()

    popt = fitter.values
    pcov = fitter.covariance
    perr = [0, 0, 0]
    try:
        perr = fitter.errors
        #print(fitter.merrors[2])
    except:
        print("Hessian could not be computed")
    
    if plot == True:
        chi2(y, dep_model(x, *popt), err_corrected, len(popt))
        plt.errorbar(x_full, histogram_areas, unc, linestyle='None', fmt='.')
        plt.plot(x, dep_model(x, *popt), color='r', label='fit')
        plt.plot(x, dep_model(x, *p0), color='g', label="Init guess")
        plt.plot(x, landau.pdf(x, popt[4], popt[5])*popt[11], color='r', ls='--')
        plt.plot(x, landau.pdf(x, p0[4], p0[5])*p0[11], color='g', ls='--')
        plt.legend()
        plt.show()

    if saveplot == True:
        fig, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        #fig.tight_layout(pad=0.8)
        fig.subplots_adjust(left=0.08, bottom=0.08, right=0.92, top=0.99, wspace=0.005, hspace=0.1)

        fig.set_size_inches(16, 9)
        axes[0].errorbar(np.array(x_full) * 10**9, histogram_areas, unc, linestyle='None', fmt='.')
        chi2(y, dep_model(x, *popt), err_corrected)
        axes[0].plot(x * 10**9, dep_model(x, *popt), color='r', label='model')
        axes[0].plot(x * 10**9, landau.pdf(x, popt[4], popt[5])*popt[11], color='r', ls='--', label="background")
        plot_residuals_norm(np.array(x_full), histogram_areas, dep_model(np.array(x_full), *popt), unc, fit_region, axes[1], 10**9)
        axes[1].set_xlabel("Charge [nV.s]", fontsize=22)
        x_span = x_full[-1] - x_full[0]

        axes[0].set_ylabel("Entries [{0:3.1f} counts / (nV.ms)]".format(x_span / len(x_full) * 10**12), fontsize=22)
        axes[1].set_ylabel("Residuals", fontsize=22)
        axes[0].legend(fontsize=22)
        axes[0].tick_params(axis='y', labelsize=15)
        axes[1].tick_params(axis='both', labelsize=15)
        if plot_title != "":
            axes[0].set_title(plot_title, fontsize=23)
        
        loc = "/home/todor/University/MPhys project/MPhys_project/analyze-lecroy/plots/"
        fig.savefig(loc+fname)
        
        plt.show()


    print(popt)
    return popt[2], perr[2], popt[3], perr[3], popt[1], perr[1]


def plot_gain(means=[], err_means=[]):
    x = np.linspace(0, 4, 5)
    popt, pcov = curve_fit(linear, x, means, sigma=err_means, absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))

    plt.plot(x, linear(x, *popt), color='b')
    plt.errorbar(x, means, err_means, fmt='.', ls='None', color='r', capsize=4)
    plt.xlabel("peak #")
    plt.ylabel("Charge[V.s]")
    plt.title("Mean of peak vs peak number")
    plt.savefig("/home/todor/University/MPhys project/MPhys_project/analyze-lecroy/plots/example_gains_fit.png")
    plt.show()
    chi2(np.array(means), linear(x, *popt), np.array(err_means), 2)
    print(popt, perr)

    return
    

def procedure_areas_save(sipm_no=411, voltage="56V", Ch=None, sipm_str='1'):
    fname = voltage + "_sipm-" + sipm_str + "_1000--"
    if Ch != None:
        fname = Ch + "--" + fname
    location = "/home/todor/University/MPhys project/Data_SiPM/"+ str(sipm_no) + "/" + voltage + "/"
    all_waveforms = reader.iterate_large_files(0, 25, fname, loc=location)
    roi_begin, roi_end, peak_loc =  determine_roi(all_waveforms, plot=False)
    roi = [roi_begin, roi_end]
    
    filtered_waveforms = filter_outliers(all_waveforms, peak_loc, roi, plot=False)
    areas, hist, bins = find_area(filtered_waveforms, roi, no_bins=300, save=True, plot=True, savename="areas_sipm-" + str(sipm_no) + "_" + voltage + ".csv")


def procedure_bkg():
    all_waveforms = reader.iterate_large_files(0, 25, "57V_sipm-1_bkg_1000--", loc="/home/todor/University/MPhys project/Data_SiPM/411/bkg/")
    roi_begin, roi_end, peak_loc =  38, 381, 110
    roi = [roi_begin, roi_end]
    filtered_waveforms = filter_outliers(all_waveforms, peak_loc, roi, plot=False)
    areas, hist, bins = find_area(filtered_waveforms, roi, no_bins=300, save=True, plot=True, savename="areas_sipm-411_bkg_57V.csv")
    plot1d(hist, bins)
    plt.xlabel("Charge[V.s]")
    plt.ylabel("Entries[2.7 counts/(nV.ms)]")
    plt.title("Data readout with no LED")
    plt.show()

def procedure_indep_fit(sipm_no=411, voltage="56V", Ch=None, bin_count=600, fit_region=[-1.5e-10, 8e-10], save=False):
    fname =  "areas_sipm-" + str(sipm_no) + "_" + voltage +".csv"
    if Ch != None:
        fname = Ch + "--" + fname
    location = "/home/todor/University/MPhys project/MPhys_project/analyze-lecroy/data/"
    areas = np.genfromtxt(location + fname, delimiter=',')
    hist, bins = np.histogram(areas, bins=bin_count)
    means, err_means = indep_gaus_fit(hist, bins, fit_region, plot=True, saveplot=save, fname="areas_fit_" + str(sipm_no) + "_" + voltage + ".png", plot_title="Independent fit for SiPM " + str(sipm_no) + " at " + voltage)
    return means, err_means

def procedure_dep_fit(sipm_no=411, voltage="56V", bin_size=2.7e-12, fit_region=[-1.5e-10, 8e-10], save=False, p0=[], plot_title=False):
    
    fname =  "areas_sipm-" + str(sipm_no) + "_" + voltage +".csv"
    #if Ch != None:
    #    fname = Ch + "--" + fname
    location = "/home/todor/University/MPhys project/MPhys_project/analyze-lecroy/data/"
    areas = np.genfromtxt(location + fname, delimiter=',')
    areas_span = np.max(areas) - np.min(areas)
    
    bin_count = int(areas_span / bin_size)
    #print(bin_count)
    hist, bins = np.histogram(areas, bins=bin_count)
    title = ""
    if plot_title == True:
        title = "Dependent fit for SiPM " + str(sipm_no) + " at " + voltage
    gain, err_gain, sigma_cell, err_sigma_cell, sigma_pedestal, err_sigma_pedestal = dep_gaus_fit(hist, bins, fit_region, plot=True, saveplot=save, fname="areas_fit_dep_" + str(sipm_no) + "_" + voltage + ".png", p0=p0, plot_title=title)
    return gain, err_gain, sigma_cell, err_sigma_cell, sigma_pedestal, err_sigma_pedestal


        
def find_all_areas(sipm_no=411, voltage=[54, 59], Ch=None, sipm_str='1'):
    for V in range(voltage[0], voltage[1] + 1):
        
        V_string = str(V) + "V"
        print("saving data for " + V_string)
        procedure_areas_save(sipm_no=sipm_no, voltage=V_string, Ch=Ch, sipm_str=sipm_str)

def do_all_fits(fitting_procedure, sipm_no=411, voltage=[54, 59], bin_size=2.7e-12, save=False):
    gains = []
    err_gains = []

    cell_sigmas = []
    err_cell_sigmas = []

    pedestal_sigmas = []
    err_pedestal_sigmas = []

    voltages = []

    #some initial parameters
    roi = [-0.5e-10, 4e-10]
    mu1 = 0
    err1 = 1e-11
    G = 0.75e-10
    erri = 1.27e-11
    mu_bkg = 0.1e-10
    err_bkg = 0.6e-10
    n = 1.8e-8 
    n_bkg = n

    for V in range(voltage[0], voltage[-1] + 1):
        voltages.append(V)
        V_string = str(V) + "V"
        print("==========================================")
        print("Fitting for " + V_string)
        p0 = [mu1, err1, G, erri, mu_bkg, err_bkg, n/1.2, n/1.5, n/3, n/10, 0, n_bkg]
        gain, error_gain, sigma_cell, error_sigma_cell, sigma_pedestal, error_sigma_pedestal = fitting_procedure(sipm_no, V_string, bin_size, roi, save, p0)
        gains.append(gain)
        err_gains.append(error_gain)
        cell_sigmas.append(sigma_cell)
        err_cell_sigmas.append(error_sigma_cell)
        pedestal_sigmas.append(sigma_pedestal)
        err_pedestal_sigmas.append(error_sigma_pedestal)
        G += 0.325e-10
        mu_bkg += 0.15e-10
        roi[0] -= 0.5e-10
        roi[1] += 2e-10
        err_bkg += 0.2e-10
        n_bkg += n / 2
        n /= np.sqrt(2)

    voltages = np.array(voltages)
    gains = np.array(gains)
    err_gains = np.array(err_gains)
    cell_sigmas = np.array(cell_sigmas)
    err_cell_sigmas = np.array(err_cell_sigmas)
    pedestal_sigmas = np.array(pedestal_sigmas)
    err_pedestal_sigmas = np.array(err_pedestal_sigmas)



    #-------------------------Gain Voltage curve----------------------------------------
    popt, pcov = curve_fit(linear, voltages, gains, sigma=err_gains, absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    ampl_factor = 1
    gains_e = gains / 1.6e-19 / ampl_factor
    err_gains_e = err_gains / 1.6e-19 /ampl_factor
    plt.errorbar(voltages, gains_e, err_gains_e , ls='None', fmt='.', capsize=5.0)
    plt.plot(voltages, linear(voltages, *popt) / 1.6e-19 /ampl_factor)
    plt.xlabel("Voltage[V]")
    plt.ylabel("Gain [#e]")
    plt.title("Gain-Voltage curve")
    #plt.plot(voltages, np.zeros(np.shape(voltages)), color='r', ls='--')
    breakdown_V = -popt[1] / popt[0]
    err_breakdown_V = breakdown_V * np.sqrt( (perr[0] / popt[0])**2 + (perr[1] / popt[1])**2 )
    chi2(gains, linear(voltages, *popt), err_gains, 2)
    print( "The breakdown voltage is: {0:3.1f} +/- {1:3.1f}".format(breakdown_V, err_breakdown_V))

    fname = "gain_voltage_sipm-" + str(sipm_no) + ".png"
    loc = "/home/todor/University/MPhys project/MPhys_project/analyze-lecroy/plots/"
    plt.savefig(loc + fname)

    plt.show()



    #-------------------------SNR Calculation------------------------------------------

    delimiter = pedestal_sigmas**2 + cell_sigmas**2 
    SNR = gains / np.sqrt(delimiter)
    part_err_delimiter =  np.sqrt( (err_pedestal_sigmas * pedestal_sigmas * 2)**2 + (err_cell_sigmas * cell_sigmas *2)**2)
    err_delimiter = part_err_delimiter / (2 * np.sqrt(delimiter))
    err_SNR = SNR * np.sqrt((err_gains / gains)**2 + (err_delimiter / np.sqrt(delimiter))**2)
    fname = "snr_sipm-" + str(sipm_no) + ".png"
    plt.errorbar(voltages, SNR, err_SNR, fmt='.', ls='None', capsize=2)
    plt.xlabel("Voltage[V]")
    plt.ylabel("SNR")
    plt.savefig(loc + fname)
    plt.show()
    


    #----------------------save results------------------------------------------------
    voltages = np.array([voltages])
    gains = np.array([gains_e])
    err_gains = np.array([err_gains_e])
    SNR = np.array([SNR])
    err_SNR = np.array([err_SNR])

    results = np.concatenate([voltages.T, gains.T, err_gains.T, SNR.T, err_SNR.T], axis=1)
    loc = "/home/todor/University/MPhys project/MPhys_project/analyze-lecroy/results/"
    fname = "results_sipm-" + str(sipm_no) + ".csv"
    np.savetxt(loc+fname, results, delimiter=',', header='voltage[V], gain[#e], err_gain[#e], SNR, err_SNR')
    print("Saved gains and SNR to file: " + loc + fname)


def overvoltages_plot(sipm_no=[411, 412, 413, 414, 417, 418, 419], voltage=4):
    G_overvoltage = []
    err_G_overvoltage = []
    breakdowns = []
    err_breakdowns = []
    for sipm in sipm_no:
        loc = "/home/todor/University/MPhys project/MPhys_project/analyze-lecroy/results/"
        fname = "results_sipm-" + str(sipm) + ".csv"
        data = np.genfromtxt(loc+fname, delimiter=',', skip_header=1)
        p0=[ 1.68893835e+08, -8.67367450e+09]
        popt, pcov = curve_fit(linear, data[:, 0], data[:, 1], sigma=data[:, 2], absolute_sigma=True, p0=p0)
        perr = np.sqrt(np.diag(pcov))
        breakdown_V = -popt[1] / popt[0]
        err_breakdown_V = breakdown_V * np.sqrt( (perr[0] / popt[0])**2 + (perr[1] / popt[1])**2 )
        V = breakdown_V + voltage
        G = popt[0] * V + popt[1]
        err_G_1 = popt[0] * V * np.sqrt( (perr[0] / popt[0])**2 + (err_breakdown_V / breakdown_V)**2 )
        err_G = np.sqrt(err_G_1**2 + perr[1]**2)
        G_overvoltage.append(G)
        err_G_overvoltage.append(err_G)
        breakdowns.append(breakdown_V)
        err_breakdowns.append(err_breakdown_V)
        #debug plot
        print(breakdown_V)
        plt.errorbar(data[:, 0], data[:, 1], data[:, 2], ls='None', fmt='.')
        plt.plot(data[:, 0], linear(data[:, 0], *popt))
        plt.show()

    x = np.linspace(1, len(sipm_no), len(sipm_no))
    plt.errorbar(x, breakdowns, err_breakdowns, fmt='.', ls='None', capsize=2)
    plt.xticks(x, sipm_no)
    plt.xlabel("SiPM #")
    plt.ylabel("Breakdown voltage [V]")
    plt.savefig("/home/todor/University/MPhys project/MPhys_project/analyze-lecroy/plots/breakdown_V_all.png", dpi=600)
    plt.show()

    plt.errorbar(x, G_overvoltage, err_G_overvoltage, ls='None', fmt='.', capsize=2)
    plt.xticks(x, sipm_no)
    plt.xlabel("SiPM #")
    plt.ylabel("Gain[#e]")
    plt.savefig("/home/todor/University/MPhys project/MPhys_project/analyze-lecroy/plots/4_overvolts_all.png", dpi=600)
    plt.show()
    
    

def make_pretty_plots():
    all_waveforms = reader.iterate_large_files(0, 25, "56V_sipm-1_1000--", loc="/home/todor/University/MPhys project/Data_SiPM/411/56V/")
    roi1, roi2, max_loc = determine_roi(all_waveforms, False)
    
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
    plt.savefig("/home/todor/University/MPhys project/MPhys_project/analyze-lecroy/plots/rejected_examples.png")
    plt.show()


    


if __name__ == "__main__":
    #procedure_bkg()
    #means, err_means = procedure_dep_fit(1, save=True)
    #print(means, err_means)
    #procedure_gain_voltage()
    #find_all_areas(414, Ch='C4', sipm_str='414417')
    #do_all_fits(procedure_dep_fit, sipm_no=414, save=False)
    #overvoltages_plot()
    
    #means, err_means = procedure_indep_fit(save=True)
    #plot_gain(means, err_means)
    #do_all_fits(procedure_dep_fit, save=True)
    
    means, err = procedure_indep_fit(save=True)
    #plot_gain(means, err)
    """
    areas = np.genfromtxt("/home/todor/University/MPhys project/MPhys_project/analyze-lecroy/data/areas_sipm-411_56V.csv")
    plt.hist(areas, 600)
    plt.xlabel("Charge[V.s]")
    plt.ylabel("Entries[2.7 counts/(nV.ms)]")
    plt.title("Waveform integrals histogram")
    plt.show()
    """



    
    


