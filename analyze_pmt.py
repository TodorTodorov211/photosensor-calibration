import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

import read_data as reader
import compute_area as analyzer

from iminuit import cost
from iminuit import Minuit
import sys

from scipy.optimize import curve_fit
from configuration import PLOTS_FOLDER, DATA_FOLDER, RESULTS_FOLDER, LOC_DATA_PMT

sys.path.append("/home/todor/University/MPhys project/MPhys_project/utils/")
from plotting_utils import plot1d
from plotting_utils import get_bin_centres
from plotting_utils import get_bin_index



def invert_waveform(waveforms):
    """A function to invert waveforms (Negative amplitudes become positive). Used for the PMT area calculation.

    Parameters
    ----------
    waveforms : list of ndarray
        All waveforms to be inverted. Every waveform has 2 culumns: time and amplitude

    Returns
    -------
    list of ndarray
        inverted waveforms
    """
    inverted_waveforms =[]
    for wave in waveforms:
        wave[:, 1] = - wave[:, 1]
        inverted_waveforms.append(wave)

    return inverted_waveforms


def model(x, mu1, mu2, mu3, err1, err2, err3, n1, n2, n3):
    return norm.pdf(x, mu1, err1) * n1 + norm.pdf(x, mu2, err2) * n2 + norm.pdf(x, mu3, err3) * n3


def model_dep(x, mu0, G, err0, err_cell, n1, n2, n3):
    mu = []
    err = []
    for i in range(0, 3):
        mu.append(mu0 + G * i)
        err.append(np.sqrt(err0**2 + i * err_cell**2))

    return model(x, mu[0], mu[1], mu[2], err[0], err[1], err[2], n1, n2, n3)


def model_4(x, mu1, mu2, mu3, mu4, err1, err2, err3, err4, n1, n2, n3, n4):
    return norm.pdf(x, mu1, err1) * n1 + norm.pdf(x, mu2, err2) * n2 + norm.pdf(x, mu3, err3) * n3 + norm.pdf(x, mu4, err4) * n4

def model_4_dep(x, mu0, G, err0, err_cell, n1, n2, n3, n4):
    mu = []
    err = []
    for i in range(0, 4):
        mu.append(mu0 + G * i)
        err.append(np.sqrt(err0**2 + i * err_cell**2))

    return model_4(x, mu[0], mu[1], mu[2], mu[3], err[0], err[1], err[2], err[3], n1, n2, n3, n4)


def power_law(x, a, b):
    """a simple power law b * x**a

    Parameters
    ----------
    x : float
        x variable
    a : float
        power parameter (slope in log plot)
    b : float
        scale parameter (x=1 intercept in log plot)

    Returns
    -------
    float
        b * x**a
    """
    return b * (x**a)


def indep_fit (histogram_areas, bins, fit_region=[-0.15e-10, 1.2e-10], method='LeastSquares', plot=False, saveplot=False, fname="", p0=None, fitting_function=model, plot_title=""):
    """Perform independent gaussian fit

    Parameters
    ----------
    histogram_areas : list of float
        list of all areas, calculated beforehand
    bins : list of float
        bin edges of the data
    fit_region : list, optional
        lower and upper bond of the data to be used for the fit, by default [-0.15e-10, 1.2e-10]
    method : str, optional
        method to use(Only LeastSquares work), by default 'LeastSquares'
    plot : bool, optional
        plot pretty plots or not, by default False
    saveplot : bool, optional
        save the plots or not, by default False
    fname : str, optional
        the filename to save the plots, by default ""
    p0 : list, optional
        Initial guess parameters. Must match the fit function, by default None
    fitting_function : callable, optional
        Fitting function. First parameter has to be x, by default model
    plot_title : str, optional
        Title to print on the plot, leave "" for no title, by default ""

    Returns
    -------
    float, float
        gain and error in gain of the system
    """
    x_full = get_bin_centres(bins)
    err_all = np.sqrt(histogram_areas)
    err_all_corrected = np.where(err_all == 0, np.full(np.shape(err_all), 1), err_all)

    x = []
    y = []
    err = []
    bins_used = [bins[0]]
    for iterator in range(0, len(x_full)):
        if x_full[iterator] >= fit_region[0] and x_full[iterator] <= fit_region[1]:
            x.append(x_full[iterator])
            y.append(histogram_areas[iterator])
            err.append(err_all_corrected[iterator])
            bins_used.append(bins[iterator + 1])

    bins_used = np.array(bins_used)
    y = np.array(y)
    x = np.array(x)
    err = np.array(err)

    if p0 == []:
        #initial params
        mu = np.array([1e-12, 3e-11, 6e-11])
        std = np.array([0.1e-11, 1.2e-11, 2e-11])
        n = np.array([2e-9 / 3, 2e-9, 2e-9])
        p0 = np.hstack([mu, std, n])
        

    

    cost_f = cost.LeastSquares(x, y, err, fitting_function)

    fitter = Minuit(cost_f, *p0)
    fitter.migrad()
    fitter.hesse()

    popt = fitter.values
    pcov = fitter.covariance
    perr = [0, 0, 0]
    try:
        perr = fitter.errors
        #print(fitter.merrors[2])
    except:
        print("Hessian could not be computed")
    
    peak_no = np.array([0, 1, 2])
    peak_no_1 = np.array([0, 1, 2, 3])
    g_opt, g_cov = curve_fit(analyzer.linear, peak_no, popt[0:3], p0=[1e-11, 0], sigma=perr[0:3], absolute_sigma=True)
    if plot == True:
        analyzer.chi2(y, fitting_function(x, *popt), err, 9)
        plt.errorbar(x, y, err, ls="None", fmt='.')
        #plt.plot(x, fitting_function(x, *p0), label="init guess")
        plt.plot(x, fitting_function(x, *popt), label="fit")

        #plot separate gaussians
        gaus_count = int(len(p0) / 3)
        for iterator in range(0, gaus_count):
            plt.plot(x, norm.pdf(x, popt[iterator], popt[iterator + gaus_count]) * popt[iterator + 2 * gaus_count], ls='--')

        plt.legend()
        plt.show()

        #gain plot
        #plt.errorbar(peak_no, popt[0:3], perr[0:3], ls='None', fmt='.')
        #plt.plot(peak_no, analyzer.linear(peak_no, *g_opt))
        #plt.show()

        #sigmas plot
        #plt.errorbar(peak_no_1, popt[4:8], perr[4:8], ls='None', fmt='.')
        #plt.show()
    
        
    gain = g_opt[0]
    #err_gain = np.sqrt( perr[0]**2 + perr[1]**2 )
    err_gain = np.sqrt(np.diag(g_cov))[0]

    if saveplot == True:
        fig, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        #fig.tight_layout(pad=0.8)
        fig.subplots_adjust(left=0.08, bottom=0.08, right=0.92, top=0.92, wspace=0.005, hspace=0.1)

        fig.set_size_inches(16, 9)
        axes[0].errorbar(np.array(x_full) * 10**9, histogram_areas, err_all, linestyle='None', fmt='.')
        analyzer.chi2(y, fitting_function(x, *popt), err)
        axes[0].plot(x * 10**9, fitting_function(x, *popt), color='r', label='model')
        gaus_count = int(len(p0) / 3)
        for iterator in range(0, gaus_count):
            axes[0].plot(x * 10**9, norm.pdf(x, popt[iterator], popt[iterator + gaus_count]) * popt[iterator + 2 * gaus_count], ls='--', color='r')
        
        analyzer.plot_residuals_norm(np.array(x_full), histogram_areas, fitting_function(np.array(x_full), *popt), err_all, fit_region, axes[1], 10**9)
        axes[1].set_xlabel("Charge [nV.s]", fontsize=20)
        x_span = x_full[-1] - x_full[0]

        axes[0].set_ylabel("Entries [{0:3.1f} counts / (nV.ms)]".format(x_span / len(x_full) * 10**12), fontsize=22)
        axes[1].set_ylabel("Residuals", fontsize=22)
        #axes[0].legend()
        axes[0].tick_params(axis='y', labelsize=15)
        axes[1].tick_params(axis='both', labelsize=15)
        if plot_title != "":
            axes[0].set_title(plot_title, fontsize=23)
        
        loc = PLOTS_FOLDER
        fig.savefig(loc+fname)
        
        plt.show()

    
    if fitting_function == model or fitting_function == model_dep:
        return gain, err_gain, popt[4], perr[4]
    else:
        return gain, err_gain, popt[5], perr[5]


    
def dep_fit(histogram_areas, bins, fit_region=[-0.15e-10, 1.2e-10], method='LeastSquares', plot=False, saveplot=False, fname="", p0=[], fitting_function=model_4_dep, plot_title=""):
    """Dependent fit where mu_i = i*G + mu_0

    Parameters
    ----------
histogram_areas : list of float
        list of all areas, calculated beforehand
    bins : list of float
        bin edges of the data
    fit_region : list, optional
        lower and upper bond of the data to be used for the fit, by default [-0.15e-10, 1.2e-10]
    method : str, optional
        method to use(Only LeastSquares work), by default 'LeastSquares'
    plot : bool, optional
        plot pretty plots or not, by default False
    saveplot : bool, optional
        save the plots or not, by default False
    fname : str, optional
        the filename to save the plots, by default ""
    p0 : list, optional
        Initial guess parameters. Must match the fit function, by default None
    fitting_function : callable, optional
        Fitting function. First parameter has to be x, by default model
    plot_title : str, optional
        Title to print on the plot, leave "" for no title, by default ""

    Returns
    -------
    float, float
        gain and error in gain of the system
    """
    x_full = get_bin_centres(bins)
    err_all = np.sqrt(histogram_areas)
    err_all_corrected = np.where(err_all == 0, np.full(np.shape(err_all), 1), err_all)

    x = []
    y = []
    err = []
    bins_used = [bins[0]]
    for iterator in range(0, len(x_full)):
        if x_full[iterator] >= fit_region[0] and x_full[iterator] <= fit_region[1]:
            x.append(x_full[iterator])
            y.append(histogram_areas[iterator])
            err.append(err_all_corrected[iterator])
            bins_used.append(bins[iterator + 1])

    bins_used = np.array(bins_used)
    y = np.array(y)
    x = np.array(x)
    err = np.array(err)

    if p0 == []:
        #initial params
        mu = np.array([1e-12, 1e+6])
        std = np.array([0.1e-11, 1e-11])
        n = np.array([2e-9 / 3, 2e-9, 2e-9, 2e-9])
        p0 = np.hstack([mu, std, n])
        

    

    cost_f = cost.LeastSquares(x, y, err, fitting_function)

    fitter = Minuit(cost_f, *p0)
    fitter.migrad()
    fitter.hesse()

    popt = fitter.values
    pcov = fitter.covariance
    perr = [0, 0, 0]
    try:
        perr = fitter.errors
        #print(fitter.merrors[2])
    except:
        print("Hessian could not be computed")

    if plot == True:
        analyzer.chi2(y, fitting_function(x, *popt), err, 9)
        plt.errorbar(x, y, err, ls="None", fmt='.')
        plt.plot(x, fitting_function(x, *p0), label="init guess")
        plt.plot(x, fitting_function(x, *popt), label="fit")
        plt.legend()
        plt.show()

    if saveplot == True:
        fig, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        #fig.tight_layout(pad=0.8)
        fig.subplots_adjust(left=0.08, bottom=0.08, right=0.92, top=0.92, wspace=0.005, hspace=0.1)

        fig.set_size_inches(16, 9)
        axes[0].errorbar(np.array(x_full) * 10**9, histogram_areas, err_all, linestyle='None', fmt='.')
        analyzer.chi2(y, fitting_function(x, *popt), err)
        axes[0].plot(x * 10**9, fitting_function(x, *popt), color='r', label='model')
        gaus_count = int(len(p0) - 4)
        for iterator in range(0, gaus_count):
            axes[0].plot(x * 10**9, norm.pdf(x, popt[0] + iterator*popt[1], np.sqrt(popt[2]**2 + iterator * popt[3]**2)) * popt[iterator + 4], ls='--', color='r')
        analyzer.plot_residuals_norm(np.array(x_full), histogram_areas, fitting_function(np.array(x_full), *popt), err_all, fit_region, axes[1], 10**9)
        axes[1].set_xlabel("Charge [nV.s]", fontsize=22)
        x_span = x_full[-1] - x_full[0]

        axes[0].set_ylabel("Entries [{0:3.1f} counts / (nV.ms)]".format(x_span / len(x_full) * 10**12), fontsize=22)
        axes[1].set_ylabel("Residuals", fontsize=22)
        axes[0].legend()
        axes[0].tick_params(axis='y', labelsize=15)
        axes[1].tick_params(axis='both', labelsize=15)
        if plot_title != "":
            axes[0].set_title(plot_title, fontsize=23)
        loc = PLOTS_FOLDER
        fig.savefig(loc+fname)
        
        plt.show()

    return popt[1], perr[1], popt[0], perr[0]

    

    



def procedure_areas_save(pmt_no = '0047', voltage="850V", ch='C1'):
    """Procedure to read data, apply all filters, calculate areas and save them to '.csv' file

    Parameters
    ----------
    pmt_no : str, optional
        The number of the pmt as recorded in the data files, by default '0047'
    voltage : str, optional
        the valtage at which data was recorded, by default "850V"
    ch : str, optional
        Channel prefix of data files added by the scope, by default 'C1'
    """
    fname = voltage + "_pmt-" + pmt_no + "_1000--"
    if ch != None:
        fname = ch + "--" + fname
    location = LOC_DATA_PMT+ str(pmt_no) + "/" + voltage + "/"
    all_waveforms = reader.iterate_large_files(0, 25, fname, loc=location)
    inv_waveforms = invert_waveform(all_waveforms)
    
    #reader.make_heatmap(inv_waveforms)  
    roi_begin, roi_end, peak = analyzer.determine_roi(inv_waveforms)
    roi = [roi_begin, roi_end]

    filtered_waveforms = analyzer.filter_outliers(inv_waveforms, peak, roi)
    savename = "areas_pmt-" + pmt_no + "_" + voltage + ".csv"
    areas, hist, bins = analyzer.find_area(filtered_waveforms, roi, 150, save=True, savename=savename)


def procedure_indep_fit(pmt_no = '0047', voltage="850V", bin_size=2.7e-13, fit_region=[-0.15e-10, 1.2e-10], save=False, p0=[], fitting_function=model, plot_title=False):
    """Procedure to read the areas from a file and perform an independent fit.  The initial guess parameters may need tweaking

    Parameters
    ----------
    pmt_no : str, optional
        The number of the pmt as recorded in the data files, by default '0047'
    voltage : str, optional
        the voltage at which data was recorded, by default "850V"
    bin_size : float, optional
        The size of each bin to ensure equal bin sizes for all histograms, by default 2.7e-13
    fit_region : list, optional
        region in which to fit, due to the histogram having a long tail, not all of the data needs
        to be included in the fit, by default [-0.15e-10, 1.2e-10]
    save : bool, optional
        Save the plots or not, by default False
    p0 : list, optional
        Initial fit parameters, by default []
    fitting_function : callable, optional
        a function to fit to (our model), by default model
    plot_title : bool, optional
        put plot title or not(no for report, yes for presentation), by default False

    Returns
    -------
    float, float, float, float
        gain, error of gain, stddev of the first photopeak(second peak), error on the stddev of the first photopeak
    """
    fname = "areas_pmt-" + pmt_no + "_" + voltage + ".csv"
    loc = "analyze-lecroy/data/"
    areas = np.genfromtxt(loc + fname, delimiter=',')
    areas_span = np.max(areas) - np.min(areas)
    
    bin_count = int(areas_span / bin_size)
    
    hist, bins = np.histogram(areas, bins=bin_count)

    str_model = ""
    if fitting_function == model_4:
        str_model = "_4gaus"
    if fitting_function == model:
        str_model = "_3gaus"

    title = ""
    if plot_title == True:
        title = "Independent fit for PMT WA" + pmt_no + " at " + voltage
    gain, err_gain, pedestal, err_pedestal = indep_fit(hist, bins, fit_region, plot=True, saveplot=save, fname="areas_fit_indep_" + str(pmt_no) + "_" + voltage + str_model + ".png", p0=p0, fitting_function=fitting_function, plot_title=title)
    return gain, err_gain, pedestal, err_pedestal

def procedure_dep_fit(pmt_no = '0047', voltage="850V", bin_size=2.7e-13, fit_region=[-0.15e-10, 1.2e-10], save=False, p0=[], fitting_function=model_4_dep, plot_title=False):
    """Procedure to read the areas from a file and perform a dependent fit.  The initial guess parameters may need tweaking

    Parameters
    ----------
    pmt_no : str, optional
        The number of the pmt as recorded in the data files, by default '0047'
    voltage : str, optional
        the voltage at which data was recorded, by default "850V"
    bin_size : float, optional
        The size of each bin to ensure equal bin sizes for all histograms, by default 2.7e-13
    fit_region : list, optional
        region in which to fit, due to the histogram having a long tail, not all of the data needs
        to be included in the fit, by default [-0.15e-10, 1.2e-10]
    save : bool, optional
        Save the plots or not, by default False
    p0 : list, optional
        Initial fit parameters, by default []
    fitting_function : callable, optional
        a function to fit to (our model), by default model_4_dep
    plot_title : bool, optional
        put plot title or not(no for report, yes for presentation), by default False

    Returns
    -------
    float, float, float, float
        gain, error of gain, stddev of the first photopeak(second peak), error on the stddev of the first photopeak
    """
    fname = "areas_pmt-" + pmt_no + "_" + voltage + ".csv"
    loc = "analyze-lecroy/data/"
    areas = np.genfromtxt(loc + fname, delimiter=',')
    areas_span = np.max(areas) - np.min(areas)
    
    bin_count = int(areas_span / bin_size)
    
    hist, bins = np.histogram(areas, bins=bin_count)

    str_model = ""
    if fitting_function == model_4_dep:
        str_model = "_4gaus"
    if fitting_function == model_dep:
        str_model = "_3gaus"

    title = ""
    if plot_title == True:
        title = "Dependent fit for PMT WA" + pmt_no + " at " + voltage
    gain, err_gain, pedestal, err_pedestal = dep_fit(hist, bins, fit_region, plot=True, saveplot=save, fname="areas_fit_dep_" + str(pmt_no) + "_" + voltage + str_model + ".png", p0=p0, fitting_function=fitting_function, plot_title=title)
    
    return gain, err_gain, pedestal, err_pedestal


def save_all_areas(pmt_no = '0047', voltage=[800, 825, 850, 875, 900], ch='C1'):
    """Iterates all voltages for a single pmt and saves all areas

    Parameters
    ----------
    pmt_no : str, optional
        Number of the pmt as written in the data files, by default '0047'
    voltage : list, optional
        A list of all voltages to iterate(they are int), by default [800, 825, 850, 875, 900]
    ch : str, optional
        A channel prefix added to the datafiel name by the scope, by default 'C1'
    """
    for V in voltage:
        V_string = str(V) + "V"
        print("Saving data for " + V_string + "...")
        procedure_areas_save(pmt_no, V_string, ch)

    
def do_all_fits(fitting_procedure, pmt_no = '0047', voltage=[800, 825, 850, 875, 900], bin_size=2.7e-13, save=False, model_fit=model_4, plot_titles=False):
    """Read the saved data from files and perform all fits for all voltages for a single pmt.  
    It saves the voltages, gains, errors of gains, SNRs and errors of SNRs to a txt file in the results folder

    Parameters
    ----------
    fitting_procedure : callable
        The fitting procedure to be called, can be procedure_dep_fit or procedure_indep_fit
    pmt_no : str, optional
        The number of the PMT as written in the data file names, by default '0047'
    voltage : list, optional
        All voltages as integers to be iterated over, by default [800, 825, 850, 875, 900]
    bin_size : float, optional
        The bin size for the histogram, to ensure that all bins on all histograms have the same size, by default 2.7e-13
    save : bool, optional
        Save the plots or not, by default False
    model_fit : callable, optional
        The model to fit, by default model_4
    plot_titles : bool, optional
        put plot titles or not, by default False

    Returns
    -------
    list, list, list
        voltage, gains, errors of gains
    """
    gains = []
    err_gains = []


    pedestal_sigmas = []
    err_pedestal_sigmas = []

    roi = [-0.15e-10, 1.2e-10]
    mu = []
    std = []
    n = []

    if model_fit == model_4:
        #init params
        
        mu = np.array([1e-12, 1.4e-11, 2.5e-11, 4e-11])
        std = np.array([0.1e-11, 0.8e-11, 2e-11, 2e-11])
        n = np.array([2e-9 / 3, 2e-9, 3e-9, 2e-9])
    if model_fit == model:
        mu = np.array([1e-12, 1.4e-11, 2.5e-11])
        std = np.array([0.1e-11, 0.8e-11, 2e-11])
        n = np.array([2e-9 / 1, 2e-9, 3e-9])
    if model_fit == model_4_dep:
        mu = np.array([1e-12, 1.0e-11])
        std = np.array([0.1e-11, 0.8e-11])
        n = np.array([2e-9 / 3, 2e-9, 3e-9, 2e-9])
    if model_fit == model_dep:
        mu = np.array([3e-12, 0.8e-11])
        std = np.array([0.1e-11, 0.6e-11])
        n = np.array([2e-9 / 1.2, 1.5e-9, 3e-9])

    for V in voltage:
        V_string = str(V) + "V"
        print("======================================================")
        print("Fitting for " + V_string + "...")
        p0 = np.hstack([mu, std, n])
        gain, err_gain, pedestal, err_pedestal = fitting_procedure(pmt_no, V_string, bin_size, roi, save, p0, model_fit, plot_title=plot_titles)
        gains.append(gain)
        err_gains.append(err_gain)
        pedestal_sigmas.append(pedestal)
        err_pedestal_sigmas.append(err_pedestal)
        #increase fit params
        if model_fit == model_4:
            mu[1] += 0.5e-11
            mu[2] += 1e-11
            mu[3] += 0.7e-11
            std[1] += 0.1e-11
        if model_fit == model:
            mu[1] += 0.5e-11
            mu[2] += 1e-11
            std[1] += 0.1e-11
        if model_fit == model_4_dep:
            mu[1] += 0.5e-11
            std[1] += 0.1e-11
        if model_fit == model_dep:
            mu[1] += 0.3e-11
            std[1] += 0.1e-11
    
    #gain-voltage curve
    voltage = np.array(voltage)
    gains = np.array(gains)
    err_gains = np.array(err_gains)
    pedestal_sigmas = np.array(pedestal_sigmas)
    err_pedestal_sigmas = np.array(err_pedestal_sigmas)

    gains /= (50 * 1.6e-19)
    err_gains /=  (50 * 1.6e-19)
    pedestal_sigmas /= (50 * 1.6e-19)
    err_pedestal_sigmas /= (50 * 1.6e-19)

    plt.errorbar(voltage, gains, err_gains, ls='None', fmt='.', capsize=2)
    popt, pcov = curve_fit(power_law, voltage, gains, p0=[2, 1e+6], sigma=err_gains, absolute_sigma=True, maxfev=30000)
    print(popt, pcov)
    plt.plot(voltage, power_law(voltage, *popt))
    plt.yscale('log')   
    plt.xscale('log')
    plt.xlabel("Voltage [V]")
    plt.ylabel("Gain [#e]")
    if plot_titles == True:
        plt.title("Gain-Voltage curve for PMT WA" + pmt_no)
    plt.tight_layout()
    if save == True:
        plt.savefig(PLOTS_FOLDER + "gain_voltage_pmt-WA" + pmt_no + ".png", dpi=600)
    plt.show()

    #====================SNR========================
    SNR = gains / pedestal_sigmas
    err_SNR = SNR * np.sqrt( (err_gains / gains)**2 + (err_pedestal_sigmas / pedestal_sigmas)**2 )



    result = np.concatenate([np.array([voltage]).T, np.array([gains]).T, np.array([err_gains]).T, np.array([SNR]).T, np.array([err_SNR]).T], axis=1)
    #print(result)
    loc = RESULTS_FOLDER
    fname = "results_pmt-" + pmt_no + ".csv"
    np.savetxt(loc + fname, result, delimiter=',', header='voltage[V], gain[#e], err_gain[#e], SNR, err_SNR')

    return voltage, gains, err_gains



def confidence_intervals(voltage, popt, perr, color='b', alpha=0.1, axes=None):
    """Plot the 1 sigma errors on the gain-voltage curves

    Parameters
    ----------
    voltage : _type_
        _description_
    popt : _type_
        _description_
    perr : _type_
        _description_
    color : str, optional
        _description_, by default 'b'
    alpha : float, optional
        _description_, by default 0.1
    axes : _type_, optional
        _description_, by default None
    """
    if perr[0] / popt[0] < 1e-3:
        perr[0] *= 1e+4
    #delta_f_minus = power_law(voltage[0], *popt) - power_law(voltage[0], popt[0] - perr[0], popt[1])
    #delta_f_plus = power_law(voltage[0], popt[0] + perr[0], popt[1]) - power_law(voltage[0], *popt)
    delta_f_minus = 0
    delta_f_plus = 0
    if axes == None:
        axes = plt.gca()
    
    axes.fill_between(voltage, power_law(voltage, popt[0] - perr[0], popt[1] - perr[1]) + delta_f_minus, power_law(voltage, popt[0] + perr[0], popt[1] + perr[1]) - delta_f_plus, color=color, alpha=alpha)


def fit_gain(voltage, gains, errors, p0=[7, 7e-14]):
    """Fit the gain to a power law using the MIGRAD optimiser

    Parameters
    ----------
    voltage : list
        voltages (x axis data)
    gains : list
        gains (y axis data)
    errors : list
        errors on gains (y-axis errors)
    p0 : list, optional
        initial fit parameters, by default [7, 7e-14]

    Returns
    -------
    list, list
        optimal fit parameters, errors on the optimal parameters from the fit
    """
    cost_f = cost.LeastSquares(voltage, gains, errors, power_law, verbose=0)
    
    
    fitter = Minuit(cost_f, *p0)
    
    fitter.simplex(ncall=30000)
    print("SIMPLEX finished")
    fitter.migrad(ncall=30000)
    print("MIGRAD finished")
    fitter.hesse()
    fitter.minos()
    popt = fitter.values
    fitter.visualize()
    print(fitter.params)
    plt.show()
    try:
        perr = fitter.errors
    except:
        print("Hessian could not be computed...")

    
    return popt, perr


def guess_datasheet():
    """Hardcoded with values from the R8520-506 PMT datasheet to provide a plot for comparison

    Returns
    -------
    list, list
        parameters of the power law in the datasheet, errors on the parameters
    """
    voltage = np.array([500, 600, 700, 800, 900])
    gain = np.array([1.6e+4, 9e+4, 3.5e+5, 1e+6, 2.5e+6])
    errors = 0.01 * gain
    popt, perr = fit_gain(voltage, gain, errors)
    

    print(popt, perr)
    return popt, perr









if __name__ == "__main__":
    save_all_areas(pmt_no="0049")
    
    




    
    