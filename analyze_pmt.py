import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

import read_data as reader
import compute_area as analyzer

from iminuit import cost
from iminuit import Minuit
import sys

from scipy.optimize import curve_fit

sys.path.append("utils/")
from plotting_utils import plot1d
from plotting_utils import get_bin_centres
from plotting_utils import get_bin_index



def invert_waveform(waveforms):
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
    return b * (x**a)


def indep_fit (histogram_areas, bins, fit_region=[-0.15e-10, 1.2e-10], method='LeastSquares', plot=False, saveplot=False, fname="", p0=None, fitting_function=model, plot_title=""):

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
        
        loc = "/home/todor/University/MPhys project/MPhys_project/analyze-lecroy/plots/"
        fig.savefig(loc+fname)
        
        plt.show()

    

    return gain, err_gain, popt[3], perr[3]


    
def dep_fit(histogram_areas, bins, fit_region=[-0.15e-10, 1.2e-10], method='LeastSquares', plot=False, saveplot=False, fname="", p0=[], fitting_function=model_4_dep, plot_title=""):

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
        loc = "/home/todor/University/MPhys project/MPhys_project/analyze-lecroy/plots/"
        fig.savefig(loc+fname)
        
        plt.show()

    return popt[1], perr[1], popt[0], perr[0]

    

    



def procedure_areas_save(pmt_no = '0047', voltage="850V", ch='C1'):
    fname = voltage + "_pmt-" + pmt_no + "_1000--"
    if ch != None:
        fname = ch + "--" + fname
    location = "/home/todor/University/MPhys project/Data_PMT/"+ str(pmt_no) + "/" + voltage + "/"
    all_waveforms = reader.iterate_large_files(0, 25, fname, loc=location)
    inv_waveforms = invert_waveform(all_waveforms)
    
    #reader.make_heatmap(inv_waveforms)  
    roi_begin, roi_end, peak = analyzer.determine_roi(inv_waveforms)
    roi = [roi_begin, roi_end]

    filtered_waveforms = analyzer.filter_outliers(inv_waveforms, peak, roi)
    savename = "areas_pmt-" + pmt_no + "_" + voltage + ".csv"
    areas, hist, bins = analyzer.find_area(filtered_waveforms, roi, 150, save=True, savename=savename)


def procedure_indep_fit(pmt_no = '0047', voltage="850V", bin_size=2.7e-13, fit_region=[-0.15e-10, 1.2e-10], save=False, p0=[], fitting_function=model, plot_title=False):
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
    for V in voltage:
        V_string = str(V) + "V"
        print("Saving data for " + V_string + "...")
        procedure_areas_save(pmt_no, V_string, ch)

    
def do_all_fits(fitting_procedure, pmt_no = '0047', voltage=[800, 825, 850, 875, 900], bin_size=2.7e-13, save=False, model_fit=model_4, plot_titles=False):

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

    gains = gains / (50 * 1.6e-19)
    err_gains = err_gains / (50 * 1.6e-19)

    plt.errorbar(voltage, gains, err_gains, ls='None', fmt='.', capsize=2)
    popt, pcov = curve_fit(power_law, voltage, gains, p0=[2, 1e+6], sigma=err_gains, absolute_sigma=True, maxfev=30000)
    plt.plot(voltage, power_law(voltage, *popt))
    plt.yscale('log')   
    plt.xscale('log')
    plt.xlabel("Voltage [V]")
    plt.ylabel("Gain [#e]")
    if plot_titles == True:
        plt.title("Gain-Voltage curve for PMT WA" + pmt_no)
    plt.tight_layout()
    if save == True:
        plt.savefig("/home/todor/University/MPhys project/MPhys_project/analyze-lecroy/plots/gain_voltage_pmt-WA" + pmt_no + ".png", dpi=600)
    plt.show()

    return voltage, gains, err_gains













if __name__ == "__main__":
    #save_all_areas(pmt_no="0049")
    voltage, gains_1, errors_1 = do_all_fits(procedure_indep_fit, pmt_no='0049', save=False, model_fit=model, plot_titles=True)
    voltage, gains_2, errors_2 = do_all_fits(procedure_indep_fit, pmt_no='0047', save=False, model_fit=model_4, plot_titles=True)
    popt_1, pcov_1 = curve_fit(power_law, voltage, gains_1, p0=[2, 1e+6], sigma=errors_1, absolute_sigma=True, maxfev=30000)
    popt_2, pcov_2 = curve_fit(power_law, voltage, gains_2, p0=[2, 1e+6], sigma=errors_2, absolute_sigma=True, maxfev=30000)
    perr_1 = np.sqrt(np.diag(pcov_1))
    perr_2 = np.sqrt(np.diag(pcov_2))
    print(popt_1, perr_1)

    plt.errorbar(voltage, gains_2, errors_2, fmt='.', ls='None', capsize=2, label="WA0047", color='b')
    plt.errorbar(voltage, gains_1, errors_1, fmt='.', ls='None', capsize=2, label="WA0049", color='r')
    plt.plot(voltage, power_law(voltage, *popt_2), color='b')
    plt.plot(voltage, power_law(voltage, *popt_1), color='r')
    plt.fill_between(voltage, power_law(voltage, popt_2[0] - perr_2[0], popt_2[1]), power_law(voltage, popt_2[0] + perr_2[0], popt_2[1]), color='b', alpha=0.1)
    plt.fill_between(voltage, power_law(voltage, popt_1[0] - perr_1[0], popt_1[1]), power_law(voltage, popt_1[0] + perr_1[0], popt_1[1]), color='r', alpha=0.1)
    plt.legend()
    plt.yscale('log')   
    plt.xscale('log')
    plt.xlabel("Voltage [V]")
    plt.ylabel("Gain [#e]")
    plt.title("Gain-Voltage curves")
    plt.tight_layout()
    plt.savefig("/home/todor/University/MPhys project/MPhys_project/analyze-lecroy/plots/gain_voltage_pmts_both.png", dpi=600)
    plt.show()



    """
    location = "/home/todor/University/MPhys project/Data_PMT/"+ "0047" + "/" + "850V" + "/"
    fname = 'C1--850V' + "_pmt-" + "0047" + "_1000--"
    all_waveforms = reader.iterate_large_files(0, 25, fname, loc=location)
    reader.make_heatmap(all_waveforms, True, "pmt_0047_850V.png", True, "Recorded waveforms for a single PMT at 850V")
    inv_waveforms = invert_waveform(all_waveforms)
    roi_begin, roi_end, peak = analyzer.determine_roi(inv_waveforms, True)
    roi = [roi_begin, roi_end]
    filtered_waveforms = analyzer.filter_outliers(inv_waveforms, peak, roi)
    areas, hist, bins = analyzer.find_area(filtered_waveforms, roi, 150, save=False, plot=True)
    """
    
    