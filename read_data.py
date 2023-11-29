import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys 
sys.path.append("utils/")
from plotting_utils import plot2d
import matplotlib.colors



def generate_counter_string(iterator):
    """Generate the counter string in every filename from an integer. (e.g. 00023 from 23)

    Parameters
    ----------
    iterator : int
        the file count 

    Returns
    -------
    string
        The file name ending
    """
    appendix = ""
    if iterator < 10:
        appendix = "0000{}".format(iterator)
    elif iterator < 100:
        appendix = "000{}".format(iterator)
    elif iterator < 1000:
        appendix = "00{}".format(iterator)
    elif iterator < 10000:
        appendix = "0{}".format(iterator)
    else:
        print("Illegal filename exception")
        appendix = "00000"  

    return appendix  



def read_large_file(filename, loc="/home/todor/University/MPhys project/MPhys_project/analyze-lecroy/data/"):
    all_waveforms = []
    raw_data = np.genfromtxt(loc + filename, skip_header=0, delimiter=',')
    new_waveform = np.array([[-1, -1]])
    for index, entry in enumerate(raw_data):
            #print(entry)
        if entry[0] < raw_data[index - 1, 0]:
            if index == 0:
                new_waveform = np.append(new_waveform, [entry], axis=0)
                continue
            new_waveform = np.delete(new_waveform, 0, axis=0)
            all_waveforms.append(new_waveform)
                #new_waveform = [entry]
            new_waveform = np.array([[-1, -1], entry])
                #print("saving...")
        else:
                

            new_waveform = np.append(new_waveform, [entry], axis=0)
                #new_waveform.append(entry)
            if index == len(raw_data) - 1:
                    #print(new_waveform)
                new_waveform = np.delete(new_waveform, 0, axis=0)
                all_waveforms.append(new_waveform)
                    #new_waveform = []
                new_waveform = np.array([[-1, -1]])    
    
    return all_waveforms



def iterate_large_files(start, stop, filename, segment_no=1000, loc="/home/todor/University/MPhys project/MPhys_project/analyze-lecroy/data/"):
    """Iterate over all files and split all the segments into a single list

    Parameters
    ----------
    start : int
        filename count start
    stop : int
        filename count stop
    filename : string
        the filename to look for
    segment_no : int, optional
        the number of segments in each file, by default 1000
    loc : str, optional
        the location of the file, by default "/home/todor/University/MPhys project/MPhys_project/analyze-lecroy/data/"

    Returns
    -------
    list<numpy.array>
        a list with each entry being an numpy array containing the time in the 0th column and the amplitude in the 1st column
    """

    header = 4 + segment_no
    all_waveforms = []
    print("Reading files...")
    for iterator in tqdm(range(start, stop)):
        file = loc + filename + generate_counter_string(iterator) + ".txt"
        raw_data = np.genfromtxt(file, skip_header=header, delimiter=',')
        #new_waveform = []
        new_waveform = np.array([[-1, -1]])
        for index, entry in enumerate(raw_data):
            #print(entry)
            if entry[0] < raw_data[index - 1, 0]:
                if index == 0:
                    new_waveform = np.append(new_waveform, [entry], axis=0)
                    continue
                new_waveform = np.delete(new_waveform, 0, axis=0)
                all_waveforms.append(new_waveform)
                #new_waveform = [entry]
                new_waveform = np.array([[-1, -1], entry])
                #print("saving...")
            else:
                

                new_waveform = np.append(new_waveform, [entry], axis=0)
                #new_waveform.append(entry)
                if index == len(raw_data) - 1:
                    #print(new_waveform)
                    new_waveform = np.delete(new_waveform, 0, axis=0)
                    all_waveforms.append(new_waveform)
                    #new_waveform = []
                    new_waveform = np.array([[-1, -1]])
        
        #print(all_waveforms)

    return all_waveforms

def make_heatmap(all_waveforms, save=False, savename="initial_data_reading_10x1000waveforms_heatmap.png", plot_title=False, title="Recorded waveforms for a single SiPM at 56V bias"):
    """Make a heatmap of the waveforms (faster than plotting all waveforms)

    Parameters
    ----------
    all_waveforms : list<numpy.array>
        a list of numpy arrays with each array being 1 waveform with timepoints in the 0th column and amplitudes in the 1st column
    save : bool, optional
        save the figure or not, by default False
    savename : str, optional
        name of the file to save, by default "initial_data_reading_10x1000waveforms_heatmap.png"
    """
    time = []
    amplitude = []
    print("Making a heatmap...")
    for index, waveform in tqdm(enumerate(all_waveforms)):
        for index_inner, single_point in enumerate(waveform):
            time.append(single_point[0])
            amplitude.append(single_point[1])
        
    
    
    image, x_edges, y_edges = np.histogram2d(time, amplitude, bins=[400, 300])
    image = np.where(image == 0, np.full(np.shape(image), np.nan), image)
    
    fig = plt.figure(figsize=(12, 9))
    axes = fig.add_subplot()

    plot2d(image, x_edges, y_edges, axes, norm=matplotlib.colors.LogNorm())
    axes.set_xlabel('time[s]', fontsize=22)
    axes.set_ylabel("Amplified signal[V]", fontsize=22)
    axes.tick_params(axis='both', which='major', labelsize=18)
    axes.tick_params(axis='both', which='minor', labelsize=18)
    
    if plot_title == True:
        axes.set_title(title, fontsize=22)

    if save == True:
        loc = "/home/todor/University/MPhys project/MPhys_project/analyze-lecroy/plots/"
        fig.savefig(loc + savename, dpi=600)

    plt.show()


    
if __name__ == "__main__":
    all_waveforms = iterate_large_files(0, 25, "56V_sipm-1_1000--", loc="/home/todor/University/MPhys project/Data_SiPM/411/56V/")
    #make_heatmap(all_waveforms, True, "sipm-411_56V_25000waveforms.png", True)
    """
    for index, waveform in tqdm(enumerate(all_waveforms)):
        #for iterator, entry in enumerate(waveform):
        
        plt.plot(waveform[:,0], waveform[:,1], color='b', alpha=0.002)
        #print(waveform[:, 0])
    plt.xlabel('time(s)')
    plt.ylabel("Amplified signal(V)")
    plt.savefig("/home/todor/University/MPhys project/MPhys_project/analyze-lecroy/plots/initial_data_reading_10x1000waveforms.png", dpi=600)
    plt.show()
    """

