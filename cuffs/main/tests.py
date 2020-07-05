import pickle
import numpy as np
log_2vMm = None

def init_gaussian_params(): 

    # ----------- setup global variables -----------------
    global log_2vMm
    #------------------------------------------------------

    log_2vMm_min = 0
    log_2vMm_max = 0
    print("Initializing Gaussian parameters", end="")
    fname = "Gaussian_minmax_" + str(len(log_2vMm)) + ".dat"
    print("fname = {0}".format(fname))

    try:
        print("inside try statement")
        #with open(fname, 'rb') as f:
        lt = pickle.load(open(fname, "rb"))
        print("inside with block")
        print(" (from cache)... ", end="\n")
        #lt = pickle.load(f)
        log_2vMm_min = lt[0]
        log_2vMm_max = lt[1]
    except (OSError, IOError) as e:
        print("inside except block")
        print("... ", end="\n")
        print("checkpoint1")
        log_2vMm_min = np.amin(log_2vMm)
        print("checkpoint 2 min = {0}".format(log_2vMm_min))
        log_2vMm_max = np.amax(log_2vMm)
        print("checkpoint 3 max = {0}".format(log_2vMm_max))
        lt = [log_2vMm_min, log_2vMm_max]
        print("lt = {0}".format(lt))
        # with open(fname, 'wb') as f:
        #     pickle.dump(lt, f)
        pickle.dump(lt, open(fname, "wb"))

    print("Done!")
    return

def start():
    global log_2vMm
    print("Loading log_2vMm.npy...")
    log_2vMm = np.load('/home/pankaj/radis-lab/data-1750-1850/log_2vMm.npy')
    print("Done!")
    #spec_h_log_2vMm = log_2vMm
    init_gaussian_params()
    print()

start()
