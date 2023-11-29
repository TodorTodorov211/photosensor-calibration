import uproot
import numpy as np
import pandas as pd
import concurrent.futures

def read_root(files, branches, tree):
    executor = concurrent.futures.ThreadPoolExecutor()
    treeName=tree
    total_df = pd.DataFrame()
    for df in uproot.iterate(files,branches=branches,entrysteps=1000000,namedecode='utf-8',executor=executor,blocking=True, library='pd'):
        total_df = pd.concat([total_df, df])
    print("Total number of events: {}".format(len(total_df.index)))
    return total_df