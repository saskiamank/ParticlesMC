# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "atooms-pp>=4.2.1",
#     "fastparquet>=2025.12.0",
#     "pandas>=3.0.0",
# ]
# ///

import atooms.postprocessing as pp
import numpy as np
import pandas as pd
#from atooms.trajectory import Trajectory

temperatures = [1.0] #[3.0, 2.0, 1.6, 1.4, 1.25, 1.2, 1.15, 1.1, 1.05, 1.0]


def compute_fskt() -> pd.DataFrame:
    print("F_s(k,t)")
    df_list = []
    for T in temperatures:
        print(f"T = {T}")
        traj = Trajectory(f"../3-run-production/{T}/chains/1/trajectory.xyz")
        print('Trajectory loaded.')

        cf = pp.SelfIntermediateScatteringFast(
            traj,
            ksamples=1,
            kmin=7.4,
            kmax=7.4,
            tgrid=sorted({round(x) for x in np.logspace(0, 6, num=55)}),   # initialize an object of class SelfIntermediateScattering (the fast i have not found in the documentation), give a trajectory and specify kvalues as well as tgrid
        )
        cf.compute()                                                       # use method .compute() to actually determine the selfintermediate scattering function 

        print(f"k = {cf.grid[0][0]}")                                      # print the kvalue in terminal 

        # The grid is a tuple of lists: the first one is the list of all the k'S while the second one is the list of ts (the time grid, which is identical for all wave-vectors)
        # Thus cf.grid[0] are all kvalues, cf.grid[1] are all tvalues 

        N = len(cf.grid[1])                  # number of timesteps? 
        df_list.append(
            pd.DataFrame(
                {
                    "t": cf.grid[1],                                 # add column of timeArray with label t
                    cf.qualified_name: np.array(cf.values)[0],       # adds a second column whose name is dynamically extracted from cf with an array of the values of the scattering function, where for cf.values the first index corresponds to different kvalues and the second index corresponds to different timevalues, so that in this case we get one timeseries
                    "T": [T] * N,                                    # adds a column for the temperature which stores the same value of T for all N timesteps
                }
            )
        )

    return pd.concat(df_list).reset_index(drop=True)                # resets the index of the dataframe to the standard one (integers starting from 0), drop= True means that the old indices are replaced instead of adding them as a new column


def main() -> None:
    df = compute_fskt()
    df.to_parquet("fskt.parquet")                  # saving the data to the fskt.parquet file


if __name__ == "__main__":
    main()
