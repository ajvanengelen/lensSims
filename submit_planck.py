import os,sys

#arrays = ['033', '044', '070', '100', '143' ,'217', '353', '545', '857']
#lmaxes = [3000,3000,3000,6000,6000,6000,6000,6000,6000]

arrays = ['545', '857']
lmaxes = [6000,6000]

for array,lmax in zip(arrays,lmaxes):
    for region in ['boss', 'deep56']:
        cmd = "mpi_niagara 1 \"python bin/make_covsqrt.py v5.2 planck_hybrid --patch %s --array %s --overwrite --no-prewhiten --mask-version padded_v1 --nsims 3 --lmax %d --nsims 5\" --walltime 00:40:00 -t 80" % (region,array,lmax)
        os.system(cmd)
