import numpy as np
from pixell import enmap,enplot

plot = lambda imap,dg=8 : enplot.show(enplot.plot(enmap.downgrade(imap,dg),grid=False))

class bin2D(object):
    def __init__(self, modrmap, bin_edges):
        self.centers = (bin_edges[1:]+bin_edges[:-1])/2.
        self.digitized = np.digitize(np.ndarray.flatten(modrmap), bin_edges,right=True)
        self.bin_edges = bin_edges
    def bin(self,data2d,weights=None):
        
        if weights is None:
            res = np.bincount(self.digitized,(data2d).reshape(-1))[1:-1]/np.bincount(self.digitized)[1:-1]
        else:
            res = np.bincount(self.digitized,(data2d*weights).reshape(-1))[1:-1]/np.bincount(self.digitized,weights.reshape(-1))[1:-1]
        return self.centers,res
    
def binned_power(pmap,modlmap,bin_edges):
    s = bin2D(modlmap,bin_edges)
    return s.bin(pmap)

    


def null_off_diagonals(cov):
    ocov = cov.copy()
    for i in range(cov.shape[0]):
        for j in range(cov.shape[0]):
            if i==j: continue
            ocov[i,j] = 0
    return ocov
    
def null_pol_off_diagonals(cov):
    ocov = cov.copy()
    for i in range(cov.shape[0]):
        for j in range(cov.shape[0]):
            if i==j or (i==0 and j==3) or (i==3 and j==0): continue
            ocov[i,j] = 0
    return ocov
    
def get_covsqrt(ps,method="multipow"):
    if method=="multipow":
        return enmap.multi_pow(ps.copy(),0.5)
    elif method=="arrayops":
        from enlib import array_ops
        return array_ops.eigpow(ps.copy(),0.5,axes=[0,1])

class Power(object):
    def __init__(self,shape,wcs,mc=False):
        self.mc = mc
        self.shape = shape
        self.wcs = wcs
        if self.mc:
            self._wts = {}
        self.nffts = 0
        
    def add_weight(self,key,weight): # will be used to initialize a mode coupling mask
        # DW, Steve, Zack? Someone fill this in?
        self._wts[key] = weight.copy()
    
    def mc_power(self,m1,w1,m2=None,w2=None):
        # Implements P^M
        # DW, Steve, Zack? Someone fill this in?
        # In this situation w1 (and optionally w2) will be strings
        # that point to a dictionary of pre-calculated mode couplings
        raise NotImplementedError
    
    
    def naive_power(self,f1,w1,f2=None,w2=None): # same API as mc_power, but calculates naive power
        # Implements P^N
        if f2 is None:
            assert w2 is None
            f2 = f1
            w2 = w1
        norm = np.mean(w1*w2,axis=(-2,-1),keepdims=True)
        return np.real(f1*np.conjugate(f2)) / norm
    
    def _expand_wts(self,wts):
        if self.mc: 
            return enmap.enmap(np.stack([self._weights[w] for w in wts]),self.wcs)
        else:
            return wts        
        
    
    def noise_power(self,kmaps1,weights1,kmaps2=None,weights2=None,
                    coadd_estimator=False,pfunc=None):
        # Implements the generalized optimal estimator N^GO_alpha_beta
        # and the sub-optimal noise estimator N^S_alpha_beta
        # All weights must include the apodized mask
        # No weights must be applied before hand
        # coadds1 = (Ny,Nx) -- the coadd map
        # cweights1 = (Ny,Nx) -- the coadd weight
        # imaps1 = (nsplits,Ny,Nx) -- the split maps
        # weights1 = (nsplits,Ny,Nx) -- the weights for the splits
        
        # select the PS function
        if pfunc is None: pfunc = self.mc_power if self.mc else self.naive_power
        if np.any(np.isnan(kmaps1)): raise ValueError
        if np.any(np.isnan(weights1)): raise ValueError
        if kmaps2 is not None:
            if np.any(np.isnan(kmaps2)): raise ValueError
            if np.any(np.isnan(weights1)): raise ValueError
        
        if coadd_estimator:
            # N^{GO} estimator
            assert kmaps1.ndim==3
            nsplits = kmaps1.shape[0]
            noise = np.sum(pfunc(kmaps1,weights1,kmaps2,weights2),axis=0)
            if np.any(np.isnan(noise)): raise ValueError
            return noise / nsplits / (nsplits-1.)
        else:
            # Cross and auto power
            assert kmaps1.ndim==3
            nsplits = kmaps1.shape[0]
            if kmaps2 is None:
                assert weights2 is None
                kmaps2 = kmaps1
                weights2 = weights1
            else: assert kmaps2.shape[0]==nsplits
            ncrosses = 0.
            nautos = 0.
            crosses = 0.
            autos = 0.
            for i in range(nsplits):
                for j in range(i,nsplits):
                    ret = pfunc(kmaps1[i],weights1[i],kmaps2[j],weights2[j])
                    if i!=j:
                        crosses = crosses + ret.copy()
                        ncrosses += 1
                    else:
                        autos = autos + ret.copy()
                        nautos += 1
            crosses = crosses / ncrosses
            autos = autos / nautos
            return (autos-crosses)/nsplits


    def get_n2d(self,ffts,wmaps,show=True,coadd_estimator=False):
        shape,wcs = ffts.shape[-2:],ffts.wcs
        modlmap = enmap.modlmap(shape,wcs)
        Ny,Nx = shape[-2:]
        nfreqs = ffts.shape[0]
        npol = ffts.shape[2]
        ncomps = nfreqs * npol


        def ncomp_to_freq_pol(index):
            ifreq = index // 3
            ipol = index % 3
            return ifreq, ipol


        n2d = enmap.zeros((ncomps,ncomps,Ny,Nx),wcs)
        pols = ['I','Q','U']
        for i in range(ncomps):
            for j in range(i,ncomps):
                ifreq,ipol = ncomp_to_freq_pol(i)
                jfreq,jpol = ncomp_to_freq_pol(j)

                isplits = ffts[ifreq,:,ipol]
                iwts = wmaps[ifreq,:,0]
                if i!=j:
                    jsplits = ffts[jfreq,:,jpol]
                    jwts = wmaps[jfreq,:,0]
                else:
                    jsplits = None ; jwts = None
                n2d[i,j] = self.noise_power(isplits,iwts,jsplits,jwts,
                                            coadd_estimator=coadd_estimator,pfunc=self.naive_power)
                if i!=j: n2d[j,i] = n2d[i,j]
                if show: plot(enmap.enmap(np.arcsinh(np.fft.fftshift(n2d[i,j])),wcs))
        return n2d

            
    
def get_coadd(imaps,wts,axis):
    # sum(w*m)/sum(w)
    twt = np.sum(wts,axis=axis)
    return np.nan_to_num(np.sum(wts*imaps,axis=axis)/twt),twt
    
    

def smooth_ps(ps,dfact=(16,16),radial_fit_lmin=300,
              radial_fit_lmax=8000,radial_fit_wnoise_annulus=500,
              radial_fit_annulus=20,radial_pairs=[],show=False):
    from tilec import covtools
    ncomps = ps.shape[0]
    assert ncomps==ps.shape[1]
    sps = ps.copy()*0.
    modlmap = ps.modlmap()
    wcs = ps.wcs
    
    # Do diagonals first
    for i in range(ncomps):
        if radial_pairs is None: do_radial = True
        else: do_radial = True if (i,i) in radial_pairs else False
        dnoise,_,_ = covtools.noise_average(ps[i,i].copy(),dfact=dfact,
                                            lmin=radial_fit_lmin,lmax=radial_fit_lmax,
                                            wnoise_annulus=radial_fit_wnoise_annulus,
                                            bin_annulus=radial_fit_annulus,radial_fit=do_radial)
        sps[i,i] = dnoise.copy()
        if show: plot(enmap.enmap(np.arcsinh(np.fft.fftshift(ps[i,i])),wcs))
        if show: plot(enmap.enmap(np.arcsinh(np.fft.fftshift(sps[i,i])),wcs))
            
        
    # Do offdiagonals
    for i in range(ncomps):
        for j in range(i+1,ncomps):
            if radial_pairs is None: do_radial = True
            else: do_radial = True if (i,j) in radial_pairs or (j,i) in radial_pairs else False
            afit = ps[i,j].copy()
            if not(do_radial):
                assert i!=j # otherwise this is a trivial operation
                mul = np.sqrt(sps[i,i]*sps[j,j])
            else:
                mul = 1.
            afit = np.nan_to_num(afit / mul)
            afit[...,modlmap<2] = 0.
            dnoise,_,_ = covtools.noise_average(afit,dfact=dfact,
                                                lmin=radial_fit_lmin,lmax=radial_fit_lmax,
                                                wnoise_annulus=radial_fit_wnoise_annulus,
                                                bin_annulus=radial_fit_annulus,radial_fit=do_radial)
            dnoise = dnoise * mul
            
            sps[i,j] = dnoise.copy()
            if i!=j: sps[j,i] = dnoise.copy()
            if show: plot(enmap.enmap(np.arcsinh(np.fft.fftshift(ps[i,j])),wcs))
            if show: plot(enmap.enmap(np.arcsinh(np.fft.fftshift(sps[i,j])),wcs))
            
    sps[...,modlmap<2] = 0.
            
    return sps



def binary_mask(mask,threshold=0.5):
    m = np.abs(mask)
    m[m<threshold] = 0
    m[m>threshold] = 1
    return m


def get_p1ds(p2d,modlmap,bin_edges):
    p1ds = np.zeros((p2d.shape[0],p2d.shape[0],bin_edges.size-1))
    for i in range(p2d.shape[0]):
        for j in range(p2d.shape[0]):
            p1ds[i,j] = binned_power(p2d[i,j],modlmap,bin_edges)[1]
    cents = (bin_edges[1:] + bin_edges[:-1])/2.
    return cents,p1ds


def compare_ps(cents,p1ds1,p1ds2):
    import matplotlib.pyplot as plt
    k = 0
    for i in range(p1ds1.shape[0]):
        for j in range(p1ds1.shape[0]):
            if not(i==j or (i==0 and j==3) or (i==3 and j==0)): continue
            plt.plot(cents,p1ds1[i,j]*cents**2./2./np.pi,label="%d-%d" % (i,j),color="C%d" % k,ls="--")
            plt.plot(cents,p1ds2[i,j]*cents**2./2./np.pi,color="C%d" % k)
            k = k+1
    plt.legend()
    plt.xlim(30,8000)
    plt.axvline(x=500,ls="--")
    plt.xlabel("$\\ell$")
    plt.ylabel("$D_{\\ell}$")
    plt.xscale('linear')
    plt.yscale('log')
    plt.show()

    k = 0
    for i in range(p1ds1.shape[0]):
        for j in range(p1ds1.shape[0]):
            if not(i==j or (i==0 and j==3) or (i==3 and j==0)): continue
            plt.plot(cents,p1ds1[i,j]/p1ds2[i,j],label="%d-%d" % (i,j),color="C%d" % k)
            k = k+1
    plt.legend()
    plt.xlim(30,8000)
    plt.ylim(0.5,1.5)
    plt.axvline(x=500,ls="--")
    plt.axhline(y=1,ls="--")
    plt.axhline(y=1.05,ls="-.")
    plt.axhline(y=0.95,ls="-.")
    plt.xlabel("$\\ell$")
    plt.ylabel("$R$")
    plt.xscale('linear')
    plt.yscale('linear')
    plt.show()                                    