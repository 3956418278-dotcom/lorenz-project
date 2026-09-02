"""
Lorenz-63 time-invariant forcing experiment with no noise

## Figure-only reproduction

### Requirements

Hardware: 4 seconds on personal computer.

### Recipe
1. `conda create --name lorenzsteady numpy=1.24.4 dask-mpi scipy matplotlib` **OR** see [requirements.txt](requirements.txt)
2. `conda activate lorenzsteady`
3. Run `python -m lorenz_steady 2>&1 |tee log`

Locate plotting code by searching ' # Figure '

## Full reproduction

### Requirements

Hardware: around 1 day with 48 processes (no noise)

### Recipe
1. `conda create --name lorenzsteady numpy=1.24.4 dask-mpi scipy matplotlib` **OR** see [requirements.txt](requirements.txt)
2. `conda activate lorenzsteady`
3. Try ONE of these commands:
`time mpirun -np 48 python -m lorenz_steady`
`sbatch -n 256 --hint=compute_bound -t 230 --mem-per-cpu=4000 -o log/slurm.out-%j --wrap='mamba activate hcmc_jas; time srun -n $SLURM_NTASKS python -m lorenz_steady'`
4. Go to `Figure-only reproduction`

DO NOT append .py at end, fail in a strange way
"""
# rsync -avzum --no-p --include "ngrp*" --exclude "*.txt" teach:~/zod/l2-lorenz/mean_std ./

import os
from pprint import pprint
#pprint(os.environ)
pmi_size = int(os.getenv('OMPI_COMM_WORLD_SIZE', os.getenv('PMI_SIZE', os.getenv('SLURM_NTASKS', 1))))
if pmi_size > 2:
    from dask_mpi import initialize
    initialize()  # block except rank 1
    from dask.distributed import Client, wait
    client = Client()

import json
import re
from timeit import default_timer as timer
tic = timer()
print('tic')
import numpy as np
import dask.array as da
from scipy.integrate import solve_ivp
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
np.set_printoptions(precision=6, linewidth=120)
plt.rcParams['font.size'] = 10.

def Lorenz(T=10000,fx=0,fy=0,fz=0,density=10,T_spinup=20,seed=0, **kwargs):
    p=10
    q=28
    r=8/3
    rng = np.random.default_rng(seed)
    lorenz_tend = lambda t, s: [p*(s[1]-s[0])+fx, q*s[0]-s[1]-s[0]*s[2]+fy,
                                s[0]*s[1]-r*s[2]+fz]
    sol = solve_ivp(lorenz_tend, [0, T+T_spinup], rng.random(3),
                    t_eval=np.r_[:T*density]/density +T_spinup,
                    **kwargs)
    return sol.y

def Lorenz_stat(**kwargs):
    xyz=Lorenz(**kwargs)
    xyz_groupmean = xyz.reshape(3,10,-1).mean(-1)
    mm = xyz_groupmean.mean(-1)
    ss = xyz_groupmean.std(-1, ddof=1)/10**0.5
    return mm, ss,xyz_groupmean[2]

def slugify(s):
    """Clean sanitize illegal invalid filename."""
    return re.sub(r'[\s]+', '_', re.sub(r'[^\.\w\s-]', '', s))

def Lorenz_cache(seed=None, **kwargs):
    s_kw = 'mean_std/' + slugify(f'cache-{sorted(kwargs.items())}')
    fn_json = s_kw + '.txt'
    if seed is not None:
        fn_json = s_kw + f'/{seed}.txt'
        os.system('mkdir -p '+s_kw)
    try:
        mm, ss = json.load(open(fn_json,'r'))
        if pmi_size > 2:
            print(f'sucesss    {fn_json}')
    except(FileNotFoundError):
        print(f'fail    {fn_json}')
        mm,ss ,t0= Lorenz_stat(seed=seed, **kwargs)
        with open(fn_json,'w') as fp:
            json.dump([mm.tolist(), ss.tolist()], fp)
    mm = np.array(mm)
    ss = np.array(ss)
    return mm, ss

def Lorenz_mpi(ngrp=None, **kwargs):
    if ngrp or pmi_size > 2:
        if ngrp is None:
            ngrp = pmi_size - 2
        s_kw = 'mean_std/' + slugify(f'cache-{sorted(kwargs.items())}')
        fn_json = s_kw + f'/ngrp{ngrp}.txt'
        try:
            mm, ss = json.load(open(fn_json,'r'))
        except(FileNotFoundError):
            xyz_groupmean = np.zeros((3, ngrp))  # dtype=np.float64
            func = lambda s: Lorenz_cache(seed=s, **kwargs)[0]
            print(f'before wait {kwargs}')
            xyz_groupmean = da.apply_gufunc(
                func, '()->(3)', da.arange(ngrp, chunks=1), axis=0, output_sizes={'3':3},
                vectorize=True, meta=xyz_groupmean).compute()
#        wait(client.map(func, range(pmi_size - 2)))
            print(f'Finish all {timer()-tic:.4f}s {kwargs}')
            mm = xyz_groupmean.mean(-1)
            ss = xyz_groupmean.std(-1, ddof=1) / xyz_groupmean.shape[-1]**0.5
            with open(fn_json,'w') as fp:
                json.dump([mm.tolist(), ss.tolist()], fp)
        mm = np.array(mm)
        ss = np.array(ss)
        return mm, ss
    else:
        return Lorenz_cache(**kwargs)
#func = lambda s: np.array([0, 10, 100])+s
#xxx = da.arange(9, chunks=1).map_blocks(func, new_axis=0, template=)


def spread_mpi(ngrp=None, T=50, density=10, T_spinup=0, **kwargs):
    s_kw = 'time_series/' + slugify(f'{sorted(dict(kwargs, ngrp=ngrp, T=T, density=density).items())}')
    fn_json = s_kw + '.txt'
    try:
        mm,ss = json.load(open(fn_json,'r'))
#    if pmi_size > 2:
#      print(f'sucesss    {fn_json}')
    except(FileNotFoundError):
#  if ngrp or pmi_size > 2:
        func = lambda s: Lorenz(seed=s, T=T, density=density, T_spinup=0, **kwargs)
        xyz_groupmean = np.zeros((ngrp, 3, T*density))  # dtype=np.float64
#    print(f'before wait {kwargs}')
        xyz_groupmean = da.apply_gufunc(
            func, '()->(3,t)', da.arange(ngrp, chunks=1), axes=[(), (1,2)], output_sizes={'3':3, 't':T*density},
            vectorize=True, meta=xyz_groupmean).compute()
#    wait(client.map(func, range(pmi_size - 2)))
#    print(f'Finish all {timer()-tic:.4f}s {kwargs}')
        mm = xyz_groupmean.mean(0)
        ss = xyz_groupmean.std(0, ddof=1)
        os.system('mkdir -p time_series')
        with open(fn_json,'w') as fp:
            json.dump([mm.tolist(), ss.tolist()], fp)
#    xyz=Lorenz(T=5, T_spinup=0)
    return mm, ss
#  else:
#    return Lorenz_cache(**kwargs)


def tol_check(rtol=1e-3, atol=1e-6, **kwargs):
    m0, s0 = Lorenz_mpi(rtol=rtol, atol=atol, **kwargs)
    m1, s1 = Lorenz_mpi(rtol=rtol/10, atol=atol/10, **kwargs)
    print((m1-m0)/s0,'<2*sqrt(2)')
    #print((m1-m0)/(s1**2+s0**2)**0.5)


#dim='fx';f=10;
def oppression(dim, f, **kwargs):
    print()
    m0, s0 = Lorenz_mpi(**kwargs)
    m0[:2] = 0
    m1, s1 = Lorenz_mpi(**kwargs, **{dim:f})
    if dim == 'fz':
        print('+f: <2,<2,>10',(m1-m0)/s0)
        m2, s2 = Lorenz_mpi(**kwargs, **{dim:-f})
        print('-f: <2,<2,>10',(m2-m0)/s0)
        print('linear, <2*sqrt(6)',(m1+m2-2*m0)/s0)
        l_col = (m1-m2) /2/f
        l_col[:2] = 0
        ci = stats.norm.ppf(0.975) * (s1**2+s2**2)**0.5 /2/f
    if dim in ('fx', 'fy'):
        print('+f: >10,>10,<2.828',(m1-m0)/s0)
        l_col = (m1-m0) / f
        l_col[2] = 0
        ci = stats.norm.ppf(0.975) * s1 / f
        m5 = Lorenz_mpi(**kwargs, **{dim:f/2})[0]
        print('+0.5f: >10,>10,<2.828',(m5-m0)/s0)
        print('linear, <2*sqrt(6)',(m1-2*m5+m0)/s0)
    return l_col, ci


#tol_check()
# kw = dict(T_spinup=50,rtol=1e-3,atol=1e-6,density=10, ngrp=30000)  # s
#kw = dict(T_spinup=20,rtol=1e-5,atol=1e-8,density=10, ngrp=46*15)  # 26s
#kw = dict(T_spinup=20,rtol=1e-5,atol=1e-8,density=10, ngrp=46*250)  # 26s, fail tol_check (3*sigma)
#kw = dict(T_spinup=50,rtol=1e-6,atol=1e-9,density=10, ngrp=30000)  # 36s
kw = dict(T_spinup=50,rtol=1e-6,atol=1e-9,density=10, ngrp=8, method='DOP853')  # 29s
#kw = dict(kw, T_spinup=100)
ngrp2 = 8 #46*5 #
#kw = dict(kw, ngrp=46*2)
#kw = dict(rtol=1e-5,atol=1e-8,density=10, ngrp=46*2)  #T=100000,
#kw = dict(rtol=1e-5,atol=1e-8,density=10, ngrp=46*60, method='RK23')  #T=100000,
if __name__ == '__main__':
#    kwargs = kw.copy(); del kwargs['ngrp']
#    kw = dict(kw, ngrp=ngrp2)
#    for method in ['DOP853', 'RK45']:
#    for kwo in [{'density': 20}, ]:  # {'density': 10}
#        tic = timer()
#        Lorenz_stat(**kwargs, method=method)
#        Lorenz_stat(**dict(kwargs, **kwo))
#        toc = timer() - tic
#        kw2 = kw if method=='RK45' else dict(kw, method=method)
#        ss = Lorenz_mpi(**kw, method=method)[1]
#        ss = Lorenz_mpi(**kw2)[1]
#        ss = Lorenz_mpi(**dict(kw, **kwo))[1]
#        print(f'{toc:.4f}s ss={ss} {ss[2]**2*toc} {kwo}')
#    quit()   # TODO

 # b = tan beta; a=tan alpha; pp; qq
#    b, a, pp, qq = 2, 1, 3, 1
    b, a, pp, qq = 3, 2, 2, 1
    tt = np.r_[:600]/100
    xt = (b/pp*(1-np.exp(-pp*tt)) - a/qq*(1-np.exp(-qq*tt))) /(b-a)
    yt = ((1-np.exp(-pp*tt))/pp - (1-np.exp(-qq*tt))/qq) *a*b/(b-a)
 # z = [1 1;tan(a) tan(b)]; z*[1/p 0;0 0]*inv(z)*[1;0]
    xyp = np.array([1,a])*b/pp/(b-a)
    xyq = -np.array([1,b])*a/qq/(b-a)
 # Figure A1
    with PdfPages(f'obs-partial.pdf') as pdf:
 # 19 (one column), 27, 33, and 39 (two columns) picas.
        fig, axs = plt.subplots(2, 1, figsize=(19*12/72.27, 13*12/72.27))
        axs[0].plot(tt, xt)
        axs[0].set_ylabel('resolved')#, labelpad=10
        axs[0].set_xticks([])
        axs[1].plot(tt, yt)
        axs[1].set_ylabel('unresolved')#, labelpad=10
        axs[-1].set_xlabel('$t$')
#        plt.show()
        fig.align_ylabels()
        fig.tight_layout(pad=0.1, h_pad=0)
        pdf.savefig()
    b, a, pp, qq = 2, 1, 3, 1
    tt = np.r_[:600]/100
    xt = (b/pp*(1-np.exp(-pp*tt)) - a/qq*(1-np.exp(-qq*tt))) /(b-a)
    yt = ((1-np.exp(-pp*tt))/pp - (1-np.exp(-qq*tt))/qq) *a*b/(b-a)
 # z = [1 1;tan(a) tan(b)]; z*[1/p 0;0 0]*inv(z)*[1;0]
    xyp = np.array([1,a])*b/pp/(b-a)
    xyq = -np.array([1,b])*a/qq/(b-a)
#    with PdfPages(f'obs-partial.pdf') as pdf:
 # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/gridspec_multicolumn.html#sphx-glr-gallery-subplots-axes-and-figures-gridspec-multicolumn-py
    fig, ax = plt.subplots(figsize=(89/25.4, 89/25.4))
    ax.set_aspect(1, 'box')
    ax.plot(xt, yt, 'k')
    kw_arrow = dict(length_includes_head=True, width=0, fc='None', head_width=0.1, head_length=0.3, overhang=1)
    ax.arrow(0, 0, *xyp, **kw_arrow, ec='C1')
    ax.arrow(*xyp, *xyq, **kw_arrow, ec='C0')
    ax.set_xlabel('resolved')
    ax.set_ylabel('unresolved')
#    plt.show()
    fig.tight_layout(pad=0.1, h_pad=0)
    plt.savefig('obs-partial-vec.png', dpi=300, transparent=True)
#        pdf.savefig()
#    quit()


    T, density = kw.get('T',50), kw.get('density',10)
# #  T = 200
# #  kws = [dict(), dict(rtol=kw['rtol']/10, atol=kw['atol']/10)]
    kws = [dict()]
    ts_plot = np.zeros((len(kws), 6, T*density))  # dtype=np.float64
    for i, kw1 in enumerate(kws):
        ts_plot[i, :3], ts_plot[i, 3:] = spread_mpi(**dict(kw, **kw1, T=T, density=density))
 #  quit()   # TODO
 # Figure 1
    with PdfPages(f'spinup.pdf') as pdf:
        fig, axs = plt.subplots(6, 1, figsize=(39*12/72.27, 39*12/72.27))  #sharex=True,
#        fig.subplots_adjust(hspace=0)
        txts = [r'$\langle x\rangle$',r'$\langle y\rangle$',r'$\langle z\rangle$',r'$\sigma_x$',r'$\sigma_y$',r'$\sigma_z$']
        for i, (ax, txt) in enumerate(zip(axs, txts)):
# #          ax.plot(np.r_[:T*density]/density, ss[i])
            ax.plot(np.r_[:T*density]/density, ts_plot[:,i,:].T)
 #            ax.plot(np.r_[100*density:T*density]/density, ts_plot[:,i,100*density:].T)
            ax.set_ylabel(txt, rotation=0)#, labelpad=10
            ax.label_outer(True)
#            if i < axs.size-1:
#                ax.set_xticks([])
# #          ax.set_xlim([20, T])
#        plt.show()
        axs[-1].set_xlabel('$t$')
        fig.align_ylabels()
        fig.tight_layout(pad=0.1, h_pad=0)
        pdf.savefig()
#    quit()

    tol_check(**kw)
    #tol_check(rtol=1e-6,atol=1e-3,density=10, ngrp=92)

    #quit()   # TODO

    m0, s0 = Lorenz_mpi(**kw)
    print(m0, s0, stats.norm.ppf(0.975)*s0)
#  oppression('fz',1)  # (5.8>2*6**.5, s/n>10, ngrp15k), s/n>5, ngrp=46*60
#  oppression('fz', 0.8)  # 6.6>2*6**.5, ngrp15k
#    kw = dict(kw, ngrp=30000)
    fz = 0.5
    tol_check(**kw, fz=fz)
    l3c, ci3 = oppression('fz', fz, **kw)  # (s/n>10, ngrp30k); s/n>5, ngrp15k
    fzs = [-fz, 0, fz]
    ms_fz = np.array([Lorenz_mpi(**kw, **wrk) for wrk in [{'fz':-fz}, {}, {'fz':fz}]])
    ms_fz[:,1,:] *= stats.norm.ppf(0.975)
    kw = dict(kw, ngrp=ngrp2)
    fy = 0.1
    l2c, ci2 = oppression('fy', fy, **kw)  # 12, 12, 1.3, ngrp92
    l2c[:2] = l2c[:2].mean()
    fys = fy * np.array([0.5, 1])
    ms_fy = np.array([Lorenz_mpi(**kw, fy=wrk) for wrk in fys])
    ms_fy[:,1,:] *= stats.norm.ppf(0.975)
    fx = 0.25
    l1c, ci1 = oppression('fx', fx, **kw)  # 12, 6, -0.3, ngrp92
    l1c[:2] = l1c[:2].mean() + 0.05*np.array([1, -1])
    fxs = fx * np.array([0.5, 1])
    ms_fx = np.array([Lorenz_mpi(**kw, fx=wrk) for wrk in fxs])
    ms_fx[:,1,:] *= stats.norm.ppf(0.975)
    lll = np.array([l1c, l2c, l3c]).T
    pprint([lll, np.array([ci1, ci2, ci3]).T])
    print(-np.linalg.inv(lll))

    uuu, sss, vh = np.linalg.svd(lll)
    uuu = uuu*np.sign(vh.sum(1))
    vvv = vh.T*np.sign(vh.sum(1))
    print(uuu)
    print(sss)
    print(vvv)  #.reshape(-1,1)
 # aa @ np.diag(sss) @ bb
    eee, www = np.linalg.eig(lll)
    ind = eee.argsort()[::-1]
    print(eee[ind])
    print((www * np.sign(www.sum(0)))[:,ind])
 # bb @ np.diag(aa) @ np.linalg.inv(bb)

 # Figure 2
    with PdfPages(f'xy.mean.pdf') as pdf:
        fig, axs = plt.subplots(1, 3, width_ratios=[1.0, 1, 0.8], figsize=(39*12/72.27, 12*12/72.27))
#        fig, axs = plt.subplots(1, 2, width_ratios=[1, 0.8], figsize=(20*12/72.27, 10*12/72.27))
        kw_anno = dict(va='top', xycoords='axes fraction', size=8, weight='bold')
        zerr = (ms_fz[1,1,2]**2 + ms_fy[0,1,2]**2) **0.5
        axs[0].axhspan(m0[2]-zerr, m0[2]+zerr, color='0.8')
        axs[0].axhspan(m0[2]-ms_fz[1,1,2], m0[2]+ms_fz[1,1,2], color='0.7')
        axs[0].plot(fzs[0::2], ms_fz[0::2,0,2] +ms_fz[:,0,2].mean()-ms_fz[0::2,0,2].mean(), c='C3')
        axs[0].errorbar(fxs, ms_fx[:,0,2], ms_fx[:,1,2], ls='', c='C1')
        axs[0].errorbar(fys, ms_fy[:,0,2], ms_fy[:,1,2], ls='', c='C0')
        axs[0].errorbar(fzs, ms_fz[:,0,2], ms_fz[:,1,2], ls='', c='C3')#, fmt='_'
        axs[0].plot(fxs, ms_fx[:,0,2], 'k.')
        axs[0].plot(fys, ms_fy[:,0,2], 'k.')
        axs[0].plot(fzs, ms_fz[:,0,2], 'k.')
        axs[0].set_xlabel('$f$')
        axs[0].set_ylabel('$z$', rotation=0)
        axs[0].ticklabel_format(useOffset=False)
        axs[-2].set_aspect(1, 'datalim')
 #        axs[-2].errorbar(0, 0, stats.norm.ppf(0.975)*s0[1], stats.norm.ppf(0.975)*s0[0])
        kw_arrow = dict(length_includes_head=True, width=0, fc='None', head_width=0.04, overhang=1)
        kw_fa = dict(kw_arrow, alpha=0.5)#ls=':'
        axs[-2].plot([0,ms_fx[-1,0,0]], [0,ms_fx[-1,0,1]], c='C1')
        axs[-2].errorbar(ms_fx[:,0,0], ms_fx[:,0,1], ms_fx[:,1,1], ms_fx[:,1,0], ls='', c='C1')
        for wrk in fxs:
            axs[-2].arrow(0, 0, wrk*0.2, 0, **dict(kw_fa, head_width=0.005), ec='C1')
        axs[-2].plot(ms_fx[:,0,0], ms_fx[:,0,1], 'k.', ms=3)
        axs[-2].plot([0,ms_fy[-1,0,0]], [0,ms_fy[-1,0,1]], c='C0')
        axs[-2].errorbar(ms_fy[:,0,0], ms_fy[:,0,1], ms_fy[:,1,1], ms_fy[:,1,0], ls='', c='C0')
        for wrk in fys:
            axs[-2].arrow(0, 0, 0, wrk*0.2, **dict(kw_fa, head_width=0.005), ec='C0')
        axs[-2].plot(ms_fy[:,0,0], ms_fy[:,0,1], 'k.', ms=3)
        axs[-2].set_xlabel('$x$')
        axs[-2].set_ylabel('$y$', rotation=0)  # r'$\overline{y}$'
#        axs[-2].annotate('b', xy=(0.05,0.95), **kw_anno)
        axs[-1].set_aspect(1, 'datalim')
        axs[-1].arrow(0, 0, *vvv[:2,0]*0.2, **kw_fa, ec='C0')
        axs[-1].arrow(0, 0, *vvv[:2,1]*0.2, **kw_fa, ec='C1')
        axs[-1].arrow(0, 0, *uuu[:2,0]*sss[0], **kw_arrow, ec='C0')
        axs[-1].arrow(0, 0, *uuu[:2,1]*sss[1], **kw_arrow, ec='C1')
        axs[-1].set_xlabel('$x$')
        axs[-1].set_ylabel('$y$', rotation=0)
#        axs[0].annotate('a', xy=(0.10,0.95), **kw_anno)#, ha='right'
#        axs[-1].annotate('c', xy=(0.05,0.95), **kw_anno)
        for m, ax, wrk in zip(map(chr, range(ord('a'), ord('z'))), axs.flat, [0.10, 0.05, 0.05]):
#        for m, ax, wrk in zip(map(chr, range(ord('a'), ord('z'))), axs.flat, [0.05, 0.05]):
            ax.annotate(m, xy=(wrk,0.95), **kw_anno)
#        fig.align_ylabels()
        fig.tight_layout(pad=0.1, w_pad=0.5)
#        plt.show()
        pdf.savefig()
#        plt.savefig('xy.mean.png', dpi=300, transparent=True)

