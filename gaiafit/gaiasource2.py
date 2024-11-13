import matplotlib.pyplot as plt
import gaiafit
import pandas as pd
import numpy as np
from nsstools import NssSource
import corner
import emcee
import utils
import astropylib.webscrape
import astropylib.rvprec
from datetime import datetime
import astropy.time
import astroplan
import astropy.units as u

def read_gsheet(sheet_id,sheet_name):
    """
    Read a google sheet and return a pandas dataframe

    EXAMPLE:
        sheet_id = '1kAxJJAfS8o1afxRW8bHyskA7s7qgU4cvnzJCHpTSSyU'
        sheet_name = 'main'
        df = read_gsheet(sheet_id,sheet_name)
    """
    url = 'https://docs.google.com/spreadsheets/d/{}/gviz/tq?tqx=out:csv&sheet={}'.format(sheet_id,sheet_name)
    df = pd.read_csv(url,dtype={'source_id': str})
    return df

def log_prob_gaia(x, mu, cov):
    """
    log_prob_gaia
    
    INPUT:
        x - parameters to be sampled over
        mu - mean of parameter values [e.g., Gaia measured values]
        cov - covariance matrix (e.g., C)
        
    EXAMPLE:
        ndim = 5
        nwalkers = 32
        means = np.random.rand(ndim)
        p0 = np.random.rand(nwalkers, ndim)
        log_prob(p0[0], means, cov)
        
    NOTES:
        https://emcee.readthedocs.io/en/stable/tutorials/quickstart/
    """
    Theta = x - mu # deviation vector
    return -0.5 * np.dot(Theta, np.linalg.solve(cov, Theta))

class GaiaSource(object):
    
    nwalkers = 50
    
    def __init__(self,filename='',df=None,source_id=None,source_index=None,st_mass=None,st_masserr=None,verbose=True,estimate_rvprec=False):
        """
        EXAMPLE:
            df = astropylib.starinfo.query_gaia_nss_two_body_orbit_from_source_id(source_id,add_mass=True)
            GS = gaiasource.GaiaSource(df=df,source_id=source_id)
            GS.plot_expected_rv_orbit(tstart='2022-01-01',tstop='2025-12-31',het_start='2024-04-01',
                                    het_stop='2025-08-01',label='24-2 Trimester',plot_het=False)
        """
        if filename is not '':
            self.filename = filename
            self.df = pd.read_csv(filename,dtype={'source_id': str})
        else:
            #self.df = read_gsheet(gsheet,gsheet_name)
            self.df = df
            self.df.source_id = self.df.source_id.astype('str')
        #self.df['source_id'] = np.array(self.df.source_id.values,dtype=int)
        #self.df['source_id'] = np.array(self.df.source_id.values,dtype=str)
        if source_index is None:
            source_id = str(source_id)
            source_index = self.find_source_index(source_id,verbose=verbose)
        self.source_index = source_index
        self.source = NssSource(self.df, indice=self.source_index)
        self.source_id = self.source._star['source_id']
        if verbose:
            print('SOURCE_ID',self.source_id)
        self.covmat = self.source.covmat()
        self.C = self.covmat[['a_thiele_innes','b_thiele_innes','f_thiele_innes','g_thiele_innes','eccentricity','period','t_periastron']].iloc[5:,:]
        self.CC = self.C.values
        self.Campbell = self.source.campbell()
        self.inclination = self.Campbell['inclination'].values[0]
        self.inclination_error = self.Campbell['inclination_error'].values[0]
        self.a0 = self.Campbell['a0'].values[0]
        self.a0_error = self.Campbell['a0_error'].values[0]
        self.arg_periastron = self.Campbell['arg_periastron'].values[0]
        self.arg_periastron_error = self.Campbell['arg_periastron_error'].values[0]
        self.nodeangle = self.Campbell['nodeangle'].values[0]
        self.nodeangle_error = self.Campbell['nodeangle_error'].values[0]
        self.parallax = self.source._star['parallax']
        self.parallax_error = self.source._star['parallax_error']
        self.ndim = self.CC.shape[0]
        self.a_thiele_innes = self.source._star['a_thiele_innes']
        self.a_thiele_innes_error = self.source._star['a_thiele_innes_error']
        self.b_thiele_innes = self.source._star['b_thiele_innes']
        self.b_thiele_innes_error = self.source._star['b_thiele_innes_error']
        self.f_thiele_innes = self.source._star['f_thiele_innes']
        self.f_thiele_innes_error = self.source._star['f_thiele_innes_error']
        self.g_thiele_innes = self.source._star['g_thiele_innes']
        self.g_thiele_innes_error = self.source._star['g_thiele_innes_error']
        self.eccentricity = self.source._star['eccentricity']
        self.eccentricity_error = self.source._star['eccentricity_error']
        self.period = self.source._star['period']
        self.period_error = self.source._star['period_error']
        self.t_periastron = self.source._star['t_periastron']
        self.t_periastron_error = self.source._star['t_periastron_error']
        if st_mass is not None:
            if verbose:
                print('Using provided mass')
            self.st_mass = st_mass
            self.st_masserr = st_masserr
        else:
            if verbose:
                print('No mass provded, using mass in df: M={:0.3f}+-{:0.3f}M_sun'.format(self.source._star['st_mass'],self.source._star['st_masserr']))
            self.st_mass = self.source._star['st_mass']
            self.st_masserr = self.source._star['st_masserr']
        self.means = np.array([self.a_thiele_innes,
                               self.b_thiele_innes,
                               self.f_thiele_innes,
                               self.g_thiele_innes,
                               self.eccentricity,
                               self.period,
                               self.t_periastron])
        self.errors = np.array([self.a_thiele_innes_error,
                                self.b_thiele_innes_error,
                                self.f_thiele_innes_error,
                                self.g_thiele_innes_error,
                                self.eccentricity_error,
                                self.period_error,
                                self.t_periastron_error])
        self.labels = ['A','B','F','G','e','P','t_peri']
        self.df_known = pd.DataFrame(list(zip(self.labels,self.means,self.errors)),columns=['label','median','error'])

        self.pl_mass = np.nan
        self.pl_mass_error = np.nan
        self.K, self.K_error, self.K_error_low, self.K_error_up = self.get_semi_amplitude_and_error()

        labels_c = ['source_id','period','period_error','eccentricity','eccentricity_error','inclination','inclination_error',
                    'K','K_error','t_periastron','t_periastron_error','a0','a0_error','nodeangle','nodeangle_error',
                    'arg_periastron','arg_periastron_error','parallax','parallax_error','pl_mass','pl_mass_error','st_mass']
        values_c = [getattr(self,i) for i in labels_c]
        self.df_campbell = pd.DataFrame(list(zip(values_c)),index=labels_c).T
        if estimate_rvprec:
            S = astropylib.starinfo.StarInfo('Gaia DR3 {}'.format(source_id))
            self.st_teff = S.st_teff
            self.st_vj_gaia = S.st_vj_gaia
            self.rv_prec = astropylib.rvprec.get_neid_rv_precision(self.st_teff,self.st_vj_gaia,exptime=900)
            self.K_sigma = self.K/self.rv_prec



    def find_source_index(self,source_id,verbose=True):
        source_id = str(source_id)
        df = self.df.source_id[self.df.source_id==source_id]
        print('Found {} inds'.format(len(df)))
        return df.index.values[0]

    def get_semi_amplitude_and_error(self,N=10000,plot=False):
        """
        Get semi-amplitudes and errors
        """
        #P = np.random.normal(self.period,self.period_error,N)
        P = utils.truncated_normal_samples(self.period,self.period_error,0,self.period*10,N)
        plx = np.random.normal(self.parallax,self.parallax_error,N)
        a0 = np.random.normal(self.a0,self.a0_error,N)
        e = utils.truncated_normal_samples(self.eccentricity,self.eccentricity_error,0,1,N)
        i = np.random.normal(self.inclination,self.inclination_error,N)
        cosi = np.cos(np.deg2rad(i))
        K = gaiafit.semi_amplitude_from_gaia(P,a0,plx,cosi,e)
        self.Ksamples = K

        if self.st_mass is not None:
            self.st_mass_samples = np.random.normal(self.st_mass,self.st_masserr,N)
            self.pl_mass_samples = utils.msini_from_rvs(self.Ksamples,self.st_mass_samples,P,e,i)#/317.81 # to jupiter masses
            self.pl_mass = np.percentile(self.pl_mass_samples,50)
            self.pl_mass_low = self.pl_mass - np.percentile(self.pl_mass_samples,16)
            self.pl_mass_up = np.percentile(self.pl_mass_samples,84) - self.pl_mass
            self.pl_mass_error = (self.pl_mass_up + self.pl_mass_low)/2
            print('M={:0.2f}-{:0.2f}+{:0.2f}M_jup'.format(self.pl_mass,self.pl_mass_low,self.pl_mass_up))

        if plot:
            corner.corner(np.array(list(zip(P,plx,a0,e,cosi,K))),show_titles=True,labels=['P','plx','a0','e','cosi','K']);

        Kmed = np.percentile(K,50)
        low = Kmed - np.percentile(K,16)
        up = np.percentile(K,84) - Kmed
        K_error = (low+up)/2
        print('K={:0.2f}-{:0.2f}+{:0.2f}m/s'.format(Kmed,low,up))
        return Kmed, K_error, up, low
        

    def run_sampler(self,nburn=1000,nit=1000,plot_corner=True,plot_zscore=True):
        """
        Use ABFGePTp parametrization
        """
        self.sampler = emcee.EnsembleSampler(nwalkers=self.nwalkers,
                                            ndim=self.ndim,
                                            log_prob_fn=log_prob_gaia,
                                            args=[self.means, self.CC])
        #p0 = np.random.rand(self.nwalkers,self.ndim)
        p0 = self.means + 1e-3 * np.random.randn(self.nwalkers, self.ndim)
        #print(p0)
        #return p0
        state = self.sampler.run_mcmc(p0, nburn)
        print('Finished burnin',)
        self.sampler.reset()
        print('Running main MCMC, it=',nit)
        self.sampler.run_mcmc(state, nit)
        
        samples = self.sampler.get_chain(flat=True)
        if plot_corner:
            print("Mean acceptance fraction: {0:.3f}".format(np.mean(self.sampler.acceptance_fraction)))
            corner.corner(samples,labels=self.labels,show_titles=True,truths=self.means);
            
        self.df_mean = self.get_mean_values_mcmc_posteriors()
        self.df_mean['error'] = (self.df_mean['minus'] + self.df_mean['plus'])/2.
        self.df_mean['zscore'] = (self.df_mean['medvals'].values - self.df_known['median'].values)/self.df_known.error.values

        if plot_zscore:
            self.plot_zscore_panel()

        return samples

    def plot_zscore_panel(self,ax=None,bx=None):
        if ax is None and bx is None:
            fig, (ax,bx) = plt.subplots(dpi=200,ncols=2,nrows=1,figsize=(6,2.5))
        ax.errorbar(range(len(self.df_known[0:4])),self.df_known['median'].values[0:4],self.df_known.error.values[0:4],
                    marker='o',lw=0,mew=0.5,capsize=4,elinewidth=0.5,label='Gaia',color='black')
        ax.errorbar(range(len(self.df_mean[0:4])),self.df_mean['medvals'].values[0:4],yerr=[self.df_mean.minus.values[0:4],self.df_mean.plus.values[0:4]],
                    marker='o',lw=0,mew=0.5,capsize=4,elinewidth=0.5,label='MCMC Sampled',color='crimson')

        bx.plot(range(len(self.df_mean)),self.df_mean['zscore'],marker='o',lw=0,color='crimson')


        ax.legend(loc='upper right',fontsize=6)
        ax.set_xticks([0,1,2,3])
        ax.set_xticklabels(['A','B','F','G'])

        bx.set_xticks(range(len(self.df_mean)))
        bx.set_xticklabels(self.labels)

        ax.set_xlabel('Thile-Innes coefficients')
        ax.set_ylabel('Milliarcsec')
        bx.set_ylabel('Z-score')

        bx.axhline(y=-1,color='red',ls='--',lw=0.5)
        bx.axhline(y=1,color='red',ls='--',lw=0.5)

        for xx in [ax,bx]:
            utils.ax_apply_settings(xx)
            
        fig.subplots_adjust(wspace=0.3)
    
    def get_mean_values_mcmc_posteriors(self,flatchain=None):
        """
        Get the mean values from the posteriors

            flatchain - if not passed, then will default using the full flatchain (will likely include burnin)

        EXAMPLE:
        """
        if flatchain is None:
            flatchain = self.sampler.flatchain
            print('No flatchain passed, defaulting to using full chains')
        df_list = [utils.get_mean_values_for_posterior(flatchain[:,i],label) for i,label in zip(range(len(self.labels)),self.labels)]
        df_list = pd.concat(df_list).reset_index(drop=True)
        df_list = df_list[['Labels','medvals','minus','plus','values']]
        return df_list

    def plot_expected_rv_orbit(self,bjd=None,rv=None,e_rv=None,ax=None,N=1000,tstart='2023-04-01',tstop='2024-01-01',
                               Nt=1000,plot_today=True,rv_offset=0.,title='',plot_het=True,het_start='2023-12-01',het_stop='2024-04-01',label='24-1 Trimester',savename=None,plot_neid_visibility=True):
        """
        Plot expected RV orbit
        """

        t = utils.iso2jd([tstart,tstop])
        t = np.linspace(t[0],t[1],Nt)

        if ax is None:
            fig, ax = plt.subplots(dpi=200)
        if bjd is not None:
            ax.errorbar(utils.jd2datetime(bjd),rv+rv_offset,e_rv,marker='o',lw=0,mew=0.5,capsize=4,color='black',elinewidth=0.5)
        rv_50 = gaiafit.get_rv_curve_peri(t,
                                          self.period,
                                          self.t_periastron+gaiafit.GAIA_TP_OFFSET,
                                          self.eccentricity,
                                          self.arg_periastron,
                                          self.K)
        P = np.random.normal(self.period,self.period_error,N)
        tp = np.random.normal(self.t_periastron+gaiafit.GAIA_TP_OFFSET,self.t_periastron_error,N)
        e = np.random.normal(self.eccentricity,self.eccentricity_error,N)
        w = np.random.normal(self.arg_periastron,self.arg_periastron_error,N)
        K = np.random.normal(self.K,self.K_error,N)
        
        rvs = []
        for i in range(N):
            r = gaiafit.get_rv_curve_peri(t,P[i],tp[i],e[i],w[i],K[i])
            rvs.append(r)
        rvs = np.array(rvs)
        rv_16 = np.percentile(rvs,16,axis=0)
        #rv_50 = np.percentile(rvs,84,axis=0)
        rv_84 = np.percentile(rvs,84,axis=0)
        ax.plot(utils.jd2datetime(t),rv_50,color='crimson',label='50% model')
        ax.fill_between(utils.jd2datetime(t),rv_16,rv_84,color='crimson',lw=0,alpha=0.1,label='$1\sigma$')
        ax.set_xlabel('Date [UT]',fontsize=14)
        ax.set_ylabel('RV [m/s]',fontsize=14)
        lab = 'Target={} ($M={:0.2f}\pm{:0.2f}M_\odot$)\n'.format(title,self.st_mass,self.st_masserr)
        lab += 'Source_id={}\n'.format(self.source_id)
        lab += '$P={:0.2f}\pm{:0.2f}$days\n'.format(self.period,self.period_error)
        lab += '$e={:0.2f}\pm{:0.2f}$\n'.format(self.eccentricity,self.eccentricity_error)
        lab += '$i={:0.1f}\pm{:0.1f}$deg\n'.format(self.inclination,self.inclination_error)
        lab += '$M_p={:0.1f}\pm{:0.1f}M_J$'.format(self.pl_mass,self.pl_mass_error)
        ax.set_title(lab)

        if plot_neid_visibility:
            df_vis = astropylib.gkastro.month_by_month_observability_for_targets(names=[self.source_id],
                                                                                 ra=[self.source._star.ra],
                                                                                 dec=[self.source._star.dec],
                                                                                 sitename='Kitt Peak National Observatory',
                                                                                 min_altitude=22,
                                                                                 max_altitude=90,
                                                                                 verbose=False)
            df_vis = df_vis.T
            df_vis['months'] = df_vis.index
            df_vis = df_vis.reset_index(drop=True)
            df_vis = df_vis[df_vis[0] == 1]

            for m in df_vis.months.values:
                m = int(m[1:])
                start = astropylib.gkastro.TIMES_MONTH[m-1][0].replace(' ','T')
                stop  = astropylib.gkastro.TIMES_MONTH[m-1][1].replace(' ','T')
                times = astroplan.time_grid_from_range([astropy.time.Time(start),astropy.time.Time(stop)],1*u.d).datetime
                
                for t in times:
                    ax.axvline(t,ymin=0,ymax=0.05,zorder=-100,color='green',lw=1,alpha=0.05)

                times = astroplan.time_grid_from_range([astropy.time.Time(start)-1*u.year,astropy.time.Time(stop)-1*u.year],1*u.d).datetime
                for t in times:
                    ax.axvline(t,ymin=0,ymax=0.05,zorder=-100,color='green',lw=1,alpha=0.05)

                times = astroplan.time_grid_from_range([astropy.time.Time(start)-2*u.year,astropy.time.Time(stop)-2*u.year],1*u.d).datetime
                for t in times:
                    ax.axvline(t,ymin=0,ymax=0.05,zorder=-100,color='green',lw=1,alpha=0.05)

                times = astroplan.time_grid_from_range([astropy.time.Time(start)-3*u.year,astropy.time.Time(stop)-3*u.year],1*u.d).datetime
                for t in times:
                    ax.axvline(t,ymin=0,ymax=0.05,zorder=-100,color='green',lw=1,alpha=0.05)

        if plot_het:
            t = utils.iso2jd([het_start,het_stop])
            YLIM = ax.get_ylim()
            ax.set_ylim(*YLIM)
            ax.fill_between(utils.jd2datetime(t),np.ones(len(t))*YLIM[0],np.ones(len(t))*YLIM[1],zorder=10,alpha=0.3,color='green',lw=0,label=label)
            ax.legend(loc='upper right')

            ra = self.source._star.ra
            dec = self.source._star.dec
            dd = astropylib.webscrape.yearly_visibility(ra,dec)
            dd_not_visible = dd[dd.visible==False]
            for i in range(len(dd_not_visible)):
                ax.axvline(dd_not_visible.datetime.values[i],zorder=-100,color='black',lw=1,alpha=0.1)
                ax.axvline(dd_not_visible.datetime.values[i]+pd.DateOffset(years= 1),zorder=-100,color='black',lw=1,alpha=0.1)
                ax.axvline(dd_not_visible.datetime.values[i]+pd.DateOffset(years= 2),zorder=-100,color='black',lw=1,alpha=0.1)


        if plot_today:
            today = datetime.now()
            ax.axvline(today,color='orange',ls='--',label='{}'.format(today.isoformat()[0:10]),zorder=100)

            ax.axvline(astropy.time.Time('2024-08-01').datetime,color='black',ls='--',label='2024-08-01',zorder=100)
            ax.axvline(astropy.time.Time('2025-02-01').datetime,color='black',ls='--',label='2025-02-01',zorder=100)
            
        ax.legend(loc='upper right')
        if savename is not None:
            fig.tight_layout()
            fig.savefig(savename,dpi=200)
            print('Saved to',savename)


class GaiaSource2(object):
    
    nwalkers = 50
    
    def __init__(self,filename,source_index=0):
        self.filename = filename
        self.df = pd.read_csv(filename)
        self.source_index = source_index
        self.source = NssSource(self.df, indice=self.source_index)
        self.covmat = self.source.covmat()
        self.C = self.covmat[['a_thiele_innes','b_thiele_innes','f_thiele_innes','g_thiele_innes','eccentricity','period','t_periastron']].iloc[5:,:]
        self.CC = self.C.values
        self.Campbell = self.source.campbell()
        self.ndim = self.CC.shape[0]
        self.a_thiele_innes = self.source._star['a_thiele_innes']
        self.a_thiele_innes_error = self.source._star['a_thiele_innes_error']
        self.b_thiele_innes = self.source._star['b_thiele_innes']
        self.b_thiele_innes_error = self.source._star['b_thiele_innes_error']
        self.f_thiele_innes = self.source._star['f_thiele_innes']
        self.f_thiele_innes_error = self.source._star['f_thiele_innes_error']
        self.g_thiele_innes = self.source._star['g_thiele_innes']
        self.g_thiele_innes_error = self.source._star['g_thiele_innes_error']
        self.eccentricity = self.source._star['eccentricity']
        self.eccentricity_error = self.source._star['eccentricity_error']
        self.period = self.source._star['period']
        self.period_error = self.source._star['period_error']
        self.t_periastron = self.source._star['t_periastron']
        self.t_periastron_error = self.source._star['t_periastron_error']
        self.means = np.array([self.a_thiele_innes,
                               self.b_thiele_innes,
                               self.f_thiele_innes,
                               self.g_thiele_innes,
                               self.eccentricity,
                               self.period,
                               self.t_periastron])
        self.errors = np.array([self.a_thiele_innes_error,
                                self.b_thiele_innes_error,
                                self.f_thiele_innes_error,
                                self.g_thiele_innes_error,
                                self.eccentricity_error,
                                self.period_error,
                                self.t_periastron_error])
        self.labels = ['A','B','F','G','e','P','t_peri']
        self.df_known = pd.DataFrame(list(zip(self.labels,self.means,self.errors)),columns=['label','median','error'])

    def run_sampler(self,nit=1000,plot_corner=True,plot_zscore=True):
        """
        Use ABFGePTp parametrization
        """
        self.sampler = emcee.EnsembleSampler(nwalkers=self.nwalkers,
                                        ndim=self.ndim,
                                        log_prob_fn=log_prob_gaia,
                                        args=[self.means, self.CC])
        p0 = np.random.rand(self.nwalkers,self.ndim)
        state = self.sampler.run_mcmc(p0, 1000)
        print('Finished burnin',)
        self.sampler.reset()
        print('Running main MCMC, it=',nit)
        self.sampler.run_mcmc(state, nit)
        
        samples = self.sampler.get_chain(flat=True)
        if plot_corner:
            print("Mean acceptance fraction: {0:.3f}".format(np.mean(self.sampler.acceptance_fraction)))
            corner.corner(samples,labels=self.labels,show_titles=True,truths=self.means);
            
        self.df_mean = self.get_mean_values_mcmc_posteriors()
        self.df_mean['error'] = (self.df_mean['minus'] + self.df_mean['plus'])/2.
        self.df_mean['zscore'] = (self.df_mean['medvals'].values - self.df_known['median'].values)/self.df_known.error.values

        if plot_zscore:
            self.plot_zscore_panel()

        return samples

    def plot_zscore_panel(self,ax=None,bx=None):
        if ax is None and bx is None:
            fig, (ax,bx) = plt.subplots(dpi=200,ncols=2,nrows=1,figsize=(6,2.5))
        ax.errorbar(range(len(self.df_known[0:4])),self.df_known['median'].values[0:4],self.df_known.error.values[0:4],
                    marker='o',lw=0,mew=0.5,capsize=4,elinewidth=0.5,label='Gaia',color='black')
        ax.errorbar(range(len(self.df_mean[0:4])),self.df_mean['medvals'].values[0:4],yerr=[self.df_mean.minus.values[0:4],self.df_mean.plus.values[0:4]],
                    marker='o',lw=0,mew=0.5,capsize=4,elinewidth=0.5,label='MCMC Sampled',color='crimson')

        bx.plot(range(len(self.df_mean)),self.df_mean['zscore'],marker='o',lw=0,color='crimson')


        ax.legend(loc='upper right',fontsize=6)
        ax.set_xticks([0,1,2,3])
        ax.set_xticklabels(['A','B','F','G'])

        bx.set_xticks(range(len(self.df_mean)))
        bx.set_xticklabels(self.labels)

        ax.set_xlabel('Thile-Innes coefficients')
        ax.set_ylabel('Milliarcsec')
        bx.set_ylabel('Z-score')

        bx.axhline(y=-1,color='red',ls='--',lw=0.5)
        bx.axhline(y=1,color='red',ls='--',lw=0.5)

        for xx in [ax,bx]:
            utils.ax_apply_settings(xx)
            
        fig.subplots_adjust(wspace=0.3)
    
    def get_mean_values_mcmc_posteriors(self,flatchain=None):
        """
        Get the mean values from the posteriors

            flatchain - if not passed, then will default using the full flatchain (will likely include burnin)

        EXAMPLE:
        """
        if flatchain is None:
            flatchain = self.sampler.flatchain
            print('No flatchain passed, defaulting to using full chains')
        df_list = [utils.get_mean_values_for_posterior(flatchain[:,i],label) for i,label in zip(range(len(self.labels)),self.labels)]
        df_list = pd.concat(df_list).reset_index(drop=True)
        df_list = df_list[['Labels','medvals','minus','plus','values']]
        return df_list
