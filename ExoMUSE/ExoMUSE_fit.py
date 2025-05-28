import numpy as np
from multiprocessing import Pool
import contextlib
import dynesty
import pandas as pd
import pyde
import pyde.de
import matplotlib.pyplot as plt
import emcee
import batman
import astropy.constants as aconst
import radvel
from priors import PriorSet, UP, NP, JP, FP
from likelihood import ll_normal_ev_py
import stats_help
import ExoMUSE_utils
import mcmc_help
from nsstools import NssSource
import pyde
import pyde.de
import corner
from scipy.stats import binned_statistic
from pytransit.utils import de

# GAIA_TP_OFFSET = 2457389.0
GAIA_TP_OFFSET = 0.0

def distance_in_pc_from_parallax(plx):
    """
    Parallax in mas
    """
    return 1000./plx

def plot_cov(C,labels,cmap='viridis'):
    fig, ax = plt.subplots(dpi=150)#,figsize=(5,5))
    ax.imshow(C,cmap=cmap)
    ax.set_xticklabels(['']+labels)
    ax.set_yticklabels(['']+labels)
    cx, cb = ExoMUSE_utils.ax_add_colorbar(ax,p=C,cmap=cmap)
    cx.set_ylabel('Value')

def semi_amplitude_from_gaia(P,a0,plx,cosi,e):
    """
    INPUT:
        P - period in days
        a0 - UNIT?
        plx - parallax in mas
        cosi - cosi
        e - eccentricity
        
    OUTPUT:
        K - m/s
    
    NOTES:
        See Equation 15 from Winn 2022
        
    # EXAMPLE:
        # HD_81040
        semi_amplitude_from_gaia(P=850.837877,a0=0.387459,plx=29.063471700504785,cosi=np.cos(np.deg2rad(107.780206)),e=0.369314)
        # 174m/s
        
        # BD-17_0063
        semi_amplitude_from_gaia(P=648.915120,a0=0.217464,plx=28.972342936808044,cosi=np.cos(np.deg2rad(80.223472)),e=0.275404)
    """
    P = P*86400. # days to s
    d = distance_in_pc_from_parallax(plx)*aconst.pc.value
    a0 = np.deg2rad((a0/(1000.*3600.)))
    K = ((np.pi*2)/P)*((((a0*d))*np.sqrt(1-cosi**2.))/(np.sqrt(1.-e**2.)))
    return K

def a0_from_m_M_P_epsilon(M,m,P,eps,plx):
    """
    INPUT:
        M - mass of star in Solar masses
        m - mass of planet in Jupiter masses
        P - period in days
        eps - light fraction
        plx - in mas
    
    OUTPUT:
        a0 - in mas
        
    EXAMPLE:
        # 
        a0_from_m_M_P_epsilon(0.74,5.1*317.82,P=648.915120,eps=0,plx=28.972342936808044)
        # 0.2517; Gaia a0 is =0.217
        
        # HD_81040
        a0_from_m_M_P_epsilon(0.96,6.87*317.82,P=1001.7,eps=0,plx=29.063471700504785)
        # 0.381; Gais a0 is =0.387
    """
    M = M*aconst.M_sun.value
    #m = m*aconst.M_earth.value
    # NOTE
    #m = m*aconst.M_sun.value
    m = m*aconst.M_jup.value
    P = P*86400.
    G = aconst.G.value
    a0_omega = (G*(M+m))**(1./3.)*((P/(2*np.pi))**(2./3.))*((m)/(M+m) - (eps)/(1+eps))
    a0_omega = a0_omega/(distance_in_pc_from_parallax(plx)*aconst.pc.value)
    a0_omega = np.rad2deg(a0_omega)*3600.*1000.
    return a0_omega

def thiele_innes_a_from_campell(a0,omega,Omega,cosi):
    omega = np.deg2rad(omega)
    Omega = np.deg2rad(Omega)
    return a0*(np.cos(omega)*np.cos(Omega) - np.sin(omega)*np.sin(Omega)*cosi)

def thiele_innes_b_from_campell(a0,omega,Omega,cosi):
    omega = np.deg2rad(omega)
    Omega = np.deg2rad(Omega)
    return a0*(np.cos(omega)*np.sin(Omega) + np.sin(omega)*np.cos(Omega)*cosi)
        
def thiele_innes_f_from_campell(a0,omega,Omega,cosi):
    omega = np.deg2rad(omega)
    Omega = np.deg2rad(Omega)
    return (-1)*a0*(np.sin(omega)*np.cos(Omega) + np.cos(omega)*np.sin(Omega)*cosi)

def thiele_innes_g_from_campell(a0,omega,Omega,cosi):
    omega = np.deg2rad(omega)
    Omega = np.deg2rad(Omega)
    return (-1)*a0*(np.sin(omega)*np.sin(Omega) + np.cos(omega)*np.cos(Omega)*cosi)


class LPFunctionRVGaia(object):
    """
    Log-Likelihood function class for RV only

    NOTES:
        Based on hpprvi's class, see: https://github.com/hpparvi/exo_tutorials
    """
    def __init__(self,x,y,yerr,file_priors,filename='',source_id=None,source_index=None,df=None,verbose=True):
        """
        INPUT:
            file_priors - prior file name
            filename - filename that has Gaia information
            source_id - Gaia source id
            source_index - source index
            df - alternative way to pass the Gaia information as a pandas df
        """
        self.data= {"x"   : x,
                    "y"   : y,
                    "error"  : yerr}
        
        if filename is not '':
            self.filename = filename
            self.df = pd.read_csv(filename,dtype={'source_id': str})
        else:
            #self.df = read_gsheet(gsheet,gsheet_name)
            self.df = df
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
        
        #self.filename = filename
        #self.df = pd.read_csv(filename)
        #self.source_index = source_index
        #self.source = NssSource(self.df, indice=self.source_index)
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
        
        # Setting priors
        self.ps_all = priorset_from_file(file_priors) # all priors
        self.ps_fixed = PriorSet(np.array(self.ps_all.priors)[np.array(self.ps_all.fixed)]) # fixed priorset
        self.ps_vary  = PriorSet(np.array(self.ps_all.priors)[~np.array(self.ps_all.fixed)]) # varying priorset
        self.ps_fixed_dict = {key: val for key, val in zip(self.ps_fixed.labels,self.ps_fixed.args1)}
        print('Reading in priorfile from {}'.format(file_priors))
        print(self.ps_all.df)
        print(self.means)

    def find_source_index(self,source_id,verbose=True):
        source_id = str(source_id)
        df = self.df.source_id[self.df.source_id==source_id]
        print('Found {} inds'.format(len(df)))
        return df.index.values[0]

    def get_jump_parameter_index(self,lab):
        """
        Get the index of a given label
        """
        return np.where(np.array(self.ps_vary.labels)==lab)[0][0]

    def get_jump_parameter_value(self,pv,lab):
        """
        Get the current value in the argument list 'pv' that has label 'lab'
        """
        # First check if we are actually varying it
        if lab in self.ps_vary.labels:
            return pv[self.get_jump_parameter_index(lab)]
        else:
            # We are not varying it
            return self.ps_fixed_dict[lab]
        
    def compute_rv_model(self,pv,times=None):
        """
        Compute the RV model

        INPUT:
            pv    - a list of parameters (only parameters that are being varied)
            times - times (optional), array of timestamps

        OUTPUT:
            rv - the rv model evaluated at 'times' if supplied, otherwise
                      defaults to original data timestamps
        """
        if times is None:
            times = self.data["x"]
        Tp      = self.get_jump_parameter_value(pv,'tp_p1') + GAIA_TP_OFFSET
        P       = self.get_jump_parameter_value(pv,'P_p1') # done
        gamma   = self.get_jump_parameter_value(pv,'gamma') # done
        cosi    = self.get_jump_parameter_value(pv,'cosi') # done
        e       = self.get_jump_parameter_value(pv,'ecc_p1') # done
        w       = self.get_jump_parameter_value(pv,'omega_p1') # done
        M       = self.get_jump_parameter_value(pv,'mstar') # done
        m       = self.get_jump_parameter_value(pv,'mp') # done
        eps     = self.get_jump_parameter_value(pv,'eps') # done
        plx     = self.get_jump_parameter_value(pv,'plx') # done

        a0      = a0_from_m_M_P_epsilon(M,m,P,eps,plx)
        K       = semi_amplitude_from_gaia(P,a0,plx,cosi,e)

        self.rv = get_rv_curve_peri(times,P=P,tp=Tp,e=e,omega=w,K=K)+gamma
        return self.rv

    def compute_polynomial_model(self,pv,times=None):
        """
        Compute the polynomial model.  Note that if gammadot and gammadotdot
        are not specified in the priors file, they both default to zero.

        INPUT:
            pv    - a list of parameters (only parameters that are being varied)
            times - times (optional), array of timestamps

        OUTPUT:
            poly - the polynomial model evaluated at 'times' if supplied,
                   otherwise defaults to original data timestamps
        """
        if times is None:
            times = self.data["x"]

        #T0 = self.get_jump_parameter_value(pv,'t0_p1')
        # Use Mean of data instead
        T0 = (self.data['x'][0] + self.data['x'][-1])/2.
        try:
            gammadot = self.get_jump_parameter_value(pv,'gammadot')
        except KeyError as e:
            gammadot = 0
        try:
            gammadotdot = self.get_jump_parameter_value(pv,'gammadotdot')
        except KeyError as e:
            gammadotdot = 0

        self.poly = (
            gammadot * (times - T0) +
            gammadotdot * (times - T0)**2
        )
        return self.poly

    def compute_gaia_model(self,pv):
        e     = self.get_jump_parameter_value(pv,'ecc_p1')
        P     = self.get_jump_parameter_value(pv,'P_p1')
        tp    = self.get_jump_parameter_value(pv,'tp_p1')
        cosi  = self.get_jump_parameter_value(pv,'cosi')
        Omega = self.get_jump_parameter_value(pv,'Omega')
        w     = self.get_jump_parameter_value(pv,'omega_p1')
        M     = self.get_jump_parameter_value(pv,'mstar') # done
        m     = self.get_jump_parameter_value(pv,'mp') # done
        eps   = self.get_jump_parameter_value(pv,'eps') # done
        plx   = self.get_jump_parameter_value(pv,'plx') # done

        a0      = a0_from_m_M_P_epsilon(M,m,P,eps,plx)
        a       = thiele_innes_a_from_campell(a0,w,Omega,cosi)
        b       = thiele_innes_b_from_campell(a0,w,Omega,cosi)
        f       = thiele_innes_f_from_campell(a0,w,Omega,cosi)
        g       = thiele_innes_g_from_campell(a0,w,Omega,cosi)

        pv = np.array([a,b,f,g,e,P,tp])
        Theta = pv - self.means # deviation vector
        log_prob_gaia = -0.5 * np.dot(Theta, np.linalg.solve(self.CC, Theta))
        return log_prob_gaia

    def compute_total_model(self,pv,times=None):
        """
        Computes the full RM model (including RM and RV and CB)

        INPUT:
            pv    - a list of parameters (only parameters that are being varied)
            times - times (optional), array of timestamps

        OUTPUT:
            rm - the rm model evaluated at 'times' if supplied, otherwise
                      defaults to original data timestamps

        NOTES:
            see compute_rm_model(), compute_rv_model(),
            compute_polynomial_model()
        """
        return self.compute_rv_model(pv,times=times) + self.compute_polynomial_model(pv,times=times) 

        
    def __call__(self,pv):
        """
        Return the log likelihood

        INPUT:
            pv - the input list of varying parameters
        """
        if any(pv < self.ps_vary.pmins) or any(pv>self.ps_vary.pmaxs):
            return -np.inf
                
        ###############
        # PRIORS
        ###############
        log_of_priors = self.ps_vary.c_log_prior(pv)

        ###############
        # RV
        ###############
        y_data = self.data['y']
        y_model = self.compute_total_model(pv)
        jitter = self.get_jump_parameter_value(pv,'sigma_rv')
        error = np.sqrt(self.data['error']**2.+jitter**2.)
        log_of_rv_model  = ll_normal_ev_py(y_data, y_model, error)
        
        ###############
        # Gaia
        ###############
        log_of_gaia_model = self.compute_gaia_model(pv)

        # FINAL COMBINED
        log_ln = log_of_priors + log_of_rv_model + log_of_gaia_model
        return log_ln



class LPFunctionRVGaia2(object):
    """
    Log-Likelihood function class for 2 RV instruments

    NOTES:
        Based on hpprvi's class, see: https://github.com/hpparvi/exo_tutorials
    """
    def __init__(self,x1,y1,yerr1,x2,y2,yerr2,file_priors,filename='',source_id=None,source_index=None,df=None,verbose=True):
        """
        INPUT:
            file_priors - prior file name
            filename - filename that has Gaia information
            source_id - Gaia source id
            source_index - source index
            df - alternative way to pass the Gaia information as a pandas df
        """
        self.data= {"x1"   : x1,
                    "y1"   : y1,
                    "error1"  : yerr1,
                    "x2"   : x2,
                    "y2"   : y2,
                    "error2"  : yerr2
                    }
        
        if filename is not '':
            self.filename = filename
            self.df = pd.read_csv(filename,dtype={'source_id': str})
        else:
            #self.df = read_gsheet(gsheet,gsheet_name)
            self.df = df
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
        
        #self.filename = filename
        #self.df = pd.read_csv(filename)
        #self.source_index = source_index
        #self.source = NssSource(self.df, indice=self.source_index)
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
        
        # Setting priors
        self.ps_all = priorset_from_file(file_priors) # all priors
        self.ps_fixed = PriorSet(np.array(self.ps_all.priors)[np.array(self.ps_all.fixed)]) # fixed priorset
        self.ps_vary  = PriorSet(np.array(self.ps_all.priors)[~np.array(self.ps_all.fixed)]) # varying priorset
        self.ps_fixed_dict = {key: val for key, val in zip(self.ps_fixed.labels,self.ps_fixed.args1)}
        print('Reading in priorfile from {}'.format(file_priors))
        print(self.ps_all.df)
        print(self.means)

    def find_source_index(self,source_id,verbose=True):
        source_id = str(source_id)
        df = self.df.source_id[self.df.source_id==source_id]
        print('Found {} inds'.format(len(df)))
        return df.index.values[0]

    def get_jump_parameter_index(self,lab):
        """
        Get the index of a given label
        """
        return np.where(np.array(self.ps_vary.labels)==lab)[0][0]

    def get_jump_parameter_value(self,pv,lab):
        """
        Get the current value in the argument list 'pv' that has label 'lab'
        """
        # First check if we are actually varying it
        if lab in self.ps_vary.labels:
            return pv[self.get_jump_parameter_index(lab)]
        else:
            # We are not varying it
            return self.ps_fixed_dict[lab]
        
    def compute_rv_model(self,pv,times1=None,times2=None):
        """
        Compute the RV model

        INPUT:
            pv    - a list of parameters (only parameters that are being varied)
            times - times (optional), array of timestamps

        OUTPUT:
            rv - the rv model evaluated at 'times' if supplied, otherwise
                      defaults to original data timestamps
        """
        if times1 is None:
            times1 = self.data["x1"]
        if times2 is None:
            times2 = self.data["x2"]
        Tp      = self.get_jump_parameter_value(pv,'tp_p1') + GAIA_TP_OFFSET
        P       = self.get_jump_parameter_value(pv,'P_p1') # done
        gamma1  = self.get_jump_parameter_value(pv,'gamma1') # done
        gamma2  = self.get_jump_parameter_value(pv,'gamma2') # done
        cosi    = self.get_jump_parameter_value(pv,'cosi') # done
        e       = self.get_jump_parameter_value(pv,'ecc_p1') # done
        w       = self.get_jump_parameter_value(pv,'omega_p1') # done
        M       = self.get_jump_parameter_value(pv,'mstar') # done
        m       = self.get_jump_parameter_value(pv,'mp') # done
        eps     = self.get_jump_parameter_value(pv,'eps') # done
        plx     = self.get_jump_parameter_value(pv,'plx') # done

        a0      = a0_from_m_M_P_epsilon(M,m,P,eps,plx)
        K       = semi_amplitude_from_gaia(P,a0,plx,cosi,e)

        self.rv1 = get_rv_curve_peri(times1,P=P,tp=Tp,e=e,omega=w,K=K)+gamma1
        self.rv2 = get_rv_curve_peri(times2,P=P,tp=Tp,e=e,omega=w,K=K)+gamma2
        return self.rv1, self.rv2

    def compute_polynomial_model(self,pv,times1=None,times2=None):
        """
        Compute the polynomial model.  Note that if gammadot and gammadotdot
        are not specified in the priors file, they both default to zero.

        INPUT:
            pv    - a list of parameters (only parameters that are being varied)
            times - times (optional), array of timestamps

        OUTPUT:
            poly - the polynomial model evaluated at 'times' if supplied,
                   otherwise defaults to original data timestamps
        """
        print('poly implemented')
        return 0.

    def compute_gaia_model(self,pv):
        e     = self.get_jump_parameter_value(pv,'ecc_p1')
        P     = self.get_jump_parameter_value(pv,'P_p1')
        tp    = self.get_jump_parameter_value(pv,'tp_p1')
        cosi  = self.get_jump_parameter_value(pv,'cosi')
        Omega = self.get_jump_parameter_value(pv,'Omega')
        w     = self.get_jump_parameter_value(pv,'omega_p1')
        M     = self.get_jump_parameter_value(pv,'mstar') # done
        m     = self.get_jump_parameter_value(pv,'mp') # done
        eps   = self.get_jump_parameter_value(pv,'eps') # done
        plx   = self.get_jump_parameter_value(pv,'plx') # done

        a0      = a0_from_m_M_P_epsilon(M,m,P,eps,plx)
        a       = thiele_innes_a_from_campell(a0,w,Omega,cosi)
        b       = thiele_innes_b_from_campell(a0,w,Omega,cosi)
        f       = thiele_innes_f_from_campell(a0,w,Omega,cosi)
        g       = thiele_innes_g_from_campell(a0,w,Omega,cosi)

        pv = np.array([a,b,f,g,e,P,tp])
        Theta = pv - self.means # deviation vector
        log_prob_gaia = -0.5 * np.dot(Theta, np.linalg.solve(self.CC, Theta))
        return log_prob_gaia

    def compute_total_model(self,pv,times1=None,times2=None):
        """
        Computes the full RM model (including RM and RV and CB)

        INPUT:
            pv    - a list of parameters (only parameters that are being varied)
            times - times (optional), array of timestamps

        OUTPUT:
            rm - the rm model evaluated at 'times' if supplied, otherwise
                      defaults to original data timestamps

        NOTES:
            see compute_rm_model(), compute_rv_model(),
            compute_polynomial_model()
        """
        return self.compute_rv_model(pv,times1=times1,times2=times2) #+ self.compute_polynomial_model(pv,times=times) 

        
    def __call__(self,pv):
        """
        Return the log likelihood

        INPUT:
            pv - the input list of varying parameters
        """
        if any(pv < self.ps_vary.pmins) or any(pv>self.ps_vary.pmaxs):
            return -np.inf
                
        ###############
        # PRIORS
        ###############
        log_of_priors = self.ps_vary.c_log_prior(pv)

        ###############
        # RV
        ###############
        y_data1 = self.data['y1']
        y_data2 = self.data['y2']
        y_model1, y_model2 = self.compute_total_model(pv)
        jitter1 = self.get_jump_parameter_value(pv,'sigma_rv1')
        jitter2 = self.get_jump_parameter_value(pv,'sigma_rv2')
        error1 = np.sqrt(self.data['error1']**2.+jitter1**2.)
        error2 = np.sqrt(self.data['error2']**2.+jitter2**2.)
        log_of_rv_model1  = ll_normal_ev_py(y_data1, y_model1, error1)
        log_of_rv_model2  = ll_normal_ev_py(y_data2, y_model2, error2)
        log_of_rv_model = log_of_rv_model1 + log_of_rv_model2
        
        ###############
        # Gaia
        ###############
        log_of_gaia_model = self.compute_gaia_model(pv)

        # FINAL COMBINED
        log_ln = log_of_priors + log_of_rv_model + log_of_gaia_model
        return log_ln


class LPFunctionRVGaia_withInflationFactor(object):
    """
    Log-Likelihood function class for Gaia+RV with 2 custom RV instruments. Ideally this could be coded more comprehensively to permit N number of input RV instruments

    NOTES:
        Based on hpprvi's class, see: https://github.com/hpparvi/exo_tutorials
    """
    def __init__(self,x1,y1,yerr1,instrument_name1,x2,y2,yerr2,instrument_name2,periastron_offset,filename,source_index,file_priors):
        """
        INPUT:
            x1 - time values in BJD for RV instrument 1
            y1 - y values in m/s for RV instrument 1
            yerr1 - yerr values in m/s for RV instrument 1
            instrument_name1 - string of the instrument name that distinguishes the 2 sets of RVs
            x2 - time values in BJD for RV instrument 2
            y2 - y values in m/s for RV instrument 2
            yerr2 - yerr values in m/s for RV instrument 2
            instrument_name2 - string of the instrument name that distinguishes the 2 sets of RVs
            periastron_offset - custom input for the periastron offset time to be used
            filename - where all of the Gaia information is stored
            source_index - what index of filename where the target of interest data is located
            file_priors - prior file name
        """
        self.data1= {"x"   : x1,
                    "y"   : y1,
                    "error"  : yerr1,
                    "instrument_name": instrument_name1}
        self.data2= {"x"   : x2,
                    "y"   : y2,
                    "error"  : yerr2,
                    "instrument_name": instrument_name2}
        
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
                                self.t_periastron_error]) #* overall_error_multiplier
        self.labels = ['A','B','F','G','e','P','t_peri']
        self.df_known = pd.DataFrame(list(zip(self.labels,self.means,self.errors)),columns=['label','median','error'])
        self.TP_OFFSET = periastron_offset
        
        # Setting priors
        self.ps_all = priorset_from_file(file_priors) # all priors
        self.ps_fixed = PriorSet(np.array(self.ps_all.priors)[np.array(self.ps_all.fixed)]) # fixed priorset
        self.ps_vary  = PriorSet(np.array(self.ps_all.priors)[~np.array(self.ps_all.fixed)]) # varying priorset
        self.ps_fixed_dict = {key: val for key, val in zip(self.ps_fixed.labels,self.ps_fixed.args1)}
        print('Reading in priorfile from {}'.format(file_priors))
        print(self.ps_all.df)
        print(self.means)

    def get_jump_parameter_index(self,lab):
        """
        Get the index of a given label
        """
        return np.where(np.array(self.ps_vary.labels)==lab)[0][0]

    def get_jump_parameter_value(self,pv,lab):
        """
        Get the current value in the argument list 'pv' that has label 'lab'
        """
        # First check if we are actually varying it
        if lab in self.ps_vary.labels:
            return pv[self.get_jump_parameter_index(lab)]
        else:
            # We are not varying it
            return self.ps_fixed_dict[lab]
        
    def compute_rv_model(self,pv,instrument,times=None):
        """
        Compute the RV model

        INPUT:
            pv    - a list of parameters (only parameters that are being varied)
            times - times (optional), array of timestamps

        OUTPUT:
            rv - the rv model evaluated at 'times' if supplied, otherwise
                      defaults to original data timestamps
        """
        if times is None:
            if instrument == self.data1["instrument_name"]:
                times = self.data1["x"]
            else:
                times = self.data2["x"]

        Tp = self.get_jump_parameter_value(pv,'tp_p1') + self.TP_OFFSET
        P = self.get_jump_parameter_value(pv,'P_p1') # done
        cosi = self.get_jump_parameter_value(pv,'cosi') # done
        e = self.get_jump_parameter_value(pv,'ecc_p1') # done
        w = self.get_jump_parameter_value(pv,'omega_p1') # done
        M = self.get_jump_parameter_value(pv,'mstar') # done
        m = self.get_jump_parameter_value(pv,'mp') # done # to update to log just do 10**() and change prior to uniform 0,2
        eps = self.get_jump_parameter_value(pv,'eps') # done
        plx = self.get_jump_parameter_value(pv,'plx') # done
        a0 = a0_from_m_M_P_epsilon(M,m,P,eps,plx)
        K = semi_amplitude_from_gaia(P,a0,plx,cosi,e)

        if instrument == self.data1["instrument_name"]:
            gamma1  = self.get_jump_parameter_value(pv,'gamma1') # done
            self.rv = get_rv_curve_peri(times,P=P,tp=Tp,e=e,omega=w,K=K)+gamma1

        else:
            gamma2  = self.get_jump_parameter_value(pv,'gamma2') # done
            self.rv = get_rv_curve_peri(times,P=P,tp=Tp,e=e,omega=w,K=K)+gamma2
        
        return self.rv

    def compute_polynomial_model(self,pv,instrument,times=None):
        """
        Compute the polynomial model.  Note that if gammadot and gammadotdot
        are not specified in the priors file, they both default to zero.

        INPUT:
            pv    - a list of parameters (only parameters that are being varied)
            times - times (optional), array of timestamps

        OUTPUT:
            poly - the polynomial model evaluated at 'times' if supplied,
                   otherwise defaults to original data timestamps
        """
        if times is None:
            if instrument == self.data1["instrument_name"]:
                times = self.data1["x"]
            else:
                times = self.data2["x"]

        #T0 = self.get_jump_parameter_value(pv,'t0_p1')
        # Use Mean of data instead
        T0 = 0
        if instrument == self.data1["instrument_name"]:
            T0 = (self.data1['x'][0] + self.data1['x'][-1])/2.
        else:
            T0 = (self.data2['x'][0] + self.data2['x'][-1])/2.

        #T0 = (data['x'][0] + data['x'][-1])/2.
        try:
            gammadot = self.get_jump_parameter_value(pv,'gammadot')
        except KeyError as e:
            gammadot = 0
        try:
            gammadotdot = self.get_jump_parameter_value(pv,'gammadotdot')
        except KeyError as e:
            gammadotdot = 0

        self.poly = (
            gammadot * (times - T0) +
            gammadotdot * (times - T0)**2
        )
        return self.poly

    def compute_gaia_model(self,pv):
        e     = self.get_jump_parameter_value(pv,'ecc_p1')
        P     = self.get_jump_parameter_value(pv,'P_p1')
        tp    = self.get_jump_parameter_value(pv,'tp_p1')
        cosi  = self.get_jump_parameter_value(pv,'cosi')
        Omega = self.get_jump_parameter_value(pv,'Omega')
        w     = self.get_jump_parameter_value(pv,'omega_p1')
        M     = self.get_jump_parameter_value(pv,'mstar') # done
        m     = self.get_jump_parameter_value(pv,'mp') # done ## to update to log just do 10**() and change prior to uniform 0,2
        eps   = self.get_jump_parameter_value(pv,'eps') # done
        plx   = self.get_jump_parameter_value(pv,'plx') # done

        a0      = a0_from_m_M_P_epsilon(M,m,P,eps,plx)
        a       = thiele_innes_a_from_campell(a0,w,Omega,cosi)
        b       = thiele_innes_b_from_campell(a0,w,Omega,cosi)
        f       = thiele_innes_f_from_campell(a0,w,Omega,cosi)
        g       = thiele_innes_g_from_campell(a0,w,Omega,cosi)

        inflation_factor = self.get_jump_parameter_value(pv,'inf_fact') # to update to log just do 10**() and change prior to uniform 0,2

        pv = np.array([a,b,f,g,e,P,tp])
        Theta = pv - self.means # deviation vector
        
        # based on the likelihood function from Winn 2022 (Joint Constraints..) and broken up into 3 parts for clarity
        # First part
        first_factor = -4.0 * np.log(2*np.pi)

        # incorporate the square of the inflation factor since we want to inflate the standard deviation linearly by the inflation factor but we are applying it to a matrix of variances/multiplied standard deviations
        CC_use = self.CC * (inflation_factor**2)

        # Second part
        detCC = np.linalg.det(CC_use) # was ...det(self.CC)
        second_factor = -0.5*np.log(np.absolute(detCC))
        #print(second_factor)

        # third part
        third_factor = -0.5 * np.dot(Theta, np.linalg.solve(CC_use, Theta)) # was ...solve(self.CC, Theta)

        # put it all together
        return_value = first_factor + second_factor + third_factor

        return return_value

    def compute_total_model(self,pv,instrument,times=None):
        """
        Computes the full RM model (including RM and RV and CB)

        INPUT:
            pv    - a list of parameters (only parameters that are being varied)
            times - times (optional), array of timestamps

        OUTPUT:
            rm - the rm model evaluated at 'times' if supplied, otherwise
                      defaults to original data timestamps

        NOTES:
            see compute_rm_model(), compute_rv_model(),
            compute_polynomial_model()
        """
        return self.compute_rv_model(pv,instrument,times=times) #+ self.compute_polynomial_model(pv,instrument,times=times) 

        
    def __call__(self,pv):
        """
        Return the log likelihood

        INPUT:
            pv - the input list of varying parameters
        """
        if any(pv < self.ps_vary.pmins) or any(pv>self.ps_vary.pmaxs):
            return -np.inf
                
        ###############
        # PRIORS
        ###############
        log_of_priors = self.ps_vary.c_log_prior(pv)

        ###############
        # RV - Instrument 1
        ###############
        data_1 = self.data1['y']
        model_1 = self.compute_total_model(pv,self.data1['instrument_name'])
        jitter_1 = self.get_jump_parameter_value(pv,'sigma_rv1')
        error_1 = np.sqrt(self.data1['error']**2.+jitter_1**2.)
        log_of_rv_model_1  = ll_normal_ev_py(data_1, model_1, error_1)

        ###############
        # RV - Instrument 2
        ###############
        data_2 = self.data2['y']
        model_2 = self.compute_total_model(pv,self.data2['instrument_name'])
        jitter_2 = self.get_jump_parameter_value(pv,'sigma_rv2')
        error_2 = np.sqrt(self.data2['error']**2.+jitter_2**2.)
        log_of_rv_model_2  = ll_normal_ev_py(data_2, model_2, error_2)
        
        ###############
        # Gaia
        ###############
        log_of_gaia_model = self.compute_gaia_model(pv)

        # FINAL COMBINED
        log_ln = log_of_priors + log_of_rv_model_1 + log_of_rv_model_2 + log_of_gaia_model
        return log_ln


class ExoMUSEfitRVGaia(object):
    """
    A class that does Gaia+RV fitting.

    NOTES:
        - Needs to have LPFunction defined
    """
    def __init__(self,LPFunction):
        self.lpf = LPFunction

    def minimize_PyDE(self,npop=100,de_iter=200,mc_iter=1000,mcmc=True,threads=8,maximize=True,plot_priors=True,
                      sample_ball=False,k=None,n=None,plot_corner=True,plot_zscore=True):
        centers = np.array(self.lpf.ps_vary.centers)
        print("Running PyDE Optimizer")
        self.de = pyde.de.DiffEvol(self.lpf, self.lpf.ps_vary.bounds, npop, maximize=maximize) # we want to maximize the likelihood
        self.min_pv, self.min_pv_lnval = self.de.optimize(ngen=de_iter)
        print("Optimized using PyDE")
        print("Final parameters:")
        self.print_param_diagnostics(self.min_pv)
        #if plot_priors:
        #    self.lpf.ps_vary.plot_all(figsize=(6,4),pv=self.min_pv)
        print("LogPost value:",-1*self.min_pv_lnval)
        self.lnl_max  = -1*self.min_pv_lnval-self.lpf.ps_vary.c_log_prior(self.min_pv)
        print("LnL value:",self.lnl_max)
        print("Log priors",self.lpf.ps_vary.c_log_prior(self.min_pv))
        #if k is not None and n is not None:
        #    print("BIC:",stats_help.bic_from_likelihood(self.lnl_max,k,n))
        #    print("AIC:",stats_help.aic(k,self.lnl_max))
        if mcmc:
            print("Running MCMC")
            self.sampler = emcee.EnsembleSampler(npop, self.lpf.ps_vary.ndim, self.lpf,threads=threads)

            #pb = ipywidgets.IntProgress(max=mc_iter/50)
            #display(pb)
            #val = 0
            print("MCMC iterations=",mc_iter)
            for i,c in enumerate(self.sampler.sample(self.de.population,iterations=mc_iter)):
                print(i,end=" ")
                #if i%50 == 0:
                    #val+=50.
                    #pb.value += 1
            print("Finished MCMC")
            self.min_pv_mcmc = self.get_mean_values_mcmc_posteriors().medvals.values
        
        #samples = self.sampler.get_chain(flat=True)
        
        #if plot_corner:
        #    print("Mean acceptance fraction: {0:.3f}".format(np.mean(self.sampler.acceptance_fraction)))
        #    corner.corner(samples,labels=self.lpf.labels,show_titles=True,truths=self.lpf.means);
        #    
        #self.df_mean = self.get_mean_values_mcmc_posteriors()
        #self.df_mean['error'] = (self.df_mean['minus'] + self.df_mean['plus'])/2.
        #self.df_mean['zscore'] = (self.df_mean['medvals'].values - self.lpf.df_known['median'].values)/self.lpf.df_known.error.values
        
        #if plot_zscore:
        #    self.plot_zscore_panel()
        
        #return samples

    def calculate_derived_params(self,df_post=None,burnin_index=200):
        if df_post is None:
            chains_after_burnin = self.sampler.chain[:,burnin_index:,:]
            flatchain = chains_after_burnin.reshape((-1,len(self.lpf.ps_vary.priors)))
            df_post = pd.DataFrame(flatchain,columns=self.lpf.ps_vary.labels)

        if 'eps' in self.lpf.ps_fixed.labels:
            eps = self.lpf.ps_fixed_dict['eps']
        else:
            eps = df_post['P_p1'].values

        df_post['i']     = np.rad2deg(np.arccos(df_post['cosi'].values))
        df_post['a0']    = a0_from_m_M_P_epsilon(df_post['mstar'].values, df_post['mp'].values, df_post['P_p1'].values, eps, df_post['plx'].values)
        df_post['K_p1']  = semi_amplitude_from_gaia(df_post['P_p1'].values, df_post['a0'].values, df_post['plx'].values, df_post['cosi'].values, df_post['ecc_p1'].values)
        df_post['A']     = thiele_innes_a_from_campell(df_post['a0'].values, df_post['omega_p1'].values, df_post['Omega'].values, df_post['cosi'].values)
        df_post['B']     = thiele_innes_b_from_campell(df_post['a0'].values, df_post['omega_p1'].values, df_post['Omega'].values, df_post['cosi'].values)
        df_post['F']     = thiele_innes_f_from_campell(df_post['a0'].values, df_post['omega_p1'].values, df_post['Omega'].values, df_post['cosi'].values)
        df_post['G']     = thiele_innes_g_from_campell(df_post['a0'].values, df_post['omega_p1'].values, df_post['Omega'].values, df_post['cosi'].values)

        self.df_post = df_post
        self.df_mean = pd.concat([ExoMUSE_utils.get_mean_values_for_posterior(df_post[lab].values,lab) for lab in df_post.columns])

        return self.df_post, self.df_mean

    
    def plot_zscore_panel(self,ax=None,bx=None):

        labels_gaia = ['A','B','F','G','ecc_p1','P_p1','tp_p1']
        df_post_gaia = self.df_post[labels_gaia]
        self.df_gaia_mean = pd.concat([ExoMUSE_utils.get_mean_values_for_posterior(df_post_gaia[lab].values,lab) for lab in df_post_gaia.columns])

        self.df_gaia_mean['error'] = (self.df_gaia_mean['minus'] + self.df_gaia_mean['plus'])/2.
        self.df_gaia_mean['zscore'] = (self.df_gaia_mean['medvals'].values - self.lpf.df_known['median'].values)/self.lpf.df_known.error.values

        if ax is None and bx is None:
            fig, (ax,bx) = plt.subplots(dpi=200,ncols=2,nrows=1,figsize=(6,2.5))
            ax.errorbar(range(len(self.lpf.df_known[0:4])),self.lpf.df_known['median'].values[0:4],self.lpf.df_known.error.values[0:4],
                        marker='o',lw=0,mew=0.5,capsize=4,elinewidth=0.5,label='Gaia',color='black')
            ax.errorbar(range(len(self.df_gaia_mean[0:4])),self.df_gaia_mean['medvals'].values[0:4],yerr=[self.df_gaia_mean.minus.values[0:4],self.df_gaia_mean.plus.values[0:4]],
                        marker='o',lw=0,mew=0.5,capsize=4,elinewidth=0.5,label='MCMC Sampled',color='crimson')

            bx.plot(range(len(self.df_gaia_mean)),self.df_gaia_mean['zscore'],marker='o',lw=0,color='crimson')


            ax.legend(loc='upper right',fontsize=6)
            ax.set_xticks([0,1,2,3])
            ax.set_xticklabels(['A','B','F','G'])

            bx.set_xticks(range(len(self.df_gaia_mean)))
            bx.set_xticklabels(self.lpf.labels)

            ax.set_xlabel('Thile-Innes coefficients')
            ax.set_ylabel('Milliarcsec')
            bx.set_ylabel('Z-score')

            bx.axhline(y=-1,color='red',ls='--',lw=0.5)
            bx.axhline(y=1,color='red',ls='--',lw=0.5)

            for xx in [ax,bx]:
                ExoMUSE_utils.ax_apply_settings(xx)
                
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
        df_list = [ExoMUSE_utils.get_mean_values_for_posterior(flatchain[:,i],label,description) for i,label,description in zip(range(len(self.lpf.ps_vary.descriptions)),self.lpf.ps_vary.labels,self.lpf.ps_vary.descriptions)]
        return pd.concat(df_list)
    
    def print_param_diagnostics(self,pv):
        """
        A function to print nice parameter diagnostics.
        """
        self.df_diagnostics = pd.DataFrame(zip(self.lpf.ps_vary.labels,self.lpf.ps_vary.centers,self.lpf.ps_vary.bounds[:,0],self.lpf.ps_vary.bounds[:,1],pv,self.lpf.ps_vary.centers-pv),columns=["labels","centers","lower","upper","pv","center_dist"])
        print(self.df_diagnostics.to_string())
        return self.df_diagnostics









class LPFunctionGaiaOnly(object):
    """
    Log-Likelihood function class for RV only

    NOTES:
        Based on hpprvi's class, see: https://github.com/hpparvi/exo_tutorials
    """
    def __init__(self,filename,source_index,file_priors):
        """
        INPUT:
            file_priors - prior file name
        """
        
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
        
        # Setting priors
        self.ps_all = priorset_from_file(file_priors) # all priors
        self.ps_fixed = PriorSet(np.array(self.ps_all.priors)[np.array(self.ps_all.fixed)]) # fixed priorset
        self.ps_vary  = PriorSet(np.array(self.ps_all.priors)[~np.array(self.ps_all.fixed)]) # varying priorset
        self.ps_fixed_dict = {key: val for key, val in zip(self.ps_fixed.labels,self.ps_fixed.args1)}
        print('Reading in priorfile from {}'.format(file_priors))
        print(self.ps_all.df)
        print(self.means)
        
    def __call__(self,pv):
        """
        Return the log likelihood

        INPUT:
            pv - the input list of varying parameters
        """
        if any(pv < self.ps_vary.pmins) or any(pv>self.ps_vary.pmaxs):
            return -np.inf
                
        ###############
        ###############
        Theta = pv - self.means # deviation vector
        log_prob_gaia = -0.5 * np.dot(Theta, np.linalg.solve(self.CC, Theta))
        ###############
        ###############
        
        # Return the log-likelihood
        log_of_priors = self.ps_vary.c_log_prior(pv)
        log_ln = log_of_priors + log_prob_gaia
        #log_ln = log_prob_gaia
        return log_ln

class ExoMUSEfitGaiaOnly(object):
    """
    A class that does Gaia fitting.

    NOTES:
        - Needs to have LPFunction defined
    """
    def __init__(self,LPFunction):
        self.lpf = LPFunction

    def run_sampler(self,nburn=1000,mc_iter=1000,nwalkers=50,plot_zscore=True,plot_corner=True):
        """
        Minimize using the PyDE

        NOTES:
            see https://github.com/hpparvi/PyDE
        """
        
        print("Running MCMC")
        self.sampler = emcee.EnsembleSampler(nwalkers=nwalkers,
                                                ndim=self.lpf.ndim,
                                                log_prob_fn=self.lpf)
        #p0 = np.random.rand(nwalkers,self.lpf.ndim)
        #centers = self.lpf.means
        centers = np.array(self.lpf.ps_vary.centers)
        p0 = centers + 1e-4 * np.random.randn(nwalkers, self.lpf.ndim)
        #print(p0)
        #return p0
        state = self.sampler.run_mcmc(p0, nburn)
        print('Finished burnin',)
        self.sampler.reset()
        print('Running main MCMC, it=',mc_iter)
        self.sampler.run_mcmc(state, mc_iter)
        
        samples = self.sampler.get_chain(flat=True)
        
        if plot_corner:
            print("Mean acceptance fraction: {0:.3f}".format(np.mean(self.sampler.acceptance_fraction)))
            corner.corner(samples,labels=self.lpf.labels,show_titles=True,truths=self.lpf.means);
            
        self.df_mean = self.get_mean_values_mcmc_posteriors()
        self.df_mean['error'] = (self.df_mean['minus'] + self.df_mean['plus'])/2.
        self.df_mean['zscore'] = (self.df_mean['medvals'].values - self.lpf.df_known['median'].values)/self.lpf.df_known.error.values
        
        if plot_zscore:
            self.plot_zscore_panel()
        
        return samples
    
    def minimize_PyDE(self,npop=100,de_iter=200,mc_iter=1000,mcmc=True,threads=8,maximize=True,plot_priors=True,
                      sample_ball=False,k=None,n=None,plot_corner=True,plot_zscore=True):
        centers = np.array(self.lpf.ps_vary.centers)
        print("Running PyDE Optimizer")
        self.de = pyde.de.DiffEvol(self.lpf, self.lpf.ps_vary.bounds, npop, maximize=maximize) # we want to maximize the likelihood
        self.min_pv, self.min_pv_lnval = self.de.optimize(ngen=de_iter)
        print("Optimized using PyDE")
        print("Final parameters:")
        self.print_param_diagnostics(self.min_pv)
        #if plot_priors:
        #    self.lpf.ps_vary.plot_all(figsize=(6,4),pv=self.min_pv)
        print("LogPost value:",-1*self.min_pv_lnval)
        self.lnl_max  = -1*self.min_pv_lnval-self.lpf.ps_vary.c_log_prior(self.min_pv)
        print("LnL value:",self.lnl_max)
        print("Log priors",self.lpf.ps_vary.c_log_prior(self.min_pv))
        #if k is not None and n is not None:
        #    print("BIC:",stats_help.bic_from_likelihood(self.lnl_max,k,n))
        #    print("AIC:",stats_help.aic(k,self.lnl_max))
        if mcmc:
            print("Running MCMC")
            self.sampler = emcee.EnsembleSampler(npop, self.lpf.ps_vary.ndim, self.lpf,threads=threads)

            #pb = ipywidgets.IntProgress(max=mc_iter/50)
            #display(pb)
            #val = 0
            print("MCMC iterations=",mc_iter)
            for i,c in enumerate(self.sampler.sample(self.de.population,iterations=mc_iter)):
                print(i,end=" ")
                #if i%50 == 0:
                    #val+=50.
                    #pb.value += 1
            print("Finished MCMC")
            self.min_pv_mcmc = self.get_mean_values_mcmc_posteriors().medvals.values
        samples = self.sampler.get_chain(flat=True)
        
        if plot_corner:
            print("Mean acceptance fraction: {0:.3f}".format(np.mean(self.sampler.acceptance_fraction)))
            corner.corner(samples,labels=self.lpf.labels,show_titles=True,truths=self.lpf.means);
            
        self.df_mean = self.get_mean_values_mcmc_posteriors()
        self.df_mean['error'] = (self.df_mean['minus'] + self.df_mean['plus'])/2.
        self.df_mean['zscore'] = (self.df_mean['medvals'].values - self.lpf.df_known['median'].values)/self.lpf.df_known.error.values
        
        if plot_zscore:
            self.plot_zscore_panel()
        
        return samples
        

    
    def plot_zscore_panel(self,ax=None,bx=None):
        if ax is None and bx is None:
            fig, (ax,bx) = plt.subplots(dpi=200,ncols=2,nrows=1,figsize=(6,2.5))
        ax.errorbar(range(len(self.lpf.df_known[0:4])),self.lpf.df_known['median'].values[0:4],self.lpf.df_known.error.values[0:4],
                    marker='o',lw=0,mew=0.5,capsize=4,elinewidth=0.5,label='Gaia',color='black')
        ax.errorbar(range(len(self.df_mean[0:4])),self.df_mean['medvals'].values[0:4],yerr=[self.df_mean.minus.values[0:4],self.df_mean.plus.values[0:4]],
                    marker='o',lw=0,mew=0.5,capsize=4,elinewidth=0.5,label='MCMC Sampled',color='crimson')

        bx.plot(range(len(self.df_mean)),self.df_mean['zscore'],marker='o',lw=0,color='crimson')


        ax.legend(loc='upper right',fontsize=6)
        ax.set_xticks([0,1,2,3])
        ax.set_xticklabels(['A','B','F','G'])

        bx.set_xticks(range(len(self.df_mean)))
        bx.set_xticklabels(self.lpf.labels)

        ax.set_xlabel('Thile-Innes coefficients')
        ax.set_ylabel('Milliarcsec')
        bx.set_ylabel('Z-score')

        bx.axhline(y=-1,color='red',ls='--',lw=0.5)
        bx.axhline(y=1,color='red',ls='--',lw=0.5)

        for xx in [ax,bx]:
            ExoMUSE_utils.ax_apply_settings(xx)
            
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
        df_list = [ExoMUSE_utils.get_mean_values_for_posterior(flatchain[:,i],label,description) for i,label,description in zip(range(len(self.lpf.ps_vary.descriptions)),self.lpf.ps_vary.labels,self.lpf.ps_vary.descriptions)]
        return pd.concat(df_list)
    
    def print_param_diagnostics(self,pv):
        """
        A function to print nice parameter diagnostics.
        """
        self.df_diagnostics = pd.DataFrame(zip(self.lpf.ps_vary.labels,self.lpf.ps_vary.centers,self.lpf.ps_vary.bounds[:,0],self.lpf.ps_vary.bounds[:,1],pv,self.lpf.ps_vary.centers-pv),columns=["labels","centers","lower","upper","pv","center_dist"])
        print(self.df_diagnostics.to_string())
        return self.df_diagnostics



class LPFunctionRVOnlyNoGAIAOffset(object):
    """
    Log-Likelihood function class for RV only

    NOTES:
        Based on hpprvi's class, see: https://github.com/hpparvi/exo_tutorials
    """
    def __init__(self,x,y,yerr,file_priors):
        """
        INPUT:
            x - time values in BJD
            y - y values in m/s
            yerr - yerr values in m/s
            file_priors - prior file name
        """
        self.data= {"x"   : x,
                    "y"   : y,
                    "error"  : yerr}
        # Setting priors
        self.ps_all = priorset_from_file(file_priors) # all priors
        self.ps_fixed = PriorSet(np.array(self.ps_all.priors)[np.array(self.ps_all.fixed)]) # fixed priorset
        self.ps_vary  = PriorSet(np.array(self.ps_all.priors)[~np.array(self.ps_all.fixed)]) # varying priorset
        self.ps_fixed_dict = {key: val for key, val in zip(self.ps_fixed.labels,self.ps_fixed.args1)}
        print('Reading in priorfile from {}'.format(file_priors))
        print(self.ps_all.df)

    def get_jump_parameter_index(self,lab):
        """
        Get the index of a given label
        """
        return np.where(np.array(self.ps_vary.labels)==lab)[0][0]

    def get_jump_parameter_value(self,pv,lab):
        """
        Get the current value in the argument list 'pv' that has label 'lab'
        """
        # First check if we are actually varying it
        if lab in self.ps_vary.labels:
            return pv[self.get_jump_parameter_index(lab)]
        else:
            # We are not varying it
            return self.ps_fixed_dict[lab]
        
    def compute_rv_model(self,pv,times=None):
        """
        Compute the RV model

        INPUT:
            pv    - a list of parameters (only parameters that are being varied)
            times - times (optional), array of timestamps

        OUTPUT:
            rv - the rv model evaluated at 'times' if supplied, otherwise
                      defaults to original data timestamps
        """
        if times is None:
            times = self.data["x"]
        Tp      = self.get_jump_parameter_value(pv,'tp_p1') 
        P       = self.get_jump_parameter_value(pv,'P_p1')
        gamma   = self.get_jump_parameter_value(pv,'gamma')
        K       = self.get_jump_parameter_value(pv,'K_p1')
        e       = self.get_jump_parameter_value(pv,'ecc_p1')
        w       = self.get_jump_parameter_value(pv,'omega_p1')
        self.rv = get_rv_curve_peri(times,P=P,tp=Tp,e=e,omega=w,K=K)+gamma
        return self.rv

    def compute_polynomial_model(self,pv,times=None):
        """
        Compute the polynomial model.  Note that if gammadot and gammadotdot
        are not specified in the priors file, they both default to zero.

        INPUT:
            pv    - a list of parameters (only parameters that are being varied)
            times - times (optional), array of timestamps

        OUTPUT:
            poly - the polynomial model evaluated at 'times' if supplied,
                   otherwise defaults to original data timestamps
        """
        if times is None:
            times = self.data["x"]

        #T0 = self.get_jump_parameter_value(pv,'t0_p1')
        # Use Mean of data instead
        T0 = (self.data['x'][0] + self.data['x'][-1])/2.
        try:
            gammadot = self.get_jump_parameter_value(pv,'gammadot')
        except KeyError as e:
            gammadot = 0
        try:
            gammadotdot = self.get_jump_parameter_value(pv,'gammadotdot')
        except KeyError as e:
            gammadotdot = 0

        self.poly = (
            gammadot * (times - T0) +
            gammadotdot * (times - T0)**2
        )
        return self.poly

    def compute_total_model(self,pv,times=None):
        """
        Computes the full RM model (including RM and RV and CB)

        INPUT:
            pv    - a list of parameters (only parameters that are being varied)
            times - times (optional), array of timestamps

        OUTPUT:
            rm - the rm model evaluated at 'times' if supplied, otherwise
                      defaults to original data timestamps

        NOTES:
            see compute_rm_model(), compute_rv_model(),
            compute_polynomial_model()
        """
        return self.compute_rv_model(pv,times=times) + self.compute_polynomial_model(pv,times=times) 

    def __call__(self,pv):
        """
        Return the log likelihood

        INPUT:
            pv - the input list of varying parameters
        """
        if any(pv < self.ps_vary.pmins) or any(pv>self.ps_vary.pmaxs):
            return -np.inf


        ###############
        # Prepare data and model and error for ingestion into likelihood
        y_data = self.data['y']
        y_model = self.compute_total_model(pv)
        # jitter in quadrature
        jitter = self.get_jump_parameter_value(pv,'sigma_rv')
        error = np.sqrt(self.data['error']**2.+jitter**2.)
        ###############

        # Return the log-likelihood
        log_of_priors = self.ps_vary.c_log_prior(pv)
        # Calculate log likelihood
        log_of_model  = ll_normal_ev_py(y_data, y_model, error)
        log_ln = log_of_priors + log_of_model
        return log_ln




class LPFunctionRVOnly2(object):
    """
    Log-Likelihood function class for RV only

    NOTES:
        Based on hpprvi's class, see: https://github.com/hpparvi/exo_tutorials
    """
    def __init__(self,x1,y1,yerr1,x2,y2,yerr2,file_priors):
        """
        INPUT:
            x - time values in BJD
            y - y values in m/s
            yerr - yerr values in m/s
            file_priors - prior file name
        """
        self.data= {"x1"    : x1,
                    "y1"    : y1,
                    "error1": yerr1,
                    "x2"    : x2,
                    "y2"    : y2,
                    "error2": yerr2
                    }
        # Setting priors
        self.ps_all = priorset_from_file(file_priors) # all priors
        self.ps_fixed = PriorSet(np.array(self.ps_all.priors)[np.array(self.ps_all.fixed)]) # fixed priorset
        self.ps_vary  = PriorSet(np.array(self.ps_all.priors)[~np.array(self.ps_all.fixed)]) # varying priorset
        self.ps_fixed_dict = {key: val for key, val in zip(self.ps_fixed.labels,self.ps_fixed.args1)}
        print('Reading in priorfile from {}'.format(file_priors))
        print(self.ps_all.df)

    def get_jump_parameter_index(self,lab):
        """
        Get the index of a given label
        """
        return np.where(np.array(self.ps_vary.labels)==lab)[0][0]

    def get_jump_parameter_value(self,pv,lab):
        """
        Get the current value in the argument list 'pv' that has label 'lab'
        """
        # First check if we are actually varying it
        if lab in self.ps_vary.labels:
            return pv[self.get_jump_parameter_index(lab)]
        else:
            # We are not varying it
            return self.ps_fixed_dict[lab]
        
    def compute_rv_model(self,pv,times1=None,times2=None):
        """
        Compute the RV model

        INPUT:
            pv    - a list of parameters (only parameters that are being varied)
            times - times (optional), array of timestamps

        OUTPUT:
            rv - the rv model evaluated at 'times' if supplied, otherwise
                      defaults to original data timestamps
        """
        if times1 is None:
            times1 = self.data["x1"]
        if times2 is None:
            times2 = self.data["x2"]
        Tp      = self.get_jump_parameter_value(pv,'tp_p1') + GAIA_TP_OFFSET
        P       = self.get_jump_parameter_value(pv,'P_p1')
        gamma1  = self.get_jump_parameter_value(pv,'gamma1')
        gamma2  = self.get_jump_parameter_value(pv,'gamma2')
        K       = self.get_jump_parameter_value(pv,'K_p1')
        e       = self.get_jump_parameter_value(pv,'ecc_p1')
        w       = self.get_jump_parameter_value(pv,'omega_p1')
        self.rv1= get_rv_curve_peri(times1,P=P,tp=Tp,e=e,omega=w,K=K)+gamma1
        self.rv2= get_rv_curve_peri(times2,P=P,tp=Tp,e=e,omega=w,K=K)+gamma2
        return self.rv1, self.rv2

    def compute_polynomial_model(self,pv,times=None):
        """
        Compute the polynomial model.  Note that if gammadot and gammadotdot
        are not specified in the priors file, they both default to zero.

        INPUT:
            pv    - a list of parameters (only parameters that are being varied)
            times - times (optional), array of timestamps

        OUTPUT:
            poly - the polynomial model evaluated at 'times' if supplied,
                   otherwise defaults to original data timestamps
        """
        if times is None:
            times = self.data["x"]

        #T0 = self.get_jump_parameter_value(pv,'t0_p1')
        # Use Mean of data instead
        T0 = (self.data['x'][0] + self.data['x'][-1])/2.
        try:
            gammadot = self.get_jump_parameter_value(pv,'gammadot')
        except KeyError as e:
            gammadot = 0
        try:
            gammadotdot = self.get_jump_parameter_value(pv,'gammadotdot')
        except KeyError as e:
            gammadotdot = 0

        self.poly = (
            gammadot * (times - T0) +
            gammadotdot * (times - T0)**2
        )
        #return self.poly
        print('poly not implemented')
        return 0.

    def compute_total_model(self,pv,times1=None,times2=None):
        """
        Computes the full RM model (including RM and RV and CB)

        INPUT:
            pv    - a list of parameters (only parameters that are being varied)
            times - times (optional), array of timestamps

        OUTPUT:
            rm - the rm model evaluated at 'times' if supplied, otherwise
                      defaults to original data timestamps

        NOTES:
            see compute_rm_model(), compute_rv_model(),
            compute_polynomial_model()
        """
        return self.compute_rv_model(pv,times1=times1,times2=times2)# + self.compute_polynomial_model(pv,times=times) 

    def __call__(self,pv):
        """
        Return the log likelihood

        INPUT:
            pv - the input list of varying parameters
        """
        if any(pv < self.ps_vary.pmins) or any(pv>self.ps_vary.pmaxs):
            return -np.inf

        ###############
        # RV
        ###############
        y_data1 = self.data['y1']
        y_data2 = self.data['y2']
        y_model1, y_model2 = self.compute_total_model(pv)
        jitter1 = self.get_jump_parameter_value(pv,'sigma_rv1')
        jitter2 = self.get_jump_parameter_value(pv,'sigma_rv2')
        error1 = np.sqrt(self.data['error1']**2.+jitter1**2.)
        error2 = np.sqrt(self.data['error2']**2.+jitter2**2.)
        log_of_rv_model1  = ll_normal_ev_py(y_data1, y_model1, error1)
        log_of_rv_model2  = ll_normal_ev_py(y_data2, y_model2, error2)
        log_of_rv_model = log_of_rv_model1 + log_of_rv_model2

        # Return the log-likelihood
        log_of_priors = self.ps_vary.c_log_prior(pv)
        # Calculate log likelihood
        log_ln = log_of_priors + log_of_rv_model
        return log_ln


class LPFunctionRVOnly(object):
    """
    Log-Likelihood function class for RV only

    NOTES:
        Based on hpprvi's class, see: https://github.com/hpparvi/exo_tutorials
    """
    def __init__(self,x,y,yerr,file_priors):
        """
        INPUT:
            x - time values in BJD
            y - y values in m/s
            yerr - yerr values in m/s
            file_priors - prior file name
        """
        self.data= {"x"   : x,
                    "y"   : y,
                    "error"  : yerr}

        # Setting priors
        self.ps_all = priorset_from_file(file_priors) # all priors
        self.ps_fixed = PriorSet(np.array(self.ps_all.priors)[np.array(self.ps_all.fixed)]) # fixed priorset
        self.ps_vary  = PriorSet(np.array(self.ps_all.priors)[~np.array(self.ps_all.fixed)]) # varying priorset
        self.ps_fixed_dict = {key: val for key, val in zip(self.ps_fixed.labels,self.ps_fixed.args1)}
        print('Reading in priorfile from {}'.format(file_priors))
        print(self.ps_all.df)

    def get_jump_parameter_index(self,lab):
        """
        Get the index of a given label
        """
        return np.where(np.array(self.ps_vary.labels)==lab)[0][0]

    def get_jump_parameter_value(self,pv,lab):
        """
        Get the current value in the argument list 'pv' that has label 'lab'
        """
        # First check if we are actually varying it
        if lab in self.ps_vary.labels:
            return pv[self.get_jump_parameter_index(lab)]
        else:
            # We are not varying it
            return self.ps_fixed_dict[lab]
        
    def compute_rv_model(self,pv,times=None):
        """
        Compute the RV model

        INPUT:
            pv    - a list of parameters (only parameters that are being varied)
            times - times (optional), array of timestamps

        OUTPUT:
            rv - the rv model evaluated at 'times' if supplied, otherwise
                      defaults to original data timestamps
        """
        if times is None:
            times = self.data["x"]
        Tp      = self.get_jump_parameter_value(pv,'tp_p1') + GAIA_TP_OFFSET
        P       = self.get_jump_parameter_value(pv,'P_p1')
        gamma   = self.get_jump_parameter_value(pv,'gamma')
        K       = self.get_jump_parameter_value(pv,'K_p1')
        e       = self.get_jump_parameter_value(pv,'ecc_p1')
        w       = self.get_jump_parameter_value(pv,'omega_p1')
        self.rv = get_rv_curve_peri(times,P=P,tp=Tp,e=e,omega=w,K=K)+gamma
        return self.rv

    def compute_polynomial_model(self,pv,times=None):
        """
        Compute the polynomial model.  Note that if gammadot and gammadotdot
        are not specified in the priors file, they both default to zero.

        INPUT:
            pv    - a list of parameters (only parameters that are being varied)
            times - times (optional), array of timestamps

        OUTPUT:
            poly - the polynomial model evaluated at 'times' if supplied,
                   otherwise defaults to original data timestamps
        """
        if times is None:
            times = self.data["x"]

        #T0 = self.get_jump_parameter_value(pv,'t0_p1')
        # Use Mean of data instead
        T0 = (self.data['x'][0] + self.data['x'][-1])/2.
        try:
            gammadot = self.get_jump_parameter_value(pv,'gammadot')
        except KeyError as e:
            gammadot = 0
        try:
            gammadotdot = self.get_jump_parameter_value(pv,'gammadotdot')
        except KeyError as e:
            gammadotdot = 0

        self.poly = (
            gammadot * (times - T0) +
            gammadotdot * (times - T0)**2
        )
        return self.poly

    def compute_total_model(self,pv,times=None):
        """
        Computes the full RM model (including RM and RV and CB)

        INPUT:
            pv    - a list of parameters (only parameters that are being varied)
            times - times (optional), array of timestamps

        OUTPUT:
            rm - the rm model evaluated at 'times' if supplied, otherwise
                      defaults to original data timestamps

        NOTES:
            see compute_rm_model(), compute_rv_model(),
            compute_polynomial_model()
        """
        return self.compute_rv_model(pv,times=times) + self.compute_polynomial_model(pv,times=times) 

    def __call__(self,pv):
        """
        Return the log likelihood

        INPUT:
            pv - the input list of varying parameters
        """
        if any(pv < self.ps_vary.pmins) or any(pv>self.ps_vary.pmaxs):
            return -np.inf


        ###############
        # Prepare data and model and error for ingestion into likelihood
        y_data = self.data['y']
        y_model = self.compute_total_model(pv)
        # jitter in quadrature
        jitter = self.get_jump_parameter_value(pv,'sigma_rv')
        error = np.sqrt(self.data['error']**2.+jitter**2.)
        ###############

        # Return the log-likelihood
        log_of_priors = self.ps_vary.c_log_prior(pv)
        # Calculate log likelihood
        log_of_model  = ll_normal_ev_py(y_data, y_model, error)
        log_ln = log_of_priors + log_of_model
        return log_ln

class LPFunctionRVOnly_1Instruments(object):
    """
    Log-Likelihood function class for RV only

    NOTES:
        Based on hpprvi's class, see: https://github.com/hpparvi/exo_tutorials
    """
    def __init__(self,x,y,yerr,instrument_name,periastron_offset,file_priors):
        """
        INPUT:
            x - time values in BJD for RV instrument
            y - y values in m/s for RV instrument
            yerr - yerr values in m/s for RV instrument
            instrument_name - string of the instrument name
            periastron_offset - custom input for the periastron offset time to be used
            file_priors - prior file name
        """
        self.data= {"x"   : x,
                    "y"   : y,
                    "error"  : yerr,
                    "instrument_name" : instrument_name}    

        self.periastron_offset = periastron_offset
        # Setting priors
        self.ps_all = priorset_from_file(file_priors) # all priors
        self.ps_fixed = PriorSet(np.array(self.ps_all.priors)[np.array(self.ps_all.fixed)]) # fixed priorset
        self.ps_vary  = PriorSet(np.array(self.ps_all.priors)[~np.array(self.ps_all.fixed)]) # varying priorset
        self.ps_fixed_dict = {key: val for key, val in zip(self.ps_fixed.labels,self.ps_fixed.args1)}
        print('Reading in priorfile from {}'.format(file_priors))
        print(self.ps_all.df)

    def get_jump_parameter_index(self,lab):
        """
        Get the index of a given label
        """
        return np.where(np.array(self.ps_vary.labels)==lab)[0][0]

    def get_jump_parameter_value(self,pv,lab):
        """
        Get the current value in the argument list 'pv' that has label 'lab'
        """
        # First check if we are actually varying it
        if lab in self.ps_vary.labels:
            return pv[self.get_jump_parameter_index(lab)]
        else:
            # We are not varying it
            return self.ps_fixed_dict[lab]

    def compute_rv_model(self,pv,instrument='TRES',times=None):
        """
        Compute the RV model

        INPUT:
            pv    - a list of parameters (only parameters that are being varied)
            times - times (optional), array of timestamps

        OUTPUT:
            rv - the rv model evaluated at 'times' if supplied, otherwise
                      defaults to original data timestamps
        """
        if times is None:
                times = self.data["x"]

        Tp = self.get_jump_parameter_value(pv,'tp_p1') + self.periastron_offset #+ GAIA_TP_OFFSET
        P = self.get_jump_parameter_value(pv,'P_p1') # done
        #sqrt_e_cos_w = self.get_jump_parameter_value(pv,'sqrt_e_cos_w')
        #sqrt_e_sin_w = self.get_jump_parameter_value(pv,'sqrt_e_sin_w')
        e = self.get_jump_parameter_value(pv,'ecc_p1') # done
        w = self.get_jump_parameter_value(pv,'omega_p1') # done
        K = self.get_jump_parameter_value(pv,'K_p1')

        gamma  = self.get_jump_parameter_value(pv,'gamma') # done
        self.rv = get_rv_curve_peri(times,P=P,tp=Tp,e=e,omega=w,K=K)+gamma
        
        return self.rv

    def compute_polynomial_model(self,pv,instrument,times=None):
        """
        Compute the polynomial model.  Note that if gammadot and gammadotdot
        are not specified in the priors file, they both default to zero.

        INPUT:
            pv    - a list of parameters (only parameters that are being varied)
            times - times (optional), array of timestamps

        OUTPUT:
            poly - the polynomial model evaluated at 'times' if supplied,
                   otherwise defaults to original data timestamps
        """
        if times is None:
            times = self.data["x"]

        #T0 = self.get_jump_parameter_value(pv,'t0_p1')
        # Use Mean of data instead
        T0 = 0
        T0 = (self.data['x'][0] + self.data['x'][-1])/2.
        try:
            gammadot = self.get_jump_parameter_value(pv,'gammadot')
        except KeyError as e:
            gammadot = 0
        try:
            gammadotdot = self.get_jump_parameter_value(pv,'gammadotdot')
        except KeyError as e:
            gammadotdot = 0
            
        self.poly = (gammadot * (times - T0) + gammadotdot * (times - T0)**2)

        return self.poly

    def compute_total_model(self,pv,instrument,times=None):
        """
        Computes the full RM model (including RM and RV and CB)

        INPUT:
            pv    - a list of parameters (only parameters that are being varied)
            times - times (optional), array of timestamps

        OUTPUT:
            rm - the rm model evaluated at 'times' if supplied, otherwise
                      defaults to original data timestamps

        NOTES:
            see compute_rm_model(), compute_rv_model(),
            compute_polynomial_model()
        """
        return self.compute_rv_model(pv,instrument,times=times) + self.compute_polynomial_model(pv,instrument,times=times) 

        
    def __call__(self,pv):
        """
        Return the log likelihood

        INPUT:
            pv - the input list of varying parameters
        """
        if any(pv < self.ps_vary.pmins) or any(pv>self.ps_vary.pmaxs):
            return -np.inf
                
        ###############
        # PRIORS
        ###############
        log_of_priors = self.ps_vary.c_log_prior(pv)

        ###############
        # RV 
        ###############
        data = self.data['y']
        model = self.compute_total_model(pv,self.data["instrument_name"])
        jitter = self.get_jump_parameter_value(pv,'sigma_rv')
        error = np.sqrt((self.data['error']**2.)+(jitter**2.))
        log_of_rv_model  = ll_normal_ev_py(data, model, error)


        # FINAL COMBINED
        log_ln = log_of_priors + log_of_rv_model
        return log_ln

class LPFunctionRVOnly_2Instruments(object):
    """
    Log-Likelihood function class for RV only

    NOTES:
        Based on hpprvi's class, see: https://github.com/hpparvi/exo_tutorials
    """
    def __init__(self,x1,y1,yerr1,x2,y2,yerr2,instrument_name1,instrument_name2,periastron_offset,file_priors):
        """
        INPUT:
            x1 - time values in BJD for RV instrument 1
            y1 - y values in m/s for RV instrument 1
            yerr1 - yerr values in m/s for RV instrument 1
            instrument_name1 - string of the instrument name that distinguishes the 2 sets of RVs
            x2 - time values in BJD for RV instrument 2
            y2 - y values in m/s for RV instrument 2
            yerr2 - yerr values in m/s for RV instrument 2
            instrument_name2 - string of the instrument name that distinguishes the 2 sets of RVs
            periastron_offset - custom input for the periastron offset time to be used
            file_priors - prior file name
        """
        self.data1= {"x"   : x1,
                    "y"   : y1,
                    "error"  : yerr1,
                    "instrument_name":instrument_name1}
        self.data2= {"x"   : x2,
                    "y"   : y2,
                    "error"  : yerr2,
                    "instrument_name":instrument_name2}                   
        # Setting priors
        self.ps_all = priorset_from_file(file_priors) # all priors
        self.ps_fixed = PriorSet(np.array(self.ps_all.priors)[np.array(self.ps_all.fixed)]) # fixed priorset
        self.ps_vary  = PriorSet(np.array(self.ps_all.priors)[~np.array(self.ps_all.fixed)]) # varying priorset
        self.ps_fixed_dict = {key: val for key, val in zip(self.ps_fixed.labels,self.ps_fixed.args1)}
        print('Reading in priorfile from {}'.format(file_priors))
        print(self.ps_all.df)

    def get_jump_parameter_index(self,lab):
        """
        Get the index of a given label
        """
        return np.where(np.array(self.ps_vary.labels)==lab)[0][0]

    def get_jump_parameter_value(self,pv,lab):
        """
        Get the current value in the argument list 'pv' that has label 'lab'
        """
        # First check if we are actually varying it
        if lab in self.ps_vary.labels:
            return pv[self.get_jump_parameter_index(lab)]
        else:
            # We are not varying it
            return self.ps_fixed_dict[lab]

    def compute_rv_model(self,pv,instrument='TRES',times=None):
        """
        Compute the RV model

        INPUT:
            pv    - a list of parameters (only parameters that are being varied)
            times - times (optional), array of timestamps

        OUTPUT:
            rv - the rv model evaluated at 'times' if supplied, otherwise
                      defaults to original data timestamps
        """
        if times is None:
            if instrument == self.data1["instrument_name"]:
                times = self.data1["x"]
            else:
                times = self.data2["x"]

        Tp = self.get_jump_parameter_value(pv,'tp_p1') + GAIA_TP_OFFSET
        P = self.get_jump_parameter_value(pv,'P_p1') # done
        #sqrt_e_cos_w = self.get_jump_parameter_value(pv,'sqrt_e_cos_w')
        #sqrt_e_sin_w = self.get_jump_parameter_value(pv,'sqrt_e_sin_w')
        e = self.get_jump_parameter_value(pv,'ecc_p1') # done
        w = self.get_jump_parameter_value(pv,'omega_p1') # done
        K = self.get_jump_parameter_value(pv,'K_p1')

        if instrument == self.data1["instrument_name"]:
            gamma1  = self.get_jump_parameter_value(pv,'gamma1') # done
            self.rv = get_rv_curve_peri(times,P=P,tp=Tp,e=e,omega=w,K=K)+gamma1

        else:
            gamma2  = self.get_jump_parameter_value(pv,'gamma2') # done
            self.rv = get_rv_curve_peri(times,P=P,tp=Tp,e=e,omega=w,K=K)+gamma2
        
        return self.rv

    def compute_polynomial_model(self,pv,instrument,times=None):
        """
        Compute the polynomial model.  Note that if gammadot and gammadotdot
        are not specified in the priors file, they both default to zero.

        INPUT:
            pv    - a list of parameters (only parameters that are being varied)
            times - times (optional), array of timestamps

        OUTPUT:
            poly - the polynomial model evaluated at 'times' if supplied,
                   otherwise defaults to original data timestamps
        """
        if times is None:
            if instrument == self.data1["instrument_name"]:
                times = self.data1["x"]
            else:
                times = self.data2["x"]

        #T0 = self.get_jump_parameter_value(pv,'t0_p1')
        # Use Mean of data instead
        T0 = 0
        if instrument == self.data1["instrument_name"]:
            T0 = (self.data1['x'][0] + self.data1['x'][-1])/2.
            try:
                gammadot = self.get_jump_parameter_value(pv,'gammadot1')
            except KeyError as e:
                gammadot = 0
            try:
                gammadotdot = self.get_jump_parameter_value(pv,'gammadotdot1')
            except KeyError as e:
                gammadotdot = 0
            
            self.poly = (gammadot * (times - T0) + gammadotdot * (times - T0)**2)

        else:
            T0 = (self.data2['x'][0] + self.data2['x'][-1])/2.

            #T0 = (data['x'][0] + data['x'][-1])/2.
            try:
                gammadot = self.get_jump_parameter_value(pv,'gammadot2')
            except KeyError as e:
                gammadot = 0
            try:
                gammadotdot = self.get_jump_parameter_value(pv,'gammadotdot2')
            except KeyError as e:
                gammadotdot = 0

            self.poly = (gammadot * (times - T0) + gammadotdot * (times - T0)**2)

        return self.poly

    def compute_total_model(self,pv,instrument,times=None):
        """
        Computes the full RM model (including RM and RV and CB)

        INPUT:
            pv    - a list of parameters (only parameters that are being varied)
            times - times (optional), array of timestamps

        OUTPUT:
            rm - the rm model evaluated at 'times' if supplied, otherwise
                      defaults to original data timestamps

        NOTES:
            see compute_rm_model(), compute_rv_model(),
            compute_polynomial_model()
        """
        return self.compute_rv_model(pv,instrument,times=times) + self.compute_polynomial_model(pv,instrument,times=times) 

        
    def __call__(self,pv):
        """
        Return the log likelihood

        INPUT:
            pv - the input list of varying parameters
        """
        if any(pv < self.ps_vary.pmins) or any(pv>self.ps_vary.pmaxs):
            return -np.inf
                
        ###############
        # PRIORS
        ###############
        log_of_priors = self.ps_vary.c_log_prior(pv)

        ###############
        # RV - TRES
        ###############
        data_inst1 = self.data1['y']
        model_inst1 = self.compute_total_model(pv,self.data1["instrument_name"])
        jitter_inst1 = self.get_jump_parameter_value(pv,'sigma_rv1')
        error_inst1 = np.sqrt((self.data1['error']**2.)+(jitter_inst1**2.))
        log_of_rv_model_inst1  = ll_normal_ev_py(data_inst1, model_inst1, error_inst1)

        ###############
        # RV - HPF
        ###############
        data_inst2 = self.data2['y']
        model_inst2 = self.compute_total_model(pv,self.data2["instrument_name"])
        jitter_inst2 = self.get_jump_parameter_value(pv,'sigma_rv2')
        error_inst2 = np.sqrt((self.data2['error']**2.)+(jitter_inst2**2.))
        log_of_rv_model_inst2  = ll_normal_ev_py(data_inst2, model_inst2, error_inst2)

        # FINAL COMBINED
        log_ln = log_of_priors + log_of_rv_model_inst1 + log_of_rv_model_inst2
        return log_ln

class ExoMUSEfitRVOnly(object):
    """
    A class that does Gaia fitting.

    NOTES:
        - Needs to have LPFunction defined
    """
    def __init__(self,LPFunction):
        self.lpf = LPFunction

    def minimize_AMOEBA(self):
        centers = np.array(self.lpf.ps_vary.centers)

        def neg_lpf(pv):
            return -1.*self.lpf(pv)
        self.min_pv = minimize(neg_lpf,centers,method='Nelder-Mead',tol=1e-9,
                                   options={'maxiter': 100000, 'maxfev': 10000, 'disp': True}).x

    def nested_sampling(self,dynesty_bound='single',dynesty_nthreads=1,nlive=1000):
        """
        Perform nested sampling with dynesty
        """
        ndim = self.lpf.ps_vary.ndim

        #################
        # Prior transforms
        def prior_transform(pv):
            return [self.lpf.ps_vary.transform_function[i](pv[i]) for i in range(len(pv))] 
        #################
        
        self.sampler = dynesty.DynamicNestedSampler(self.lpf,
                                                    prior_transform,
                                                    ndim,
                                                    bound=dynesty_bound,
                                                    nlive=nlive,queue_size=dynesty_nthreads)
        self.sampler.run_nested() 
        self.results = self.sampler.results

        # Starting to code up some of the parallel aspects, but running into some pickling errors
        #with contextlib.closing(Pool(processes=dynesty_nthreads - 1)) as executor:
        #    self.sampler = dynesty.DynamicNestedSampler(self.lpf,
        #                                                prior_transform,
        #                                                ndim,
        #                                                pool=executor,
        #                                                nlive=nlive,
        #                                                queue_size=dynesty_nthreads)
        #    self.sampler.run_nested()
        #    self.results = self.sampler.results



    def minimize_PyDE(self,npop=100,de_iter=200,mc_iter=1000,mcmc=True,threads=8,maximize=True,plot_priors=True,sample_ball=False,k=None,n=None):
        """
        Minimize using the PyDE

        NOTES:
            see https://github.com/hpparvi/PyDE
        """
        centers = np.array(self.lpf.ps_vary.centers)
        print("Running PyDE Optimizer")
        self.de = pyde.de.DiffEvol(self.lpf, self.lpf.ps_vary.bounds, npop, maximize=maximize) # we want to maximize the likelihood
        self.min_pv, self.min_pv_lnval = self.de.optimize(ngen=de_iter)
        print("Optimized using PyDE")
        print("Final parameters:")
        self.print_param_diagnostics(self.min_pv)
        #self.lpf.ps.plot_all(figsize=(6,4),pv=self.min_pv)
        print("LogPost value:",-1*self.min_pv_lnval)
        self.lnl_max  = -1*self.min_pv_lnval-self.lpf.ps_vary.c_log_prior(self.min_pv)
        print("LnL value:",self.lnl_max)
        print("Log priors",self.lpf.ps_vary.c_log_prior(self.min_pv))
        if k is not None and n is not None:
            print("BIC:",stats_help.bic_from_likelihood(self.lnl_max,k,n))
            print("AIC:",stats_help.aic(k,self.lnl_max))
        if mcmc:
            print("Running MCMC")
            self.sampler = emcee.EnsembleSampler(npop, self.lpf.ps_vary.ndim, self.lpf,threads=threads)

            #pb = ipywidgets.IntProgress(max=mc_iter/50)
            #display(pb)
            #val = 0
            print("MCMC iterations=",mc_iter)
            for i,c in enumerate(self.sampler.sample(self.de.population,iterations=mc_iter)):
                print(i,end=" ")
                #if i%50 == 0:
                    #val+=50.
                    #pb.value += 1
            print("Finished MCMC")
            self.min_pv_mcmc = self.get_mean_values_mcmc_posteriors().medvals.values

    def get_mean_values_mcmc_posteriors(self,flatchain=None):
        """
        Get the mean values from the posteriors

            flatchain - if not passed, then will default using the full flatchain (will likely include burnin)

        EXAMPLE:
        """
        if flatchain is None:
            flatchain = self.sampler.flatchain
            print('No flatchain passed, defaulting to using full chains')
        df_list = [ExoMUSE_utils.get_mean_values_for_posterior(flatchain[:,i],label,description) for i,label,description in zip(range(len(self.lpf.ps_vary.descriptions)),self.lpf.ps_vary.labels,self.lpf.ps_vary.descriptions)]
        return pd.concat(df_list)

    def print_param_diagnostics(self,pv):
        """
        A function to print nice parameter diagnostics.
        """
        self.df_diagnostics = pd.DataFrame(zip(self.lpf.ps_vary.labels,self.lpf.ps_vary.centers,self.lpf.ps_vary.bounds[:,0],self.lpf.ps_vary.bounds[:,1],pv,self.lpf.ps_vary.centers-pv),columns=["labels","centers","lower","upper","pv","center_dist"])
        print(self.df_diagnostics.to_string())
        return self.df_diagnostics

    def plot_fit(self,pv=None,times=None):
        """
        Plot the model curve for a given set of parameters pv

        INPUT:
            pv - an array containing a sample draw of the parameters defined in self.lpf.ps_vary
               - will default to best-fit parameters if none are supplied
        """
        if pv is None:
            print('Plotting curve with best-fit values')
            pv = self.min_pv
        x = self.lpf.data['x']
        y = self.lpf.data['y']
        jitter = self.lpf.get_jump_parameter_value(pv,'sigma_rv')
        yerr = np.sqrt(self.lpf.data['error']**2.+jitter**2.)
        model_obs = self.lpf.compute_total_model(pv)
        residuals = y-model_obs
            
        model = self.lpf.compute_total_model(pv)
        residuals = y-model

        # Plot
        nrows = 2
        self.fig, self.ax = plt.subplots(nrows=nrows,sharex=True,figsize=(10,6),gridspec_kw={'height_ratios': [5, 2]})
        self.ax[0].errorbar(x,y,yerr=yerr,elinewidth=1,lw=0,alpha=1,capsize=5,mew=1,marker="o",barsabove=True,markersize=8,label="Data")
        if times is not None:
            model = self.lpf.compute_total_model(pv,times=times)
            self.ax[0].plot(times,model,label="Model",color='crimson')
        else:
            self.ax[0].plot(x,model_obs,label="Model",color='crimson')
            
        self.ax[1].errorbar(x,residuals,yerr=yerr,elinewidth=1,lw=0,alpha=1,capsize=5,mew=1,marker="o",barsabove=True,markersize=8,
                            label="Residuals, std="+str(np.std(residuals)))
        for xx in self.ax:
            xx.minorticks_on()
            xx.legend(loc='lower left',fontsize=9)
            xx.set_ylabel("RV [m/s]",labelpad=2)
        self.ax[-1].set_xlabel("Time (BJD)",labelpad=2)
        self.ax[0].set_title("RM Effect")
        self.fig.subplots_adjust(wspace=0.05,hspace=0.05)

    def plot_mcmc_fit(self,times=None):
        df = self.get_mean_values_mcmc_posteriors()
        print('Plotting curve with best-fit mcmc values')
        self.plot_fit(pv=df.medvals.values)

def read_priors(priorname):
    """
    Read a prior file as in juliet.py style

    OUTPUT:
        priors - prior dictionary
        n_params - number of parameters

    EXAMPLE:
        P, numpriors = read_priors('../data/priors.dat')
    """
    fin = open(priorname)
    priors = {}
    n_transit = 0
    n_rv = 0
    n_params = 0
    numbering_transit = np.array([])
    numbering_rv = np.array([])
    while True:
        line = fin.readline()
        if line != '':
            if line[0] != '#':
                line = line.split('#')[0] # remove things after comment
                out = line.split()
                parameter,prior_name,vals = out
                parameter = parameter.split()[0]
                prior_name = prior_name.split()[0]
                vals = vals.split()[0]
                priors[parameter] = {}
                pvector = parameter.split('_')
                # Check if parameter/planet is from a transiting planet:
                if pvector[0] == 'r1' or pvector[0] == 'p':
                    pnumber = int(pvector[1][1:])
                    numbering_transit = np.append(numbering_transit,pnumber)
                    n_transit += 1
                # Check if parameter/planet is from a RV planet:
                if pvector[0] == 'K':
                    pnumber = int(pvector[1][1:])
                    numbering_rv = np.append(numbering_rv,pnumber)
                    n_rv += 1
                if prior_name.lower() == 'fixed':
                    priors[parameter]['type'] = prior_name.lower()
                    priors[parameter]['value'] = np.double(vals)
                    priors[parameter]['cvalue'] = np.double(vals)
                else:
                    n_params += 1
                    priors[parameter]['type'] = prior_name.lower()
                    if priors[parameter]['type'] != 'truncatednormal':
                        v1,v2 = vals.split(',')
                        priors[parameter]['value'] = [np.double(v1),np.double(v2)]
                    else:
                        v1,v2,v3,v4 = vals.split(',')
                        priors[parameter]['value'] = [np.double(v1),np.double(v2),np.double(v3),np.double(v4)]
                    priors[parameter]['cvalue'] = 0.
        else:
            break
    #return priors, n_transit, n_rv, numbering_transit.astype('int'), numbering_rv.astype('int'), n_params
    return priors, n_params

def priordict_to_priorset(priordict,verbose=True):
    """
    Get a PriorSet from prior diectionary

    EXAMPLE:
        P, numpriors = readpriors('../data/priors.dat')
        ps = priordict_to_priorset(priors)
        ps.df
    """
    priors = []
    for key in priordict.keys():
        inp = priordict[key]
        if verbose: print(key)
        val = inp['value']
        if inp['type'] == 'normal':
            outp = NP(val[0],val[1],key,key,priortype='model')
        elif inp['type'] == 'uniform':
            outp = UP(val[0],val[1],key,key,priortype='model')
        elif inp['type'] == 'fixed':
            outp = FP(val,key,key,priortype='model')
        else:
            print('Error, ptype {} not supported'.format(inp['type']))
        priors.append(outp)
    return PriorSet(priors)

def priorset_from_file(filename,verbose=False):
    """
    Get a PriorSet() from a filename
    """
    priordict, num_priors = read_priors(filename)
    return priordict_to_priorset(priordict,verbose)

class RMHirano(object):
    def __init__(self,lam,vsini,P,T0,aRs,i,RpRs,e,w,u,beta,sigma,supersample_factor=7,exp_time=0.00035,limb_dark='linear'):
        """
        Evaluate Rossiter McLaughlin effect using the model of Hirano et al. 2010, 2011

        INPUT:
            lam - sky-projected obliquity in deg
            vsini - sky-projected rotational velocity in km/s
            P - Period in days
            T0 - Transit center
            aRs - a/R*
            i - inclination in degrees
            RpRs - radius ratio
            e - eccentricity
            w - omega in degrees
            u - [u1,u2] where u1 and u2 are the quadratic limb-dark coefficients
            beta - Gaussian dispersion of spectral lines, in km/s, typically 2.5-4.5km/s (see Hirano+11)
            sigma - Gaussian broadening kernel from Hirano+10. Hirano+10 found vsini/1.31 as an approximation that sometimes works (Why?).

        EXAMPLE:
            times = np.linspace(-0.05,0.05,200)
            T0 = 0.
            P = 3.48456408
            aRs = 21.09
            i = 89.
            vsini = 8.7
            rprs = np.sqrt(0.01)
            e = 0.
            w = 90.
            lam = 45.
            u = [0.3,0.3]
            R = RMHirano(lam,vsini,P,T0,aRs,i,rprs,e,w,u,limb_dark='quadratic')
            rm = R.evaluate(times)

            fig, ax = plt.subplots()
            ax.plot(times,rm)
        """
        self.lam = lam
        self.vsini = vsini
        self.P = P
        self.T0 = T0
        self.aRs = aRs
        self.i = i
        self.iS = 90. # not really using anymore, as it cancels out, should just remove
        self.RpRs = RpRs
        self.e = e
        self.w = w
        self.u = u
        self.limb_dark = limb_dark
        self.rstar = 1. # not really using anymore, as it cancels out, should just remove
        self.beta = beta
        self.sigma = sigma
        self.exp_time = exp_time
        self.supersample_factor = int(supersample_factor)
        self._Omega = (self.vsini/np.sin(np.deg2rad(self.iS)))/(self.rstar*aconst.R_sun.value/1000.)

    def true_anomaly(self,times):
        """
        Calculate the true anomaly
        """
        f = true_anomaly(times,self.T0,self.P,self.aRs,self.i,self.e,self.w)
        return f

    def calc_transit(self,times):
        """
        Calculate transit model of planet
        """
        params = batman.TransitParams()
        params.t0 = self.T0
        params.per = self.P
        params.inc = self.i
        params.rp = self.RpRs
        params.a = self.aRs
        params.ecc = self.e
        params.w = self.w
        params.u = self.u
        params.limb_dark = self.limb_dark
        params.fp = 0.001
        transitmodel = batman.TransitModel(params, times, transittype='primary',exp_time=self.exp_time,
                                         supersample_factor=self.supersample_factor)
        return transitmodel.light_curve(params)

    def planet_XYZ_position(self,times):
        """
        Get the planet XYZ position at times
        """
        X, Y, Z = planet_XYZ_position(times,self.T0,self.P,self.aRs,self.i,self.e,self.w)
        return X, Y, Z

    def Xp(self,times):
        lam, w, i = np.deg2rad(self.lam), np.deg2rad(self.w), np.deg2rad(self.i)
        f = self.true_anomaly(times)
        r = self.aRs*(1.-self.e**2.)/(1.+self.e*np.cos(f)) # distance
        x = -r*np.cos(f+w)*np.cos(lam) + r*np.sin(lam)*np.sin(f+w)*np.cos(i)
        return x

    def evaluate(self,times,base_error=0.):
        sigma = self.sigma
        beta = self.beta
        X = self.Xp(times)
        F = 1.-self.calc_transit(times)
        vp = X*self._Omega*np.sin(self.iS)*self.rstar*aconst.R_sun.value/1000.
        v = -1000.*vp*F*((2.*beta**2.+2.*sigma**2)/(2.*beta**2+sigma**2))**(3./2.) * (1.-(vp**2.)/(2.*beta**2+sigma**2) + (vp**4.)/(2.*(2.*beta**2+sigma**2)**2.))
        # For diagnostics
        self.vp = vp
        self.X = X
        self.F = F
        if base_error >0:
            return v + np.random.normal(loc=0.,scale=base_error,size=len(v))
        else:
            return v

def true_anomaly(time,T0,P,aRs,inc,ecc,omega):
    """
    Uses the batman function to get the true anomaly. Note that some 

    INPUT:
        time - in days
        T0 - in days
        P - in days
        aRs - in a/R*
        inc - in deg
        ecc - eccentricity
        omega - omega in deg

    OUTPUT:
        True anomaly in radians
    """
    # Some of the values here are just dummy values (limb dark etc.) to allow us to get the true anomaly
    params = batman.TransitParams()
    params.t0 = T0                           #time of inferior conjunction
    params.per = P                           #orbital period
    params.rp = 0.1                          #planet radius (in units of stellar radii)
    params.a = aRs                           #semi-major axis (in units of stellar radii)
    params.inc = inc                         #orbital inclination (in degrees)
    params.ecc = ecc                         #eccentricity
    params.w = omega                         #longitude of periastron (in degrees)
    params.u = [0.3,0.3]                     #limb darkening coefficients [u1, u2]
    params.limb_dark = "quadratic"           #limb darkening model
    m = batman.TransitModel(params, time)    #initializes model
    return m.get_true_anomaly()

def planet_XYZ_position(time,T0,P,aRs,inc,ecc,omega):
    """
    Get planet XYZ position

    INPUT:
        time - in days
        T0 - in days
        P - in days
        aRs - in a/R*
        inc - in deg
        ecc - eccentricity
        omega - omega in deg

    OUTPUT:
        X - planet X position
        Y - planet Y position
        Z - planet Z position

    EXAMPLE:
        Rstar = 1.
        Mstar = 1.
        inc = 90.
        ecc = 0.9
        omega = -90.
        P = 1.1
        T0 = 1.1
        Rp = 0.1
        aRs =
        print(aRs)
        x_1, y_1, z_1 = planet_XYZ_position(time,T0,P,aRs,inc,ecc,omega)
        fig, ax = plt.subplots()
        ax.plot(time,x_1)
    """
    f = true_anomaly(time,T0,P,aRs,inc,ecc,omega) # true anomaly in radiance
    omega = np.deg2rad(omega)
    inc = np.deg2rad(inc)
    r = aRs*(1.-ecc**2.)/(1.+ecc*np.cos(f)) # distance
    X = -r*np.cos(omega+f)
    Y = -r*np.sin(omega+f)*np.cos(inc)
    Z = r*np.sin(omega+f)*np.sin(inc)
    return X, Y, Z

def get_rv_curve_peri(times_jd,P,tp,e,omega,K):
    """
    A function to calculate an RV curve as a function of time

    INPUT:
        times_jd - times in bjd
        P - period in days
        tp - T_periastron 
        e - eccentricity
        omega - omega in degrees
        K - semi-amplitude in m/s

    OUTPUT:
        RV curve in m/s
    """
    rvs = radvel.kepler.rv_drive(times_jd,[P,tp,e,np.deg2rad(omega),K])
    return rvs

def get_rv_curve(times_jd,P,tc,e,omega,K):
    """
    A function to calculate an RV curve as a function of time

    INPUT:
        times_jd - times in bjd
        P - period in days
        tc - transit center
        e - eccentricity
        omega - omega in degrees
        K - semi-amplitude in m/s

    OUTPUT:
        RV curve in m/s
    """
    t_peri = radvel.orbit.timetrans_to_timeperi(tc=tc,per=P,ecc=e,omega=np.deg2rad(omega))
    rvs = radvel.kepler.rv_drive(times_jd,[P,t_peri,e,np.deg2rad(omega),K])
    return rvs

def u1_u2_from_q1_q2(q1,q2):
    u1, u2 = 2.*np.sqrt(q1)*q2, np.sqrt(q1)*(1.-2*q2)
    return u1, u2

def b_from_aRs_and_i(aRs,i):
    return aRs*np.cos(np.deg2rad(i))

###########################################################################################################################################################################################
###########################################################################################################################################################################################
###########################################################################################################################################################################################
###########################################################################################################################################################################################
###########################################################################################################################################################################################

#The Transit Addition
def get_lc_batman(times,t0,P,i,rprs,aRs,e,omega,u,supersample_factor=1,exp_time=0.,limbdark="quadratic"):
    """
    Calls BATMAN and returns the transit model

    INPUT:
        times - times to evaluate lc
        t0 - transit midpoint
        P - period in days
        i - inclination in degrees
        rprs - Rp/R*
        aRs - a/R*
        e - eccentricity 
        omega - argument of periastron
        u - limb darkening [u1,u2]
        supersample_factor=1
        exp_time=0. - exposure time in days
        limbdark="quadratic"

    OUTPUT:
        lc - the lightcurve model at *times*
    """
    supersample_factor = int(supersample_factor)
    params = batman.TransitParams()
    params.t0 = t0
    params.per = P
    params.inc = i
    params.rp = rprs
    params.a = aRs
    params.ecc = e
    params.w = omega
    params.u = u
    params.limb_dark = limbdark
    params.fp = 0.001     
    transitmodel = batman.TransitModel(params, times, transittype='primary',
                                       supersample_factor=supersample_factor,
                                       exp_time=exp_time)
    lc = transitmodel.light_curve(params)
    return lc 


def calc_aRs_from_rprs_b_tdur_P(rprs,b,tdur,P):
    """

    INPUT:
        rprs
        b
        tdur - in days
        P - in days

    OUTPUT:
        a/R*

    See equation 8 in Seager et al. 2002
    """
    aa = (1+rprs)**2.
    bb = b*b*(1-np.sin(tdur*np.pi/P))
    cc = (np.sin(tdur*np.pi/P))**2.
    return np.sqrt((aa - bb)/cc)

def i_from_aRs_and_b(aRs,b):
    return np.rad2deg(np.arccos(b/aRs))#aRs*np.cos(np.deg2rad(i))

def aflare1(t, tpeak, fwhm, ampl, upsample=False, uptime=10):
    '''
    The Analytic Flare Model evaluated for a single-peak (classical).
    Reference Davenport et al. (2014) http://arxiv.org/abs/1411.3723
    Use this function for fitting classical flares with most curve_fit
    tools.

    Note: this model assumes the flux before the flare is zero centered

    Function from Allesfitter: https://github.com/MNGuenther/allesfitter

    Parameters
    ----------
    t : 1-d array
        The time array to evaluate the flare over
    tpeak : float
        The time of the flare peak
    fwhm : float
        The "Full Width at Half Maximum", timescale of the flare
    ampl : float
        The amplitude of the flare
    upsample : bool
        If True up-sample the model flare to ensure more precise energies.
    uptime : float
        How many times to up-sample the data (Default is 10)
    Returns
    -------
    flare : 1-d array
        The flux of the flare model evaluated at each time
    '''
    _fr = [1.00000, 1.94053, -0.175084, -2.24588, -1.12498]
    _fd = [0.689008, -1.60053, 0.302963, -0.278318]

    if upsample:
        dt = np.nanmedian(np.diff(t))
        timeup = np.linspace(min(t)-dt, max(t)+dt, t.size * uptime)

        flareup = np.piecewise(timeup, [(timeup<= tpeak) * (timeup-tpeak)/fwhm > -1.,
                                        (timeup > tpeak)],
                                    [lambda x: (_fr[0]+                       # 0th order
                                                _fr[1]*((x-tpeak)/fwhm)+      # 1st order
                                                _fr[2]*((x-tpeak)/fwhm)**2.+  # 2nd order
                                                _fr[3]*((x-tpeak)/fwhm)**3.+  # 3rd order
                                                _fr[4]*((x-tpeak)/fwhm)**4. ),# 4th order
                                     lambda x: (_fd[0]*np.exp( ((x-tpeak)/fwhm)*_fd[1] ) +
                                                _fd[2]*np.exp( ((x-tpeak)/fwhm)*_fd[3] ))]
                                    ) * np.abs(ampl) # amplitude

        # and now downsample back to the original time...
        ## this way might be better, but makes assumption of uniform time bins
        # flare = np.nanmean(flareup.reshape(-1, uptime), axis=1)

        ## This way does linear interp. back to any input time grid
        # flare = np.interp(t, timeup, flareup)

        ## this was uses "binned statistic"
        downbins = np.concatenate((t-dt/2.,[max(t)+dt/2.]))
        flare,_,_ = binned_statistic(timeup, flareup, statistic='mean',
                                 bins=downbins)

    else:
        flare = np.piecewise(t, [(t<= tpeak) * (t-tpeak)/fwhm > -1.,
                                 (t > tpeak)],
                                [lambda x: (_fr[0]+                       # 0th order
                                            _fr[1]*((x-tpeak)/fwhm)+      # 1st order
                                            _fr[2]*((x-tpeak)/fwhm)**2.+  # 2nd order
                                            _fr[3]*((x-tpeak)/fwhm)**3.+  # 3rd order
                                            _fr[4]*((x-tpeak)/fwhm)**4. ),# 4th order
                                 lambda x: (_fd[0]*np.exp( ((x-tpeak)/fwhm)*_fd[1] ) +
                                            _fd[2]*np.exp( ((x-tpeak)/fwhm)*_fd[3] ))]
                                ) * np.abs(ampl) # amplitude

    return flare

def psi_from_lambda_iS_i(lam,iS,i):
    """
    Calculate the true obliquity from lambda, is and i
    
    EXAMPLE:
        N = 10000
        iS = np.random.normal(39,10.,size=N)
        i = np.random.normal(86.858,0.05,size=N)
        lam = np.random.normal(72,30.,size=N)
        psi = psi_from_lambda_iS_i(lam,iS,i)
        corner(zip(iS,i,lam,psi),labels=["$i_*$","$i$","$\lambda$","$\psi$"],show_titles=True,quantiles=[0.16,0.5,0.84]);
    """
    lam = np.deg2rad(lam)
    iS = np.deg2rad(iS)
    i = np.deg2rad(i)
    cospsi = np.sin(iS)*np.cos(lam)*np.sin(i)+np.cos(iS)*np.cos(i)
    psi = np.rad2deg(np.arccos(cospsi))
    return psi

class LPFunctionTransit(object):
    """
    Log-Likelihood function class

    NOTES:
        Based on hpprvi's class, see: https://github.com/hpparvi/exo_tutorials
    """
    def __init__(self,x,y,yerr,file_priors,airmass=None):
        """
        INPUT:
            x - time values in BJD
            y - normalized flux
            yerr - error on normalized flux
            file_priors - prior file name
        """
        self.data= {"x"   : x,
                    "y"   : y,
                    "error"  : yerr,
                    "airmass" : airmass}
        # Setting priors
        self.ps_all = priorset_from_file(file_priors) # all priors
        self.ps_fixed = PriorSet(np.array(self.ps_all.priors)[np.array(self.ps_all.fixed)]) # fixed priorset
        self.ps_vary  = PriorSet(np.array(self.ps_all.priors)[~np.array(self.ps_all.fixed)]) # varying priorset
        self.ps_fixed_dict = {key: val for key, val in zip(self.ps_fixed.labels,self.ps_fixed.args1)}
        print('Reading in priorfile from {}'.format(file_priors))
        print(self.ps_all.df)

    def get_jump_parameter_index(self,lab):
        """
        Get the index of a given label
        """
        return np.where(np.array(self.ps_vary.labels)==lab)[0][0]

    def get_jump_parameter_value(self,pv,lab):
        """
        Get the current value in the argument list 'pv' that has label 'lab'
        """
        # First check if we are actually varying it
        if lab in self.ps_vary.labels:
            return pv[self.get_jump_parameter_index(lab)]
        else:
            # We are not varying it
            return self.ps_fixed_dict[lab]

    def compute_transit_model_main(self,pv,times=None):
        """
        Calls RM model and returns the transit model

        INPUT:
            pv    - parameters passed to the function
            times - times, and array of timestamps

        OUTPUT:
            lc - the lightcurve model at *times*
        """
        T0     =self.get_jump_parameter_value(pv,'t0_p1')
        P      =self.get_jump_parameter_value(pv,'P_p1')
        ii     =self.get_jump_parameter_value(pv,'inc_p1')
        rprs   =self.get_jump_parameter_value(pv,'p_p1')
        aRs    =self.get_jump_parameter_value(pv,'a_p1')
        u1     =self.get_jump_parameter_value(pv,'u1')
        u2     =self.get_jump_parameter_value(pv,'u2')
        u = [u1,u2]
        gamma  =self.get_jump_parameter_value(pv,'gamma')
        e      =self.get_jump_parameter_value(pv,'ecc_p1')
        omega  =self.get_jump_parameter_value(pv,'omega_p1')
        exptime=self.get_jump_parameter_value(pv,'exptime')/86400. # exptime in days
        if times is None:
            times = self.data["x"]
        self.lc = get_lc_batman(times,T0,P,ii,rprs,aRs,e,omega,u,supersample_factor=7,exp_time=exptime,limbdark="quadratic")
        return self.lc + gamma

    def compute_flare(self,pv,times=None):
        tpeak     =self.get_jump_parameter_value(pv,'flare_tpeak')
        fwhm      =self.get_jump_parameter_value(pv,'flare_fwhm')
        A         =self.get_jump_parameter_value(pv,'flare_ampl')
        if times is None:
            times = self.data["x"]
        self.flare = aflare1(times,tpeak,fwhm,A)
        return self.flare


    def compute_transit_model(self,pv,times=None):
        """
        Calls RM model and returns the transit model

        INPUT:
            pv    - parameters passed to the function
            times - times, and array of timestamps

        OUTPUT:
            lc - the lightcurve model at *times*
        """
        T0     =self.get_jump_parameter_value(pv,'t0_p1')
        P      =self.get_jump_parameter_value(pv,'P_p1')
        tdur   =self.get_jump_parameter_value(pv,'tdur_p1')
        b      =self.get_jump_parameter_value(pv,'b_p1')
        rprs   =self.get_jump_parameter_value(pv,'p_p1')
        u1     =self.get_jump_parameter_value(pv,'u1')
        u2     =self.get_jump_parameter_value(pv,'u2')
        #q1     =self.get_jump_parameter_value(pv,'q1')
        #q2     =self.get_jump_parameter_value(pv,'q2')
        #u1, u2 = u1_u2_from_q1_q2(q1,q2)
        u = [u1,u2]
        gamma  =self.get_jump_parameter_value(pv,'gamma')
        e      =self.get_jump_parameter_value(pv,'ecc_p1')
        omega  =self.get_jump_parameter_value(pv,'omega_p1')
        aRs    = calc_aRs_from_rprs_b_tdur_P(rprs,b,tdur,P)
        ii     = i_from_aRs_and_b(aRs,b)
        exptime=self.get_jump_parameter_value(pv,'exptime')/86400. # exptime in days
        if times is None:
            times = self.data["x"]
        self.lc = get_lc_batman(times,T0,P,ii,rprs,aRs,e,omega,u,supersample_factor=7,exp_time=exptime,limbdark="quadratic")
        return self.lc + gamma

    def compute_polynomial_model(self,pv,times=None):
        """
        Compute the polynomial model.  Note that if gammadot and gammadotdot
        are not specified in the priors file, they both default to zero.

        INPUT:
            pv    - a list of parameters (only parameters that are being varied)
            times - times (optional), array of timestamps

        OUTPUT:
            poly - the polynomial model evaluated at 'times' if supplied,
                   otherwise defaults to original data timestamps
        """
        if times is None:
            times = self.data["x"]

        #T0 = self.get_jump_parameter_value(pv,'t0_p1')
        # Use Mean of data instead
        T0 = (self.data['x'][0] + self.data['x'][-1])/2.
        try:
            gammadot = self.get_jump_parameter_value(pv,'gammadot')
        except KeyError as e:
            gammadot = 0
        try:
            gammadotdot = self.get_jump_parameter_value(pv,'gammadotdot')
        except KeyError as e:
            gammadotdot = 0

        self.poly = (
            gammadot * (times - T0) +
            gammadotdot * (times - T0)**2
        )
        return self.poly

    def detrend(self,pv):
        """
        A function to detrend.
        
        INPUT:
        pv    - an array containing a sample draw of the parameters defined in self.lpf.ps
        
        OUTPUT:
        detrend/pv[self.number_pv_baseline] - the additional trend in the data (no including transit)
        """
        detrend = np.zeros(len(self.data["flux"]))

        c_airm     =self.get_jump_parameter_value(pv,'c_airm')
        
        # loop over detrend parameters
        #for i in self.ps.get_param_type_indices(paramtype="detrend"):
        #    #print(i)
        #    detrend += pv[i]*(self.data[self.ps.labels[i]]-1.)
        detrend = c_airm*self.data['airmass']
        return detrend

    def compute_total_model(self,pv,times=None):
        """
        Computes the full RM model (including RM and RV and CB)

        INPUT:
            pv    - a list of parameters (only parameters that are being varied)
            times - times (optional), array of timestamps

        OUTPUT:
            rm - the rm model evaluated at 'times' if supplied, otherwise
                      defaults to original data timestamps

        NOTES:
            see compute_rm_model(), compute_rv_model(),
            compute_polynomial_model()
        """
        #return self.compute_transit_model(pv,times=times) + self.detrend(pv) + self.compute_polynomial_model(pv,times=times) 
        #return self.compute_transit_model(pv,times=times) + self.compute_polynomial_model(pv,times=times) + self.compute_flare(pv,times=times)
        return self.compute_transit_model(pv,times=times) + self.compute_polynomial_model(pv,times=times) 

    def __call__(self,pv):
        """
        Return the log likelihood

        INPUT:
            pv - the input list of varying parameters
        """
        if any(pv < self.ps_vary.pmins) or any(pv>self.ps_vary.pmaxs):
            return -np.inf

        #ii =self.get_jump_parameter_value(pv,'inc_p1')
        #if ii > 90:
        #    return -np.inf
        b =self.get_jump_parameter_value(pv,'b_p1')
        if b < 0.:
            return -np.inf

        ###############
        # Prepare data and model and error for ingestion into likelihood
        y_data = self.data['y']
        y_model = self.compute_total_model(pv)
        # jitter in quadrature
        jitter = self.get_jump_parameter_value(pv,'sigma_rv')
        error = np.sqrt(self.data['error']**2.+jitter**2.)
        ###############

        # Return the log-likelihood
        log_of_priors = self.ps_vary.c_log_prior(pv)
        # Calculate log likelihood
        log_of_model  = ll_normal_ev_py(y_data, y_model, error)
        log_ln = log_of_priors + log_of_model
        return log_ln

class RMFit(object):
    """
    A class that does RM fitting.

    NOTES:
        - Needs to have LPFunction defined
    """
    def __init__(self,LPFunction):
        self.lpf = LPFunction

    def minimize_AMOEBA(self):
        centers = np.array(self.lpf.ps_vary.centers)

        def neg_lpf(pv):
            return -1.*self.lpf(pv)
        self.min_pv = minimize(neg_lpf,centers,method='Nelder-Mead',tol=1e-9,
                                   options={'maxiter': 100000, 'maxfev': 10000, 'disp': True}).x

    def minimize_PyDE(self,npop=100,de_iter=200,mc_iter=1000,mcmc=True,threads=8,maximize=True,plot_priors=True,sample_ball=False,k=None,n=None):
        """
        Minimize using the PyDE

        NOTES:
            see https://github.com/hpparvi/PyDE
        """
        centers = np.array(self.lpf.ps_vary.centers)
        print("Running PyDE Optimizer")
        self.de = de.DiffEvol(self.lpf, self.lpf.ps_vary.bounds, npop, maximize=maximize) # we want to maximize the likelihood
        self.min_pv, self.min_pv_lnval = self.de.optimize(ngen=de_iter)
        print("Optimized using PyDE")
        print("Final parameters:")
        self.print_param_diagnostics(self.min_pv)
        #self.lpf.ps.plot_all(figsize=(6,4),pv=self.min_pv)
        print("LogPost value:",-1*self.min_pv_lnval)
        self.lnl_max  = -1*self.min_pv_lnval-self.lpf.ps_vary.c_log_prior(self.min_pv)
        print("LnL value:",self.lnl_max)
        print("Log priors",self.lpf.ps_vary.c_log_prior(self.min_pv))
        if k is not None and n is not None:
            print("BIC:",stats_help.bic_from_likelihood(self.lnl_max,k,n))
            print("AIC:",stats_help.aic(k,self.lnl_max))
        if mcmc:
            print("Running MCMC")
            self.sampler = emcee.EnsembleSampler(npop, self.lpf.ps_vary.ndim, self.lpf,threads=threads)

            #pb = ipywidgets.IntProgress(max=mc_iter/50)
            #display(pb)
            #val = 0
            print("MCMC iterations=",mc_iter)
            for i,c in enumerate(self.sampler.sample(self.de.population,iterations=mc_iter)):
                print(i,end=" ")
                #if i%50 == 0:
                    #val+=50.
                    #pb.value += 1
            print("Finished MCMC")
            self.min_pv_mcmc = self.get_mean_values_mcmc_posteriors().medvals.values

    def get_mean_values_mcmc_posteriors(self,flatchain=None):
        """
        Get the mean values from the posteriors

            flatchain - if not passed, then will default using the full flatchain (will likely include burnin)

        EXAMPLE:
        """
        if flatchain is None:
            flatchain = self.sampler.flatchain
            print('No flatchain passed, defaulting to using full chains')
        df_list = [ExoMUSE_utils.get_mean_values_for_posterior(flatchain[:,i],label,description) for i,label,description in zip(range(len(self.lpf.ps_vary.descriptions)),self.lpf.ps_vary.labels,self.lpf.ps_vary.descriptions)]
        return pd.concat(df_list)

    def print_param_diagnostics(self,pv):
        """
        A function to print nice parameter diagnostics.
        """
        self.df_diagnostics = pd.DataFrame(zip(self.lpf.ps_vary.labels,self.lpf.ps_vary.centers,self.lpf.ps_vary.bounds[:,0],self.lpf.ps_vary.bounds[:,1],pv,self.lpf.ps_vary.centers-pv),columns=["labels","centers","lower","upper","pv","center_dist"])
        print(self.df_diagnostics.to_string())
        return self.df_diagnostics

    def plot_fit(self,pv=None,times=None):
        """
        Plot the model curve for a given set of parameters pv

        INPUT:
            pv - an array containing a sample draw of the parameters defined in self.lpf.ps_vary
               - will default to best-fit parameters if none are supplied
        """
        if pv is None:
            print('Plotting curve with best-fit values')
            pv = self.min_pv
        x = self.lpf.data['x']
        y = self.lpf.data['y']
        jitter = self.lpf.get_jump_parameter_value(pv,'sigma_rv')
        yerr = np.sqrt(self.lpf.data['error']**2.+jitter**2.)
        model_obs = self.lpf.compute_total_model(pv)
        residuals = y-model_obs
            
        model = self.lpf.compute_total_model(pv)
        residuals = y-model

        # Plot
        nrows = 2
        self.fig, self.ax = plt.subplots(nrows=nrows,sharex=True,figsize=(10,6),gridspec_kw={'height_ratios': [5, 2]})
        self.ax[0].errorbar(x,y,yerr=yerr,elinewidth=1,lw=0,alpha=1,capsize=5,mew=1,marker="o",barsabove=True,markersize=8,label="Data")
        if times is not None:
            model = self.lpf.compute_total_model(pv,times=times)
            self.ax[0].plot(times,model,label="Model",color='crimson')
        else:
            self.ax[0].plot(x,model_obs,label="Model",color='crimson')
            
        self.ax[1].errorbar(x,residuals,yerr=yerr,elinewidth=1,lw=0,alpha=1,capsize=5,mew=1,marker="o",barsabove=True,markersize=8,
                            label="Residuals, std="+str(np.std(residuals)))
        for xx in self.ax:
            xx.minorticks_on()
            xx.legend(loc='lower left',fontsize=9)
            xx.set_ylabel("RV [m/s]",labelpad=2)
        self.ax[-1].set_xlabel("Time (BJD)",labelpad=2)
        self.ax[0].set_title("RM Effect")
        self.fig.subplots_adjust(wspace=0.05,hspace=0.05)

    def plot_mcmc_fit(self,times=None):
        df = self.get_mean_values_mcmc_posteriors()
        print('Plotting curve with best-fit mcmc values')
        self.plot_fit(pv=df.medvals.values)

#RM Effect
class LPFunctionRM(object):
    """
    Log-Likelihood function class

    NOTES:
        Based on hpprvi's class, see: https://github.com/hpparvi/exo_tutorials
    """
    def __init__(self,x,y,yerr,file_priors):
        """
        INPUT:
            x - time values in BJD
            y - y values in m/s
            yerr - yerr values in m/s
            file_priors - prior file name
        """
        self.data= {"x"   : x,
                    "y"   : y,
                    "error"  : yerr}
        # Setting priors
        self.ps_all = priorset_from_file(file_priors) # all priors
        self.ps_fixed = PriorSet(np.array(self.ps_all.priors)[np.array(self.ps_all.fixed)]) # fixed priorset
        self.ps_vary  = PriorSet(np.array(self.ps_all.priors)[~np.array(self.ps_all.fixed)]) # varying priorset
        self.ps_fixed_dict = {key: val for key, val in zip(self.ps_fixed.labels,self.ps_fixed.args1)}
        print('Reading in priorfile from {}'.format(file_priors))
        print(self.ps_all.df)

    def get_jump_parameter_index(self,lab):
        """
        Get the index of a given label
        """
        return np.where(np.array(self.ps_vary.labels)==lab)[0][0]

    def get_jump_parameter_value(self,pv,lab):
        """
        Get the current value in the argument list 'pv' that has label 'lab'
        """
        # First check if we are actually varying it
        if lab in self.ps_vary.labels:
            return pv[self.get_jump_parameter_index(lab)]
        else:
            # We are not varying it
            return self.ps_fixed_dict[lab]

    def compute_rm_model(self,pv,times=None):
        """
        Calls RM model and returns the transit model

        INPUT:
            pv    - parameters passed to the function
            times - times, and array of timestamps

        OUTPUT:
            lc - the lightcurve model at *times*
        """
        T0     =self.get_jump_parameter_value(pv,'t0_p1')
        P      =self.get_jump_parameter_value(pv,'P_p1')
        lam    =self.get_jump_parameter_value(pv,'lam_p1')
        vsini  =self.get_jump_parameter_value(pv,'vsini')
        ii     =self.get_jump_parameter_value(pv,'inc_p1')
        rprs   =self.get_jump_parameter_value(pv,'p_p1')
        aRs    =self.get_jump_parameter_value(pv,'a_p1')
        u1     =self.get_jump_parameter_value(pv,'u1')
        u2     =self.get_jump_parameter_value(pv,'u2')
        gamma  =self.get_jump_parameter_value(pv,'gamma')
        beta   =self.get_jump_parameter_value(pv,'vbeta')
        #sigma  =self.get_jump_parameter_value(pv,'sigma')
        sigma  = vsini /1.31 # assume sigma is vsini/1.31 (see Hirano et al. 2010, 2011)
        e      =self.get_jump_parameter_value(pv,'ecc_p1')
        omega  =self.get_jump_parameter_value(pv,'omega_p1')
        exptime=self.get_jump_parameter_value(pv,'exptime')/86400. # exptime in days
        if times is None:
            times = self.data["x"]
        self.RH = RMHirano(lam,vsini,P,T0,aRs,ii,rprs,e,omega,[u1,u2],beta,
                            sigma,supersample_factor=7,exp_time=exptime,limb_dark='quadratic')
        self.rm = self.RH.evaluate(times)
        return self.rm

    def compute_cb_model(self,pv,times=None):
        """
        Compute convective blueshift model

        NOTES:
            See Shporer & Brown 2011
        """
        ################
        # adding v_cb
        if times is None:
            times = self.data["x"]
        vcb    =self.get_jump_parameter_value(pv,'vcb')
        if vcb!=0:
            T0     =self.get_jump_parameter_value(pv,'t0_p1')
            P      =self.get_jump_parameter_value(pv,'P_p1')
            ii     =self.get_jump_parameter_value(pv,'inc_p1')
            rprs   =self.get_jump_parameter_value(pv,'p_p1')
            aRs    =self.get_jump_parameter_value(pv,'a_p1')
            u1     =self.get_jump_parameter_value(pv,'u1')
            u2     =self.get_jump_parameter_value(pv,'u2')
            e      =self.get_jump_parameter_value(pv,'ecc_p1')
            omega  =self.get_jump_parameter_value(pv,'omega_p1')

            # Calculate separation of centers
            x_1, y_1, z_1 = planet_XYZ_position(times,T0,P,aRs,ii,e,omega)
            ds = np.sqrt(x_1**2.+y_1**2.)

            self.vels_cb = convective_blueshift.cb_limbdark(ds,rprs,u1,u2,vcb,epsabs=1.49e-1,epsrel=1.49e-1)
            return self.vels_cb
        else:
            return np.zeros(len(times))
        ################
        
    def compute_rv_model(self,pv,times=None):
        """
        Compute the RV model

        INPUT:
            pv    - a list of parameters (only parameters that are being varied)
            times - times (optional), array of timestamps

        OUTPUT:
            rv - the rv model evaluated at 'times' if supplied, otherwise
                      defaults to original data timestamps
        """
        if times is None:
            times = self.data["x"]
        T0      = self.get_jump_parameter_value(pv,'t0_p1')
        P       = self.get_jump_parameter_value(pv,'P_p1')
        gamma   = self.get_jump_parameter_value(pv,'gamma')
        K       = self.get_jump_parameter_value(pv,'K_p1')
        e       = self.get_jump_parameter_value(pv,'ecc_p1')
        w       = self.get_jump_parameter_value(pv,'omega_p1')
        self.rv = get_rv_curve(times,P=P,tc=T0,e=e,omega=w,K=K)+gamma
        return self.rv

    def compute_polynomial_model(self,pv,times=None):
        """
        Compute the polynomial model.  Note that if gammadot and gammadotdot
        are not specified in the priors file, they both default to zero.

        INPUT:
            pv    - a list of parameters (only parameters that are being varied)
            times - times (optional), array of timestamps

        OUTPUT:
            poly - the polynomial model evaluated at 'times' if supplied,
                   otherwise defaults to original data timestamps
        """
        if times is None:
            times = self.data["x"]

        #T0 = self.get_jump_parameter_value(pv,'t0_p1')
        # Use Mean of data instead
        T0 = (self.data['x'][0] + self.data['x'][-1])/2.
        try:
            gammadot = self.get_jump_parameter_value(pv,'gammadot')
        except KeyError as e:
            gammadot = 0
        try:
            gammadotdot = self.get_jump_parameter_value(pv,'gammadotdot')
        except KeyError as e:
            gammadotdot = 0

        self.poly = (
            gammadot * (times - T0) +
            gammadotdot * (times - T0)**2
        )
        return self.poly

    def compute_total_model(self,pv,times=None):
        """
        Computes the full RM model (including RM and RV and CB)

        INPUT:
            pv    - a list of parameters (only parameters that are being varied)
            times - times (optional), array of timestamps

        OUTPUT:
            rm - the rm model evaluated at 'times' if supplied, otherwise
                      defaults to original data timestamps

        NOTES:
            see compute_rm_model(), compute_rv_model(),
            compute_polynomial_model()
        """
        return (
            self.compute_rm_model(pv,times=times) +
            self.compute_rv_model(pv,times=times) +
            self.compute_polynomial_model(pv,times=times) +
            self.compute_cb_model(pv,times=times)
        )

    def __call__(self,pv):
        """
        Return the log likelihood

        INPUT:
            pv - the input list of varying parameters
        """
        if any(pv < self.ps_vary.pmins) or any(pv>self.ps_vary.pmaxs):
            return -np.inf

        ii =self.get_jump_parameter_value(pv,'inc_p1')
        if ii > 90:
            return -np.inf

        ###############
        # Prepare data and model and error for ingestion into likelihood
        y_data = self.data['y']
        y_model = self.compute_total_model(pv)
        # jitter in quadrature
        jitter = self.get_jump_parameter_value(pv,'sigma_rv')
        error = np.sqrt(self.data['error']**2.+jitter**2.)
        ###############

        # Return the log-likelihood
        log_of_priors = self.ps_vary.c_log_prior(pv)
        # Calculate log likelihood
        log_of_model  = ll_normal_ev_py(y_data, y_model, error)
        log_ln = log_of_priors + log_of_model
        return log_ln

#RM + Light Curve Joint Fitting
class LPFunction1RM1Phot(object):
    """
    Log-Likelihood function class
    """
    def __init__(self,inp,file_priors):
        """
        INPUT:
            x - time values in BJD
            y - y values in m/s
            yerr - yerr values in m/s
            file_priors - prior file name
        """
        self.data_rv1= {"time"   : inp['time_rv1'],  
                        "y"      : inp['rv_rv1'],   
                        "error"  : inp['e_rv1']}
        self.data_tr1= {"time"   : inp['time_tr1'],  
                        "y"      : inp['f_tr1'],   
                        "error"  : inp['e_tr1']}
        # Setting priors
        self.ps_all = priorset_from_file(file_priors) # all priors
        self.ps_fixed = PriorSet(np.array(self.ps_all.priors)[np.array(self.ps_all.fixed)]) # fixed priorset
        self.ps_vary  = PriorSet(np.array(self.ps_all.priors)[~np.array(self.ps_all.fixed)]) # varying priorset
        self.ps_fixed_dict = {key: val for key, val in zip(self.ps_fixed.labels,self.ps_fixed.args1)}
        print('Reading in priorfile from {}'.format(file_priors))
        print(self.ps_all.df)
        
    def get_jump_parameter_index(self,lab):
        """
        Get the index of a given label
        """
        return np.where(np.array(self.ps_vary.labels)==lab)[0][0]
    
    def get_jump_parameter_value(self,pv,lab):
        """
        Get the current value in the argument list 'pv' that has label 'lab'
        """
        # First check if we are actually varying it
        if lab in self.ps_vary.labels:
            return pv[self.get_jump_parameter_index(lab)]
        else:
            # We are not varying it
            return self.ps_fixed_dict[lab]
        
    def compute_rm_model(self,pv,times1=None):
        """
        Calls RM model and returns the transit model
        
        INPUT:
            pv    - parameters passed to the function 
            times - times, and array of timestamps 
        
        OUTPUT:
            lc - the lightcurve model at *times*
        """
        T0      =self.get_jump_parameter_value(pv,'t0_p1')
        P       =self.get_jump_parameter_value(pv,'P_p1')
        lam     =self.get_jump_parameter_value(pv,'lam_p1')
        #cosi    = self.get_jump_parameter_value(pv,'cosi')
        #R       = self.get_jump_parameter_value(pv,'rstar')
        #Prot    = self.get_jump_parameter_value(pv,'Prot')
        #veq = (2.*np.pi*R*aconst.R_sun.value)/(Prot*86400.*1000.) # km/s
        #vsini   = veq*np.sqrt(1.-cosi**2.)
        vsini   =self.get_jump_parameter_value(pv,'vsini') 
        ii      =self.get_jump_parameter_value(pv,'inc_p1')
        rprs    =self.get_jump_parameter_value(pv,'p_p1')
        aRs     =self.get_jump_parameter_value(pv,'a_p1')
        u1      =self.get_jump_parameter_value(pv,'u1_rv1')
        u2      =self.get_jump_parameter_value(pv,'u2_rv1')
        #q1      =self.get_jump_parameter_value(pv,'q1')
        #q2      =self.get_jump_parameter_value(pv,'q2')
        #u1, u2 = u1_u2_from_q1_q2(q1,q2)
        beta    =self.get_jump_parameter_value(pv,'vbeta')
        #sigma   =self.get_jump_parameter_value(pv,'sigma')
        sigma   = vsini /1.31 # assume sigma is vsini/1.31 (see Hirano et al. 2010, 2011)
        e       =self.get_jump_parameter_value(pv,'ecc_p1')
        omega   =self.get_jump_parameter_value(pv,'omega_p1')
        exptime1=self.get_jump_parameter_value(pv,'exptime_rv1')/86400. # exptime in days
        if times1 is None: 
            times1 = self.data_rv1["time"]
        self.rm1 = RMHirano(lam,vsini,P,T0,aRs,ii,rprs,e,omega,[u1,u2],beta,
                            sigma,supersample_factor=7,exp_time=exptime1,limb_dark='quadratic').evaluate(times1)
        return self.rm1

    def compute_transit_model(self,pv,times1=None):
        """
        Calls BATMAN and returns the transit model
        """
        T0     =self.get_jump_parameter_value(pv,'t0_p1')
        P      =self.get_jump_parameter_value(pv,'P_p1')
        ii     =self.get_jump_parameter_value(pv,'inc_p1')
        rprs   =self.get_jump_parameter_value(pv,'p_p1')
        aRs    =self.get_jump_parameter_value(pv,'a_p1')
        e      =self.get_jump_parameter_value(pv,'ecc_p1')
        omega  =self.get_jump_parameter_value(pv,'omega_p1')
        u1     =self.get_jump_parameter_value(pv,'u1_tr1')
        u2     =self.get_jump_parameter_value(pv,'u1_tr1')
        u = [u1,u2]
        mflux  =self.get_jump_parameter_value(pv,'mflux_tr1')
        exptime=self.get_jump_parameter_value(pv,'exptime_tr1')/86400. # exptime in days
        if times1 is None:
            times1 = self.data_tr1["time"]
        self.lc = get_lc_batman(times1,T0,P,ii,rprs,aRs,e,omega,u,supersample_factor=7,exp_time=exptime,limbdark="quadratic")*mflux
        return self.lc

    def compute_polynomial_transit_model(self,pv,times1=None):
        """
        Compute the polynomial model.  Note that if gammadot and gammadotdot
        are not specified in the priors file, they both default to zero.

        INPUT:
            pv    - a list of parameters (only parameters that are being varied)
            times - times (optional), array of timestamps

        OUTPUT:
            poly - the polynomial model evaluated at 'times' if supplied,
                   otherwise defaults to original data timestamps
        """
        if times1 is None:
            times1 = self.data_tr1["time"]

        T0 = (self.data_tr1['time'][0] + self.data_tr1['time'][-1])/2.
        try:
            mfluxdot = self.get_jump_parameter_value(pv,'mfluxdot_tr1')
        except KeyError as e:
            mfluxdot = 0
        try:
            mfluxdotdot = self.get_jump_parameter_value(pv,'mfluxdotdot_tr1')
        except KeyError as e:
            mfluxdotdot = 0

        self.poly = (
            mfluxdot * (times1 - T0) +
            mfluxdotdot * (times1 - T0)**2
        )
        return self.poly

    def compute_rv_model(self,pv,times1=None):
        """
        Compute the RV model

        INPUT:
            pv    - a list of parameters (only parameters that are being varied)
            times - times (optional), array of timestamps 
        
        OUTPUT:
            rv - the rv model evaluated at 'times' if supplied, otherwise 
                      defaults to original data timestamps
        """
        if times1 is None: times1 = self.data_rv1["time"]
        T0      = self.get_jump_parameter_value(pv,'t0_p1')
        P       = self.get_jump_parameter_value(pv,'P_p1')
        gamma1  = self.get_jump_parameter_value(pv,'gamma_rv1')
        K       = self.get_jump_parameter_value(pv,'K_p1')
        e       = self.get_jump_parameter_value(pv,'ecc_p1')
        w       = self.get_jump_parameter_value(pv,'omega_p1')
        self.rv1 = get_rv_curve(times1,P=P,tc=T0,e=e,omega=w,K=K)+gamma1
        return self.rv1
        
    def compute_total_model(self,pv,times_tr1=None,times_rv1=None):
        """
        Computes the full RM model (including RM and RV and CB)

        INPUT:
            pv    - a list of parameters (only parameters that are being varied)
            times - times (optional), array of timestamps 
        
        OUTPUT:
            rm - the rm model evaluated at 'times' if supplied, otherwise 
                      defaults to original data timestamps

        NOTES:
            see compute_rm_model(), compute_rv_model()
        """
        rv1 = self.compute_rm_model(pv,times1=times_rv1) + self.compute_rv_model(pv,times1=times_rv1) 
        ff1 = self.compute_transit_model(pv,times1=times_tr1)
        ff1_trend = self.compute_polynomial_transit_model(pv,times1=times_tr1)
        return rv1, ff1 + ff1_trend

    def __call__(self,pv):
        """
        Return the log likelihood

        INPUT:
            pv - the input list of varying parameters
        """
        if any(pv < self.ps_vary.pmins) or any(pv>self.ps_vary.pmaxs):
            return -np.inf

        ii =self.get_jump_parameter_value(pv,'inc_p1')
        if ii > 90:
            return -np.inf

        ###############
        # Prepare data and model and error for ingestion into likelihood
        #y_data = self.data['y']
        rv1, ff1 = self.compute_total_model(pv)
        # jitter in quadrature
        jitter_rv1 = self.get_jump_parameter_value(pv,'sigma_rv1')
        jitter_tr1 = self.get_jump_parameter_value(pv,'sigma_tr1')
        error_rv1 = np.sqrt(self.data_rv1['error']**2.+jitter_rv1**2.)
        error_tr1 = np.sqrt(self.data_tr1['error']**2.+jitter_tr1**2.)
        ###############

        # Return the log-likelihood
        log_of_priors = self.ps_vary.c_log_prior(pv)
        # Calculate log likelihood
        #log_of_model  = ll_normal_ev_py(y_data, y_model, error)
        log_of_model1  = ll_normal_ev_py(self.data_rv1["y"], rv1, error_rv1)
        log_of_model2  = ll_normal_ev_py(self.data_tr1["y"], ff1, error_tr1)
        log_ln = log_of_priors + log_of_model1 + log_of_model2 
        return log_ln
                    

def transit_time_from_ephem(tobs,P,T0,verbose=False):
    """
    Get difference between observed transit time and the 
    
    INPUT:
        tobs - observed transit time
        P - period in days
        T0 - ephem transit midpoint
        
    OUTPUT:
        T_expected in days
        
    EXAMPLE:
        P = 3.336649
        T0 = 2456340.72559
        x = [2456340.72559,2458819.8585888706,2458839.8782432866]
        y = np.copy(x)
        yerr = [0.00010,0.0002467413,0.0004495690]
        model = get_expected_transit_time_from_ephem(x,P,T0,verbose=True)

        fig, ax = plt.subplots(dpi=200)
        ax.errorbar(x,y,yerr,marker='o',lw=0,mew=0.5,capsize=4,elinewidth=0.5,)
    """
    n = np.round((tobs-T0)/P)
    if verbose:
        print('n={}'.format(n))
    T_expected = P*n+T0
    return T_expected


############################################################################################################################################################################################
############################################################################################################################################################################################
############################################################################################################################################################################################
############################################################################################################################################################################################

class TransitFit(object):
    """
    A class that does Transit fitting.

    NOTES:
        - Needs to have LPFunction defined
    """
    def __init__(self,LPFunction):
        self.lpf = LPFunction

    def minimize_AMOEBA(self):
        centers = np.array(self.lpf.ps_vary.centers)

        def neg_lpf(pv):
            return -1.*self.lpf(pv)
        self.min_pv = minimize(neg_lpf,centers,method='Nelder-Mead',tol=1e-9,
                                   options={'maxiter': 100000, 'maxfev': 10000, 'disp': True}).x

    def minimize_PyDE(self,npop=100,de_iter=200,mc_iter=1000,mcmc=True,threads=8,maximize=True,plot_priors=True,sample_ball=False,k=None,n=None):
        """
        Minimize using the PyDE

        NOTES:
            see https://github.com/hpparvi/PyDE
        """
        centers = np.array(self.lpf.ps_vary.centers)
        print("Running PyDE Optimizer")
        self.de = de.DiffEvol(self.lpf, self.lpf.ps_vary.bounds, npop, maximize=maximize) # we want to maximize the likelihood
        self.min_pv, self.min_pv_lnval = self.de.optimize(ngen=de_iter)
        print("Optimized using PyDE")
        print("Final parameters:")
        self.print_param_diagnostics(self.min_pv)
        #self.lpf.ps.plot_all(figsize=(6,4),pv=self.min_pv)
        print("LogPost value:",-1*self.min_pv_lnval)
        self.lnl_max  = -1*self.min_pv_lnval-self.lpf.ps_vary.c_log_prior(self.min_pv)
        print("LnL value:",self.lnl_max)
        print("Log priors",self.lpf.ps_vary.c_log_prior(self.min_pv))
        if k is not None and n is not None:
            print("BIC:",stats_help.bic_from_likelihood(self.lnl_max,k,n))
            print("AIC:",stats_help.aic(k,self.lnl_max))
        if mcmc:
            print("Running MCMC")
            self.sampler = emcee.EnsembleSampler(npop, self.lpf.ps_vary.ndim, self.lpf,threads=threads)

            #pb = ipywidgets.IntProgress(max=mc_iter/50)
            #display(pb)
            #val = 0
            print("MCMC iterations=",mc_iter)
            for i,c in enumerate(self.sampler.sample(self.de.population,iterations=mc_iter)):
                print(i,end=" ")
                #if i%50 == 0:
                    #val+=50.
                    #pb.value += 1
            print("Finished MCMC")
            self.min_pv_mcmc = self.get_mean_values_mcmc_posteriors().medvals.values

    def get_mean_values_mcmc_posteriors(self,flatchain=None):
        """
        Get the mean values from the posteriors

            flatchain - if not passed, then will default using the full flatchain (will likely include burnin)

        EXAMPLE:
        """
        if flatchain is None:
            flatchain = self.sampler.flatchain
            print('No flatchain passed, defaulting to using full chains')
        df_list = [ExoMUSE_utils.get_mean_values_for_posterior(flatchain[:,i],label,description) for i,label,description in zip(range(len(self.lpf.ps_vary.descriptions)),self.lpf.ps_vary.labels,self.lpf.ps_vary.descriptions)]
        return pd.concat(df_list)

    def print_param_diagnostics(self,pv):
        """
        A function to print nice parameter diagnostics.
        """
        self.df_diagnostics = pd.DataFrame(zip(self.lpf.ps_vary.labels,self.lpf.ps_vary.centers,self.lpf.ps_vary.bounds[:,0],self.lpf.ps_vary.bounds[:,1],pv,self.lpf.ps_vary.centers-pv),columns=["labels","centers","lower","upper","pv","center_dist"])
        print(self.df_diagnostics.to_string())
        return self.df_diagnostics

    def plot_fit(self,pv=None,times=None):
        """
        Plot the model curve for a given set of parameters pv

        INPUT:
            pv - an array containing a sample draw of the parameters defined in self.lpf.ps_vary
               - will default to best-fit parameters if none are supplied
        """
        if pv is None:
            print('Plotting curve with best-fit values')
            pv = self.min_pv
        x = self.lpf.data['x']
        y = self.lpf.data['y']
        jitter = self.lpf.get_jump_parameter_value(pv,'sigma_rv')
        yerr = np.sqrt(self.lpf.data['error']**2.+jitter**2.)
        model_obs = self.lpf.compute_total_model(pv)
        residuals = y-model_obs
            
        model = self.lpf.compute_total_model(pv)
        residuals = y-model

        # Plot
        nrows = 2
        self.fig, self.ax = plt.subplots(nrows=nrows,sharex=True,figsize=(10,6),gridspec_kw={'height_ratios': [5, 2]})
        self.ax[0].errorbar(x,y,yerr=yerr,elinewidth=1,lw=0,alpha=1,capsize=5,mew=1,marker="o",barsabove=True,markersize=8,label="Data")
        if times is not None:
            model = self.lpf.compute_total_model(pv,times=times)
            self.ax[0].plot(times,model,label="Model",color='crimson')
        else:
            self.ax[0].plot(x,model_obs,label="Model",color='crimson')
            
        self.ax[1].errorbar(x,residuals,yerr=yerr,elinewidth=1,lw=0,alpha=1,capsize=5,mew=1,marker="o",barsabove=True,markersize=8,
                            label="Residuals, std="+str(np.std(residuals)))
        for xx in self.ax:
            xx.minorticks_on()
            xx.legend(loc='lower left',fontsize=9)
            xx.set_ylabel("RV [m/s]",labelpad=2)
        self.ax[-1].set_xlabel("Time (BJD)",labelpad=2)
        self.ax[0].set_title("RM Effect")
        self.fig.subplots_adjust(wspace=0.05,hspace=0.05)

    def plot_mcmc_fit(self,times=None):
        df = self.get_mean_values_mcmc_posteriors()
        print('Plotting curve with best-fit mcmc values')
        self.plot_fit(pv=df.medvals.values)

class LPFunctionLC(object):
    """
    Log-Likelihood function class

    NOTES:
        Based on hpprvi's class, see: https://github.com/hpparvi/exo_tutorials
    """
    def __init__(self,x,y,yerr,file_priors,airmass=None):
        """
        INPUT:
            x - time values in BJD
            y - normalized flux
            yerr - error on normalized flux
            file_priors - prior file name
        """
        self.data= {"x"   : x,
                    "y"   : y,
                    "error"  : yerr,
                    "airmass" : airmass}
        # Setting priors
        self.ps_all = priorset_from_file(file_priors) # all priors
        self.ps_fixed = PriorSet(np.array(self.ps_all.priors)[np.array(self.ps_all.fixed)]) # fixed priorset
        self.ps_vary  = PriorSet(np.array(self.ps_all.priors)[~np.array(self.ps_all.fixed)]) # varying priorset
        self.ps_fixed_dict = {key: val for key, val in zip(self.ps_fixed.labels,self.ps_fixed.args1)}
        print('Reading in priorfile from {}'.format(file_priors))
        print(self.ps_all.df)

    def get_jump_parameter_index(self,lab):
        """
        Get the index of a given label
        """
        return np.where(np.array(self.ps_vary.labels)==lab)[0][0]

    def get_jump_parameter_value(self,pv,lab):
        """
        Get the current value in the argument list 'pv' that has label 'lab'
        """
        # First check if we are actually varying it
        if lab in self.ps_vary.labels:
            return pv[self.get_jump_parameter_index(lab)]
        else:
            # We are not varying it
            return self.ps_fixed_dict[lab]

    def compute_transit_model_main(self,pv,times=None):
        """
        Calls RM model and returns the transit model

        INPUT:
            pv    - parameters passed to the function
            times - times, and array of timestamps

        OUTPUT:
            lc - the lightcurve model at *times*
        """
        T0     =self.get_jump_parameter_value(pv,'t0_p1')
        P      =self.get_jump_parameter_value(pv,'P_p1')
        ii     =self.get_jump_parameter_value(pv,'inc_p1')
        rprs   =self.get_jump_parameter_value(pv,'p_p1')
        aRs    =self.get_jump_parameter_value(pv,'a_p1')
        u1     =self.get_jump_parameter_value(pv,'u1')
        u2     =self.get_jump_parameter_value(pv,'u2')
        u = [u1,u2]
        gamma  =self.get_jump_parameter_value(pv,'gamma')
        e      =self.get_jump_parameter_value(pv,'ecc_p1')
        omega  =self.get_jump_parameter_value(pv,'omega_p1')
        exptime=self.get_jump_parameter_value(pv,'exptime')/86400. # exptime in days
        if times is None:
            times = self.data["x"]
        self.lc = get_lc_batman(times,T0,P,ii,rprs,aRs,e,omega,u,supersample_factor=7,exp_time=exptime,limbdark="quadratic")
        return self.lc + gamma

    def compute_transit_model(self,pv,times=None):
        """
        Calls RM model and returns the transit model

        INPUT:
            pv    - parameters passed to the function
            times - times, and array of timestamps

        OUTPUT:
            lc - the lightcurve model at *times*
        """
        T0     =self.get_jump_parameter_value(pv,'t0_p1')
        P      =self.get_jump_parameter_value(pv,'P_p1')
        tdur   =self.get_jump_parameter_value(pv,'tdur_p1')
        b      =self.get_jump_parameter_value(pv,'b_p1')
        rprs   =self.get_jump_parameter_value(pv,'p_p1')
        u1     =self.get_jump_parameter_value(pv,'u1')
        u2     =self.get_jump_parameter_value(pv,'u2')
        #q1     =self.get_jump_parameter_value(pv,'q1')
        #q2     =self.get_jump_parameter_value(pv,'q2')
        #u1, u2 = u1_u2_from_q1_q2(q1,q2)
        u = [u1,u2]
        gamma  =self.get_jump_parameter_value(pv,'gamma')
        e      =self.get_jump_parameter_value(pv,'ecc_p1')
        omega  =self.get_jump_parameter_value(pv,'omega_p1')
        aRs    = calc_aRs_from_rprs_b_tdur_P(rprs,b,tdur,P)
        ii     = i_from_aRs_and_b(aRs,b)
        exptime=self.get_jump_parameter_value(pv,'exptime')/86400. # exptime in days
        if times is None:
            times = self.data["x"]
        self.lc = get_lc_batman(times,T0,P,ii,rprs,aRs,e,omega,u,supersample_factor=7,exp_time=exptime,limbdark="quadratic")
        return self.lc + gamma

    def compute_polynomial_model(self,pv,times=None):
        """
        Compute the polynomial model.  Note that if gammadot and gammadotdot
        are not specified in the priors file, they both default to zero.

        INPUT:
            pv    - a list of parameters (only parameters that are being varied)
            times - times (optional), array of timestamps

        OUTPUT:
            poly - the polynomial model evaluated at 'times' if supplied,
                   otherwise defaults to original data timestamps
        """
        if times is None:
            times = self.data["x"]

        #T0 = self.get_jump_parameter_value(pv,'t0_p1')
        # Use Mean of data instead
        T0 = (self.data['x'][0] + self.data['x'][-1])/2.
        try:
            gammadot = self.get_jump_parameter_value(pv,'gammadot')
        except KeyError as e:
            gammadot = 0
        try:
            gammadotdot = self.get_jump_parameter_value(pv,'gammadotdot')
        except KeyError as e:
            gammadotdot = 0

        self.poly = (
            gammadot * (times - T0) +
            gammadotdot * (times - T0)**2
        )
        return self.poly

    def detrend(self,pv):
        """
        A function to detrend.
        
        INPUT:
        pv    - an array containing a sample draw of the parameters defined in self.lpf.ps
        
        OUTPUT:
        detrend/pv[self.number_pv_baseline] - the additional trend in the data (no including transit)
        """
        detrend = np.zeros(len(self.data["flux"]))

        c_airm     =self.get_jump_parameter_value(pv,'c_airm')
        
        # loop over detrend parameters
        #for i in self.ps.get_param_type_indices(paramtype="detrend"):
        #    #print(i)
        #    detrend += pv[i]*(self.data[self.ps.labels[i]]-1.)
        detrend = c_airm*self.data['airmass']
        return detrend

    def compute_total_model(self,pv,times=None):
        """
        Computes the full RM model (including RM and RV and CB)

        INPUT:
            pv    - a list of parameters (only parameters that are being varied)
            times - times (optional), array of timestamps

        OUTPUT:
            rm - the rm model evaluated at 'times' if supplied, otherwise
                      defaults to original data timestamps

        NOTES:
            see compute_rm_model(), compute_rv_model(),
            compute_polynomial_model()
        """
        #return self.compute_transit_model(pv,times=times) + self.detrend(pv) + self.compute_polynomial_model(pv,times=times) 
        #return self.compute_transit_model(pv,times=times) + self.compute_polynomial_model(pv,times=times) + self.compute_flare(pv,times=times)
        return self.compute_transit_model(pv,times=times) + self.compute_polynomial_model(pv,times=times) 

    def __call__(self,pv):
        """
        Return the log likelihood

        INPUT:
            pv - the input list of varying parameters
        """
        if any(pv < self.ps_vary.pmins) or any(pv>self.ps_vary.pmaxs):
            return -np.inf

        #ii =self.get_jump_parameter_value(pv,'inc_p1')
        #if ii > 90:
        #    return -np.inf
        b =self.get_jump_parameter_value(pv,'b_p1')
        if b < 0.:
            return -np.inf

        ###############
        # Prepare data and model and error for ingestion into likelihood
        y_data = self.data['y']
        y_model = self.compute_total_model(pv)
        # jitter in quadrature
        jitter = self.get_jump_parameter_value(pv,'sigma_rv')
        error = np.sqrt(self.data['error']**2.+jitter**2.)
        ###############

        # Return the log-likelihood
        log_of_priors = self.ps_vary.c_log_prior(pv)
        # Calculate log likelihood
        log_of_model  = ll_normal_ev_py(y_data, y_model, error)
        log_ln = log_of_priors + log_of_model
        return log_ln

#Joint Transit & RV Fitting
#Likelihood of RV and LC
class LPFunctionRVLC(object):
    """
    Log-Likelihood function class

    NOTES:
        Based on hpprvi's class, see: https://github.com/hpparvi/exo_tutorials
    """
    def __init__(self,inp,file_priors):
        """
        INPUT:
            x - time values in BJD
            y - y values in m/s
            yerr - yerr values in m/s
            file_priors - prior file name
        """
        self.data_rv1= {"time"   : inp['time_rv1'],  
                        "y"      : inp['rv_rv1'],   
                        "error"  : inp['e_rv1']}
        self.data_tr1= {"time"   : inp['time_tr1'],  
                        "y"      : inp['f_tr1'],   
                        "error"  : inp['e_tr1']}
        # Setting priors
        self.ps_all = priorset_from_file(file_priors) # all priors
        self.ps_fixed = PriorSet(np.array(self.ps_all.priors)[np.array(self.ps_all.fixed)]) # fixed priorset
        self.ps_vary  = PriorSet(np.array(self.ps_all.priors)[~np.array(self.ps_all.fixed)]) # varying priorset
        self.ps_fixed_dict = {key: val for key, val in zip(self.ps_fixed.labels,self.ps_fixed.args1)}
        print('Reading in priorfile from {}'.format(file_priors))
        print(self.ps_all.df)

    def get_jump_parameter_index(self,lab):
        """
        Get the index of a given label
        """
        return np.where(np.array(self.ps_vary.labels)==lab)[0][0]

    def get_jump_parameter_value(self,pv,lab):
        """
        Get the current value in the argument list 'pv' that has label 'lab'
        """
        # First check if we are actually varying it
        if lab in self.ps_vary.labels:
            return pv[self.get_jump_parameter_index(lab)]
        else:
            # We are not varying it
            return self.ps_fixed_dict[lab]

    #LIGHT CURVE MODELLING
    def compute_transit_model_main(self,pv,times=None):
        """
        Calls RM model and returns the transit model

        INPUT:
            pv    - parameters passed to the function
            times - times, and array of timestamps

        OUTPUT:
            lc - the lightcurve model at *times*
        """
        T0     =self.get_jump_parameter_value(pv,'t0_p1')
        P      =self.get_jump_parameter_value(pv,'P_p1')
        ii     =self.get_jump_parameter_value(pv,'inc_p1')
        rprs   =self.get_jump_parameter_value(pv,'p_p1')
        aRs    =self.get_jump_parameter_value(pv,'a_p1')
        u1     =self.get_jump_parameter_value(pv,'u1')
        u2     =self.get_jump_parameter_value(pv,'u2')
        u = [u1,u2]
        gamma  =self.get_jump_parameter_value(pv,'gamma')
        e      =self.get_jump_parameter_value(pv,'ecc_p1')
        omega  =self.get_jump_parameter_value(pv,'omega_p1')
        exptime=self.get_jump_parameter_value(pv,'exptime')/86400. # exptime in days
        if times is None:
            times = self.data_tr1["x"]
        self.lc = get_lc_batman(times,T0,P,ii,rprs,aRs,e,omega,u,supersample_factor=7,exp_time=exptime,limbdark="quadratic")
        return self.lc + gamma

    def compute_flare(self,pv,times=None):
        tpeak     =self.get_jump_parameter_value(pv,'flare_tpeak')
        fwhm      =self.get_jump_parameter_value(pv,'flare_fwhm')
        A         =self.get_jump_parameter_value(pv,'flare_ampl')
        if times is None:
            times = self.data_tr1["x"]
        self.flare = aflare1(times,tpeak,fwhm,A)
        return self.flare


    def compute_transit_model(self,pv,times=None):
        """
        Calls RM model and returns the transit model

        INPUT:
            pv    - parameters passed to the function
            times - times, and array of timestamps

        OUTPUT:
            lc - the lightcurve model at *times*
        """
        T0     =self.get_jump_parameter_value(pv,'t0_p1')
        P      =self.get_jump_parameter_value(pv,'P_p1')
        tdur   =self.get_jump_parameter_value(pv,'tdur_p1')
        b      =self.get_jump_parameter_value(pv,'b_p1')
        rprs   =self.get_jump_parameter_value(pv,'p_p1')
        u1     =self.get_jump_parameter_value(pv,'u1')
        u2     =self.get_jump_parameter_value(pv,'u2')
        #q1     =self.get_jump_parameter_value(pv,'q1')
        #q2     =self.get_jump_parameter_value(pv,'q2')
        #u1, u2 = u1_u2_from_q1_q2(q1,q2)
        u = [u1,u2]
        gamma  =self.get_jump_parameter_value(pv,'gamma')
        e      =self.get_jump_parameter_value(pv,'ecc_p1')
        omega  =self.get_jump_parameter_value(pv,'omega_p1')
        aRs    = calc_aRs_from_rprs_b_tdur_P(rprs,b,tdur,P)
        ii     = i_from_aRs_and_b(aRs,b)
        exptime=self.get_jump_parameter_value(pv,'exptime')/86400. # exptime in days
        if times is None:
            times = self.data_tr1["x"]
        self.lc = get_lc_batman(times,T0,P,ii,rprs,aRs,e,omega,u,supersample_factor=7,exp_time=exptime,limbdark="quadratic")
        return self.lc + gamma

    def compute_polynomial_model_transit(self,pv,times=None):
        """
        Compute the polynomial model.  Note that if gammadot and gammadotdot
        are not specified in the priors file, they both default to zero.

        INPUT:
            pv    - a list of parameters (only parameters that are being varied)
            times - times (optional), array of timestamps

        OUTPUT:
            poly - the polynomial model evaluated at 'times' if supplied,
                   otherwise defaults to original data timestamps
        """
        if times is None:
            times = self.data_tr1["x"]

        #T0 = self.get_jump_parameter_value(pv,'t0_p1')
        # Use Mean of data instead
        T0 = (self.data_tr1['x'][0] + self.data_tr1['x'][-1])/2.
        try:
            gammadot = self.get_jump_parameter_value(pv,'gammadot')
        except KeyError as e:
            gammadot = 0
        try:
            gammadotdot = self.get_jump_parameter_value(pv,'gammadotdot')
        except KeyError as e:
            gammadotdot = 0

        self.poly = (
            gammadot * (times - T0) +
            gammadotdot * (times - T0)**2
        )
        return self.poly

    def detrend(self,pv):
        """
        A function to detrend.
        
        INPUT:
        pv    - an array containing a sample draw of the parameters defined in self.lpf.ps
        
        OUTPUT:
        detrend/pv[self.number_pv_baseline] - the additional trend in the data (no including transit)
        """
        detrend = np.zeros(len(self.data_tr1["flux"]))

        c_airm     =self.get_jump_parameter_value(pv,'c_airm')
        
        # loop over detrend parameters
        #for i in self.ps.get_param_type_indices(paramtype="detrend"):
        #    #print(i)
        #    detrend += pv[i]*(self.data[self.ps.labels[i]]-1.)
        detrend = c_airm*self.data_tr1['airmass']
        return detrend

    def compute_total_model_transit(self,pv,times=None):
        """
        Computes the full RM model (including RM and RV and CB)

        INPUT:
            pv    - a list of parameters (only parameters that are being varied)
            times - times (optional), array of timestamps

        OUTPUT:
            rm - the rm model evaluated at 'times' if supplied, otherwise
                      defaults to original data timestamps

        NOTES:
            see compute_rm_model(), compute_rv_model(),
            compute_polynomial_model()
        """
        #return self.compute_transit_model(pv,times=times) + self.detrend(pv) + self.compute_polynomial_model(pv,times=times) 
        #return self.compute_transit_model(pv,times=times) + self.compute_polynomial_model(pv,times=times) + self.compute_flare(pv,times=times)
        return self.compute_transit_model(pv,times=times) + self.compute_polynomial_model_transit(pv,times=times) 

    #RADIAL VELOCITY MODELLING
    def compute_rv_model(self,pv,times=None):
        """
        Compute the RV model

        INPUT:
            pv    - a list of parameters (only parameters that are being varied)
            times - times (optional), array of timestamps

        OUTPUT:
            rv - the rv model evaluated at 'times' if supplied, otherwise
                      defaults to original data timestamps
        """
        if times is None:
            times = self.data_rv1["x"]
        Tp      = self.get_jump_parameter_value(pv,'tp_p1') + GAIA_TP_OFFSET
        P       = self.get_jump_parameter_value(pv,'P_p1') # done
        gamma   = self.get_jump_parameter_value(pv,'gamma') # done
        cosi    = self.get_jump_parameter_value(pv,'cosi') # done
        e       = self.get_jump_parameter_value(pv,'ecc_p1') # done
        w       = self.get_jump_parameter_value(pv,'omega_p1') # done
        M       = self.get_jump_parameter_value(pv,'mstar') # done
        m       = self.get_jump_parameter_value(pv,'mp') # done
        eps     = self.get_jump_parameter_value(pv,'eps') # done
        plx     = self.get_jump_parameter_value(pv,'plx') # done

        a0      = a0_from_m_M_P_epsilon(M,m,P,eps,plx)
        K       = semi_amplitude_from_gaia(P,a0,plx,cosi,e)

        self.rv = get_rv_curve_peri(times,P=P,tp=Tp,e=e,omega=w,K=K)+gamma
        return self.rv

    def compute_polynomial_model_rv(self,pv,times=None):
        """
        Compute the polynomial model.  Note that if gammadot and gammadotdot
        are not specified in the priors file, they both default to zero.

        INPUT:
            pv    - a list of parameters (only parameters that are being varied)
            times - times (optional), array of timestamps

        OUTPUT:
            poly - the polynomial model evaluated at 'times' if supplied,
                   otherwise defaults to original data timestamps
        """
        if times is None:
            times = self.data_rv1["x"]

        #T0 = self.get_jump_parameter_value(pv,'t0_p1')
        # Use Mean of data instead
        T0 = (self.data_rv1['x'][0] + self.data_rv1['x'][-1])/2.
        try:
            gammadot = self.get_jump_parameter_value(pv,'gammadot')
        except KeyError as e:
            gammadot = 0
        try:
            gammadotdot = self.get_jump_parameter_value(pv,'gammadotdot')
        except KeyError as e:
            gammadotdot = 0

        self.poly = (
            gammadot * (times - T0) +
            gammadotdot * (times - T0)**2
        )
        return self.poly
    
    def compute_total_model(self,pv,times=None):
        """
        Computes the full RM model (including RM and RV and CB)

        INPUT:
            pv    - a list of parameters (only parameters that are being varied)
            times - times (optional), array of timestamps

        OUTPUT:
            rm - the rm model evaluated at 'times' if supplied, otherwise
                      defaults to original data timestamps

        NOTES:
            see compute_rm_model(), compute_rv_model(),
            compute_polynomial_model()
        """
        return self.compute_rv_model(pv,times=times) + self.compute_polynomial_model_rv(pv,times=times) 
    
    def __call__(self,pv):
        """
        Return the log likelihood

        INPUT:
            pv - the input list of varying parameters
        """
        if any(pv < self.ps_vary.pmins) or any(pv>self.ps_vary.pmaxs):
            return -np.inf

        #ii =self.get_jump_parameter_value(pv,'inc_p1')
        #if ii > 90:
        #    return -np.inf
        b =self.get_jump_parameter_value(pv,'b_p1')
        if b < 0.:
            return -np.inf
        
        ###############
        # PRIORS
        ###############
        log_of_priors = self.ps_vary.c_log_prior(pv)

        ###############
        # RV
        ###############
        y_data = self.data_rv1['y']
        y_model = self.compute_total_model(pv)
        jitter = self.get_jump_parameter_value(pv,'sigma_rv')
        error = np.sqrt(self.data['error']**2.+jitter**2.)
        log_of_rv_model  = ll_normal_ev_py(y_data, y_model, error)
        
        ###############
        # LC
        ###############
        y_data = self.data_tr1['y']
        y_model = self.compute_total_model_transit(pv)
        # jitter in quadrature
        jitter = self.get_jump_parameter_value(pv,'sigma_rv')
        error = np.sqrt(self.data['error']**2.+jitter**2.)
        log_of_priors = self.ps_vary.c_log_prior(pv)
        log_of_lc_model  = ll_normal_ev_py(y_data, y_model, error)

        # FINAL COMBINED
        log_ln = log_of_priors + log_of_rv_model + log_of_lc_model
        return log_ln

class ExoMUSEfitRVLC(object):
    """
    A class that does RV+LC fitting.

    NOTES:
        - Needs to have LPFunction defined
    """
    def __init__(self,LPFunction):
        self.lpf = LPFunction

    def minimize_PyDE(self,npop=100,de_iter=200,mc_iter=1000,mcmc=True,threads=8,maximize=True,plot_priors=True,
                      sample_ball=False,k=None,n=None,plot_corner=True,plot_zscore=True):
        centers = np.array(self.lpf.ps_vary.centers)
        print("Running PyDE Optimizer")
        self.de = pyde.de.DiffEvol(self.lpf, self.lpf.ps_vary.bounds, npop, maximize=maximize) # we want to maximize the likelihood
        self.min_pv, self.min_pv_lnval = self.de.optimize(ngen=de_iter)
        print("Optimized using PyDE")
        print("Final parameters:")
        self.print_param_diagnostics(self.min_pv)
        #if plot_priors:
        #    self.lpf.ps_vary.plot_all(figsize=(6,4),pv=self.min_pv)
        print("LogPost value:",-1*self.min_pv_lnval)
        self.lnl_max  = -1*self.min_pv_lnval-self.lpf.ps_vary.c_log_prior(self.min_pv)
        print("LnL value:",self.lnl_max)
        print("Log priors",self.lpf.ps_vary.c_log_prior(self.min_pv))
        #if k is not None and n is not None:
        #    print("BIC:",stats_help.bic_from_likelihood(self.lnl_max,k,n))
        #    print("AIC:",stats_help.aic(k,self.lnl_max))
        if mcmc:
            print("Running MCMC")
            self.sampler = emcee.EnsembleSampler(npop, self.lpf.ps_vary.ndim, self.lpf,threads=threads)

            #pb = ipywidgets.IntProgress(max=mc_iter/50)
            #display(pb)
            #val = 0
            print("MCMC iterations=",mc_iter)
            for i,c in enumerate(self.sampler.sample(self.de.population,iterations=mc_iter)):
                print(i,end=" ")
                #if i%50 == 0:
                    #val+=50.
                    #pb.value += 1
            print("Finished MCMC")
            self.min_pv_mcmc = self.get_mean_values_mcmc_posteriors().medvals.values
        
        #samples = self.sampler.get_chain(flat=True)
        
        #if plot_corner:
        #    print("Mean acceptance fraction: {0:.3f}".format(np.mean(self.sampler.acceptance_fraction)))
        #    corner.corner(samples,labels=self.lpf.labels,show_titles=True,truths=self.lpf.means);
        #    
        #self.df_mean = self.get_mean_values_mcmc_posteriors()
        #self.df_mean['error'] = (self.df_mean['minus'] + self.df_mean['plus'])/2.
        #self.df_mean['zscore'] = (self.df_mean['medvals'].values - self.lpf.df_known['median'].values)/self.lpf.df_known.error.values
        
        #if plot_zscore:
        #    self.plot_zscore_panel()
        
        #return samples

    def calculate_derived_params(self,df_post=None,burnin_index=200):
        if df_post is None:
            chains_after_burnin = self.sampler.chain[:,burnin_index:,:]
            flatchain = chains_after_burnin.reshape((-1,len(self.lpf.ps_vary.priors)))
            df_post = pd.DataFrame(flatchain,columns=self.lpf.ps_vary.labels)

        if 'eps' in self.lpf.ps_fixed.labels:
            eps = self.lpf.ps_fixed_dict['eps']
        else:
            eps = df_post['P_p1'].values

        # df_post['i']     = np.rad2deg(np.arccos(df_post['cosi'].values))
        # df_post['a0']    = a0_from_m_M_P_epsilon(df_post['mstar'].values, df_post['mp'].values, df_post['P_p1'].values, eps, df_post['plx'].values)
        # df_post['K_p1']  = semi_amplitude_from_gaia(df_post['P_p1'].values, df_post['a0'].values, df_post['plx'].values, df_post['cosi'].values, df_post['ecc_p1'].values)
        # df_post['A']     = thiele_innes_a_from_campell(df_post['a0'].values, df_post['omega_p1'].values, df_post['Omega'].values, df_post['cosi'].values)
        # df_post['B']     = thiele_innes_b_from_campell(df_post['a0'].values, df_post['omega_p1'].values, df_post['Omega'].values, df_post['cosi'].values)
        # df_post['F']     = thiele_innes_f_from_campell(df_post['a0'].values, df_post['omega_p1'].values, df_post['Omega'].values, df_post['cosi'].values)
        # df_post['G']     = thiele_innes_g_from_campell(df_post['a0'].values, df_post['omega_p1'].values, df_post['Omega'].values, df_post['cosi'].values)

        self.df_post = df_post
        self.df_mean = pd.concat([ExoMUSE_utils.get_mean_values_for_posterior(df_post[lab].values,lab) for lab in df_post.columns])

        return self.df_post, self.df_mean
    
    def get_mean_values_mcmc_posteriors(self,flatchain=None):
        """
        Get the mean values from the posteriors

            flatchain - if not passed, then will default using the full flatchain (will likely include burnin)

        EXAMPLE:
        """
        if flatchain is None:
            flatchain = self.sampler.flatchain
            print('No flatchain passed, defaulting to using full chains')
        df_list = [ExoMUSE_utils.get_mean_values_for_posterior(flatchain[:,i],label,description) for i,label,description in zip(range(len(self.lpf.ps_vary.descriptions)),self.lpf.ps_vary.labels,self.lpf.ps_vary.descriptions)]
        return pd.concat(df_list)
    
    def print_param_diagnostics(self,pv):
        """
        A function to print nice parameter diagnostics.
        """
        self.df_diagnostics = pd.DataFrame(zip(self.lpf.ps_vary.labels,self.lpf.ps_vary.centers,self.lpf.ps_vary.bounds[:,0],self.lpf.ps_vary.bounds[:,1],pv,self.lpf.ps_vary.centers-pv),columns=["labels","centers","lower","upper","pv","center_dist"])
        print(self.df_diagnostics.to_string())
        return self.df_diagnostics

##### Only defining the Likelihood function since RMFit is already works for the fitting process.
class LPFunction1RV1Phot(object):
    """
    Log-Likelihood function class
    """
    def __init__(self,inp,file_priors):
        """
        INPUT:
            x - time values in BJD
            y - y values in m/s
            yerr - yerr values in m/s
            file_priors - prior file name
        """
        self.data_rv1= {"time"   : inp['time_rv1'],  
                        "y"      : inp['rv_rv1'],   
                        "error"  : inp['e_rv1']}
        self.data_tr1= {"time"   : inp['time_tr1'],  
                        "y"      : inp['f_tr1'],   
                        "error"  : inp['e_tr1']}
        # Setting priors
        self.ps_all = priorset_from_file(file_priors) # all priors
        self.ps_fixed = PriorSet(np.array(self.ps_all.priors)[np.array(self.ps_all.fixed)]) # fixed priorset
        self.ps_vary  = PriorSet(np.array(self.ps_all.priors)[~np.array(self.ps_all.fixed)]) # varying priorset
        self.ps_fixed_dict = {key: val for key, val in zip(self.ps_fixed.labels,self.ps_fixed.args1)}
        print('Reading in priorfile from {}'.format(file_priors))
        print(self.ps_all.df)
        
    def get_jump_parameter_index(self,lab):
        """
        Get the index of a given label
        """
        return np.where(np.array(self.ps_vary.labels)==lab)[0][0]
    
    def get_jump_parameter_value(self,pv,lab):
        """
        Get the current value in the argument list 'pv' that has label 'lab'
        """
        # First check if we are actually varying it
        if lab in self.ps_vary.labels:
            return pv[self.get_jump_parameter_index(lab)]
        else:
            # We are not varying it
            return self.ps_fixed_dict[lab]
        
    #LIGHT CURVE MODELLING
    def compute_transit_model_main(self,pv,times=None):
        """
        Calls RM model and returns the transit model

        INPUT:
            pv    - parameters passed to the function
            times - times, and array of timestamps

        OUTPUT:
            lc - the lightcurve model at *times*
        """
        T0     =self.get_jump_parameter_value(pv,'t0_p1')
        P      =self.get_jump_parameter_value(pv,'P_p1')
        ii     =self.get_jump_parameter_value(pv,'inc_p1')
        rprs   =self.get_jump_parameter_value(pv,'p_p1')
        aRs    =self.get_jump_parameter_value(pv,'a_p1')
        u1     =self.get_jump_parameter_value(pv,'u1')
        u2     =self.get_jump_parameter_value(pv,'u2')
        u = [u1,u2]
        gamma  =self.get_jump_parameter_value(pv,'gamma')
        e      =self.get_jump_parameter_value(pv,'ecc_p1')
        omega  =self.get_jump_parameter_value(pv,'omega_p1')
        exptime=self.get_jump_parameter_value(pv,'exptime')/86400. # exptime in days
        if times is None:
            times = self.data_tr1["time"]
        self.lc = get_lc_batman(times,T0,P,ii,rprs,aRs,e,omega,u,supersample_factor=7,exp_time=exptime,limbdark="quadratic")
        return self.lc + gamma

    def compute_flare(self,pv,times=None):
        tpeak     =self.get_jump_parameter_value(pv,'flare_tpeak')
        fwhm      =self.get_jump_parameter_value(pv,'flare_fwhm')
        A         =self.get_jump_parameter_value(pv,'flare_ampl')
        if times is None:
            times = self.data_tr1["x"]
        self.flare = aflare1(times,tpeak,fwhm,A)
        return self.flare


    def compute_transit_model(self,pv,times=None):
        """
        Calls RM model and returns the transit model

        INPUT:
            pv    - parameters passed to the function
            times - times, and array of timestamps

        OUTPUT:
            lc - the lightcurve model at *times*
        """
        T0     =self.get_jump_parameter_value(pv,'t0_p1')
        P      =self.get_jump_parameter_value(pv,'P_p1')
        tdur   =self.get_jump_parameter_value(pv,'tdur_p1')
        b      =self.get_jump_parameter_value(pv,'b_p1')
        rprs   =self.get_jump_parameter_value(pv,'p_p1')
        u1     =self.get_jump_parameter_value(pv,'u1')
        u2     =self.get_jump_parameter_value(pv,'u2')
        #q1     =self.get_jump_parameter_value(pv,'q1')
        #q2     =self.get_jump_parameter_value(pv,'q2')
        #u1, u2 = u1_u2_from_q1_q2(q1,q2)
        u = [u1,u2]
        gamma  =self.get_jump_parameter_value(pv,'gamma')
        e      =self.get_jump_parameter_value(pv,'ecc_p1')
        omega  =self.get_jump_parameter_value(pv,'omega_p1')
        aRs    = calc_aRs_from_rprs_b_tdur_P(rprs,b,tdur,P)
        ii     = i_from_aRs_and_b(aRs,b)
        exptime=self.get_jump_parameter_value(pv,'exptime')/86400. # exptime in days
        if times is None:
            times = self.data_tr1["time"]
        self.lc = get_lc_batman(times,T0,P,ii,rprs,aRs,e,omega,u,supersample_factor=7,exp_time=exptime,limbdark="quadratic")
        return self.lc + gamma

    def compute_polynomial_model_transit(self,pv,times=None):
        """
        Compute the polynomial model.  Note that if gammadot and gammadotdot
        are not specified in the priors file, they both default to zero.

        INPUT:
            pv    - a list of parameters (only parameters that are being varied)
            times - times (optional), array of timestamps

        OUTPUT:
            poly - the polynomial model evaluated at 'times' if supplied,
                   otherwise defaults to original data timestamps
        """
        if times is None:
            times = self.data_tr1["time"]

        T0 = (self.data_tr1['time'][0] + self.data_tr1['time'][-1])/2.
        try:
            mfluxdot = self.get_jump_parameter_value(pv,'mfluxdot_tr1')
        except KeyError as e:
            mfluxdot = 0
        try:
            mfluxdotdot = self.get_jump_parameter_value(pv,'mfluxdotdot_tr1')
        except KeyError as e:
            mfluxdotdot = 0

        self.poly = (
            mfluxdot * (times - T0) +
            mfluxdotdot * (times - T0)**2
        )
        return self.poly

    def detrend(self,pv):
        """
        A function to detrend.
        
        INPUT:
        pv    - an array containing a sample draw of the parameters defined in self.lpf.ps
        
        OUTPUT:
        detrend/pv[self.number_pv_baseline] - the additional trend in the data (no including transit)
        """
        detrend = np.zeros(len(self.data_tr1["flux"]))

        c_airm     =self.get_jump_parameter_value(pv,'c_airm')
        
        # loop over detrend parameters
        #for i in self.ps.get_param_type_indices(paramtype="detrend"):
        #    #print(i)
        #    detrend += pv[i]*(self.data[self.ps.labels[i]]-1.)
        detrend = c_airm*self.data_tr1['airmass']
        return detrend

    def compute_total_model_transit(self,pv,times=None):
        """
        Computes the full RM model (including RM and RV and CB)

        INPUT:
            pv    - a list of parameters (only parameters that are being varied)
            times - times (optional), array of timestamps

        OUTPUT:
            rm - the rm model evaluated at 'times' if supplied, otherwise
                      defaults to original data timestamps

        NOTES:
            see compute_rm_model(), compute_rv_model(),
            compute_polynomial_model()
        """
        #return self.compute_transit_model(pv,times=times) + self.detrend(pv) + self.compute_polynomial_model(pv,times=times) 
        #return self.compute_transit_model(pv,times=times) + self.compute_polynomial_model(pv,times=times) + self.compute_flare(pv,times=times)
        return self.compute_transit_model(pv,times=times) + self.compute_polynomial_model_transit(pv,times=times)

    def compute_rv_model(self,pv,times1=None):
        """
        Compute the RV model

        INPUT:
            pv    - a list of parameters (only parameters that are being varied)
            times - times (optional), array of timestamps 
        
        OUTPUT:
            rv - the rv model evaluated at 'times' if supplied, otherwise 
                      defaults to original data timestamps
        """
        if times1 is None: times1 = self.data_rv1["time"]
        T0      = self.get_jump_parameter_value(pv,'t0_p1')
        P       = self.get_jump_parameter_value(pv,'P_p1')
        gamma1  = self.get_jump_parameter_value(pv,'gamma_rv1')
        K       = self.get_jump_parameter_value(pv,'K_p1')
        e       = self.get_jump_parameter_value(pv,'ecc_p1')
        w       = self.get_jump_parameter_value(pv,'omega_p1')
        self.rv1 = get_rv_curve(times1,P=P,tc=T0,e=e,omega=w,K=K)+gamma1
        return self.rv1
        
    def compute_total_model(self,pv,times_tr1=None,times_rv1=None):
        """
        Computes the full RM model (including RM and RV and CB)

        INPUT:
            pv    - a list of parameters (only parameters that are being varied)
            times - times (optional), array of timestamps 
        
        OUTPUT:
            rm - the rm model evaluated at 'times' if supplied, otherwise 
                      defaults to original data timestamps

        NOTES:
            see compute_rm_model(), compute_rv_model()
        """
        rv1 = self.compute_rv_model(pv,times1=times_rv1) 
        ff1 = self.compute_transit_model(pv,times=times_tr1)
        ff1_trend = self.compute_polynomial_model_transit(pv,times=times_tr1)
        return rv1, ff1 + ff1_trend

    def __call__(self,pv):
        """
        Return the log likelihood

        INPUT:
            pv - the input list of varying parameters
        """
        if any(pv < self.ps_vary.pmins) or any(pv>self.ps_vary.pmaxs):
            return -np.inf

        ii =self.get_jump_parameter_value(pv,'inc_p1')
        if ii > 90:
            return -np.inf

        ###############
        # Prepare data and model and error for ingestion into likelihood
        #y_data = self.data['y']
        rv1, ff1 = self.compute_total_model(pv)
        # jitter in quadrature
        jitter_rv1 = self.get_jump_parameter_value(pv,'sigma_rv1')
        jitter_tr1 = self.get_jump_parameter_value(pv,'sigma_tr1')
        error_rv1 = np.sqrt(self.data_rv1['error']**2.+jitter_rv1**2.)
        error_tr1 = np.sqrt(self.data_tr1['error']**2.+jitter_tr1**2.)
        ###############

        # Return the log-likelihood
        log_of_priors = self.ps_vary.c_log_prior(pv)
        # Calculate log likelihood
        #log_of_model  = ll_normal_ev_py(y_data, y_model, error)
        log_of_model1  = ll_normal_ev_py(self.data_rv1["y"], rv1, error_rv1)
        log_of_model2  = ll_normal_ev_py(self.data_tr1["y"], ff1, error_tr1)
        log_ln = log_of_priors + log_of_model1 + log_of_model2 
        return log_ln