#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# Import Libraries and Dependencies
import sys

import file_handling

sys.path.append('../')

import numpy as np
from scipy.interpolate import griddata
from features import Feature
import features

import pandas as pd
import cell_display_lib as cdl
from scipy import ndimage, io, spatial, stats
from scipy.signal import savgol_filter, spectrogram
from skimage.transform import resize
from sklearn.decomposition import PCA
from features import Feature
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon
from scipy.ndimage.filters import gaussian_filter
import scipy

import importlib, os


import matplotlib.pyplot as plt
from IPython.core.display import display
from icecream import ic

sys.path.append('../')
sys.path.append('/Volumes/Lab/Users/scooler/classification/')
sys.path.append("/Volumes/Lab/Users/mads/artificial-retina-software-pipeline/artificial-retina-software-pipeline/utilities/")
sys.path.append("/Volumes/Lab/Users/mads/cell_class/moosa_share/")


import scipy.signal as signal
import visionloader as vl
from conduction_velocity_code import get_axonal_conduction_velocity, upsample_ei, filter_ei_for_electrode_types
import eilib as el
import math
from cell_display_lib import show
import cv2
        
        


# In[ ]:

# Feature that transforms piece id into an integer. Likely a useful feature as it can demonstrate the spatial relationship of cell types within a single piece
class Feature_int_piece_id(features.Feature):
    name = 'int_piece_id'
    requires = {'dataset':set(),'unit':set()}
    provides = {'unit':{'int_piece_id','int_run_id'}}
    input = set()
    version = 0 # eventually, we'll track these to keep values in sync
    maintainer = 'Mads' # so we can blame you when it goes wrong and praise you when it's great

    def generate(self, ct, unit_indices, inpt, dtab=None):
        if dtab is None:
            dtab = ct.unit_table
        di = unit_indices[0][0:2]
        if (missing := self.check_requirements(ct, di)) is not None:
            print('Feature {}: missing requirements {}'.format(self.name, missing))
            return

        piece_id = ct.dataset_table.loc[di,'piece_id']
        run_id = ct.dataset_table.loc[di,'run_id']
        compString = piece_id.replace('-','')
        dtab.loc[unit_indices, 'int_piece_id'] = int(compString)
        dtab.loc[unit_indices, 'int_run_id'] = int(run_id)
        

        # mark these columns as valid
        self.update_valid_columns(ct, di)


# In[ ]:

# Feature for the average spike waveform broken down into its power spectrum
class Feature_spec_spike_waveform(features.Feature):
    name = 'spec_spike_waveform'
    requires = {'unit':{'spike_waveform_maxamplitude'}}
    provides = {'unit':{'spec_spike_waveform','spec_freq','spec_timeoverlap'}}
    input = set()
    version = 0 # eventually, we'll track these to keep values in sync
    maintainer = 'Mads' # so we can blame you when it goes wrong and praise you when it's great

    def generate(self, ct, unit_indices, inpt, dtab=None):
        if dtab is None:
            dtab = ct.unit_table
        di = unit_indices[0][0:2]
        if (missing := self.check_requirements(ct, di)) is not None:
            print('Feature {}: missing requirements {}'.format(self.name, missing))
            return
        
        spec = []
        f = []
        t = []

        for ci in unit_indices:
            wave = dtab.at[ci,'spike_waveform_maxamplitude'].a
        
            t_wave = np.arange(len(wave)) / 20 - 5
            t_wave = t_wave / 1000
            fs = 20000


            n = min(256,len(wave))
            f2, t2, spec2 = spectrogram(wave, nperseg = n, fs=fs, window = ('hanning'), noverlap=np.ceil(n/2))
            spec.append(file_handling.wrapper(spec2))
            f.append(file_handling.wrapper(f2))
            t.append(file_handling.wrapper(t2))

        dtab.loc[unit_indices,'spec_spike_waveform'] = spec
        dtab.loc[unit_indices,'spec_freq'] = f
        dtab.loc[unit_indices,'spec_timeoverlap'] = t

        # mark these columns as valid
        self.update_valid_columns(ct, di)


# In[ ]:

# Feature that takes interspike interval and finds its power spectrum
class Feature_spec_acf(features.Feature):
    name = 'spec_acf'
    # NOTE: This is assuming ISI is being calculated from spike times (generate_acf_from_spikes)
    requires = {'unit':{'acf'}}
    provides = {'unit':{'spec_acf','spec_freq_acf','spec_timeoverlap_acf'}}
    input = set()
    version = 0 # eventually, we'll track these to keep values in sync
    maintainer = 'Mads' # so we can blame you when it goes wrong and praise you when it's great

    def generate(self, ct, unit_indices, inpt, dtab=None):
        if dtab is None:
            dtab = ct.unit_table
        di = unit_indices[0][0:2]
        if (missing := self.check_requirements(ct, di)) is not None:
            print('Feature {}: missing requirements {}'.format(self.name, missing))
            return
        
        spec = []
        f = []
        t = []

        for ci in unit_indices:
            wave = dtab.at[ci,'acf'].a
        
            t_wave = np.arange(len(wave)) / 20 - 5
            t_wave = t_wave / 1000
            fs = 20000


            n = min(256,len(wave))
            f2, t2, spec2 = spectrogram(wave, nperseg = n, fs=fs, window = ('hanning'), noverlap=np.ceil(n/2))
            spec.append(file_handling.wrapper(spec2))
            f.append(file_handling.wrapper(f2))
            # t.append(file_handling.wrapper(t2))
            t.append(t2)

        dtab.loc[unit_indices,'spec_acf'] = spec
        dtab.loc[unit_indices,'spec_freq_acf'] = f
        dtab.loc[unit_indices,'spec_timeoverlap_acf'] = t

        # mark these columns as valid
        self.update_valid_columns(ct, di)


# In[ ]:

# Feature to calculate axon conduction velocity where cell types are known to be faster (Parsols), regular (majority types), or slow (amacrines)
class Feature_axon_vel(features.Feature):
    name = 'axon_vel'
    requires = {'unit':{'ei'}}
    provides = {'unit':{'axon_vel'}}
    input = set()
    version = 0 # eventually, we'll track these to keep values in sync
    maintainer = 'Maddy' # so we can blame you when it goes wrong and praise you when it's great

    def generate(self, ct, unit_indices, inpt, dtab=None):
        if dtab is None:
            dtab = ct.unit_table
        di = unit_indices[0][0:2]
        if (missing := self.check_requirements(ct, di)) is not None:
            print('Feature {}: missing requirements {}'.format(self.name, missing))
            return
        
        velocity = []

        for ci in unit_indices:
            ei = dtab.at[ci,'ei'].a
            try:
                vel = get_axonal_conduction_velocity(ei,10,60,6,10)
            except:
                vel = 0.0
            if math.isnan(vel):
                vel = 0.0
            velocity.append(vel)


        dtab.loc[unit_indices,'axon_vel'] = velocity

        # mark these columns as valid
        self.update_valid_columns(ct, di)


# In[ ]:

# Feature for retinal eccentricity as cell size and density and type ratios vary with eccentricity
#Needs help with none and the word values
class Feature_retinal_eccentricity(features.Feature):
    name = 'retinal_eccentricity'
    requires = {'dataset':set(),'unit':set()}
    provides = {'unit':{'retinal_eccentricity'}}
    input = set()
    version = 0 # eventually, we'll track these to keep values in sync
    maintainer = 'Maddy' # so we can blame you when it goes wrong and praise you when it's great

    def generate(self, ct, unit_indices, inpt, dtab=None):
        if dtab is None:
            dtab = ct.unit_table
        di = unit_indices[0][0:2]
        if (missing := self.check_requirements(ct, di)) is not None:
            print('Feature {}: missing requirements {}'.format(self.name, missing))
            return
        
        piece_id = ct.dataset_table.loc[di,'piece_id']
        run_id = ct.dataset_table.loc[di,'run_id']
        if 'location_eccentricity' in ct.dataset_table.columns:
            dtab.loc[unit_indices,'retinal_eccentricity']=ct.dataset_table.location_eccentricity[str(piece_id)][str(run_id)]
        else:
            dtab['retinal_eccentricity']=100 #Unknown

        # mark these columns as valid
        self.update_valid_columns(ct, di)


# In[ ]:

# Number of features calculated from the early phases of the EI including estimates of soma location, areas, perimeters, circularity which may catch different cell type morphologies
class Feature_map_early_ei_char(features.Feature):
    name = 'map_early_ei_char'
    requires = {'unit':{'map_ei_energy_early'}}
    provides = {'unit':{'total_energy_ptnorm','early_area','early_peri','early_circularity','early_centroid','soma_area','soma_peri','soma_circularity','soma_centroid'}}
    input = set()
    version = 0 # eventually, we'll track these to keep values in sync
    maintainer = 'Maddy' # so we can blame you when it goes wrong and praise you when it's great

    def generate(self, ct, unit_indices, inpt, dtab=None):
        if dtab is None:
            dtab = ct.unit_table
        di = unit_indices[0][0:2]
        if (missing := self.check_requirements(ct, di)) is not None:
            print('Feature {}: missing requirements {}'.format(self.name, missing))
            return
        
        total_energy_ptnorm = []
        early_area = []
        early_peri = []
        early_circularity = []
        early_centroid = []
        soma_area = []
        soma_peri = []
        soma_circularity = []
        soma_centroid = []

        for ci in unit_indices:
            ei_energy= dtab.at[ci,'map_ei_energy_early'].a
            
            # Find Total Energy
            total_energy = np.sum(ei_energy.flatten()) / (ei_energy.shape[0]*ei_energy.shape[1])
            total_energy_ptnorm.append(total_energy)

            # Next Find Contours
            ei_energy = np.flip(ei_energy)
            thresholds = (np.percentile(ei_energy,90), np.percentile(ei_energy,99.8))
            ei_energy = np.flip(ei_energy)
            segments_all_ei = [measure.find_contours(ei_energy, level = thresholds[i]) for i in range(len(thresholds))]
            
            for ssi, segments in enumerate(segments_all_ei):
                if ssi==0:
                    for si, seg in enumerate(segments):
                        if si == 0:
                            
                            ## Set up for cv2
                            seg_proc = []
                            for point in seg:
                                point_proc = [int(point[0]), int(point[1])]
                                seg_proc.append([point_proc])
                            seg_proc = np.array(seg_proc)
                            
                            ## FIND EARLY AREA
                            early_a = cv2.contourArea(seg_proc)
                            if math.isnan(early_a):
                                early_a = 0.0
                            early_area.append(early_a)

                            ## FIND EARLY PERIMETER
                            early_p = cv2.arcLength(seg_proc,True)
                            if math.isnan(early_p):
                                early_p = 0.0
                            early_peri.append(early_p)

                            ## FIND EARLY CENTROID
                            M = cv2.moments(seg_proc)
                            if M['m00']==0.0:
                                cx = 0
                                cy = 0
                            else:
                                cx = int(M['m10']/M['m00'])
                                cy = int(M['m01']/M['m00'])
                            early_cen = [cx, cy]
                            early_centroid.append(file_handling.wrapper(early_cen))
                            
                            ## FIND EARLY CIRCULARITY
                            if early_a != 0 and early_p != 0:
                                early_cir = (4*np.pi*early_a) / (early_p**2)
                            else:
                                early_cir = 0.0
                            early_circularity.append(early_cir)


                if ssi==1:
                    for si, seg in enumerate(segments):
                        
                        ## Set up for cv2
                        if si == 0:
                            seg_proc = []
                            for point in seg:
                                point_proc = [int(point[0]), int(point[1])]
                                seg_proc.append([point_proc])
                            seg_proc = np.array(seg_proc)
                            
                            ## FIND SOMA AREA
                            soma_a = cv2.contourArea(seg_proc)
                            if math.isnan(soma_a):
                                soma_a = 0.0
                            soma_area.append(soma_a)

                            ## FIND SOMA PERIMETER
                            soma_p = cv2.arcLength(seg_proc,True)
                            if math.isnan(soma_p):
                                soma_p = 0.0
                            soma_peri.append(soma_p)

                            ## FIND SOMA CENTROID
                            M = cv2.moments(seg_proc)
                            if M['m00']==0.0:
                                cx = 0
                                cy = 0
                            else:
                                cx = int(M['m10']/M['m00'])
                                cy = int(M['m01']/M['m00'])
                            soma_cen = [cx, cy]
                            soma_centroid.append(file_handling.wrapper(soma_cen))

                            ## FIND SOMA CIRCULARITY
                            if soma_a != 0 and soma_p != 0:
                                soma_cir = (4*np.pi*soma_a) / (soma_p**2)
                            else:
                                soma_cir = 0.0
                            soma_circularity.append(soma_cir)
     
            
        dtab.loc[unit_indices,'total_energy_ptnorm'] = total_energy_ptnorm
        dtab.loc[unit_indices,'early_area'] = early_area
        dtab.loc[unit_indices,'early_peri'] = early_peri
        dtab.loc[unit_indices,'early_centroid'] = early_centroid
        dtab.loc[unit_indices,'early_circularity'] = early_circularity
        dtab.loc[unit_indices,'soma_area'] = soma_area
        dtab.loc[unit_indices,'soma_peri'] = soma_peri
        dtab.loc[unit_indices,'soma_centroid'] = soma_centroid
        dtab.loc[unit_indices,'soma_circularity'] = soma_circularity

        # mark these columns as valid
        self.update_valid_columns(ct, di)


# In[ ]:

# Characteristics calculated from spike waveform which have been used before previously in logistic regression of ON Parasol and OFF Parasol identification
class Feature_spike_char(features.Feature):
    name = 'spike_char'
    requires = {'unit':{'spike_waveform_maxamplitude'}}
    provides = {'unit':{'spike_min_h','spike_max_l','spike_minmax_ratio','spike_width_half_min_amp','spike_width_half_max_amp','spike_area_amp_upper','spike_area_amp_lower','spike_area_tr_amp_upper','spike_area_tr_amp_lower','spike_slope_amp_upper','spike_slope_amp_lower', 'spike_area2'}}
    input = set()
    version = 0 # eventually, we'll track these to keep values in sync
    maintainer = 'Maddy' # so we can blame you when it goes wrong and praise you when it's great

    def generate(self, ct, unit_indices, inpt, dtab=None):
        if dtab is None:
            dtab = ct.unit_table
        di = unit_indices[0][0:2]
        if (missing := self.check_requirements(ct, di)) is not None:
            print('Feature {}: missing requirements {}'.format(self.name, missing))
            return
        
        spike_min_h = []
        spike_max_l = []
        spike_minmax_ratio = []
        spike_width_half_min_amp = []
        spike_width_half_max_amp = []
        spike_area_amp_upper = []
        spike_area_amp_lower = []
        spike_area_tr_amp_upper = []
        spike_area_tr_amp_lower = []
        spike_slope_amp_upper = []
        spike_slope_amp_lower = []
        spike_area2 = []

        for ci in unit_indices:
            spike_wave = dtab.at[ci,'spike_waveform_maxamplitude'].a
            spike_wave = signal.resample(spike_wave, len(spike_wave)*10)
            
            # Max, Min, Max Min Ratio
            h = np.abs(np.min(spike_wave))
            l = np.abs(np.max(spike_wave))
            l_h_ratio = np.abs(np.max(spike_wave)/np.min(spike_wave))
            spike_min_h.append(h)
            spike_max_l.append(l)
            spike_minmax_ratio.append(l_h_ratio)
            
            # Min Amp Area Calc's
            amp_idx = np.argmin(spike_wave)
            lower_idx = amp_idx
            upper_idx = amp_idx
            while(spike_wave[lower_idx] < spike_wave[amp_idx]/2):
                if(lower_idx == 0):
                    break
                lower_idx = lower_idx -1
            while(spike_wave[upper_idx] < spike_wave[amp_idx]/2):
                if(upper_idx == len(spike_wave)-1):
                    break
                upper_idx = upper_idx +1
                
            width_half_min_amp = upper_idx - lower_idx
            spike_width_half_min_amp.append(width_half_min_amp)
            
            area_amp_upper = np.abs(np.sum([spike_wave[idx] for idx in range(amp_idx,upper_idx)]))
            area_amp_lower = np.abs(np.sum([spike_wave[idx] for idx in range(lower_idx+1,amp_idx)]))
            spike_area_amp_upper.append(area_amp_upper)
            spike_area_amp_lower.append(area_amp_lower)
            
            area_tr_amp_upper = h*(3/4)*(upper_idx-amp_idx)
            area_tr_amp_lower = h*(3/4)*(amp_idx-lower_idx)
            spike_area_tr_amp_upper.append(area_tr_amp_upper)
            spike_area_tr_amp_lower.append(area_tr_amp_lower)
            
            if upper_idx-amp_idx != 0:
                slope_amp_upper = h / (upper_idx-amp_idx)
                slope_amp_lower = h / (amp_idx-lower_idx)
            else:
                slope_amp_upper = 0
                slope_amp_lower = 0
            spike_slope_amp_upper.append(slope_amp_upper)
            spike_slope_amp_lower.append(slope_amp_lower)
            
            # Max Amp Area Calc's
            amp_idx = np.argmax(spike_wave)
            upper_idx_max = amp_idx
            lower_idx_max = amp_idx
            while(spike_wave[lower_idx_max] > spike_wave[amp_idx]/2):
                if(lower_idx_max==0):
                    break
                lower_idx_max = lower_idx_max -1
            while(spike_wave[upper_idx_max] > spike_wave[amp_idx]/2):
                if(upper_idx_max == len(spike_wave)-1):
                    break
                upper_idx_max = upper_idx_max +1
                
            area2 = np.sum([spike_wave[idx] for idx in range(lower_idx_max+1,upper_idx_max)])
            spike_area2.append(area2)
            width_half_max_amp = upper_idx_max - lower_idx_max
            spike_width_half_max_amp.append(width_half_max_amp)


        dtab.loc[unit_indices,'spike_min_h'] = spike_min_h
        dtab.loc[unit_indices,'spike_max_l'] = spike_max_l
        dtab.loc[unit_indices,'spike_minmax_ratio'] = spike_minmax_ratio
        dtab.loc[unit_indices,'spike_width_half_min_amp'] = spike_width_half_min_amp
        dtab.loc[unit_indices,'spike_area_amp_upper'] = spike_area_amp_upper
        dtab.loc[unit_indices,'spike_area_amp_lower'] = spike_area_amp_lower
        dtab.loc[unit_indices,'spike_area_tr_amp_upper'] = spike_area_tr_amp_upper
        dtab.loc[unit_indices,'spike_area_tr_amp_lower'] = spike_area_tr_amp_lower
        dtab.loc[unit_indices,'spike_slope_amp_upper'] = spike_slope_amp_upper
        dtab.loc[unit_indices,'spike_slope_amp_lower'] = spike_slope_amp_lower
        dtab.loc[unit_indices,'spike_area2'] = spike_area2
        dtab.loc[unit_indices,'spike_width_half_max_amp'] = spike_width_half_max_amp
        

        # mark these columns as valid
        self.update_valid_columns(ct, di)


# In[ ]:




