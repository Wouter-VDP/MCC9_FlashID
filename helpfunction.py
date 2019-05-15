import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

gr = 1.618      
outdir= './output/helper/'   
nrPMT = 32
rangePMT = range(nrPMT)
 
lower = [-1.55, -115.53, 0.1]
upper = [254.8, 117.47, 1036.9]

# Return true if the point is in the TPC with a tolerance.    
def inTPC_df(df, str_x, str_y, str_z, fidvol=[0]*6):
    global upper, lower
    mask_x = df[str_x].between(lower[0]+fidvol[0], upper[0]-fidvol[1])
    mask_y = df[str_y].between(lower[1]+fidvol[2], upper[1]-fidvol[3])
    mask_z = df[str_z].between(lower[2]+fidvol[4], upper[2]-fidvol[5])
    mask = mask_x & mask_y & mask_z
    return mask


# Formatting
def sciNot(x):
    x=float(x)
    return "{:.1f}".format(x)

def sciNot2(x):
    x=float(x)
    return "{:.2f}".format(x)


# @in N: number of bins 
# @in x_min, x_max: range of the plot
# @in data: list of data arrays
# @in weights: list of weights
# @in if where='post': duplicate the last bin
# @in log=True: return x-axis log

def histHelper(N,x_min,x_max,data,weights=0, where='mid', log=False):
    if log:
        edges = np.logspace(np.log10(x_min), np.log10(x_max), N+1)
    else:
        edges = np.linspace(x_min,x_max,N+1)
    edges_mid = [ edges[i]+(edges[i+1]-edges[i])/2 for i in range(N)]
    bins = [np.histogram(data_i,bins=edges)[0] for data_i in data]
    max_val = [max(x) for x in bins]
    if where=='post':
        bins = [ np.append(b,b[-1]) for b in bins]
    err = np.sqrt(bins)
    if weights!=0:
        bins = [b*s for b,s in zip(bins,weights)]
        err = [e*s for e,s in zip(err,weights)]
        max_val = [v*s for v,s in zip(max_val,weights)]
    return edges, edges_mid, bins, err, max_val
    

# efficiency error unweighted
def effErr(teller,noemer):
    return np.sqrt(teller*(1-teller/noemer))/noemer
 
# weighted histogram error  
def hist_bin_uncertainty(data, weights, bin_edges):
    # Bound the data and weights to be within the bin edges
    in_range_index = [idx for idx in range(len(data)) if data[idx] > min(bin_edges) and data[idx] < max(bin_edges)]
    in_range_data = np.asarray([data[idx] for idx in in_range_index])
    in_range_weights = np.asarray([weights[idx] for idx in in_range_index])

    # Bin the weights with the same binning as the data
    bin_index = np.digitize(in_range_data, bin_edges)
    # N.B.: range(1, bin_edges.size) is used instead of set(bin_index) as if
    # there is a gap in the data such that a bin is skipped no index would appear
    # for it in the set
    binned_weights = np.asarray(
        [in_range_weights[np.where(bin_index == idx)[0]] for idx in range(1, len(bin_edges))])
    bin_uncertainties = np.asarray(
        [np.sqrt(np.sum(np.square(w))) for w in binned_weights])
    return bin_uncertainties


# This cell is a single event viewer!
def SingleEventViewer(run,subrun,event, sample_dict, save_plot=False):
    str_pause = '------------------------------'
    event_keys = ['nFlashes', 'hasBeamFlash', 'nSlices', 'nSlicesAfterPrecuts', 'foundATargetSlice', 
                  'nuCCNC', 'nuEnergy', 'leptonEnergy', 'nuInteractionTime', 'nuPdgCode', 'nuVertexX', 'nuVertexY', 'nuVertexZ']
    flash_keys = ['time', 'centerY', 'centerZ', 'widthY', 'widthZ', 'totalPE', 'inBeamWindow', 'isBeamFlash']
    slice_keys = ['hasDeposition', 'totalCharge', 'centerX', 'centerY', 'centerZ', 'minX', 'nHits',
                  'deltaY', 'deltaZ', 'deltaYSigma', 'deltaZSigma', 'chargeToLightRatio', 
                  'passesPreCuts', 'flashMatchScore', 'flashMatchX', 'totalPEHypothesis', 
                  'isTaggedAsTarget', 'isConsideredByFlashId', 'topologicalScore', 'hasBestTopologicalScore', 
                  'purity', 'completeness']
    # Event Info:
    event_dict = sample_dict['events']
    events_index = np.where( (event_dict.array('run')==run) & (event_dict.array('subRun')==subrun) & (event_dict.array('event')==event) ) 
    if len(events_index[0])!=1:
        print('The combination of event, subrun and run was found '+str(len(events_index[0]))+" times in the sample.")
        if len(events_index[0])>1:
            print("Taking the first one...\n")
        else:
            return
    print('Run',run,', Subrun', subrun,', Event',event,' found!','\n',str_pause)
    print('\n--- EVENT INFO ---')
    events_index = events_index[0][0]
    event_time = event_dict.array('evt_time_sec')[events_index]
    for key in event_keys:
        print(key, ':\t', event_dict.array(key)[events_index])
        
    # Flash Info:
    flash_dict = sample_dict['flashes']
    flashes_indices = np.where( (flash_dict.array('evt_time_sec')==event_time) & (flash_dict.array('run')==run) & (flash_dict.array('subRun')==subrun) & (flash_dict.array('event')==event))[0]
    print('\n--- FLASH INFO ---')
    for key in flash_keys:
        print(key, ':\t', flash_dict.array(key)[flashes_indices])
    
    # Slice Info:
    slice_dict = sample_dict['slices']
    slices_indices = np.where( (slice_dict.array('evt_time_sec')==event_time) & (slice_dict.array('run')==run) & (slice_dict.array('subRun')==subrun) & (slice_dict.array('event')==event))[0]
    print('\n--- SLICE INFO ---')
    for key in slice_keys:
        print(key, ':\t', slice_dict.array(key)[slices_indices])
    
    if not np.any(flash_dict.array('isBeamFlash')[flashes_indices]):
        print('\nUnable to plot: There was no beamflash in the selected event!')
        return
    # Make the plot!
    fig, ax = plt.subplots(figsize=(5.5*gr,6.5))
    ax.set_xlabel('PMT identification number')
    ax.set_ylabel('Number of Photo-electrons per PMT')
    ax.grid(alpha=.3)
    
    beam_flash_index = flashes_indices[flash_dict.array('isBeamFlash')[flashes_indices]][0]
    beam_flash_pe = flash_dict.array('peSpectrum')[beam_flash_index]
    
    lab_flash = "\nOptical Flash\n  PE: {0:.0f} \n  ".format(flash_dict.array('totalPE')[beam_flash_index]) \
                +r"$z$: {0:.0f} cm".format(flash_dict.array('centerZ')[beam_flash_index])+"\n"
    
    ax.errorbar(rangePMT,beam_flash_pe, yerr=np.sqrt(beam_flash_pe), fmt="none")
    ax.fill_between(rangePMT, beam_flash_pe, alpha=.5,label=lab_flash)
    
    
    #slice_hypo_index = slices_indices[slice_dict.array('passesPreCuts')[slices_indices]]
    slice_hypo_index = slices_indices[slice_dict.array('isConsideredByFlashId')[slices_indices]]
    flash_hypo_pe = slice_dict.array('peHypothesisSpectrum')[slice_hypo_index]
    for i,(idx, spectrum) in enumerate(zip(slice_hypo_index, flash_hypo_pe)):
        slice_lab = 'Slice Hypothesis '+str(i)+"\n  "
        slice_lab+=r"Purity: "+ str(round(slice_dict.array('purity')[idx],1)) + "\n  "
        slice_lab+=r"Completeness: "+ str(round(slice_dict.array('completeness')[idx],3)) + "\n  "
        slice_lab+=r"Topo score: "+ str(round(slice_dict.array('topologicalScore')[idx],3)) + "\n  "
        slice_lab+=r"Flash $\chi^2$: {0:.0f}".format(slice_dict.array('flashMatchScore')[idx])
        slice_lab+="\n  "+r"$\Delta$z: "+ str(round(slice_dict.array('deltaZ')[idx],1)) + "cm\n"
        if(len(spectrum)==nrPMT):
            ax.errorbar(rangePMT,spectrum, yerr= np.sqrt(spectrum), label=slice_lab) 

    ax.legend(bbox_to_anchor=(1.02,0.3,.25,.8),loc=2)
    
    ax.set_title('Run '+str(run)+', Subrun '+str(subrun)+', Event '+str(event), loc='left')
    if (b'nuPdgCode' in event_dict.keys()):
        d_pdg = {12: r"$\nu_e$", 14: r"$\nu_\mu$",-12: r"$\bar{\nu_e}$", -14: r"$\bar{\nu_\mu}$"}
        print(str_pause,"\nProducing plot for MC event!")
        ax.set_title("MicroBooNE Simulation", loc='right')
        txt =  d_pdg[event_dict.array('nuPdgCode')[events_index]]
        txt+= " with {0:.2f} GeV energy\n".format(event_dict.array('nuEnergy')[events_index])
        txt+= "and vertex: ({0:.0f}, {1:.0f}, {2:.0f}) cm".format(event_dict.array('nuVertexX')[events_index],
                                     event_dict.array('nuVertexY')[events_index],
                                     event_dict.array('nuVertexZ')[events_index])
        x_txt_start = 1
        if np.argmax(beam_flash_pe)<nrPMT/2:
            x_txt_start = 17
        
        ax.text(x_txt_start, 0.8*ax.get_ylim()[1], txt, fontsize=12)
        ax.set_ylim(0,ax.get_ylim()[1])
        ax.set_xlim(-0.5,31.5)
    else:
        print(str_pause,"\nProducing plot for Data event!")
        ax.set_title("MicroBooNE Data", loc='right')
        
    fig.tight_layout()
    if 1:
        tag = "run"+str(run)+"_subrun"+str(subrun)+"_event"+str(event)
        fig.savefig(outdir+"event_viewer_"+tag+".pdf", bbox_inches="tight")
        print("Image saved: "+outdir+"event_viewer_"+tag+".pdf" )
