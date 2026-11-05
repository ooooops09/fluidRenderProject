"""Workload shifts and controlled-replay configurations."""
import copy
from .core import merge_config


def generalization_configs(cfg):
    net=cfg['network']
    return [(name,merge_config(cfg,override)) for name,override in [
        ('arrival',{'workload':{'lambda_per_session':.25}}),
        ('cache',{'cluster':{'cache_mb_per_node':8192}}),
        ('popularity',{'workload':{'zipf_alpha':.4}}),
        ('network',{'network':{k:net[k]*.5 for k in ('worker_access_gbps','rack_uplink_gbps','object_store_gbps')}}),
        ('complexity',{'workload':{'render_demand_scale':1.5}}),
        ('telemetry',{'telemetry':{'extra_delay_ms':300.}}),
    ]]


def replay_cases(cfg,suite,start=600.,end=720.):
    def case(name,overrides,method='RR'):
        c=merge_config(cfg,overrides)
        c['measurement_window_s']=[start,end]
        return name,c,method
    delay={'start_s':start,'end_s':end,'extra_delay_ms':800.}
    if suite=='network':
        out=[case('nominal',{})]
        for link in ('rack_up:0','store','in:e5'):
            for factor in (.3,.1,.03):
                out.append(case(link.replace(':','_')+'_'+str(factor),{'network':{'degradations':[
                    {'start_s':start,'end_s':end,'link':link,'factor':factor}]}}))
        for rho in (1,3):
            out.append(case('qos_1_'+str(rho),{'network':{'background_weight_rho':rho},
                                               'telemetry':{'delay_events':[delay]}}))
        return out
    if suite=='tail':
        return [case('nominal',{}),case('ingress',{'network':{'degradations':[
            {'start_s':start,'end_s':end,'link':'in:e5','factor':.1}]}}),case('hotspot',{},'Hotspot')]
    if suite=='freshness':
        return [case('timely',{})]+[case('expiry_'+str(t),{'telemetry':{'cache_stale_ms':t,'delay_events':[delay]}})
                                           for t in (100,250,500,1000,None)]
    if suite=='recovery':
        return [case('delay',{'telemetry':{'delay_events':[delay]}}),case('compound',{
                'telemetry':{'delay_events':[delay]},'network':{'degradations':[
                {'start_s':start,'end_s':end,'link':'store','factor':6./200.}]}})]
    raise ValueError(suite)
