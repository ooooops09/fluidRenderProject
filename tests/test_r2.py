import copy,json,math,tempfile,unittest,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fluidrender.core import load_config,Asset,Task,percentile
from fluidrender.network import progressive_fill,FluidNetwork
from fluidrender.telemetry import Report,ValidatedDirectory,sign,verify_chunk
from fluidrender.training import masked_softmax,product_of_experts,team_reward,counterfactual_baseline
from fluidrender.learning import Policy,MLP,LearnedScheduler,train,stage_at
from fluidrender.simulator import CloudRenderingSimulator,SimulationState,PlacementRecord,build_trace
from fluidrender.schedulers import make_scheduler
from fluidrender.metrics import summarize,aggregate,hierarchical_aggregate,tail_composition


def config():
    c=load_config(str(ROOT/'configs/default.json'))
    c['simulation'].update(duration_s=1.,warmup_s=0.)
    c['workload'].update(active_sessions=10)
    return c


class NetworkTests(unittest.TestCase):
    def test_shared_link(self):
        self.assertEqual(progressive_fill({'x':10},{'a':('x',),'b':('x',)}),{'a':5,'b':5})
    def test_independent_bottlenecks(self):
        r=progressive_fill({'a':2,'b':20,'x':10},{0:('a','x'),1:('b','x')})
        self.assertAlmostEqual(r[0],2);self.assertAlmostEqual(r[1],8)
    def test_weighted_redistribution(self):
        r=progressive_fill({'x':20},{0:('x',),1:('x',)}, {0:1,1:3})
        self.assertAlmostEqual(r[0],5);self.assertAlmostEqual(r[1],15)
    def test_capacity_invariants(self):
        rng=np.random.default_rng(7)
        for _ in range(40):
            cap={i:float(rng.uniform(0,10)) for i in range(5)}
            paths={j:tuple(int(x) for x in rng.choice(5,3,replace=False)) for j in range(12)}
            r=progressive_fill(cap,paths)
            for link,c in cap.items():
                self.assertLessEqual(sum(r[f] for f,p in paths.items() if link in p),c+1e-8)
    def test_receiver_in_store_path(self):
        s=SimulationState(config(),{},7)
        self.assertIn('in:e5',s.network.path(None,'e5')[0])
        self.assertIn('out:e1',s.network.path('e1','e10')[0])
    def test_measurement_bytes_partial_transfer(self):
        c=config();c['measurement_window_s']=[.001,.002]
        s=SimulationState(c,{},7);f=s.network.add(0,Asset('a',100.,0,None,'x'),None,'e1')
        s.network.advance(0.,3.)
        self.assertAlmostEqual(s.network.window_mb,12.5)
        self.assertAlmostEqual(f.sent_mb,37.5)


class TelemetryTests(unittest.TestCase):
    def setUp(self):
        self.key=b'k'*32;self.d=ValidatedDirectory({'e1':self.key},500,1000)
        self.d.heartbeat('e1',0)
        self.r=Report('e1',0,1,0.,1,True,('a',))
    def test_bad_auth_and_replay(self):
        self.assertFalse(self.d.receive(self.r,b'x'*32,0))
        self.assertTrue(self.d.receive(self.r,sign(self.r,self.key),0))
        self.assertFalse(self.d.receive(self.r,sign(self.r,self.key),1))
    def test_generation_time_expiry_and_heartbeat(self):
        self.d.receive(self.r,sign(self.r,self.key),400)
        self.d.heartbeat('e1',600)
        self.assertFalse(self.d.observed_has('e1','a',600))
    def test_gap_requires_refresh(self):
        self.d.receive(self.r,sign(self.r,self.key),0)
        gap=Report('e1',0,3,3.,3,False,('b',))
        self.assertFalse(self.d.receive(gap,sign(gap,self.key),3))
        self.assertFalse(self.d.observed_has('e1','a',3))
        full=Report('e1',0,4,4.,4,True,('b',))
        self.assertTrue(self.d.receive(full,sign(full,self.key),4))
        self.assertTrue(self.d.observed_has('e1','b',4))
    def test_old_boot_rejected(self):
        new=Report('e1',2,1,0.,1,True,('b',))
        self.d.receive(new,sign(new,self.key),0)
        self.assertFalse(self.d.receive(self.r,sign(self.r,self.key),1))
    def test_digest_and_version(self):
        import hashlib
        digest=hashlib.sha256(b'good').hexdigest()
        self.assertTrue(verify_chunk(b'good',2,2,digest))
        self.assertFalse(verify_chunk(b'bad',2,2,digest))
        self.assertFalse(verify_chunk(b'good',1,2,digest))


class LearningTests(unittest.TestCase):
    def test_mask_empty_is_not_uniform(self):
        self.assertEqual(masked_softmax([1.,2.],[False,False]),[0.,0.])
    def test_poe(self):
        a=[1.,3.,2.];b=[3.,1.,4.];mask=[True,False,True]
        x=product_of_experts(a,b,mask)
        self.assertEqual(x[1],0);self.assertAlmostEqual(x[2]/x[0],math.exp(2))
    def test_reward_fixed_reference(self):
        self.assertEqual(team_reward(50,80,.1),-.6)
        self.assertEqual(team_reward(150,80,0),-3.)
        self.assertEqual(team_reward(None,80,.1,completed=False),-3.1)
    def test_top_b_renormalization(self):
        self.assertAlmostEqual(counterfactual_baseline([.2,.8],[1.,3.],1),3.)
    def test_mlp_finite_difference(self):
        net=MLP(3,4,2,np.random.default_rng(7));x=np.array([.1,-.2,.7]);dy=np.array([.5,-.9])
        grad=net.gradients(x,dy)
        for key in net.p:
            idx=tuple(0 for _ in net.p[key].shape);old=net.p[key][idx];eps=1e-6
            net.p[key][idx]=old+eps;plus=float(net.forward(x)@dy)
            net.p[key][idx]=old-eps;minus=float(net.forward(x)@dy);net.p[key][idx]=old
            self.assertAlmostEqual(grad[key][idx],(plus-minus)/(2*eps),places=7)
    def test_fused_gradient_finite_difference(self):
        z=np.array([.2,.4,-.1]);mask=[True,True,False];a=1
        p=np.array(masked_softmax(z.tolist(),mask));expected=-p;expected[a]+=1
        for i in range(3):
            plus=z.copy();minus=z.copy();plus[i]+=1e-6;minus[i]-=1e-6
            numerical=(math.log(masked_softmax(plus.tolist(),mask)[a])-math.log(masked_softmax(minus.tolist(),mask)[a]))/2e-6
            self.assertAlmostEqual(expected[i],numerical,places=7)
    def test_model_parameter_match(self):
        counts=[]
        for method in ('FluidRender','JointFromScratch','1A-Scalar','1A-MO','3A-SQL'):
            p=Policy(method,10);counts.append(sum(n.parameters for n in p.actors+p.critics))
        self.assertLess(max(counts)/min(counts),1.05)
    def test_checkpoint_optimizer_rng_roundtrip(self):
        p=Policy('FluidRender',2,7,hidden=8);x=np.ones(29)
        p.actors[0].update(x,np.array([.1,-.1]),.001);p.rng.random()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'c.npz';p.save(path);q=Policy.load(path)
            np.testing.assert_array_equal(p.actors[0].forward(x),q.actors[0].forward(x))
            self.assertEqual(p.rng.random(),q.rng.random())
            p.actors[0].update(x,np.array([.2,-.2]),.001);q.actors[0].update(x,np.array([.2,-.2]),.001)
            np.testing.assert_array_equal(p.actors[0].p['w1'],q.actors[0].p['w1'])
    def test_stage_budget(self):
        p=Policy('FluidRender',2)
        self.assertEqual(stage_at(p,0,20),'0');self.assertEqual(stage_at(p,10,20),'1')
        self.assertEqual(stage_at(p,20,20),'joint')
    def test_real_parameter_updates(self):
        c=config();p=Policy('FluidRender',10,hidden=8);old=p.actors[0].p['w1'].copy()
        train(p,c,12,8,batch_size=4)
        self.assertEqual(p.interactions,12);self.assertFalse(np.array_equal(old,p.actors[0].p['w1']))
        self.assertEqual({r['stage'] for r in p.history},{'0','1','joint'})


class SimulatorTests(unittest.TestCase):
    def test_all_rejected_counted(self):
        c=config();c['cluster']['initial_cache_fill']=0
        a={'a':Asset('a',32,0,None,'test')};t=Task(0,5,0,0,0,('a',),10000,1,1,'low')
        sim=CloudRenderingSimulator(c,a,[t],7);r=sim.run(make_scheduler('RR',c))[0]
        self.assertEqual(r.status,'rejected');self.assertEqual(r.slo_miss,1)
        row=summarize([r],1);self.assertIsNone(row['p99_latency_ms']);self.assertEqual(row['slo_miss_rate'],1)
    def test_drain_after_window(self):
        c=config();c['cluster']['initial_cache_fill']=0
        a={'a':Asset('a',1,0,None,'test')};t=Task(0,999,0,0,0,('a',),10,100,1,'low')
        r=CloudRenderingSimulator(c,a,[t],7).run(make_scheduler('RR',c))[0]
        self.assertEqual(r.status,'completed');self.assertGreater(r.finish_ms,1000)
        self.assertAlmostEqual(r.latency_ms,r.sync_ms+r.queue_ms+r.render_ms)
    def test_stale_peer_store_fallback(self):
        c=config();c['cluster']['initial_cache_fill']=0
        c['cluster']['nodes']=c['cluster']['nodes'][:2]
        a=Asset('a',32.,0,None,'test');state=SimulationState(c,{'a':a},7)
        peer=state.nodes[1];peer.cache.insert(a,0);state._report(peer,full=True,immediate=True)
        peer.cache.items.clear();peer.cache.used_mb=0.
        t=Task(0,5,0,0,0,('a',),1,100,1,'low')
        r=state.run([t],make_scheduler('FIFO',c))[0]
        self.assertEqual(r.status,'completed');self.assertEqual(r.peer_failures,1)
        self.assertGreaterEqual(r.sync_ms,5.)
        self.assertAlmostEqual(state.network.window_store_mb,32.)
    def test_sync_deadline_cancel(self):
        c=config();c['cluster']['initial_cache_fill']=0
        c['controlled_replay']={'enabled':True,'resident_chunks':0}
        c['network']['object_store_gbps']=.001
        a=Asset('a',32.,0,None,'test');t=Task(0,5,0,0,0,('a',),1,10,1,'low')
        sim=CloudRenderingSimulator(c,{'a':a},[t],7)
        r=sim.run(make_scheduler('RR',c))[0]
        self.assertEqual(r.status,'failed');self.assertEqual(r.finish_ms,15.)
        self.assertFalse(sim.state.network.flows)
    def test_numerical_inference_fallback(self):
        c=config();assets,tasks=build_trace(c,7);state=SimulationState(c,assets,7)
        pol=Policy('FluidRender',10);pol.actors[0].p['w1'][0,0]=float('nan')
        fallback=make_scheduler('A-LALL',c)
        learned=LearnedScheduler(pol,fallback=fallback)
        self.assertEqual(learned.select(tasks[0],state),fallback.select(tasks[0],state))
    def test_shared_trace_determinism(self):
        c=config();a,t=build_trace(c,7)
        r1=CloudRenderingSimulator(c,a,t,7).run(make_scheduler('RR',c))
        r2=CloudRenderingSimulator(c,a,t,7).run(make_scheduler('RR',c))
        self.assertEqual([(r.status,r.latency_ms) for r in r1],[(r.status,r.latency_ms) for r in r2])
    def test_run_percentiles_not_pooled(self):
        rows=[{'tasks':2,'p99_latency_ms':10.},{'tasks':100,'p99_latency_ms':100.}]
        self.assertEqual(aggregate(rows)['p99_latency_ms'],55.)
    def test_variance_decomposition(self):
        rows=[]
        for i,(node,lat) in enumerate([('a',1),('a',3),('b',7),('b',9)]):
            r=PlacementRecord('RR',7,i,0,node,100,status='completed',latency_ms=lat,sync_ms=lat)
            rows.append(r)
        out=tail_composition(rows,0,1)
        self.assertEqual(out['within_worker_variance_ms2'],1.)
        self.assertEqual(out['between_worker_variance_ms2'],9.)
    def test_missing_checkpoint_not_scoring_proxy(self):
        with self.assertRaises(ValueError):make_scheduler('FluidRender',config())
        with self.assertRaises(NotImplementedError):make_scheduler('MADDPG',config())


if __name__=='__main__':unittest.main()
