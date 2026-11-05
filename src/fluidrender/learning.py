"""Trainable NumPy actor-critic policies with Adam and restartable checkpoints.

Both experts consume the same flattened global state. Training differentiates
log pi_coop, including its normalization, not the unfused expert policies.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
from .training import masked_softmax, counterfactual_baseline, latency_reward, team_reward

FEATURES_PER_NODE = 12
TASK_FEATURES = 5
LEARNED = ('FluidRender', 'JointFromScratch', '1A-Scalar', '1A-MO',
           'LatencyOnly', 'LoadStabilityOnly', '2A-CacheLoad', '3A-SQL', 'MAPPO')


def encode(task, state):
    """Global compute, queue, cache, network, and task summary, identically shared."""
    rows = []
    for n in state.nodes:
        c = state.predicted_components(task, n.node_id)
        stamps = state.directory.entries[n.node_id].values()
        age = min((state.now_ms-s for s in stamps), default=1000.)
        incoming = sum(f.remaining_mb for f in state.network.flows.values() if f.destination == n.node_id)
        rows += [n.capacity/5.2, n.physical_utilization(state.now_ms), c['queue_ms']/100.,
                 c['sync_ms']/100., c['render_ms']/100., c['missing_mb']/2048.,
                 n.cache.free_mb()/max(n.cache.capacity_mb, 1.), n.active_slots(state.now_ms)/n.slots,
                 min(age/1000., 10.), incoming/2048.,
                 state.network.capacities(state.now_ms)['in:'+n.node_id]/12.5,
                 state.post_load_skew(n.node_id, task)]
    rows += [task.demand_ms/100., task.deadline_ms/100., state.working_set_mb(task)/2048.,
             len(task.working_set)/64., task.app/max(1, state.cfg['workload']['applications']-1)]
    return np.nan_to_num(np.asarray(rows, dtype=float), nan=0., posinf=10., neginf=-10.).clip(-10,10)


class MLP:
    def __init__(self, inputs, hidden, outputs, rng):
        self.p = {'w1': rng.normal(0, 1/np.sqrt(inputs), (inputs, hidden)), 'b1': np.zeros(hidden),
                  'w2': rng.normal(0, .05/np.sqrt(hidden), (hidden, outputs)), 'b2': np.zeros(outputs)}
        self.m = {k: np.zeros_like(v) for k,v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k,v in self.p.items()}
        self.step = 0

    @property
    def parameters(self):
        return sum(v.size for v in self.p.values())

    def forward(self, x):
        h = np.tanh(x @ self.p['w1'] + self.p['b1'])
        return h @ self.p['w2'] + self.p['b2']

    def gradients(self, x, dy):
        x = np.atleast_2d(x)
        dy = np.atleast_2d(dy)
        h = np.tanh(x @ self.p['w1'] + self.p['b1'])
        dh = (dy @ self.p['w2'].T) * (1-h*h)
        return {'w1': x.T@dh, 'b1': dh.sum(0), 'w2': h.T@dy, 'b2': dy.sum(0)}

    def update(self, x, dy, lr):
        grad = self.gradients(x, dy)
        norm = np.sqrt(sum(np.square(v).sum() for v in grad.values()))
        self.step += 1
        for k,g in grad.items():
            g = g * min(1., 10./max(norm, 1e-12))
            self.m[k] = .9*self.m[k]+.1*g
            self.v[k] = .999*self.v[k]+.001*g*g
            m = self.m[k]/(1-.9**self.step)
            v = self.v[k]/(1-.999**self.step)
            self.p[k] -= lr*m/(np.sqrt(v)+1e-8)


class Policy:
    def __init__(self, method, nodes, seed=7, hidden=64, actor_lr=1e-4, critic_lr=3e-4,
                 entropy=1e-3, gamma=.99, top_b=0, model='backend'):
        if method not in LEARNED:
            raise ValueError('Unsupported learned method: '+method)
        self.method, self.nodes, self.hidden, self.seed = method, nodes, hidden, seed
        self.actor_lr, self.critic_lr, self.entropy = actor_lr, critic_lr, entropy
        self.gamma, self.top_b, self.model = gamma, top_b, model
        self.rng = np.random.default_rng(seed)
        inputs = nodes*FEATURES_PER_NODE+TASK_FEATURES
        count = 3 if method == '3A-SQL' else 2 if method in ('FluidRender','JointFromScratch','2A-CacheLoad') else 1
        # Match total actor and active critic parameter count to two actors + one Q.
        value_count = 2 if method == '1A-MO' else 1
        target = 3 * (inputs*hidden+hidden+hidden*nodes+nodes)
        choices = range(4, hidden*4+1)
        width = min(choices, key=lambda h: abs((count+value_count)*(inputs*h+h+h*nodes+nodes)-target))
        self.actors = [MLP(inputs, width, nodes, self.rng) for _ in range(count)]
        self.critics = [MLP(inputs, width, nodes, self.rng) for _ in range(value_count)]
        self.precritics = [MLP(inputs, width, nodes, self.rng) for _ in range(count)]
        self.interactions = 0
        self.elapsed_s = 0.
        self.history = []
        self.config = None

    def probabilities(self, x, mask, stage='joint'):
        logits = [a.forward(x) for a in self.actors]
        combined = logits[int(stage)] if stage != 'joint' else np.sum(logits, axis=0)
        if not np.all(np.isfinite(combined)):
            raise FloatingPointError('Actor produced non-finite logits')
        return np.asarray(masked_softmax(combined.tolist(), list(mask)))

    def action(self, x, mask, stage='joint', greedy=True):
        p = self.probabilities(x, mask, stage)
        if not p.any():
            return None
        return int(np.argmax(p)) if greedy else int(self.rng.choice(self.nodes, p=p))

    def reward_vector(self, record, stage, reward_cfg):
        done = record.status == 'completed'
        if stage != 'joint':
            idx = int(stage)
            if self.method == '2A-CacheLoad':
                return [-min(record.sync_ms/100., 1.) if idx == 0 else
                        -(record.queue_ms+record.render_ms)/100.-record.load_skew-2*record.slo_miss]
            if self.method == '3A-SQL':
                return [-min(record.sync_ms/100.,1.) if idx == 0 else
                        -(record.queue_ms+record.render_ms)/100.-2*record.slo_miss if idx == 1 else -record.load_skew]
            if self.method == 'LoadStabilityOnly' or idx == 1:
                return [-record.load_skew]
            return [latency_reward(record.latency_ms, record.deadline_ms, done,
                                   reward_cfg.get('t_ref_ms',100.), reward_cfg.get('w_fail',-1.))]
        if self.method == '1A-MO':
            load = -reward_cfg.get('lambda_load',1.)*record.load_skew
            return [record.team_reward-load, load]
        return [record.team_reward]

    def learn(self, transitions, records, stage, reward_cfg):
        if not transitions:
            return
        byid = {r.task_id:r for r in records}
        xs = np.stack([t['x'] for t in transitions])
        critics = self.critics if stage == 'joint' else [self.precritics[int(stage)]]
        values = [c.forward(xs) for c in critics]
        target = np.zeros((len(xs),len(critics)))
        for i,t in enumerate(transitions):
            target[i] = self.reward_vector(byid[t['task_id']], stage, reward_cfg)
            if i+1 < len(transitions):
                nxt = transitions[i+1]
                pnext = self.probabilities(nxt['x'], nxt['mask'], stage)
                for k in range(len(critics)):
                    target[i,k] += self.gamma * np.dot(pnext, values[k][i+1])
        chosen = [i for i,t in enumerate(transitions) if t['action'] is not None]
        # Rejections contribute costs to predecessor returns but have no placement update.
        for i in range(len(transitions)-2,-1,-1):
            if transitions[i+1]['action'] is None:
                target[i] = self.reward_vector(byid[transitions[i]['task_id']], stage, reward_cfg) + self.gamma*target[i+1]
        for k,c in enumerate(critics):
            dy = np.zeros_like(values[k])
            for i in chosen:
                a = transitions[i]['action']
                dy[i,a] = (values[k][i,a]-target[i,k])/max(1,len(chosen))
            c.update(xs, dy, self.critic_lr)
        if not chosen:
            return
        qtot = sum(c.forward(xs) for c in critics)
        if self.method == 'MAPPO':
            # PPO ratios compare the current policy with the rollout policy.
            actor = self.actors[0]
            advantages = {i: float(target[i,0]-values[0][i,transitions[i]['action']]) for i in chosen}
            for epoch in range(4):
                dy = np.zeros((len(xs),self.nodes))
                for i in chosen:
                    t=transitions[i];p=self.probabilities(t['x'],t['mask'])
                    old=max(t['old_probability'],1e-300);ratio=p[t['action']]/old
                    adv=advantages[i]
                    clipped=(adv>=0 and ratio>1.2) or (adv<0 and ratio<.8)
                    grad=-p.copy();grad[t['action']]+=1.
                    logp=np.log(np.maximum(p,1e-300));entropy=-np.sum(p*logp)
                    dy[i]=((0. if clipped else -adv*ratio)*grad+self.entropy*p*(logp+entropy))/len(chosen)
                actor.update(xs,dy,self.actor_lr)
            return
        actor_indices = range(len(self.actors)) if stage == 'joint' else [int(stage)]
        # Accumulate all expert gradients before either actor is changed.
        gradients = {}
        for k in actor_indices:
            dy = np.zeros((len(xs),self.nodes))
            for i in chosen:
                t = transitions[i]
                p = self.probabilities(t['x'], t['mask'], stage)
                own = masked_softmax(self.actors[k].forward(t['x']).tolist(), t['mask'])
                baseline = counterfactual_baseline(own, qtot[i], self.top_b)
                advantage = qtot[i,t['action']]-baseline
                grad = -p.copy()
                grad[t['action']] += 1.
                # dH/dz = -p*(log(p)+H); minimized loss = -adv*logpi - beta*H.
                logp = np.log(np.maximum(p,1e-300))
                entropy = -np.sum(p*logp)
                dy[i] = (-advantage*grad+self.entropy*p*(logp+entropy))/len(chosen)
            gradients[k] = dy
        for k,dy in gradients.items():
            self.actors[k].update(xs, dy, self.actor_lr)

    def save(self, path, cfg=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.config = cfg or self.config
        meta = {k:getattr(self,k) for k in ('method','nodes','hidden','seed','actor_lr','critic_lr',
                                            'entropy','gamma','top_b','model','interactions','elapsed_s','history')}
        meta.update(format_version=2, rng=self.rng.bit_generator.state, config=self.config)
        arrays = {'metadata':np.asarray(json.dumps(meta))}
        for group in ('actors','critics','precritics'):
            for i,net in enumerate(getattr(self,group)):
                arrays[f'{group}.{i}.step'] = np.asarray(net.step)
                for part in ('p','m','v'):
                    for key,v in getattr(net,part).items():
                        arrays[f'{group}.{i}.{part}.{key}'] = v
        with path.open('wb') as f:
            np.savez_compressed(f, **arrays)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z['metadata']))
            if meta['format_version'] != 2:
                raise ValueError('Unsupported checkpoint version')
            p = cls(**{k:meta[k] for k in ('method','nodes','hidden','seed','actor_lr','critic_lr','entropy','gamma','top_b','model')})
            for key in ('interactions','elapsed_s','history','config'):
                setattr(p,key,meta[key])
            p.rng.bit_generator.state = meta['rng']
            for group in ('actors','critics','precritics'):
                for i,net in enumerate(getattr(p,group)):
                    net.step = int(z[f'{group}.{i}.step'])
                    for part in ('p','m','v'):
                        for key in getattr(net,part):
                            getattr(net,part)[key] = z[f'{group}.{i}.{part}.{key}'].copy()
            return p


class LearnedScheduler:
    def __init__(self, policy, stage='joint', training=False, fallback=None):
        self.policy, self.stage, self.training = policy, stage, training
        self.name = policy.method
        self.transitions = []
        self.fallback = fallback

    def select(self, task, state):
        x, mask = encode(task,state), state.mask(task)
        try:
            action = self.policy.action(x,mask,self.stage,greedy=not self.training)
        except (FloatingPointError, ArithmeticError):
            if self.training or self.fallback is None:
                raise
            return self.fallback.select(task,state)
        placeholder = action
        used_mask = mask
        if self.training and action is None and self.policy.model == 'reference':
            used_mask = [True]*len(mask)
            placeholder = self.policy.action(x,used_mask,self.stage,greedy=False)
        self.transitions.append(dict(x=x, mask=used_mask, action=placeholder, task_id=task.task_id,
            old_probability=float(self.policy.probabilities(x,used_mask,self.stage)[placeholder]) if placeholder is not None else 0.))
        return state.nodes[action].node_id if action is not None else None

    def observe(self, task, record, state):
        pass


def stage_at(policy, completed, pretrain_total):
    if policy.method == 'LatencyOnly' or policy.method == 'LoadStabilityOnly':
        return '0'
    if policy.method in ('JointFromScratch','1A-Scalar','1A-MO','MAPPO') or completed >= pretrain_total:
        return 'joint'
    per = pretrain_total // len(policy.actors)
    return str(min(len(policy.actors)-1,completed//max(1,per)))


def train(policy, cfg, budget, pretrain_total=0, batch_size=256, checkpoint=None,
          checkpoint_every=0, callback=None):
    from .simulator import CloudRenderingSimulator, build_trace
    from .reference import ReferenceSimulator
    start = time.perf_counter()
    while policy.interactions < budget:
        stage = stage_at(policy, policy.interactions, pretrain_total)
        remaining = budget-policy.interactions
        if stage != 'joint' and policy.method not in ('LatencyOnly','LoadStabilityOnly'):
            boundary = pretrain_total*(int(stage)+1)//len(policy.actors)
            remaining = min(remaining,boundary-policy.interactions)
        if checkpoint_every:
            remaining = min(remaining,checkpoint_every-policy.interactions%checkpoint_every)
        seed = policy.seed*1000003 + policy.interactions + 10000
        assets,tasks = build_trace(cfg,seed)
        tasks = tasks[:min(batch_size,remaining)]
        if not tasks:
            raise ValueError('Training trace has no arrivals; increase rollout duration')
        scheduler = LearnedScheduler(policy,stage,training=True)
        engine = ReferenceSimulator if policy.model == 'reference' else CloudRenderingSimulator
        records = engine(cfg,assets,tasks,seed).run(scheduler)
        policy.learn(scheduler.transitions,records,stage,cfg.get('reward',{}))
        policy.interactions += len(tasks)
        policy.elapsed_s += time.perf_counter()-start
        start = time.perf_counter()
        policy.history.append({'interactions':policy.interactions,'stage':stage,
                               'training_cost':-float(np.mean([r.team_reward for r in records]))})
        if callback and (policy.interactions == budget or checkpoint_every and policy.interactions%checkpoint_every == 0):
            callback(policy)
            start = time.perf_counter()  # validation excluded from training wall time
        if checkpoint:
            policy.save(checkpoint,cfg)
    return policy
