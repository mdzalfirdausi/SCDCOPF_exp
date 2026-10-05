"""Run: python -m unittest -v test_reproduction.py. Torch tests skip if absent."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
import numpy as np
from case_data import Grid,load_case,Sampler,apr_numpy,assess_numpy,capacity_deficits
from reference import solve_reference


def toy():
    return Grid('synthetic3',100.,np.array([10,20,30]),np.array([1,2,3]),np.array([0,1,2]),
        np.array([.8,.7,.5]),np.array([0.,0.,0.]),np.array([2.,2.,1.]),np.array([10.,20.,30.]),
        np.array([0,1,0]),np.array([1,2,2]),np.array([.2,.3,.4]),np.ones(3),np.zeros(3),
        np.array([.4,.4,.4]),0,'synthetic-test-only')


class NumericalTests(unittest.TestCase):
    def test_affine_ptdf_and_lodf_against_angle_solve(self):
        g=toy(); g.tap[0]=1.1; g.shift[1]=.07
        P,off,L,kg,ke=g.matrices(); injection=np.array([.8,-.3,-.5])
        def flow(omit=None):
            keep=np.array([i for i in range(3) if i!=omit]); A=np.zeros((len(keep),3))
            A[np.arange(len(keep)),g.source[keep]]=1; A[np.arange(len(keep)),g.target[keep]]=-1
            b=1/(g.reactance[keep]*g.tap[keep]); shift=-b*g.shift[keep]
            B=A.T@(b[:,None]*A); th=np.zeros(3)
            th[1:]=np.linalg.solve(B[1:,1:],(injection-A.T@shift)[1:])
            f=np.zeros(3); f[keep]=b*(A@th)+shift; return f
        f=P@injection+off
        np.testing.assert_allclose(f,flow(),atol=1e-12)
        for j,k in enumerate(ke): np.testing.assert_allclose(f+L[:,j]*f[k],flow(k),atol=1e-12)

    def test_apr_feasible_and_infeasible(self):
        lower=np.array([0.,.1,0.]); u=np.array([[1.5,1.,1.5]])
        g=np.array([[.8,.9,.3]]); d=np.array([[1.,1.]])
        q,n=apr_numpy(g,d,u,lower,1.,np.arange(3))
        np.testing.assert_allclose(q.sum(2),2,atol=1e-10)
        np.testing.assert_allclose(np.diagonal(q,axis1=1,axis2=2),0)
        self.assertTrue(np.all(q<=u[:,None,:]+1e-12))
        q,n=apr_numpy([[3,1.5,.5]],[[5]],[[4,2,1]],np.zeros(3),1,np.arange(3))
        np.testing.assert_allclose(q[0,0],[0,2,1],atol=1e-10)
        self.assertTrue(np.all(n<=1))

    def test_capacity_deficit_explains_constant_old_residual(self):
        D=5.; u=np.array([4.,2.,1.]); values=[]
        for g in (np.array([3,1.5,.5]),np.array([2.5,1.8,.7])):
            r=u-g; R=r[1:].sum(); raw=g[0]/R; n=1+.1*(raw-1)
            values.append((g[1:]+n*r[1:]).sum()-D)
        np.testing.assert_allclose(values,[-1.8,-1.8])

    def test_sampler_joint_truncation_matches_rejection(self):
        grid=toy(); s=Sampler(grid,123,.1)
        d,c,u=s.sample(20000)
        noise=(d-grid.demand)/(.5*np.abs(grid.demand)/s.z)
        self.assertTrue(np.all(abs(noise)<=s.z+1e-10))
        self.assertEqual(np.count_nonzero(np.isclose(abs(noise),s.z,atol=1e-12)),0)
        rng=np.random.default_rng(741)
        z=np.sqrt(.5)*rng.normal(size=(80000,1))+np.sqrt(.5)*rng.normal(size=(80000,3))
        z=z[np.all(abs(z)<=s.z,axis=1)]
        np.testing.assert_allclose(noise.mean(0),z.mean(0),atol=.03)
        np.testing.assert_allclose(np.cov(noise.T),np.cov(z.T),atol=.035)
        self.assertTrue(np.all(c>=0)); self.assertTrue(np.all(u>=grid.lower+.01*(grid.upper-grid.lower)))

    def test_ccga_matches_extensive_and_full_feasibility(self):
        grid=toy(); mat=grid.matrices()
        results=[solve_reference(grid,mat,grid.demand,grid.cost,grid.upper,.5,
                  backend='scipy',seconds=30,gap=1e-7,extensive=full) for full in (False,True)]
        for result in results:
            self.assertTrue(result['certified'],str(result))
            r=assess_numpy(grid,mat,result['dispatch'],grid.demand,grid.cost,grid.upper,.5)
            self.assertLess(r['contingency_balance_max_pu'],1e-6)
            self.assertLess(r['generator_bounds_pu'],1e-6)
            self.assertAlmostEqual(r['objective'],result['objective'],places=5)
        self.assertAlmostEqual(results[0]['objective'],results[1]['objective'],places=4)

    def test_infeasible_reference_not_certified(self):
        grid=toy(); d=grid.demand*3
        r=solve_reference(grid,grid.matrices(),d,grid.cost,grid.upper,1,backend='scipy',seconds=10)
        self.assertFalse(r['certified']); self.assertIsNone(r['dispatch'])

    def test_parser_retains_zero_capacity_generator(self):
        text='''function mpc = synthetic
mpc.baseMVA=100;
mpc.bus=[10 3 80 0 0 0 1 1 0 1 1 1.1 .9;20 1 70 0 0 0 1 1 0 1 1 1.1 .9;30 1 50 0 0 0 1 1 0 1 1 1.1 .9;];
mpc.gen=[10 0 0 0 0 1 100 1 200 0;20 0 0 0 0 1 100 1 200 0;30 0 0 0 0 1 100 1 0 0;];
mpc.gencost=[2 0 0 3 .1 10 7;2 0 0 3 .1 20 9;2 0 0 3 0 0 0;];
mpc.branch=[10 20 0 .2 0 40 0 0 0 0 1 -360 360;20 30 0 .3 0 40 0 0 0 0 1 -360 360;10 30 0 .4 0 40 0 0 0 0 1 -360 360;];
'''
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'case.m'; p.write_text(text); g=load_case(p)
        self.assertEqual(len(g.upper),3)
        np.testing.assert_array_equal(g.matrices()[3],[0,1])
        np.testing.assert_allclose(g.cost,[10,20,0])


@unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch is not installed')
class TorchTests(unittest.TestCase):
    def test_repair_apr_and_chunked_losses(self):
        import torch
        from learning import repair,apr,Physics,MLP,GIN
        torch.set_default_dtype(torch.float64)
        grid=toy(); mat=grid.matrices()
        d=torch.tensor(grid.demand[None,:]); u=torch.tensor(grid.upper[None,:]); c=torch.tensor(grid.cost[None,:])
        lower=torch.tensor(grid.lower); raw=torch.tensor([[.6,.5,.2]],requires_grad=True)
        g=repair(raw,d,lower,u)
        self.assertLess(abs(g.sum().item()-d.sum().item()),1e-12)
        for cls in (MLP,GIN):
            for dual in (False,True):
                net=cls(grid,mat[3],dual,hidden=16)
                self.assertEqual(tuple(net(d,c,u).shape),(1,3))
        p=Physics(grid,mat,.5,chunk=1)
        obj,h,eta=p.terms(g,d,c,u)
        ref=assess_numpy(grid,mat,g.detach().numpy()[0],grid.demand,grid.cost,grid.upper,.5)
        self.assertAlmostEqual(obj.item(),ref['objective'],places=6)
        self.assertAlmostEqual(eta.item(),ref['total_slack_pu'],places=6)
        (obj.mean()+h.square().mean()).backward()
        self.assertTrue(torch.isfinite(raw.grad).all())
        # Fixed-signal APR backward: check the expression derivative, not a
        # finite difference through a root solver whose derivative is different.
        raw=torch.tensor([[.8,.9,.3]],requires_grad=True)
        q,n=apr(raw,torch.tensor([[1.,1.]]),torch.tensor([[1.5,1.,1.5]]),
                 torch.tensor([0.,.1,0.]),1,torch.tensor([0]))
        self.assertFalse(n.requires_grad)
        q.sum().backward(); self.assertEqual(raw.grad[0,0].item(),0)
        self.assertTrue(torch.isfinite(raw.grad).all())

if __name__=='__main__': unittest.main()
