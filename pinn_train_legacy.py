import argparse, json, pathlib
import numpy as np, pandas as pd, torch
from torch import nn

class PINN(nn.Module):
    def __init__(self,nin=6,nout=5,width=128,depth=6):
        super().__init__(); layers=[nn.Linear(nin,width),nn.Tanh()]
        for _ in range(depth-1): layers += [nn.Linear(width,width),nn.Tanh()]
        layers += [nn.Linear(width,nout)]; self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)

def grads(y,x): return torch.autograd.grad(y,x,torch.ones_like(y),create_graph=True,retain_graph=True)[0]
def train(csv,out,epochs=5000,nu=1e-6):
    df=pd.read_csv(csv); ins=['x','y','z','rpm','fill_height','viscosity']; outs=['Ux','Uy','Uz','p','k']
    miss=[c for c in ins+outs if c not in df]
    if miss: raise ValueError('Missing columns: '+','.join(miss))
    X=torch.tensor(df[ins].values,dtype=torch.float32); Y=torch.tensor(df[outs].values,dtype=torch.float32)
    xm,xs=X.mean(0),X.std(0).clamp_min(1e-8); ym,ys=Y.mean(0),Y.std(0).clamp_min(1e-8)
    Xn=((X-xm)/xs).requires_grad_(True); Yn=(Y-ym)/ys
    model=PINN(); opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    hist=[]
    for e in range(epochs):
        opt.zero_grad(); pred=model(Xn); data=((pred-Yn)**2).mean()
        phys=pred*ys+ym; u,v,w,p=phys[:,0:1],phys[:,1:2],phys[:,2:3],phys[:,3:4]
        gu,gv,gw,gp=grads(u,Xn)/xs,grads(v,Xn)/xs,grads(w,Xn)/xs,grads(p,Xn)/xs
        cont=gu[:,0:1]+gv[:,1:2]+gw[:,2:3]
        # first-order steady incompressible momentum residual; viscous second derivatives intentionally omitted in this lightweight app trainer
        rx=u*gu[:,0:1]+v*gu[:,1:2]+w*gu[:,2:3]+gp[:,0:1]
        ry=u*gv[:,0:1]+v*gv[:,1:2]+w*gv[:,2:3]+gp[:,1:2]
        rz=u*gw[:,0:1]+v*gw[:,1:2]+w*gw[:,2:3]+gp[:,2:3]
        residual=(cont.square().mean()+rx.square().mean()+ry.square().mean()+rz.square().mean())
        loss=data+1e-3*residual; loss.backward(); opt.step()
        if e%100==0: hist.append([e,float(loss),float(data),float(residual)])
    out=pathlib.Path(out);out.mkdir(parents=True,exist_ok=True)
    torch.save({'state':model.state_dict(),'xm':xm,'xs':xs,'ym':ym,'ys':ys,'inputs':ins,'outputs':outs},out/'pinn.pt')
    pd.DataFrame(hist,columns=['epoch','loss','data_loss','physics_loss']).to_csv(out/'history.csv',index=False)
    (out/'metadata.json').write_text(json.dumps({'note':'Hybrid CFD-PINN screening model; validate on held-out operating conditions.','epochs':epochs},indent=2))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--csv',required=True);a.add_argument('--out',default='pinn_model');a.add_argument('--epochs',type=int,default=5000);q=a.parse_args();train(q.csv,q.out,q.epochs)
