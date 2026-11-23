import io, math, zipfile
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title='Bottom-Mount GMP Mixer Simulator', layout='wide')
st.title('Bottom-Mount GMP Mixer: Hydrodynamic Screening Simulator')
st.caption('Engineering surrogate for screening and study design. Not a replacement for geometry-specific CFD or GMP qualification.')

# -------------------- model --------------------
def model(par):
    T,H,D,C,rpm,rho,mu,Np,Nq,Pmotor,eta,Trated,sf,nr,nz = [par[k] for k in ['T','H','D','C','rpm','rho','mu','Np','Nq','Pmotor','eta','Trated','sf','nr','nz']]
    R=T/2; V=math.pi*R*R*H; N=rpm/60; tip=math.pi*D*N
    Re=rho*N*D**2/mu
    P=Np*rho*N**3*D**5
    Q=Nq*N*D**3
    torque=P/(2*math.pi*N) if N>0 else 0
    r=np.linspace(0,R,int(nr)); z=np.linspace(0,H,int(nz)); rr,zz=np.meshgrid(r,z)
    ri=D/2; safe=np.maximum(rr, max(R/500,1e-5))
    # Divergence-consistent axisymmetric meridional field generated from streamfunction.
    psi=(0.020*(rr/R)**2*(1-rr/R)**2*np.sin(np.pi*zz/H)*np.exp(-0.15*zz/H)
         +0.006*(rr/R)**2*(1-rr/R)**2*np.sin(2*np.pi*zz/H)*np.exp(-2.2*zz/H)
         +0.013*(rr/R)**2*np.exp(-(rr/max(0.43*D,1e-6))**2)*(1-np.exp(-zz/max(0.8*C,1e-6)))*np.exp(-zz/max(0.8*H,1e-6)))
    ur=-np.gradient(psi,z,axis=0)/safe; uz=np.gradient(psi,r,axis=1)/safe
    ur[:,0]=0; uz[:,0]=uz[:,1]
    vm=np.sqrt(ur**2+uz**2); s=0.42*tip/max(np.percentile(vm,99.5),1e-9); ur*=s; uz*=s
    ut=0.72*tip*(rr/max(ri,1e-6))*np.exp(-((rr-ri)/max(0.34*D,1e-6))**2-((zz-C)/max(0.30*D,1e-6))**2)
    ut+=0.10*tip*(rr/R)*np.exp(-zz/max(0.44*H,1e-6))*(1-rr/R)
    vel=np.sqrt(ur**2+uz**2+ut**2)
    durdr=np.gradient(ur,r,axis=1); durdz=np.gradient(ur,z,axis=0)
    duzdr=np.gradient(uz,r,axis=1); duzdz=np.gradient(uz,z,axis=0)
    dutdr=np.gradient(ut,r,axis=1); dutdz=np.gradient(ut,z,axis=0)
    shear=np.sqrt(np.maximum(0,2*(durdr**2+duzdz**2+(ur/safe)**2)+(durdz+duzdr)**2+(dutdr-ut/safe)**2+dutdz**2))
    tau=mu*shear
    k=0.015*vel**2 + 1.5*(P/(rho*V))**(2/3)*np.exp(-zz/max(0.35*H,1e-6))
    eps=(P/(rho*V))*(0.25+5*np.exp(-((rr-ri)/max(0.35*D,1e-6))**2-((zz-C)/max(0.30*D,1e-6))**2))
    # volume weights for axisymmetric cells
    w=2*np.pi*rr; w/=w.sum()
    threshold=par['dead_threshold']; dead=float(w[vel<threshold].sum()*100)
    # Lumped tracer mixing: normalized concentration at worst probe, exponential approach to 1
    turnover=V/max(Q,1e-12); tau_mix=1.65*turnover*(1+0.20*(H/T)**1.2)*(1+0.10*max(0,1e4/Re)**0.25)
    t=np.linspace(0,6*tau_mix,240); conc=1-np.exp(-t/tau_mix)
    times={p:-tau_mix*np.log(1-p/100) for p in (90,95,99)}
    available_torque=min(Trated, Pmotor*1000*eta/max(2*math.pi*N,1e-12)); margin=available_torque/max(sf*torque,1e-12)
    return locals()

def contour(x,y,f,title,unit,colors='Turbo', vectors=None):
    fig=go.Figure(go.Contour(x=x,y=y,z=f,colorscale=colors,colorbar_title=unit,contours=dict(coloring='heatmap',showlines=False)))
    if vectors is not None:
        ur,uz=vectors; si=max(1,len(y)//22); sj=max(1,len(x)//18)
        xs=[];ys=[]
        scale=0.035/max(np.percentile(np.sqrt(ur**2+uz**2),95),1e-9)
        for i in range(0,len(y),si):
            for j in range(0,len(x),sj):
                xs += [x[j],x[j]+ur[i,j]*scale,None]; ys += [y[i],y[i]+uz[i,j]*scale,None]
        fig.add_trace(go.Scatter(x=xs,y=ys,mode='lines',line=dict(color='white',width=1),hoverinfo='skip'))
    fig.update_layout(title=title,xaxis_title='Radius (m)',yaxis_title='Liquid height (m)',height=520,margin=dict(l=30,r=20,t=55,b=35))
    fig.update_yaxes(scaleanchor='x',scaleratio=1)
    return fig

with st.sidebar:
    st.header('Inputs')
    T=st.number_input('Vessel diameter T (m)',0.1,10.0,0.80,0.01)
    H=st.number_input('Liquid height H (m)',0.05,10.0,0.80,0.01)
    D=st.number_input('Impeller diameter D (m)',0.01,float(T*0.95),min(0.28,float(T*0.9)),0.01)
    C=st.number_input('Impeller center clearance C (m)',0.005,float(H*0.5),min(0.07,float(H*0.4)),0.005)
    rpm=st.number_input('Speed (rpm)',1.0,3000.0,250.0,5.0)
    rho=st.number_input('Density (kg/m3)',100.0,3000.0,1000.0,10.0)
    mu_mpas=st.number_input('Dynamic viscosity (mPa.s)',0.1,1e6,10.0,1.0)
    Np=st.number_input('Power number Np',0.01,20.0,0.32,0.01)
    Nq=st.number_input('Flow number Nq',0.01,5.0,0.60,0.01)
    dead_threshold=st.number_input('Dead-zone velocity threshold (m/s)',0.0001,5.0,0.05,0.01,format='%.4f')
    st.subheader('Drive / magnetic coupling')
    Pmotor=st.number_input('Motor power (kW)',0.01,1000.0,1.50,0.1)
    eta=st.number_input('Drive efficiency',0.1,1.0,0.85,0.01)
    Trated=st.number_input('Vendor coupling torque limit (N.m)',0.1,10000.0,20.0,0.5)
    sf=st.number_input('Required torque safety factor',1.0,3.0,1.50,0.05)

par=dict(T=T,H=H,D=D,C=C,rpm=rpm,rho=rho,mu=mu_mpas/1000,Np=Np,Nq=Nq,Pmotor=Pmotor,eta=eta,Trated=Trated,sf=sf,dead_threshold=dead_threshold,nr=100,nz=150)
m=model(par)

tabs=st.tabs(['Dashboard','2D fields','3D view','Tracer & blend time','Volume envelope','Equations, assumptions & calculations','Data export'])
with tabs[0]:
    c=st.columns(6)
    vals=[('Reynolds',m['Re']),('Power',m['P']/1000),('P/V',m['P']/m['V']),('Tip speed',m['tip']),('Dead zone',m['dead']),('Torque margin',m['margin'])]
    units=['','kW','W/m3','m/s','%','x']
    for col,(lab,val),u in zip(c,vals,units): col.metric(lab,f'{val:,.3g} {u}')
    st.plotly_chart(contour(m['r'],m['z'],m['vel'],'Velocity magnitude with radial/axial vectors','m/s','Turbo',(m['ur'],m['uz'])),use_container_width=True)
    if m['margin']<1: st.error('Magnetic coupling torque margin is below 1.0. The assumed drive/coupling is inadequate at the selected safety factor.')
    elif m['margin']<1.5: st.warning('Torque margin is limited. Obtain vendor torque-versus-speed and temperature derating data.')
    else: st.success('Screening torque margin is acceptable, subject to vendor coupling limits and start-up torque verification.')
with tabs[1]:
    choices={'Velocity magnitude':(m['vel'],'m/s','Turbo'),'Radial velocity ur':(m['ur'],'m/s','RdBu'),'Axial velocity uz':(m['uz'],'m/s','RdBu'),'Tangential velocity uθ':(m['ut'],'m/s','Viridis'),'Shear-rate magnitude':(m['shear'],'1/s','Inferno'),'Viscous shear stress μγ':(m['tau'],'Pa','Magma'),'Turbulent kinetic energy k':(m['k'],'m2/s2','Cividis'),'Turbulent dissipation ε':(m['eps'],'m2/s3','Plasma')}
    selected=st.multiselect('Fields',list(choices),default=['Velocity magnitude','Shear-rate magnitude','Turbulent kinetic energy k','Turbulent dissipation ε'])
    for name in selected:
        f,u,cs=choices[name]; st.plotly_chart(contour(m['r'],m['z'],f,name,u,cs,(m['ur'],m['uz']) if name=='Velocity magnitude' else None),use_container_width=True)
with tabs[2]:
    st.info('The 3D display is an axisymmetric revolution of the 2D surrogate, not a blade-resolved CFD result.')
    field_name=st.selectbox('3D field',['Velocity magnitude','Shear-rate magnitude','Turbulent kinetic energy k','Turbulent dissipation ε'])
    key={'Velocity magnitude':'vel','Shear-rate magnitude':'shear','Turbulent kinetic energy k':'k','Turbulent dissipation ε':'eps'}[field_name]
    th=np.linspace(0,2*np.pi,70); ridx=np.arange(0,len(m['r']),3); zidx=np.arange(0,len(m['z']),4)
    rr=m['rr'][np.ix_(zidx,ridx)]; zz=m['zz'][np.ix_(zidx,ridx)]; ff=m[key][np.ix_(zidx,ridx)]
    # render several meridional sheets for readable 3D screening
    fig=go.Figure()
    for ang in np.linspace(0,2*np.pi,8,endpoint=False):
        x=rr*np.cos(ang); y=rr*np.sin(ang)
        fig.add_trace(go.Surface(x=x,y=y,z=zz,surfacecolor=ff,colorscale='Turbo',cmin=np.nanmin(ff),cmax=np.nanpercentile(ff,99.5),showscale=(ang==0),colorbar_title=field_name))
    fig.update_layout(height=720,scene=dict(aspectmode='data',xaxis_title='x (m)',yaxis_title='y (m)',zaxis_title='z (m)'),title='3D meridional-sheet reconstruction')
    st.plotly_chart(fig,use_container_width=True)
with tabs[3]:
    fig=go.Figure(go.Scatter(x=m['t'],y=m['conc'],mode='lines',name='Worst-point normalized concentration'))
    for p in (90,95,99): fig.add_vline(x=m['times'][p],line_dash='dash',annotation_text=f't{p}={m["times"][p]:.1f}s')
    fig.update_layout(xaxis_title='Time (s)',yaxis_title='C/C∞',yaxis_range=[0,1.03],height=500)
    st.plotly_chart(fig,use_container_width=True)
    st.dataframe(pd.DataFrame({'Metric':['Turnover V/Q','t90','t95','t99'],'Time (s)':[m['turnover'],m['times'][90],m['times'][95],m['times'][99]]}),hide_index=True,use_container_width=True)
with tabs[4]:
    fmin,fmax=st.slider('Minimum and maximum liquid-height fraction',0.15,1.50,(0.30,1.00),0.05)
    hs=np.linspace(fmin*H,fmax*H,12); rows=[]
    for hh in hs:
        pp=par.copy();pp['H']=float(hh); mm=model(pp)
        rows.append([hh,mm['V']*1000,mm['P']/mm['V'],mm['times'][95],mm['dead'],mm['margin']])
    df=pd.DataFrame(rows,columns=['Liquid height (m)','Volume (L)','P/V (W/m3)','t95 (s)','Dead-zone (%)','Torque margin (x)'])
    st.dataframe(df.style.format(precision=2),use_container_width=True)
    st.line_chart(df.set_index('Volume (L)')[['t95 (s)','Dead-zone (%)']])
with tabs[5]:
    st.header('Model basis, equations and assumptions')
    st.markdown(r'''### Governing screening equations
- Rotational speed: $N=\mathrm{rpm}/60$
- Impeller Reynolds number: $Re=\rho N D^2/\mu$
- Tip speed: $u_{tip}=\pi DN$
- Power draw: $P=N_p\rho N^3D^5$
- Pumping rate: $Q=N_qND^3$
- Vessel volume: $V=\pi(T/2)^2H$
- Bulk energy dissipation: $\bar{\varepsilon}=P/(\rho V)$
- Power density: $P/V$
- Shaft torque: $\tau=P/(2\pi N)$
- Available drive torque: $\tau_{avail}=\min[\tau_{vendor},P_{motor}\eta/(2\pi N)]$
- Torque margin: $M=\tau_{avail}/(SF\,\tau)$

### Velocity field surrogate
A cylindrical streamfunction $\psi(r,z)$ creates a divergence-consistent recirculation field:
$u_r=-(1/r)\partial\psi/\partial z$ and $u_z=(1/r)\partial\psi/\partial r$.
The tangential component is an empirical Gaussian blade-zone field scaled to tip speed. The magnitude is
$|\mathbf{u}|=\sqrt{u_r^2+u_z^2+u_\theta^2}$.

### Shear, stress and turbulence surrogates
Shear-rate magnitude is computed from the axisymmetric rate-of-deformation tensor:
$\dot\gamma=\sqrt{2\mathbf{D}:\mathbf{D}}$. For a Newtonian liquid, local viscous stress is $\tau_v=\mu\dot\gamma$.
The displayed $k$ and $\varepsilon$ are screening correlations tied to local velocity and $P/(\rho V)$, not solved RANS transport equations.

### Tracer and blend time
A lumped first-order approach is used: $C/C_\infty=1-\exp(-t/\tau_m)$.
The characteristic time is correlated to turnover time $V/Q$, aspect ratio and a Reynolds correction. Therefore
$t_p=-\tau_m\ln(1-p)$ for $p=0.90,0.95,0.99$.

### Dead-zone calculation
Axisymmetric volume weighting uses $dV=2\pi r\,dr\,dz$. Dead-zone percentage is the weighted fraction where $|\mathbf{u}|$ is below the selected threshold.

### Assumptions
1. Cylindrical, unbaffled tank; flat liquid surface; centered bottom impeller.
2. Single-phase, incompressible, isothermal Newtonian liquid.
3. Constant $N_p$ and $N_q$ supplied by the user; no Reynolds-dependent vendor curve.
4. Axisymmetric mean-flow approximation; no blade-resolved wakes, free-surface vortex, gas entrainment or solids.
5. No-slip wall effects are only represented qualitatively by the surrogate field.
6. $k$, $\varepsilon$, wall stress and tracer results are estimates for study design, not qualification evidence.
7. Magnetic torque assessment uses steady-state hydraulic torque. Start-up, bearing drag, coupling slip, temperature derating and transient overload require vendor data.

### Current-case calculations''')
    calc=pd.DataFrame({'Quantity':['N','V','Re','Tip speed','P','Q','V/Q','P/V','P/(rho V)','Hydraulic torque','Available torque','Torque margin','t90','t95','t99'],
                       'Value':[m['N'],m['V'],m['Re'],m['tip'],m['P'],m['Q'],m['turnover'],m['P']/m['V'],m['P']/(rho*m['V']),m['torque'],m['available_torque'],m['margin'],m['times'][90],m['times'][95],m['times'][99]],
                       'Unit':['1/s','m3','-','m/s','W','m3/s','s','W/m3','W/kg','N.m','N.m','x','s','s','s']})
    st.dataframe(calc.style.format({'Value':'{:.5g}'}),hide_index=True,use_container_width=True)
    st.warning('For a defensible final model, replace the surrogate with blade-resolved CAD, mesh-independence testing, MRF/sliding mesh, appropriate laminar or turbulence closure, passive-scalar transport and experimental conductivity validation.')
with tabs[6]:
    flat=pd.DataFrame({'r_m':m['rr'].ravel(),'z_m':m['zz'].ravel(),'ur_m_s':m['ur'].ravel(),'uz_m_s':m['uz'].ravel(),'ut_m_s':m['ut'].ravel(),'velocity_m_s':m['vel'].ravel(),'shear_1_s':m['shear'].ravel(),'stress_Pa':m['tau'].ravel(),'k_m2_s2':m['k'].ravel(),'epsilon_m2_s3':m['eps'].ravel()})
    st.download_button('Download field data CSV',flat.to_csv(index=False).encode(),'gmp_mixer_fields.csv','text/csv')
    st.download_button('Download current inputs CSV',pd.DataFrame([par]).to_csv(index=False).encode(),'gmp_mixer_inputs.csv','text/csv')
