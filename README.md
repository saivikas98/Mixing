# Mixing & Heat Transfer Workbench

A Streamlit screening package reverse-engineered from the workflow and visible calculation structure of **Mixing_and_heat_transfer_toolbox (2).xlsm**. It is an independent implementation, not a byte-for-byte clone.

## Run locally
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Included
- Single-vessel mixing: Re, Np, power, P/V, torque, tip speed, Froude number, blend-time estimate
- Scale-up by constant tip speed, P/V, Reynolds number or Froude number
- Transient heat-up/cool-down with UA, heat generation and ambient loss
- Back-calculation of UA required for a target time/temperature
- Editable multi-case comparison and CSV downloads

## Engineering basis
- `Re = rho*N*D^2/mu`
- `P = Np*rho*N^3*D^5`
- `tip speed = pi*D*N`
- `Fr = N^2*D/g`
- Lumped energy balance: `m*Cp*dT/dt = UA*(Tj-T) + Qgen - UAloss*(T-Tamb)`

Default impeller constants are screening values and must be replaced with vendor/experimental constants for design work.

## Important limitation
The supplied workbook includes signed VBA, a vessel database, many hidden/support calculations and specialist Scale-up Systems/Dynochem workflows. This app intentionally provides a transparent minimum viable engineering core. Exact parity requires an agreed validation dataset of workbook input/output cases and permission to reproduce any proprietary algorithms/data.
