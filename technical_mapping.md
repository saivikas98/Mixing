# Reverse-engineering map

| Workbook area | App equivalent | Status |
|---|---|---|
| Start Here vessel/recipe inputs | Mixing and Heat transfer tabs | Implemented, simplified |
| Mixing results | Re, Np, P, P/V, torque, tip speed, blend time, Fr | Implemented |
| Heat transfer results / HT analysis | Dynamic temperature profile and UA back-calculation | Implemented, lumped model |
| Vessel 1 / Vessel 2 comparison | Scale-up tab and case comparison | Implemented |
| Impeller properties | Small editable-in-code impeller constant dictionary | Partial |
| VesselDB / equipment data service | User-entered geometry | Not copied |
| VBA/ribbon automation | Native Streamlit interaction | Replaced |
| Gas-liquid, liquid-liquid, solids, fed-batch and reaction simulation | None | Future modules |

## Recommended validation
1. Select 8 to 12 workbook cases spanning laminar, transitional and turbulent regimes.
2. Compare Re, Np, power, P/V, torque and tip speed; target exact equality for definitional metrics and document correlation differences.
3. Fit impeller-specific Np/Re and blend-time constants against approved workbook or plant data.
4. Fit UA from at least three heat-up/cool-down batches per vessel and assess RMSE and maximum temperature error.
5. Lock model version, units, constants and test cases before GMP-adjacent use.
