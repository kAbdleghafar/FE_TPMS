
### thermo-mechanical `base` (83 min; T_dep = 1350.0 C, T_plate = 20.0 C, 8 heat steps / super-layer, anneal = None, nu = 0.278)

| quantity (measurement) | points | measured RMS | RMSE | bias | correlation r |
|---|---:|---:|---:|---:|---:|
| ridge-1 deflection after leg cut (measured 1.276 mm) | 1 | | | -0.042 mm | predicted 1.234 mm |
| sigma_xx at x = 31 mm (contour, UC Davis) | 987 | 199 MPa | 168 MPa | +10 MPa | 0.71 |
| sigma_xx in the plate strip at x = 31 mm (contour) | 2548 | 124 MPa | 67 MPa | -8 MPa | 0.87 |
| eps_xx (neutron, NIST NCNR) | 204 | 721 µε | 631 µε | +308 µε | 0.77 |
|   eps_xx, solid body z >= 7 mm (neutron) | 129 | 840 µε | 608 µε | +510 µε | 0.88 |
|   eps_xx, legs z < 5 mm (neutron) | 45 | 394 µε | 526 µε | -49 µε | 0.22 |
| eps_yy (neutron) | 204 | 642 µε | 478 µε | -115 µε | 0.58 |
| eps_zz (neutron) | 204 | 1391 µε | 1158 µε | -637 µε | 0.51 |
| eps_xx (synchrotron EDXRD, CHESS) | 2219 | 916 µε | 667 µε | +287 µε | 0.80 |
| eps_zz (synchrotron EDXRD) | 2220 | 1838 µε | 643 µε | -194 µε | 0.92 |

Predicted ridge deflections 1..11 (mm): 1.234, 0.949, 0.712, 0.520, 0.366, 0.240, 0.142, 0.066, 0.016, -0.000, -0.004

For comparison, elastic inherent strain calibrated on ridge 1: legs neutron eps_xx r = 0.13, body r = 0.81, contour r = 0.77.

### thermo-mechanical `steps4` (57 min; T_dep = 1350.0 C, T_plate = 20.0 C, 4 heat steps / super-layer, anneal = None, nu = 0.278)

| quantity (measurement) | points | measured RMS | RMSE | bias | correlation r |
|---|---:|---:|---:|---:|---:|
| ridge-1 deflection after leg cut (measured 1.276 mm) | 1 | | | -0.076 mm | predicted 1.200 mm |
| sigma_xx at x = 31 mm (contour, UC Davis) | 987 | 199 MPa | 167 MPa | +6 MPa | 0.71 |
| sigma_xx in the plate strip at x = 31 mm (contour) | 2548 | 124 MPa | 66 MPa | -6 MPa | 0.86 |
| eps_xx (neutron, NIST NCNR) | 204 | 721 µε | 617 µε | +306 µε | 0.77 |
|   eps_xx, solid body z >= 7 mm (neutron) | 129 | 840 µε | 593 µε | +495 µε | 0.88 |
|   eps_xx, legs z < 5 mm (neutron) | 45 | 394 µε | 513 µε | -41 µε | 0.24 |
| eps_yy (neutron) | 204 | 642 µε | 490 µε | -84 µε | 0.55 |
| eps_zz (neutron) | 204 | 1391 µε | 1203 µε | -721 µε | 0.52 |
| eps_xx (synchrotron EDXRD, CHESS) | 2219 | 916 µε | 669 µε | +292 µε | 0.80 |
| eps_zz (synchrotron EDXRD) | 2220 | 1838 µε | 682 µε | -270 µε | 0.92 |

Predicted ridge deflections 1..11 (mm): 1.200, 0.922, 0.692, 0.505, 0.355, 0.233, 0.137, 0.063, 0.015, -0.000, -0.003

For comparison, elastic inherent strain calibrated on ridge 1: legs neutron eps_xx r = 0.13, body r = 0.81, contour r = 0.77.
