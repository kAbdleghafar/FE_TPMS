### all thermo-mechanical runs: correlation r (RMSE) with the measurements

| run | part hardening | ridge-1 deflection | contour σ_xx part | contour σ_xx plate | neutron ε_xx body | neutron ε_xx legs | synchrotron ε_xx | synchrotron ε_zz |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `base` | macto | 1.234 mm (-3.3 %) | 0.71 (168 MPa) | 0.87 (67 MPa) | 0.88 (608 µε) | 0.22 (526 µε) | 0.80 (667 µε) | 0.92 (643 µε) |
| `flow_baseline` | baseline | 0.771 mm (-39.6 %) | 0.69 (139 MPa) | 0.87 (64 MPa) | 0.88 (307 µε) | 0.15 (449 µε) | 0.79 (605 µε) | 0.91 (700 µε) |
| `flow_e8` | e8 | 1.384 mm (+8.5 %) | 0.71 (191 MPa) | 0.86 (76 MPa) | 0.88 (732 µε) | 0.21 (570 µε) | 0.80 (725 µε) | 0.92 (834 µε) |
| `steps4` | macto | 1.200 mm (-6.0 %) | 0.71 (167 MPa) | 0.86 (66 MPa) | 0.88 (593 µε) | 0.24 (513 µε) | 0.80 (669 µε) | 0.92 (682 µε) |

### thermo-mechanical `base` (83 min; T_dep = 1350.0 C, T_plate = 20.0 C, 8 heat steps / super-layer, anneal = None, nu = 0.278, part hardening = macto)

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

### thermo-mechanical `flow_baseline` (156 min; T_dep = 1350.0 C, T_plate = 20.0 C, 8 heat steps / super-layer, anneal = None, nu = 0.278, part hardening = baseline)

| quantity (measurement) | points | measured RMS | RMSE | bias | correlation r |
|---|---:|---:|---:|---:|---:|
| ridge-1 deflection after leg cut (measured 1.276 mm) | 1 | | | -0.505 mm | predicted 0.771 mm |
| sigma_xx at x = 31 mm (contour, UC Davis) | 987 | 199 MPa | 139 MPa | -18 MPa | 0.69 |
| sigma_xx in the plate strip at x = 31 mm (contour) | 2548 | 124 MPa | 64 MPa | +1 MPa | 0.87 |
| eps_xx (neutron, NIST NCNR) | 204 | 721 µε | 409 µε | +49 µε | 0.76 |
|   eps_xx, solid body z >= 7 mm (neutron) | 129 | 840 µε | 307 µε | +94 µε | 0.88 |
|   eps_xx, legs z < 5 mm (neutron) | 45 | 394 µε | 449 µε | -18 µε | 0.15 |
| eps_yy (neutron) | 204 | 642 µε | 489 µε | -226 µε | 0.59 |
| eps_zz (neutron) | 204 | 1391 µε | 888 µε | +7 µε | 0.49 |
| eps_xx (synchrotron EDXRD, CHESS) | 2219 | 916 µε | 605 µε | +211 µε | 0.79 |
| eps_zz (synchrotron EDXRD) | 2220 | 1838 µε | 700 µε | +404 µε | 0.91 |

Predicted ridge deflections 1..11 (mm): 0.771, 0.592, 0.444, 0.325, 0.229, 0.151, 0.090, 0.042, 0.011, -0.000, -0.002

For comparison, elastic inherent strain calibrated on ridge 1: legs neutron eps_xx r = 0.13, body r = 0.81, contour r = 0.77.

### thermo-mechanical `flow_e8` (73 min; T_dep = 1350.0 C, T_plate = 20.0 C, 8 heat steps / super-layer, anneal = None, nu = 0.278, part hardening = e8)

| quantity (measurement) | points | measured RMS | RMSE | bias | correlation r |
|---|---:|---:|---:|---:|---:|
| ridge-1 deflection after leg cut (measured 1.276 mm) | 1 | | | +0.108 mm | predicted 1.384 mm |
| sigma_xx at x = 31 mm (contour, UC Davis) | 987 | 199 MPa | 191 MPa | +18 MPa | 0.71 |
| sigma_xx in the plate strip at x = 31 mm (contour) | 2548 | 124 MPa | 76 MPa | -10 MPa | 0.86 |
| eps_xx (neutron, NIST NCNR) | 204 | 721 µε | 729 µε | +371 µε | 0.77 |
|   eps_xx, solid body z >= 7 mm (neutron) | 129 | 840 µε | 732 µε | +625 µε | 0.88 |
|   eps_xx, legs z < 5 mm (neutron) | 45 | 394 µε | 570 µε | -85 µε | 0.21 |
| eps_yy (neutron) | 204 | 642 µε | 500 µε | -72 µε | 0.57 |
| eps_zz (neutron) | 204 | 1391 µε | 1329 µε | -859 µε | 0.51 |
| eps_xx (synchrotron EDXRD, CHESS) | 2219 | 916 µε | 725 µε | +292 µε | 0.80 |
| eps_zz (synchrotron EDXRD) | 2220 | 1838 µε | 834 µε | -400 µε | 0.92 |

Predicted ridge deflections 1..11 (mm): 1.384, 1.065, 0.798, 0.583, 0.410, 0.269, 0.158, 0.073, 0.018, -0.000, -0.004

For comparison, elastic inherent strain calibrated on ridge 1: legs neutron eps_xx r = 0.13, body r = 0.81, contour r = 0.77.

### thermo-mechanical `steps4` (57 min; T_dep = 1350.0 C, T_plate = 20.0 C, 4 heat steps / super-layer, anneal = None, nu = 0.278, part hardening = macto)

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
