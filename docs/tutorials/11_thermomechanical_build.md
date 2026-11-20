# Tutorial 11: A thermo-mechanical AM build

**Goal.** Deposit a block layer by layer hot, conduct and lose the heat, solve the
temperature-dependent elasto-plastic stresses, cool down and remove from the plate. About
1 minute.

**You will learn**
* temperature-dependent materials (`TemperatureTable`);
* `ThermoMechanicalAM`: deposition, cooling, removal;
* where the residual stresses come from.

## Step 1: a temperature-dependent material

```python
import numpy as np
import tpmsfem as tf
from tpmsfem.fem.materials import Material, TemperatureTable, ThermalProperties
from tpmsfem.geometry import box
from tpmsfem.mesh import voxelize
from tpmsfem.residual.thermomech import ThermoMechanicalAM
from tpmsfem.thermal import HeatModel

th = ThermalProperties.from_si([20.0, 1600.0], [7.0, 30.0], [530.0, 750.0], 4430.0, 2.9e5, 1604.0, 1660.0)
temp = TemperatureTable(T=[20.0, 800.0, 1600.0], E=[110e3, 70e3, 10e3], alpha=[8.6e-6, 10e-6, 11e-6],
                        T_ref=20.0, T_cap=1604.0)
mat = Material("Ti64", E=110e3, nu=0.32, yield_stress=900.0, hardening=1000.0, temperature=temp, thermal=th)
for T in (20.0, 800.0, 1500.0):
    print(f"T = {T:6.0f} C: E = {temp.E_at(T) / 1e3:5.1f} GPa, thermal strain {temp.thermal_strain(T) * 1e3:.2f} e-3")
```

Output (from the test run):

```text
T =     20 C: E = 110.0 GPa, thermal strain 0.00 e-3
T =    800 C: E =  70.0 GPa, thermal strain 7.80 e-3
T =   1500 C: E =  17.5 GPa, thermal strain 16.10 e-3
```

## Step 2: build, cool, remove

```python
mesh = voxelize(box((0, 0, 0), (3, 3, 2)), (0, 0, 0), (3, 3, 2), 0.25)
md = tf.SolidModel(mesh, mat, T_initial=100.0)
hm = HeatModel(mesh, mat, T_initial=100.0)
hm.set_surface_losses(h=0.01, emissivity=0.3, T_amb=100.0)
tm = ThermoMechanicalAM(md, hm, layer_height=0.5, T_deposit=1650.0, layer_time=2.0, n_steps=3, T_plate=100.0)
tm.build(verbose=False)
s_build = md.stress().mean(axis=1).copy()
tm.cool_down(T_room=20.0, n_steps=3)
tm.remove_from_plate(verbose=False)
s = md.stress().mean(axis=1)
print(f"{tm.n_layers} layers")
print(f"sigma_xx on the plate after the build: {s_build[:, 0].min():.0f} .. {s_build[:, 0].max():.0f} MPa")
print(f"after cooling and removal:              {s[:, 0].min():.0f} .. {s[:, 0].max():.0f} MPa")
```

Output (from the test run):

```text
cool-down to 20.0 C: vm_max=936  newton=[1] 0.9s
4 layers
sigma_xx on the plate after the build: -430 .. 905 MPa
after cooling and removal:              -560 .. 599 MPa
```

Each new layer shrinks on cooling while held by the colder layers below: tension at the top
surface, compression below, the classic AM residual-stress profile.

## Check yourself
* More layers or a colder plate raise the stresses.
* The full-scale application is the AM-Bench IN625 bridge:
  `benchmarks/literature/nist_ambench2018_tm.py`.

## Next
[Tutorial 12: export and benchmarking](12_export_benchmark.md).
